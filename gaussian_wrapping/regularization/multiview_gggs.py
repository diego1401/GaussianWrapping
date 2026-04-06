'''
We take the losses implemented by the paper Geometry Grounded Gaussian Splatting (GGGS), and slightly adapt it to use our rasterizer
https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting
'''
import os
from typing import Tuple, List, Dict, Any, Callable
import numpy as np
import torch
import torch.nn.functional as F
import warp_patch_ncc
from arguments import PipelineParams
from gaussian_renderer.ours import sample_depth_with_ours
from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.camera_utils import get_cameras_spatial_extent


def compute_nearest_cameras(
    train_cameras: List[Camera],
    multi_view_max_angle: float = 30.0,
    multi_view_min_dis_relative: float = 0.002,  # = 0.01 / 5.0
    multi_view_max_dis_relative: float = 0.3,  # = 1.5 / 5.0
    multi_view_num: int = 8,
) -> Dict[int, Dict[str, List[int]]]:
    
    # Get the spatial extent of the cameras and compute absolute thresholds
    scene_radius = get_cameras_spatial_extent(train_cameras)['radius'].item()
    multi_view_min_dis = scene_radius * multi_view_min_dis_relative
    multi_view_max_dis = scene_radius * multi_view_max_dis_relative
    
    # For each camera, get the world view transform, camera center, and center ray
    world_view_transforms = []
    camera_centers = []
    center_rays = []
    for id, cur_cam in enumerate(train_cameras):
        world_view_transforms.append(cur_cam.world_view_transform)
        camera_centers.append(cur_cam.camera_center)
        R = torch.tensor(cur_cam.R).float().cuda()
        center_ray = torch.tensor([0.0,0.0,1.0]).float().cuda()
        center_ray = center_ray @ R.transpose(-1,-2)
        center_rays.append(center_ray)
    
    # Stack
    world_view_transforms = torch.stack(world_view_transforms)
    camera_centers = torch.stack(camera_centers, dim=0)
    center_rays = torch.stack(center_rays, dim=0)
    center_rays = torch.nn.functional.normalize(center_rays, dim=-1)
    
    # Compute distances and angles between all cameras
    diss = torch.norm(camera_centers[:,None] - camera_centers[None], dim=-1).detach().cpu().numpy()
    tmp = torch.sum(center_rays[:,None] * center_rays[None], dim=-1)
    angles = torch.arccos(tmp) * 180 / np.pi
    angles = angles.detach().cpu().numpy()
    
    # Store the nearest cameras for each camera
    nearest_cameras = {}
    for id, cur_cam in enumerate(train_cameras):
        # Sort the potential neighbor cameras by angle and distance
        sorted_indices = np.lexsort((angles[id], diss[id]))
        # sorted_indices = np.lexsort((diss[id], angles[id]))
        
        # Filter the potential neighbor cameras by angle and distance
        mask = (angles[id][sorted_indices] < multi_view_max_angle) & \
            (diss[id][sorted_indices] > multi_view_min_dis) & \
            (diss[id][sorted_indices] < multi_view_max_dis)
        sorted_indices = sorted_indices[mask]
        
        # Get the actual number of neighbor cameras
        multi_view_num = min(multi_view_num, len(sorted_indices))
        
        # Update the nearest cameras for the current camera
        cur_nearest_id = []
        cur_nearest_names = []
        for index in sorted_indices[:multi_view_num]:
            cur_nearest_id.append(index)
            cur_nearest_names.append(train_cameras[index].image_name)
        nearest_cameras[id] = {
            'nearest_id': cur_nearest_id,
            'nearest_names': cur_nearest_names,
        }
    
    return nearest_cameras


def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device, dtype=torch.float32)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing="xy")[::-1], dim=-1).view(1, -1, 2)


def sample_depth(
    fov_camera: Camera, depth: torch.Tensor, points_in_camera_space: torch.Tensor, znear=0.1, interpolation='bilinear',
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # FIXME: Problem in this function. A shift of 0.5 is needed in the pixel space. See our fonction in geometry_utils.py
    
    points_shape = points_in_camera_space.shape
    points_in_camera_space = points_in_camera_space.view(-1, 3)

    Fx = fov_camera.Fx
    Fy = fov_camera.Fy
    Cx = fov_camera.Cx
    Cy = fov_camera.Cy
    
    # Compute point projections
    _, H, W = depth.shape
    pts_projections = torch.stack(
        [
            points_in_camera_space[:,0] * Fx / points_in_camera_space[:,2] + Cx,
            points_in_camera_space[:,1] * Fy / points_in_camera_space[:,2] + Cy
        ],
        -1
    ).float()
    
    # Compute frustum mask
    mask = (
        (pts_projections[:, 0] > 0) 
        & (pts_projections[:, 0] < W) 
        & (pts_projections[:, 1] > 0) 
        & (pts_projections[:, 1] < H) 
        & (points_in_camera_space[:,2] > znear)
    )

    # Normalize point projections
    pts_projections[..., 0] /= ((W - 1) / 2)
    pts_projections[..., 1] /= ((H - 1) / 2)
    pts_projections -= 1
    
    # Sample depth
    pts_projections = pts_projections.view(1, -1, 1, 2)
    map_z = torch.nn.functional.grid_sample(input=depth[None, :, :, :],
                                            grid=pts_projections,
                                            mode=interpolation,
                                            padding_mode='border',
                                            align_corners=True
                                            )[0, :, :, 0]
    map_z = map_z.transpose(-1, -2)  # (N, 1)
    
    # Update mask
    mask = mask & (map_z[..., -1] > 0.)  # (N,)
    
    # Compute depth points
    depth_points = torch.where(
        mask[..., None],  # (N, 1)
        points_in_camera_space / points_in_camera_space[..., 2:3] * map_z,  # (N, 3)
        torch.zeros_like(points_in_camera_space)  # (N, 3)
    )
    
    sampled_pkg = {
        "sampled_depth": depth_points.view(points_shape),  # (..., 3)
        "inside": mask.view(points_shape[:-1]),  # (...,)
    }
    return sampled_pkg


class PatchMatch:
    def __init__(self, patch_size, pixel_noise_th, kernel_size, pipe, debug=False, model_path=None):
        self.patch_size = patch_size
        self.total_patch_size = (patch_size * 2 + 1) ** 2
        self.pixel_noise_th = pixel_noise_th
        self.offsets = patch_offsets(patch_size, device="cuda") * 0.5
        self.offsets.requires_grad_(False)
        self.kernel_size = kernel_size
        self.pipe = pipe
        self.debug = debug
        self.model_path = model_path
        if debug:
            os.makedirs(os.path.join(model_path, "debug"), exist_ok=True)

    def __call__(
        self, 
        viewpoint_cam: Camera, 
        nearest_cam: Camera, 
        render_pkg: dict, 
        nearest_render_pkg: dict,
        depth_ratio=1.0,
        znear=0.1,
        interpolation='bilinear',
        return_masks=False,
    ):
        if nearest_cam is None:
            return torch.tensor([0], dtype=torch.float32, device="cuda"), torch.tensor([0], dtype=torch.float32, device="cuda")
        
        H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
        ## compute geometry consistency mask
        with torch.no_grad():
            nearest_R = torch.from_numpy(nearest_cam.R.astype(np.float32)).cuda()
            nearest_T = torch.from_numpy(nearest_cam.T.astype(np.float32)).cuda()
            ix = (torch.arange(W, device="cuda", dtype=torch.float32) - viewpoint_cam.Cx) / viewpoint_cam.Fx
            iy = (torch.arange(H, device="cuda", dtype=torch.float32) - viewpoint_cam.Cy) / viewpoint_cam.Fy
            view_to_nearest_T = (
                -viewpoint_cam.world_view_transform[:3, :3].T @ nearest_R @ nearest_T + viewpoint_cam.world_view_transform[3, :3]
            )
            nearest_to_view_R = nearest_R.transpose(1, 0) @ viewpoint_cam.world_view_transform[:3, :3]

        # Points in view space
        depth_reshape = render_pkg["median_depth"].squeeze().unsqueeze(-1)
        pts = torch.cat([depth_reshape * ix[None, :, None], depth_reshape * iy[:, None, None], depth_reshape], dim=-1)

        # # Points in world space
        # R = torch.from_numpy(viewpoint_cam.R.astype(np.float32)).to(pts.device)
        # T = torch.from_numpy(viewpoint_cam.T.astype(np.float32)).to(pts.device)
        # pts = (pts - T) @ R.T
        
        # Points in nearest view space
        pts = (pts - view_to_nearest_T) @ nearest_to_view_R.T
        
        # Compute nearest depth
        nearest_depth = (
            (1. - depth_ratio) * nearest_render_pkg['expected_depth'] 
            + depth_ratio * nearest_render_pkg['median_depth']
        )
        
        # Sample points in nearest depth
        sampled_pkg = sample_depth(
            fov_camera=nearest_cam, 
            depth=nearest_depth, 
            points_in_camera_space=pts, 
            znear=znear,
            interpolation=interpolation,
        )
        pts_in_nearest_cam = sampled_pkg["sampled_depth"]

        pts_in_view_cam = view_to_nearest_T + pts_in_nearest_cam @ nearest_to_view_R
        pts_projections = pts_in_view_cam[..., :2] / torch.clamp_min(pts_in_view_cam[..., 2:], 1e-7)
        pts_projections = torch.addcmul(
            pts_projections.new_tensor([viewpoint_cam.Cx, viewpoint_cam.Cy]),
            pts_projections.new_tensor([viewpoint_cam.Fx, viewpoint_cam.Fy]),
            pts_projections,
        )

        ix, iy = torch.meshgrid(
            torch.arange(W, device="cuda", dtype=torch.int32),
            torch.arange(H, device="cuda", dtype=torch.int32),
            indexing="xy",
        )
        pixels = torch.stack([ix, iy], dim=-1)
        pixel_f = pixels.type(torch.float32).requires_grad_(False)
        pixel_noise = torch.pairwise_distance(pts_projections, pixel_f)

        with torch.no_grad():
            d_mask = (
                sampled_pkg["inside"]
                & (pts_in_nearest_cam[..., -1] > 0.2)  # FIXME: Makes this threshold relative to the scene radius
                & (pts_in_view_cam[..., -1] > 0.2)  # FIXME: Makes this threshold relative to the scene radius
                & (pixel_noise < self.pixel_noise_th)
                & (render_pkg["median_depth"].squeeze() > 0)
            )
            weights = torch.exp(-pixel_noise)
            weights[~d_mask] = 0
        
        if return_masks:
            d_mask_img = d_mask.clone()
            weight_img = weights.clone()
        
        # Compute NCC for warped patches
        if not d_mask.any():
            return torch.tensor([0], dtype=torch.float32, device="cuda"), torch.tensor([0], dtype=torch.float32, device="cuda")

        geo_loss = ((weights * pixel_noise)[d_mask]).mean()
        with torch.no_grad():
            d_mask = torch.flatten(d_mask)
            valid_indices = torch.argwhere(d_mask).squeeze(1)
            weights = torch.flatten(weights)[valid_indices]
            pixels = torch.index_select(pixels.view(-1, 2), dim=0, index=valid_indices)
            ref_to_neareast_r = nearest_cam.world_view_transform[:3, :3].transpose(-1, -2) @ viewpoint_cam.world_view_transform[:3, :3]
            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3, :3] + nearest_cam.world_view_transform[3, :3]

        depth_select = torch.index_select(render_pkg["median_depth"].view(-1), dim=0, index=valid_indices)
        normal_select = torch.index_select(render_pkg["normal"].view(3, -1), dim=1, index=valid_indices).transpose(1, 0)
        normal_select = F.normalize(normal_select, dim=-1)

        cc, valid_mask = warp_patch_ncc.warp_patch_ncc(
            depth_select,
            normal_select,
            pixels,
            ref_to_neareast_r.T,
            ref_to_neareast_t,
            viewpoint_cam.gray_image.to("cuda").squeeze(),
            nearest_cam.gray_image.to("cuda").squeeze(),
            viewpoint_cam.Fx,
            viewpoint_cam.Fy,
            viewpoint_cam.Cx,
            viewpoint_cam.Cy,
            nearest_cam.Fx,
            nearest_cam.Fy,
            nearest_cam.Cx,
            nearest_cam.Cy,
            False,
        )
        ncc = torch.clamp(1 - cc, 0.0, 2.0)
        ncc_mask = (ncc < 0.9) & valid_mask

        ncc = ncc.squeeze() * weights
        ncc = ncc[ncc_mask.squeeze()]

        if ncc_mask.any():
            ncc_loss = ncc.mean()
        else:
            ncc_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
            
        if return_masks:
            return ncc_loss, geo_loss, weight_img, d_mask_img
        return ncc_loss, geo_loss


############################################


class PatchMatchFast:
    def __init__(self, patch_size, pixel_noise_th, kernel_size, pipe, debug=True, model_path=None):
        self.patch_size = patch_size
        self.total_patch_size = (patch_size * 2 + 1) ** 2
        self.pixel_noise_th = pixel_noise_th
        self.offsets = patch_offsets(patch_size, device="cuda") * 0.5
        self.offsets.requires_grad_(False)
        self.kernel_size = kernel_size
        self.pipe = pipe
        self.debug = debug
        self.model_path = model_path
        if debug:
            os.makedirs(os.path.join(model_path, "debug"), exist_ok=True)

    def __call__(self, gaussians: GaussianModel, render_pkg: dict, viewpoint_cam: Camera, nearest_cam: Camera, iteration=0, depth_normal=None, 
                 rasterizer="ours"):
        
        if nearest_cam is None:
            return torch.tensor([0], dtype=torch.float32, device="cuda"), torch.tensor([0], dtype=torch.float32, device="cuda")
        H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
        ## compute geometry consistency mask
        with torch.no_grad():
            nearest_R = torch.from_numpy(nearest_cam.R.astype(np.float32)).cuda()
            nearest_T = torch.from_numpy(nearest_cam.T.astype(np.float32)).cuda()
            ix = (torch.arange(W, device="cuda", dtype=torch.float32) - viewpoint_cam.Cx) / viewpoint_cam.Fx
            iy = (torch.arange(H, device="cuda", dtype=torch.float32) - viewpoint_cam.Cy) / viewpoint_cam.Fy
            view_to_nearest_T = (
                -viewpoint_cam.world_view_transform[:3, :3].T @ nearest_R @ nearest_T + viewpoint_cam.world_view_transform[3, :3]
            )
            nearest_to_view_R = nearest_R.transpose(1, 0) @ viewpoint_cam.world_view_transform[:3, :3]

        # pts = (rays_d * render_pkg["median_depth"].squeeze().unsqueeze(-1)).reshape(-1, 3)
        depth_reshape = render_pkg["median_depth"].squeeze().unsqueeze(-1)
        pts = torch.cat([depth_reshape * ix[None, :, None], depth_reshape * iy[:, None, None], depth_reshape], dim=-1)

        R = torch.from_numpy(viewpoint_cam.R.astype(np.float32)).cuda()
        T = torch.from_numpy(viewpoint_cam.T.astype(np.float32)).cuda()
        pts = (pts - T) @ R.T
        
        sample_depth_func = sample_depth_with_ours
        sampled_pkg = sample_depth_func(
            pts,
            nearest_cam,
            gaussians,
            self.pipe,
            self.kernel_size,
        )

        pts_in_nearest_cam = sampled_pkg["sampled_depth"]
        R = nearest_R
        T = nearest_T

        pts_in_view_cam = view_to_nearest_T + pts_in_nearest_cam @ nearest_to_view_R
        pts_projections = pts_in_view_cam[..., :2] / torch.clamp_min(pts_in_view_cam[..., 2:], 1e-7)
        pts_projections = torch.addcmul(
            pts_projections.new_tensor([viewpoint_cam.Cx, viewpoint_cam.Cy]),
            pts_projections.new_tensor([viewpoint_cam.Fx, viewpoint_cam.Fy]),
            pts_projections,
        )

        ix, iy = torch.meshgrid(
            torch.arange(W, device="cuda", dtype=torch.int32),
            torch.arange(H, device="cuda", dtype=torch.int32),
            indexing="xy",
        )
        pixels = torch.stack([ix, iy], dim=-1)
        pixel_f = pixels.type(torch.float32).requires_grad_(False)
        pixel_noise = torch.pairwise_distance(pts_projections, pixel_f)

        with torch.no_grad():
            d_mask = (
                sampled_pkg["inside"]
                & (pts_in_nearest_cam[..., -1] > 0.2)
                & (pts_in_view_cam[..., -1] > 0.2)
                & (pixel_noise < self.pixel_noise_th)
                & (render_pkg["median_depth"].squeeze() > 0)
            )
            weights = torch.exp(-pixel_noise)
            weights[~d_mask] = 0
        ##############################################

        # if iteration % 200 == 0 and self.debug:
        #     with torch.no_grad():
        #         gt_img_show = (viewpoint_cam.original_image.permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
        #         img_show = ((render_pkg["render"]).permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
        #         normal_show = (((render_pkg["normal"] + 1.0) * 0.5).permute(1, 2, 0).clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
        #         if depth_normal is None:
        #             depth_normal_show = (
        #                 (nearest_cam.original_image.permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
        #             )
        #         else:
        #             depth_normal_show = (((depth_normal + 1.0) * 0.5).permute(1, 2, 0).clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
        #         d_mask_show = (weights.float() * 255).detach().cpu().numpy().astype(np.uint8)
        #         d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
        #         depth = render_pkg["median_depth"].squeeze().detach().cpu().numpy()
        #         depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        #         depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        #         depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
        #         row0 = np.concatenate([gt_img_show, img_show, depth_normal_show], axis=1)
        #         row1 = np.concatenate([d_mask_show_color, depth_color, normal_show], axis=1)
        #         image_to_show = np.concatenate([row0, row1], axis=0)
        #         cv2.imwrite(os.path.join(self.model_path, "debug", "%05d" % iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)
        ################## Compute NCC for warped patches ##################
        if not d_mask.any():
            return torch.tensor([0], dtype=torch.float32, device="cuda"), torch.tensor([0], dtype=torch.float32, device="cuda")

        geo_loss = ((weights * pixel_noise)[d_mask]).mean()
        with torch.no_grad():
            d_mask = torch.flatten(d_mask)
            valid_indices = torch.argwhere(d_mask).squeeze(1)
            weights = torch.flatten(weights)[valid_indices]
            pixels = torch.index_select(pixels.view(-1, 2), dim=0, index=valid_indices)
            ref_to_neareast_r = nearest_cam.world_view_transform[:3, :3].transpose(-1, -2) @ viewpoint_cam.world_view_transform[:3, :3]
            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3, :3] + nearest_cam.world_view_transform[3, :3]

        depth_select = torch.index_select(render_pkg["median_depth"].view(-1), dim=0, index=valid_indices)
        normal_select = torch.index_select(render_pkg["normal"].view(3, -1), dim=1, index=valid_indices).transpose(1, 0)
        normal_select = F.normalize(normal_select, dim=-1)

        cc, valid_mask = warp_patch_ncc.warp_patch_ncc(
            depth_select,
            normal_select,
            pixels,
            ref_to_neareast_r.T,
            ref_to_neareast_t,
            viewpoint_cam.gray_image.to("cuda").squeeze(),
            nearest_cam.gray_image.to("cuda").squeeze(),
            viewpoint_cam.Fx,
            viewpoint_cam.Fy,
            viewpoint_cam.Cx,
            viewpoint_cam.Cy,
            nearest_cam.Fx,
            nearest_cam.Fy,
            nearest_cam.Cx,
            nearest_cam.Cy,
            False,
        )
        ncc = torch.clamp(1 - cc, 0.0, 2.0)
        ncc_mask = (ncc < 0.9) & valid_mask

        ncc = ncc.squeeze() * weights
        ncc = ncc[ncc_mask.squeeze()]

        if ncc_mask.any():
            ncc_loss = ncc.mean()
        else:
            ncc_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        return ncc_loss, geo_loss