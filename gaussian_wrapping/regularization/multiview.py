from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import random
import torch
from arguments import PipelineParams, ModelParams, OptimizationParams
from scene import Scene
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.camera_utils import get_cameras_spatial_extent
from utils.graphics_utils import fov2focal


def get_points_from_depth(
    fov_camera: Camera, depth: torch.Tensor, scale: int = 1
) -> torch.Tensor:
    st = int(max(int(scale/2)-1,0))
    depth_view = depth.squeeze()[st::scale,st::scale]
    rays_d = get_rays(fov_camera, scale=scale)
    depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
    pts = (rays_d * depth_view[..., None]).reshape(-1,3)
    R = torch.tensor(fov_camera.R).float().cuda()
    T = torch.tensor(fov_camera.T).float().cuda()
    pts = (pts - T) @ R.transpose(-1,-2)
    return pts


def get_points_depth_in_depth_map(
    fov_camera: Camera, depth: torch.Tensor, points_in_camera_space: torch.Tensor, scale=1
) -> Tuple[torch.Tensor, torch.Tensor]:
    Fx = fov2focal(fov_camera.FoVx, fov_camera.image_width)
    Fy = fov2focal(fov_camera.FoVy, fov_camera.image_height)
    Cx = 0.5 * fov_camera.image_width
    Cy = 0.5 * fov_camera.image_height
    
    st = max(int(scale/2)-1,0)
    depth_view = depth[None,:,st::scale,st::scale]
    W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
    depth_view = depth_view[:H, :W]
    pts_projections = torch.stack(
                    [points_in_camera_space[:,0] * Fx / points_in_camera_space[:,2] + Cx,
                        points_in_camera_space[:,1] * Fy / points_in_camera_space[:,2] + Cy], -1).float()/scale
    mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
            (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)

    pts_projections[..., 0] /= ((W - 1) / 2)
    pts_projections[..., 1] /= ((H - 1) / 2)
    pts_projections -= 1
    pts_projections = pts_projections.view(1, -1, 1, 2)
    map_z = torch.nn.functional.grid_sample(input=depth_view,
                                            grid=pts_projections,
                                            mode='bilinear',
                                            padding_mode='border',
                                            align_corners=True
                                            )[0, :, :, 0]
    return map_z, mask


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
        T = torch.tensor(cur_cam.T).float().cuda()
        center_ray = torch.tensor([0.0,0.0,1.0]).float().cuda()
        center_ray = center_ray@R.transpose(-1,-2)
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


def get_gray_image(view: Camera, device: torch.device = "cuda") -> torch.Tensor:
    gt_image = view.original_image.to(device)
    return (0.299 * gt_image[0] + 0.587 * gt_image[1] + 0.114 * gt_image[2])[None]


def get_rays(view: Camera, scale=1.0):
    W, H = int(view.image_width/scale), int(view.image_height/scale)
    Fx = fov2focal(view.FoVx, view.image_width)
    Fy = fov2focal(view.FoVy, view.image_height)
    Cx = 0.5 * view.image_width
    Cy = 0.5 * view.image_height
    ix, iy = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy')
    rays_d = torch.stack(
                [(ix-Cx/scale) / Fx * scale,
                (iy-Cy/scale) / Fy * scale,
                torch.ones_like(ix)], -1).float().cuda()
    return rays_d


def get_k(view: Camera, scale=1.0):
    Fx = fov2focal(view.FoVx, view.image_width)
    Fy = fov2focal(view.FoVy, view.image_height)
    Cx = 0.5 * view.image_width
    Cy = 0.5 * view.image_height
    K = torch.tensor([[Fx / scale, 0, Cx / scale],
                    [0, Fy / scale, Cy / scale],
                    [0, 0, 1]]).cuda()
    return K


def get_inv_k(view: Camera, scale=1.0):
    Fx = fov2focal(view.FoVx, view.image_width)
    Fy = fov2focal(view.FoVy, view.image_height)
    Cx = 0.5 * view.image_width
    Cy = 0.5 * view.image_height
    K_T = torch.tensor([[scale/Fx, 0, -Cx/Fx],
                        [0, scale/Fy, -Cy/Fy],
                        [0, 0, 1]]).cuda()
    return K_T


def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing='xy')[::-1], dim=-1).view(1, -1, 2)


def patch_warp(H, uv):
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B,P,1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid


def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = torch.nn.functional.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = torch.nn.functional.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = torch.nn.functional.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = torch.nn.functional.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = torch.nn.functional.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask


def compute_multiview_losses(
    # dataset: ModelParams,
    scene: Scene,
    nearest_id: List[int],
    viewpoint_cam: Camera,
    view_depth: torch.Tensor,
    view_normal: torch.Tensor,
    gaussians: GaussianModel,
    render: Callable,
    pipe: PipelineParams,
    background: torch.Tensor,
    multi_view_args: Dict[str, Any],
    kernel_size: float = 0.0,
) -> torch.Tensor:
    """
    Computes the multiview regularization loss.

    Args:
        scene (Scene): The scene.
        nearest_id (List[int]): The nearest camera ids.
        viewpoint_cam (Camera): The viewpoint camera.
        view_depth (torch.Tensor): The view depth. Has shape (1, H, W).
        view_normal (torch.Tensor): The view normal. Has shape (3, H, W).
        gaussians (GaussianModel): The gaussians.
        render (Callable): The render function.
        pipe (PipelineParams): The pipeline parameters.
        background (torch.Tensor): The background color.
        kernel_size (float): The kernel size.
        multi_view_args (Dict[str, Any]): The multiview arguments.
    """
    gt_image_gray = get_gray_image(viewpoint_cam, device=view_depth.device)
    Fx = fov2focal(viewpoint_cam.FoVx, viewpoint_cam.image_width)
    Fy = fov2focal(viewpoint_cam.FoVy, viewpoint_cam.image_height)
    Cx = 0.5 * viewpoint_cam.image_width
    Cy = 0.5 * viewpoint_cam.image_height
    ncc_scale = 1.0
    
    nearest_cam = None if len(nearest_id) == 0 else scene.getTrainCameras()[random.sample(nearest_id, 1)[0]]
    use_virtul_cam = False
    # if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
    #     nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
    #     use_virtul_cam = True
    
    losses = {}
    losses["multiview_loss"] = torch.tensor(0.0, device=view_depth.device)
    if nearest_cam is not None:
        patch_size = multi_view_args["multi_view_patch_size"]
        sample_num = multi_view_args["multi_view_sample_num"]
        pixel_noise_th = multi_view_args["multi_view_pixel_noise_th"]
        total_patch_size = (patch_size * 2 + 1) ** 2
        ncc_weight = multi_view_args["multi_view_ncc_weight"]
        geo_weight = multi_view_args["multi_view_geo_weight"]
        
        # Compute geometry consistency mask and loss
        H, W = view_depth.squeeze().shape
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        pixels = torch.stack([ix, iy], dim=-1).float().to(view_depth.device)

        nearest_render_pkg = render(
            viewpoint_camera=nearest_cam, 
            pc=gaussians, 
            pipe=pipe, 
            bg_color=background, 
            kernel_size=kernel_size,
            require_depth=True,
        )

        pts = get_points_from_depth(viewpoint_cam, view_depth)
        pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
        map_z, d_mask = get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['expected_depth'], pts_in_nearest_cam)
        
        pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
        pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
        R = torch.tensor(nearest_cam.R).float().cuda()
        T = torch.tensor(nearest_cam.T).float().cuda()
        pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
        pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
        pts_projections = torch.stack(
                    [pts_in_view_cam[:,0] * Fx / pts_in_view_cam[:,2] + Cx,
                    pts_in_view_cam[:,1] * Fy / pts_in_view_cam[:,2] + Cy], -1).float()
        pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
        if not multi_view_args["wo_use_geo_occ_aware"]:
            d_mask = d_mask & (pixel_noise < pixel_noise_th)
            weights = (1.0 / torch.exp(pixel_noise)).detach()
            weights[~d_mask] = 0
        else:
            d_mask = d_mask
            weights = torch.ones_like(pixel_noise)
            weights[~d_mask] = 0

        if d_mask.sum() > 0:
            geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
            losses["geo_loss"] = geo_loss
            losses["multiview_loss"] = losses["multiview_loss"] + geo_loss
            
            if use_virtul_cam is False:
                with torch.no_grad():
                    ## sample mask
                    d_mask = d_mask.reshape(-1)
                    valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                    if d_mask.sum() > sample_num:
                        index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                        valid_indices = valid_indices[index]

                    weights = weights.reshape(-1)[valid_indices]
                    ## sample ref frame patch
                    pixels = pixels.reshape(-1,2)[valid_indices]
                    offsets = patch_offsets(patch_size, pixels.device)
                    ori_pixels_patch = pixels.reshape(-1, 1, 2) / ncc_scale + offsets.float()
                    
                    H, W = gt_image_gray.squeeze().shape
                    pixels_patch = ori_pixels_patch.clone()
                    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                    ref_gray_val = torch.nn.functional.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                    ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                    ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                    ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                ## compute Homography
                ref_local_n = view_normal.permute(1, 2, 0)  # (3, H, W) -> (H, W, 3)
                ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices]

                # ref_local_d = render_pkg['rendered_distance'].squeeze()
                rays_d = get_rays(viewpoint_cam)
                rendered_normal2 = view_normal.permute(1, 2, 0).reshape(-1,3)
                ref_local_d = view_depth.view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                ref_local_d = ref_local_d.reshape(*view_depth.shape)  # (1, H, W)

                ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                H_ref_to_neareast = ref_to_neareast_r[None] - \
                    torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                H_ref_to_neareast = torch.matmul(get_k(nearest_cam, ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                H_ref_to_neareast = H_ref_to_neareast @ get_inv_k(viewpoint_cam, ncc_scale)
                
                ## compute neareast frame patch
                grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                nearest_image_gray = get_gray_image(nearest_cam, device=view_depth.device)
                sampled_gray_val = torch.nn.functional.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                
                ## compute loss
                ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                mask = ncc_mask.reshape(-1)
                ncc = ncc.reshape(-1) * weights
                ncc = ncc[mask].squeeze()
                
                ncc_loss = ncc_weight * ncc.mean()
                losses["ncc_loss"] = ncc_loss
                losses["multiview_loss"] = losses["multiview_loss"] + ncc_loss
    
    return losses
