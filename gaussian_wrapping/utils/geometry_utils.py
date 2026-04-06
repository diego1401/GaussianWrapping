import math
import torch
from typing import List, Optional, Tuple
from utils.general_utils import build_rotation
from scene.cameras import Camera
from tqdm import tqdm


def vec_dot(
    x:torch.Tensor, 
    y:torch.Tensor=None, 
    keepdim:bool=False,
) -> torch.Tensor:
    """
    Dot product of two vectors.

    Args:
        x (torch.Tensor): Vector of shape (..., 3)
        y (torch.Tensor): Vector of shape (..., 3)

    Returns:
        torch.Tensor: Dot product of shape (..., 1) if keepdim is True, (..., ) otherwise
    """
    if y is None:
        y = x
    return (x * y).sum(dim=-1, keepdim=keepdim)


def get_gaussian_normals_from_view(view, gaussians, in_view_space=True):
    # Build rotation matrices of Gaussians
    gaussian_rots = build_rotation(gaussians._rotation)
    
    # Get the minimum scale index for each Gaussian
    gaussian_min_scale_idx = gaussians.get_scaling_with_3D_filter.min(dim=-1)[1][:, None, None].repeat(1, 3, 1)
    
    # Gather the normals as the shortest axis of the covariance matrices
    gaussian_normals = torch.gather(gaussian_rots, dim=2, index=gaussian_min_scale_idx).squeeze()
    
    # Flip the normals if they are pointing away from the camera
    gaussian_normals = gaussian_normals * torch.sign(
        (
            gaussian_normals * (view.camera_center[None] - gaussians.get_xyz)
        ).sum(dim=-1, keepdim=True)
    )
    
    if in_view_space:
        gaussian_normals = (gaussian_normals @ view.world_view_transform[:3,:3])
    
    return gaussian_normals


def transform_points_world_to_view(
    points:torch.Tensor,
    cameras:List[Camera],
    use_p3d_convention:bool=False,
):
    """Transform points from world space to view space.

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
        use_p3d_convention (bool, optional): Defaults to False.
        
    Returns:
        torch.Tensor: Has shape (n_cameras, N, 3).
    """
    world_view_transforms = torch.stack([camera.world_view_transform for camera in cameras], dim=0)  # (n_cameras, 4, 4)
    
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    view_points = (points_h @ world_view_transforms)[..., :3]  # (n_cameras, N, 3)
    if use_p3d_convention:
        factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
        view_points = factors * view_points  # (n_cameras, N, 3)
    return view_points


def transform_points_view_to_world(
    points:torch.Tensor,
    cameras:List[Camera],
    use_p3d_convention:bool=False,
):
    """Transform points from view space to world space.

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
        use_p3d_convention (bool, optional): Defaults to False.
        
    Returns:
        torch.Tensor: Has shape (n_cameras, N, 3).
    """
    view_world_transforms = torch.stack([camera.world_view_transform.inverse() for camera in cameras], dim=0)  # (n_cameras, 4, 4)
    
    if use_p3d_convention:
        factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
        points = factors * points  # (n_cameras, N, 3)
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    world_points = (points_h @ view_world_transforms)[..., :3]  # (n_cameras, N, 3)
    return world_points

        
def transform_points_to_pixel_space(
        points:torch.Tensor,
        cameras:List[Camera],
        points_are_already_in_view_space:bool=False,
        use_p3d_convention:bool=False,
        znear:float=1e-6,
        keep_float:bool=False,
    ):
        """Transform points from world space (3 coordinates) to pixel space (2 coordinates).

        Args:
            points (torch.Tensor): Should have shape (n_cameras, N, 3).
            cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
            points_are_already_in_view_space (bool, optional): Defaults to False.
            use_p3d_convention (bool, optional): Defaults to False.
            znear (float, optional): Defaults to 1e-6.

        Returns:
            torch.Tensor: Has shape (n_cameras, N, 2). 
                In pixel space, (0, 0) is the center of the left-top pixel,
                and (W-1, H-1) is the center of the right-bottom pixel.
        """
        if points_are_already_in_view_space:
            full_proj_transforms = torch.stack([camera.projection_matrix for camera in cameras])  # (n_depth, 4, 4)
            if use_p3d_convention:
                points = torch.tensor([[[-1, -1, 1]]], device=points.device) * points
        else:
            full_proj_transforms = torch.stack([camera.full_proj_transform for camera in cameras])  # (n_cameras, 4, 4)
        
        points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
        proj_points = points_h @ full_proj_transforms  # (n_cameras, N, 4)
        proj_points = proj_points[..., :2] / proj_points[..., 3:4].clamp_min(znear)  # (n_cameras, N, 2)
        # proj_points is currently in a normalized space where 
        # (-1, -1) is the left-top corner of the left-top pixel,
        # and (1, 1) is the right-bottom corner of the right-bottom pixel.

        # For converting to pixel space, we need to scale and shift the normalized coordinates
        # such that (-1/2, -1/2) is the left-top corner of the left-top pixel, 
        # and (H-1/2, W-1/2) is the right-bottom corner of the right-bottom pixel.
        
        height, width = cameras[0].image_height, cameras[0].image_width
        image_size = torch.tensor([[width, height]], device=points.device)
        
        # proj_points = (1. + proj_points) * image_size / 2
        proj_points = (1. + proj_points) / 2 * image_size - 1./2.

        if keep_float:
            return proj_points        
        else:
            return torch.round(proj_points).long()
        
        
def get_pixel_coordinates(
    points: torch.Tensor, 
    camera: Camera, 
    znear: Optional[float] = None,
    zfar: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the pixel coordinates of the points in the camera view.

    Args:
        points (torch.Tensor): The points to project. Has shape (N, 3).
        camera (Camera): The camera to project the points to.
        znear (Optional[float], optional): The near clipping plane. Defaults to None.
        zfar (Optional[float], optional): The far clipping plane. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - The pixel coordinates of the points, with shape (N, 2).
            - The integer pixel coordinates of the points, with shape (N, 2).
            - The depth of the points, with shape (N,).
            - A mask of the points that are inside the view frustum and depth range, with shape (N,).
    """
    # Transform points to view space
    view_points = transform_points_world_to_view(
        points=points.view(1, -1, 3),
        cameras=[camera],
    )[0]  # (N, 3)

    # Project points to pixel space
    pix_points = transform_points_to_pixel_space(
        points=view_points.view(1, -1, 3),
        cameras=[camera],
        points_are_already_in_view_space=True,
        keep_float=True,
    )[0]  # (N, 2)
    int_pix_points = pix_points.round().to(torch.int32)  # (N, 2)
    pix_x, pix_y, depth = pix_points[..., 0], pix_points[..., 1], view_points[..., 2]

    # Remove points outside view frustum and outside depth range
    valid_mask = (
        (pix_x >= 0) & (pix_x <= camera.image_width-1) 
        & (pix_y >= 0) & (pix_y <= camera.image_height-1) 
        & (depth > (camera.znear if znear is None else znear)) 
        & (depth < (camera.zfar if zfar is None else zfar))
    )  # (N,)
    
    return pix_points, int_pix_points, depth, valid_mask


# the following functions are adopted from RaDe-GS: 
def depths_to_points(view, depthmap1, depthmap2=None):
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins_inv = torch.tensor(
        [[1/fx, 0.,-W/(2 * fx)],
        [0., 1/fy, -H/(2 * fy),],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).float().cuda()
    rays_d = intrins_inv @ points
    points1 = depthmap1.reshape(1,-1) * rays_d
    if depthmap2 is not None:
        points2 = depthmap2.reshape(1,-1) * rays_d
        return points1.reshape(3,H,W), points2.reshape(3,H,W)
    else:
        return points1.reshape(3,H,W)


def point_to_normal(view, points1, points2=None):
    points = (
        points1[None] if points2 is None 
        else torch.stack([points1, points2],dim=0)
    )
    output = torch.zeros_like(points)
    dx = points[...,2:, 1:-1] - points[...,:-2, 1:-1]
    dy = points[...,1:-1, 2:] - points[...,1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=1), dim=1)
    output[...,1:-1, 1:-1] = normal_map
    return (
        output[0] if points2 is None 
        else output
    )


def depth_to_normal(view, depth1, depth2=None):
    points = depths_to_points(view, depth1, depth2)
    points = points[None] if depth2 is None else points
    return point_to_normal(view, *points)


def depth_to_normal_with_mask(view, depth:torch.Tensor):
    W, H = view.image_width, view.image_height
    x = (torch.arange(W, device="cuda", dtype=torch.float32) - view.Cx) / view.Fx
    y = (torch.arange(H, device="cuda", dtype=torch.float32) - view.Cy) / view.Fy
    points = torch.cat([depth * x[None, None], depth * y[None, :, None], depth], dim=0)
    dy = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    dx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dy, dx, dim=0), dim=0)
    output = torch.nn.functional.pad(normal_map, (1, 1, 1, 1))

    valid_depths = depth > 0
    valid_depths = (
        valid_depths[:, 2:, 1:-1] & valid_depths[:, :-2, 1:-1] & valid_depths[:, 1:-1, 2:] & valid_depths[:, 1:-1, :-2] & valid_depths[:, 1:-1, 1:-1]
    )
    valid_points = torch.zeros_like(depth, dtype=torch.bool)
    valid_points[:, 1:-1, 1:-1] = valid_depths
    return output, valid_points


def is_in_view_frustum(
    points:torch.Tensor,
    camera:Camera,
    znear:float=None,
) -> torch.Tensor:
    """_summary_

    Args:
        points (torch.Tensor): Tensor with shape (N, 3)
        camera (Camera): _description_
    """
    H, W = camera.image_height, camera.image_width
    
    _znear = camera.znear if znear is None else znear
    
    view_points = transform_points_world_to_view(
        points.view(1, -1, 3),
        cameras=[camera],
    )[0]  # (N, 3)
    
    pix_pts = transform_points_to_pixel_space(
        view_points.view(1, -1, 3),
        points_are_already_in_view_space=True,
        cameras=[camera],
    )[0]  # (N, 2)
    
    pix_x, pix_y, pix_z = pix_pts[..., 0], pix_pts[..., 1], view_points[..., 2]
    
    valid_mask = (
        (pix_x >= 0) & (pix_x <= W-1) 
        & (pix_y >= 0) & (pix_y <= H-1) 
        & (pix_z > _znear) & (pix_z < camera.zfar)
    )  # (N,)
    
    return valid_mask


def unflatten_voronoi_features(voronoi_features:torch.Tensor, n_voronoi_per_gaussians:int=9):
    """Unflatten the voronoi features into a 3D tensor with shape (n_gaussians, n_voronoi_per_gaussians, *voronoi_features.shape[1:]).

    Args:
        voronoi_features (torch.Tensor): Tensor with shape (n_gaussians * n_voronoi_per_gaussians, *voronoi_features.shape[1:]).

    Returns:
        torch.Tensor: Tensor with shape (n_gaussians, n_voronoi_per_gaussians, *voronoi_features.shape[1:]).
    """
    n_gaussians = len(voronoi_features) // n_voronoi_per_gaussians
    return torch.cat(
        [
            voronoi_features[:-n_gaussians].reshape(n_gaussians, n_voronoi_per_gaussians-1, *voronoi_features.shape[1:]),
            voronoi_features[-n_gaussians:].reshape(n_gaussians, 1, *voronoi_features.shape[1:])
        ],
        dim=1
    ).reshape(n_gaussians, n_voronoi_per_gaussians, *voronoi_features.shape[1:])
    
    
def flatten_voronoi_features(voronoi_features:torch.Tensor, n_voronoi_per_gaussians:int=9):
    return torch.cat(
        [
            voronoi_features[:, :n_voronoi_per_gaussians-1].reshape(-1, *voronoi_features.shape[2:]),
            voronoi_features[:, n_voronoi_per_gaussians-1:].reshape(-1, *voronoi_features.shape[2:])
        ],
        dim=0
    )


def identify_out_of_field_points(
    points:torch.Tensor,
    views:List[Camera],
):  
    n_points = points.shape[0]
    out_of_field_mask = torch.ones_like(points[:, 0], dtype=torch.bool, device=points.device)
    
    for camera in tqdm(views, desc="Identifying out of field points"):
        H, W = camera.image_height, camera.image_width

        # Transform points to view space
        view_points = transform_points_world_to_view(
            points=points.view(1, n_points, 3),
            cameras=[camera],
        )[0]  # (N, 3)
        
        # Project points to pixel space
        pix_points = transform_points_to_pixel_space(
            points=view_points.view(1, n_points, 3),
            cameras=[camera],
            points_are_already_in_view_space=True,
            keep_float=True,
        )[0]  # (N, 2)
        int_pix_points = pix_points.round().long()  # (N, 2)
        pix_x, pix_y, pix_z = pix_points[..., 0], pix_points[..., 1], view_points[..., 2]  # (N,)
        int_pix_x, int_pix_y = int_pix_points[..., 0], int_pix_points[..., 1]  # (N,)
        
        # Remove points outside view frustum and outside depth range
        valid_mask = (
            (pix_x >= 0) & (pix_x <= W-1) 
            & (pix_y >= 0) & (pix_y <= H-1) 
        )  # (N,)
        
        out_of_field_mask[valid_mask] = False
        
    return out_of_field_mask


def sample_depth_normal(
    fov_camera: Camera, 
    depth: torch.Tensor, 
    normal: torch.Tensor, 
    points: torch.Tensor, 
    znear=0.1, 
    interpolation='bilinear',
):
    
    points_in_camera_space = transform_points_world_to_view(
        points=points.unsqueeze(0),
        cameras=[fov_camera],
    ).squeeze(0)
    
    points_shape = points_in_camera_space.shape
    points_in_camera_space = points_in_camera_space.view(-1, 3)
    
    # Compute point projections
    _, H, W = depth.shape
    full_proj_transforms = torch.stack([camera.projection_matrix for camera in [fov_camera]])  # (1, 4, 4)
    
    points_h = torch.cat([points_in_camera_space, torch.ones_like(points_in_camera_space[..., :1])], dim=-1)  # (N, 4)
    points_h = points_h.unsqueeze(0)  # (1, N, 4)
    pts_projections = points_h @ full_proj_transforms  # (1, N, 4)
    pts_projections = pts_projections[..., :2] / pts_projections[..., 3:4].clamp_min(znear)  # (1, N, 2)
    pts_projections = pts_projections.squeeze(0)  # (N, 2)
    
    # Compute frustum mask
    mask = (
        (pts_projections[:, 0] > -1) 
        & (pts_projections[:, 0] < 1) 
        & (pts_projections[:, 1] > -1) 
        & (pts_projections[:, 1] < 1) 
        & (points_in_camera_space[:,2] > znear)
    )
    
    # Sample depth
    pts_projections = pts_projections.view(1, -1, 1, 2)
    map_z = torch.nn.functional.grid_sample(input=depth[None, :, :, :],
                                            grid=pts_projections,
                                            mode=interpolation,
                                            padding_mode='border',
                                            align_corners=True
                                            )[0, :, :, 0]
    map_z = map_z.transpose(-1, -2)  # (N, 1)
    
    # Sample normal
    map_normal = torch.nn.functional.grid_sample(input=normal[None, :, :, :],
                                                grid=pts_projections,
                                                mode=interpolation,
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
    map_normal = map_normal.transpose(-1, -2)  # (N, 3)
    
    # Sample mask
    map_z_nearest = torch.nn.functional.grid_sample(input=depth[None, :, :, :],
                                            grid=pts_projections,
                                            mode='nearest',
                                            padding_mode='border',
                                            align_corners=True
                                            )[0, :, :, 0]
    map_z_nearest = map_z_nearest.transpose(-1, -2)  # (N, 1)
    
    # Update mask
    mask = mask & (map_z_nearest[..., -1] > 0.)  # (N,)
    
    # Compute depth points
    depth_points = torch.where(
        mask[..., None],  # (N, 1)
        points_in_camera_space / points_in_camera_space[..., 2:3] * map_z,  # (N, 3)
        torch.zeros_like(points_in_camera_space)  # (N, 3)
    )
    
    normals = torch.where(
        mask[..., None],  # (N, 1)
        map_normal,  # (N, 3)
        torch.zeros_like(map_normal)  # (N, 3)
    )
    
    sampled_pkg = {
        "sampled_depth": depth_points.view(points_shape),  # (..., 3)
        "sampled_normal": normals.view(points_shape),  # (..., 3)
        "valid": mask.view(points_shape[:-1]),  # (...,)
    }
    return sampled_pkg
