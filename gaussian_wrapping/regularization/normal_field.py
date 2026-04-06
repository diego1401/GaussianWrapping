from typing import Dict, Any, Tuple, Optional, List, Callable
import numpy as np
import torch
from argparse import Namespace
from arguments import PipelineParams
from scene import Scene
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render_simp
from regularization.sdf.depth_fusion import (
    transform_points_world_to_view, 
    transform_points_to_pixel_space, 
    get_interpolated_value_from_pixel_coordinates,
)
from utils.general_utils import build_scaling_rotation


def get_gaussian_std_in_direction(
    directions: torch.Tensor,
    gaussians: Optional[GaussianModel]=None,
    gaussian_scaling: Optional[torch.Tensor]=None,
    gaussian_rotation: Optional[torch.Tensor]=None,
    normalize_directions: bool = True,
) -> torch.Tensor:
    """
    Get the standard deviation of the Gaussian in the given directions.
    
    If gaussians is provided, the scaling and rotation are extracted from the Gaussians.
    If gaussian_scaling and gaussian_rotation are provided, they are used instead of the Gaussians.
    
    Args:
        directions (torch.Tensor): A vector of shape (N_gaussians, n_directions, 3).
        gaussians (GaussianModel): The Gaussian model, with N_gaussians Gaussians.
        gaussian_scaling (torch.Tensor): The scaling of the Gaussians. Has shape (N_gaussians, 3).
        gaussian_rotation (torch.Tensor): The rotation of the Gaussians. Has shape (N_gaussians, 3, 3).
        normalize_directions (bool): Whether to normalize the directions.

    Returns:
        torch.Tensor: The standard deviation of the Gaussian in the direction of the given vector, of shape (N_gaussians, n_directions).
    """
    assert (
        gaussians is not None
        or (gaussian_scaling is not None and gaussian_rotation is not None)
    )
    
    if False:
        # Get transposed invscaled rotation S^-1 @ R^T
        if gaussian_scaling is None:
            gaussian_scaling = gaussians.get_scaling_with_3D_filter.detach()
        if gaussian_rotation is None:
            gaussian_rotation = gaussians._rotation.detach()
        transposed_invscaled_rotation = build_scaling_rotation(
            s=1. / gaussian_scaling,  # (N_gaussians, 3)
            r=gaussian_rotation,  # (N_gaussians, 3, 3)
        ).transpose(-1, -2)  # (N_gaussians, 3, 3)
        
        # Normalize directions
        if normalize_directions:
            directions_to_use = torch.nn.functional.normalize(directions, dim=-1)
        else:
            directions_to_use = directions
        
        # Compute S^-1 @ R^T @ directions
        invscaled_directions = torch.bmm(
            transposed_invscaled_rotation,  # (N_gaussians, 3, 3)
            directions_to_use.permute(0, 2, 1),  # (N_gaussians, 3, n_directions)
        ).permute(0, 2, 1)  # (N_gaussians, n_directions, 3)
        
        # Std is 1. / ||S^-1 @ R^T @ directions||
        direction_stds = 1. / invscaled_directions.norm(dim=-1)  # (N_gaussians, n_directions)
        return direction_stds  # (N_gaussians, n_directions)
    else:
        # Get transposed scaled rotation
        if gaussian_scaling is None:
            gaussian_scaling = gaussians.get_scaling_with_3D_filter.detach()
        if gaussian_rotation is None:
            gaussian_rotation = gaussians._rotation.detach()
        transposed_scaled_rotation = build_scaling_rotation(
            s=gaussian_scaling,  # (N_gaussians, 3)
            r=gaussian_rotation,  # (N_gaussians, 3, 3)
        ).transpose(-1, -2)  # (N_gaussians, 3, 3)
        
        if normalize_directions:
            directions_to_use = torch.nn.functional.normalize(directions, dim=-1)
        else:
            directions_to_use = directions
        
        scaled_directions = torch.bmm(
            transposed_scaled_rotation,  # (N_gaussians, 3, 3)
            directions_to_use.permute(0, 2, 1),  # (N_gaussians, 3, n_directions)
        ).permute(0, 2, 1)  # (N_gaussians, n_directions, 3)
        
        direction_stds = scaled_directions.norm(dim=-1)  # (N_gaussians, n_directions)
        return direction_stds  # (N_gaussians, n_directions)


# FIXME: Should we also add the center point of the Gaussian to the pivots?
def get_pivots_from_normals(
    gaussians: GaussianModel,
    normals: Optional[torch.Tensor]=None,
    std_factor: float = 3.0,
    return_scales: bool = True,
) -> torch.Tensor:
    
    # If normals are not provided, convert features to normals
    if normals is None:
        normals = gaussians.convert_features_to_normals()
    
    # Normalize the normals
    normals = torch.nn.functional.normalize(normals, dim=-1)  # (N_gaussians, 3)
    
    # Get the standard deviation of the Gaussian in the direction of the normal
    normal_stds = get_gaussian_std_in_direction(
        directions=normals.unsqueeze(1),  # (N_gaussians, 1, 3)
        gaussians=gaussians, 
        normalize_directions=False,
    )  # (N_gaussians, 1)

    # Get the pivots from the normals
    pivots = gaussians.get_xyz.unsqueeze(1).repeat(1, 2, 1)  # (N_gaussians, 2, 3)
    pivots[:, 0, :] = pivots[:, 0, :] + std_factor * normal_stds * normals
    pivots[:, 1, :] = pivots[:, 1, :] - std_factor * normal_stds * normals
    
    # Get pivot scales
    if return_scales:
        pivot_scales = 3. * gaussians.get_scaling_with_3D_filter.detach().max(dim=-1).values.unsqueeze(1).repeat(1, 2, 1)  # (N_gaussians, 2, 1)
        return pivots, pivot_scales
    
    return pivots  # (N_gaussians, 2, 3)


def get_signed_distance_to_depthmap(
    depth: torch.Tensor, 
    points: torch.Tensor,
    camera: Camera,
    interpolate_depth:bool=True,
    interpolation_mode:str='bilinear',
    padding_mode:str='border',
    align_corners:bool=True,
    znear:float=None,
    zfar:float=None,
):
    """
    Interpolate the depthmap at the points.

    Args:
        depthmap (torch.Tensor): The depthmap to interpolate. Has shape (H, W), (1, H, W), or (H, W, 1).
        points (torch.Tensor): The points to interpolate at. Has shape (N, 3)
        camera (GSCamera): Camera.
        interpolate_depth (bool): Whether to interpolate the depth.
        interpolation_mode (str): Interpolation mode.
        padding_mode (str): Padding mode for interpolation.
        align_corners (bool): Whether to align corners for interpolation.
        
    Returns:
        torch.Tensor: The signed distance to the depthmap, of shape (N, 1).
        torch.Tensor: The valid mask, of shape (N,).
    """
    
    # Reshape image and depth to (H, W, 3) and (H, W) respectively
    depth = depth.squeeze()
    H, W = depth.shape
    n_points = points.shape[0]
    
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
    pix_x, pix_y, pix_z = pix_points[..., 0], pix_points[..., 1], view_points[..., 2]
    int_pix_x, int_pix_y = int_pix_points[..., 0], int_pix_points[..., 1]
    
    # Remove points outside view frustum and outside depth range
    valid_mask = (
        (pix_x >= 0) & (pix_x <= W-1) 
        & (pix_y >= 0) & (pix_y <= H-1) 
        & (pix_z > (camera.znear if znear is None else znear)) 
        & (pix_z < (camera.zfar if zfar is None else zfar))
    )  # (N,)
    
    if valid_mask.sum() > 0:
        # Get depth and image values at pixel locations
        packed_values = torch.cat(
            [
                -torch.ones(len(valid_mask), 1, device=points.device),  # Depth values
            ], 
            dim=-1
        )  # (N, 1)
        if interpolate_depth:
            packed_values[valid_mask] = get_interpolated_value_from_pixel_coordinates(
                value_img=depth.unsqueeze(-1),  # (H, W, 1)
                pix_coords=pix_points[valid_mask],
                interpolation_mode=interpolation_mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )  # (N_valid, 1)
        else:
            packed_values[valid_mask] = depth.unsqueeze(-1)[int_pix_y[valid_mask], int_pix_x[valid_mask]]  # (N_valid, 1)
        depth_values = packed_values[..., :1]  # (N, 1)
        valid_mask = valid_mask & (depth_values[..., 0] > 0.)  # (N,)
        
        # Compute distance
        sdf = (depth_values - pix_z.unsqueeze(-1))  # (N, 1)

    return sdf, valid_mask