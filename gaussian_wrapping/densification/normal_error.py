from typing import List, Callable
import torch
from arguments import PipelineParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import (
    render_depth,
    render_simp,
)
from utils.geometry_utils import (
    depth_to_normal, 
    depth_to_normal_with_mask
)


@torch.no_grad()
def compute_normal_error(
    gaussians: GaussianModel,
    cameras: List[Camera],
    render_func: Callable,
    pipe: PipelineParams,
    background: torch.Tensor = torch.zeros(3, device="cuda"),
    method: str ="area",  # "count" or "area" or "none"
    normal_to_use: str ="expected_depth",  # "rendered" or "median_depth" or "expected_depth"
    average_method_over_cameras: str ="all",  # "all" or "visible"
    mask_error_at_zero_depth: bool = True,
) -> torch.Tensor:
    """
    Compute, for each Gaussian, the average normal error between the Gaussian rendering and the Normal Field rendering.

    Args:
        gaussians (GaussianModel): The Gaussian model.
        cameras (List[Camera]): The cameras.
        render_func (Callable): The rendering function.
        pipe (PipelineParams): The pipeline parameters.
        background (torch.Tensor, optional): The background color. Defaults to torch.zeros(3, device="cuda").
        method (str, optional): The method to use for normalization. Defaults to "count".
        normal_to_use (str, optional): The normal to use for the error computation. Defaults to "median_depth".

    Raises:
        ValueError: If the method is not "count", "area" or "none".
        ValueError: If the normal_to_use is not "rendered", "median_depth" or "expected_depth".

    Returns:
        torch.Tensor: The average normal error between the Gaussian rendering and the Normal Field rendering. Has shape (N_gaussians,).
    """
    
    assert normal_to_use in ["rendered", "median_depth", "expected_depth"], "Invalid normal to use"
    assert method in ["count", "area", "none"], "Invalid method"
    assert average_method_over_cameras in ["all", "visible"], "Invalid average method over cameras"
    
    gaussian_errors = torch.zeros_like(gaussians._xyz[:, 0])
    gaussian_normals = gaussians.convert_features_to_normals()
    gaussian_normals = torch.nn.functional.normalize(gaussian_normals, dim=-1)
    # Number of visible cameras for each Gaussian
    if average_method_over_cameras == "visible":
        gaussian_visible_cameras = torch.zeros_like(gaussian_errors)
    
    for i_img in range(len(cameras)):
        # Get Gaussian idx
        msv2_render_pkg = render_depth(
            viewpoint_camera=cameras[i_img], 
            pc=gaussians, 
            pipe=pipe, 
            bg_color=background,
            culling=None
        )
        msv2_idx = msv2_render_pkg["gidx"]
        
        # Get projected Gaussian areas
        msv2_render_pkg_simp = render_simp(
            viewpoint_camera=cameras[i_img], 
            pc=gaussians, 
            pipe=pipe, 
            bg_color=background,
            culling=None
        )
        gaussians_proj_area = msv2_render_pkg_simp['area_proj']  # (N_gaussians,)
        
        # Get Gaussian Splatting rendering and rendered Normal Field
        gaussian_render_pkg = render_func(
            viewpoint_camera=cameras[i_img], 
            pc=gaussians, 
            pipe=pipe, 
            bg_color=background, 
            colors_precomp=gaussian_normals,
            require_coord=False, 
            require_depth=True
        )
        normal_field_render = gaussian_render_pkg["render"]  # (3, H, W)
        view_to_world_transform = cameras[i_img].world_view_transform[:3, :3].permute(-1, -2)
        
        # Select a normal map to compare the Normal Field rendering to
        if normal_to_use == "rendered":
            rendered_normal = gaussian_render_pkg["normal"]  # (3, H, W)
            error_mask = None
            rendered_normal = (rendered_normal.permute(1, 2, 0) @ view_to_world_transform).permute(2, 0, 1)  # (3, H, W)
            normal_render_to_use = rendered_normal  # (3, H, W)

        elif normal_to_use == "median_depth":
            median_depth = gaussian_render_pkg["median_depth"]  # (1, H, W)
            if mask_error_at_zero_depth:
                median_depth_normal, error_mask = depth_to_normal_with_mask(cameras[i_img], median_depth)  # (3, H, W) and (1, H, W)
                error_mask = error_mask.squeeze(0)  # (H, W)
            else:
                median_depth_normal = depth_to_normal(cameras[i_img], median_depth, None)  # (3, H, W)
                error_mask = None
            median_depth_normal = (median_depth_normal.permute(1, 2, 0) @ view_to_world_transform).permute(2, 0, 1)  # (3, H, W)
            normal_render_to_use = median_depth_normal  # (3, H, W)

        elif normal_to_use == "expected_depth":
            expected_depth = gaussian_render_pkg["expected_depth"]  # (1, H, W)
            if mask_error_at_zero_depth:
                expected_depth_normal, error_mask = depth_to_normal_with_mask(cameras[i_img], expected_depth)  # (3, H, W) and (1, H, W)
                error_mask = error_mask.squeeze(0)  # (H, W)
            else:
                expected_depth_normal = depth_to_normal(cameras[i_img], expected_depth, None)  # (3, H, W)
                error_mask = None
            expected_depth_normal = (expected_depth_normal.permute(1, 2, 0) @ view_to_world_transform).permute(2, 0, 1)  # (3, H, W)
            normal_render_to_use = expected_depth_normal  # (3, H, W)

        else:
            raise ValueError(f"Invalid normal to use: {normal_to_use}")

        # Compute normal error
        normal_error = 1. - (normal_field_render * normal_render_to_use).sum(dim=0)  # (H, W)
        if mask_error_at_zero_depth and (error_mask is not None):
            if i_img == 0:
                print("[INFO] Masking error at zero depth.")
            normal_error = torch.where(error_mask, normal_error, torch.zeros_like(normal_error))
        
        # Compute per-Gaussian error
        gaussian_errors_i = torch.zeros_like(gaussians_proj_area, dtype=torch.float32)  # (N_gaussians,)
        gaussian_errors_i.index_add_(0, msv2_idx.flatten(), normal_error.flatten())
        
        # If count, we normalize by the number of pixels in which the Gaussian is visible
        if method == "count":
            gaussian_count = torch.zeros_like(gaussians_proj_area, dtype=torch.float32)
            gaussian_count.index_add_(0, msv2_idx.flatten(), torch.ones_like(normal_error.flatten(), dtype=torch.float32))
            
            valid_mask = gaussian_count > 0
            gaussian_errors_i = torch.where(valid_mask, gaussian_errors_i / gaussian_count, torch.zeros_like(gaussian_errors_i))
        
        # If area, we normalize by the area of the projected Gaussian splat
        elif method == "area":
            valid_area_mask = gaussians_proj_area > 0
            gaussian_errors_i = torch.where(valid_area_mask, gaussian_errors_i / gaussians_proj_area, torch.zeros_like(gaussian_errors_i))
        
        # If none, we don't normalize
        elif method == "none":
            pass
        
        else:
            raise ValueError(f"Invalid method: {method}")
        
        gaussian_errors = gaussian_errors + gaussian_errors_i
        
        if average_method_over_cameras == "visible":
            visible_gaussian_idx = msv2_idx.unique()
            gaussian_visible_cameras.index_add_(0, visible_gaussian_idx, torch.ones_like(visible_gaussian_idx, dtype=torch.float32))
    
    if average_method_over_cameras == "all":
        gaussian_errors = gaussian_errors / len(cameras)
    elif average_method_over_cameras == "visible":
        gaussian_errors = torch.where(gaussian_visible_cameras > 0, gaussian_errors / gaussian_visible_cameras, torch.zeros_like(gaussian_errors))
    else:
        raise ValueError(f"Invalid average method over cameras: {average_method_over_cameras}")
    
    return gaussian_errors