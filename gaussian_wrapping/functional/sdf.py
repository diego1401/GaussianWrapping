from typing import (
    Callable, Union, Dict, List, Tuple
)
from functools import partial
import torch
from scene.cameras import Camera
from regularization.sdf.depth_fusion import AdaptiveTSDF
from utils.camera_utils import get_cameras_spatial_extent
from functional.pivots import extract_gaussian_pivots
from tqdm import tqdm


def _evaluate_sdf_values(
    pivots:torch.Tensor, 
    views:List[Camera], 
    render_func:Callable, 
    return_colors:bool=False,
    trunc_margin:Union[float, None]=None, 
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Evaluate truncated SDF values at the given pivots using Depth Fusion.

    Args:
        pivots (torch.Tensor): Points at which to compute the SDF values. Has shape (N, 3).
        views (List[Camera]): List of cameras.
        render_func (Callable): Function that returns the Gaussian Splatting rendering 
            and depth map for a given viewpoint. The function only takes the camera as input.
            The result should be formatted as a dictionary with keys "render" and "depth", 
            containing the rendered image and depth map as tensors with shape (3, H, W) 
            and (1, H, W) respectively. Below is a template for render_func:
            #
            def render_func(
                view:Camera,
            ) -> Dict[str, torch.Tensor]:
                return {
                    "render": torch.randn(3, 900, 1600),
                    "depth": torch.randn(1, 900, 1600)
                }
            #
            We believe passing a Callable as input is more efficient than providing all renderings 
            and depth maps as inputs, as the dataset might contain hundreds or thousands of viewpoints.
        return_colors (bool, optional): Whether to return fused colors for each pivot. Defaults to False.
        trunc_margin (float): Truncation margin for depth fusion.
        
    Returns:
        sdf (torch.Tensor): Truncated SDF values. Has shape (N,).
        colors (torch.Tensor, optional): Fused colors. Has shape (N, 3).
    """
    
    if trunc_margin is None:
        trunc_margin = 2e-3 * get_cameras_spatial_extent(views)["radius"]
    
    tsdf_volume = AdaptiveTSDF(
        points=pivots,
        trunc_margin=trunc_margin,
        use_binary_opacity=False,
    )
    
    for cam_id, view in enumerate(
        tqdm(views, desc=f"Fusing depth maps with trunc margin {trunc_margin:.6f}")
    ):
        render_pkg = render_func(view)
        tsdf_volume.integrate(
            img=render_pkg["render"], 
            depth=render_pkg["depth"],
            camera=view, 
            obs_weight=1.0,
            override_points=None,
            interpolate_depth=True,
            interpolation_mode='bilinear',
            padding_mode='border',
            align_corners=True,
            weight_by_softmax=False,
            softmax_temperature=1.0,
        )

    field_values = tsdf_volume.return_field_values()
    if return_colors:
        return field_values["tsdf"].squeeze(), field_values["colors"]
    return field_values["tsdf"].squeeze()


@torch.no_grad()
def compute_initial_sdf_values(
    views:List[Camera],
    render_func:Callable,
    means:torch.Tensor,
    scales:torch.Tensor,
    rotations:torch.Tensor,
    method:str="depth_fusion",
    gaussian_idx:Union[torch.Tensor, None]=None,
    scale_pivots_with_downsample_ratio:bool=True,
    scale_pivots_factor:float=None,
    override_pivots:Union[torch.Tensor, None]=None,
) -> torch.Tensor:
    """Compute initial SDF values for a set of Gaussians, or precomputed pivots.

    Args:
        views (List[Camera]): List of cameras.
        render_func (Callable): Function that returns the Gaussian Splatting rendering 
            and depth map for a given viewpoint. The function only takes the camera as input.
            The result should be formatted as a dictionary with keys "render" and "depth", 
            containing the rendered image and depth map as tensors with shape (3, H, W) 
            and (1, H, W) respectively. Below is a template for render_func:
            #
            def render_func(
                view:Camera,
            ) -> Dict[str, torch.Tensor]:
                return {
                    "render": torch.randn(3, 900, 1600),
                    "depth": torch.randn(1, 900, 1600)
                }
            #
            We believe passing a Callable as input is more efficient than providing all renderings 
            and depth maps as inputs, as the dataset might contain hundreds or thousands of viewpoints.
        means (torch.Tensor): Means of the Gaussians. Shape: (N, 3).
        scales (torch.Tensor): Scales of the Gaussians. Shape: (N, 3).
        rotations (torch.Tensor): Rotations of the Gaussians as quaternions. Shape: (N, 4).
        method (str, optional): Method to compute the SDF values. Defaults to "depth_fusion".
        gaussian_idx (Union[torch.Tensor, None], optional): Indices of the Gaussians to be used for generating pivots. 
            Shape: (N_selected,). Defaults to None.
        scale_pivots_with_downsample_ratio (bool, optional): If True, the scale of the pivots will be adjusted to match the downsample ratio. Defaults to True.
        scale_pivots_factor (float, optional): If provided, the scale of the pivots will be multiplied by this factor. Defaults to None.
        override_pivots (Union[torch.Tensor, None], optional): Override pivots. Shape: (N_pivots, 3). Defaults to None.

    Raises:
        NotImplementedError: Integration method not implemented yet.

    Returns:
        pivots_sdf (torch.Tensor): SDF values for the pivots. Shape: (N_pivots,).
    """
    assert method in ["depth_fusion", "integration"]
    
    if override_pivots is None:
        pivots, _ = extract_gaussian_pivots(
            means=means,
            scales=scales,
            rotations=rotations,
            gaussian_idx=gaussian_idx,
            scale_pivots_with_downsample_ratio=scale_pivots_with_downsample_ratio,
            scale_pivots_factor=scale_pivots_factor,
        )
    else:
        pivots = override_pivots
    
    # Get base occupancy values for all voronoi points    
    if method == 'depth_fusion':                        
        sdf_function = partial(
            _evaluate_sdf_values,
            views=views.copy(), 
            render_func=render_func, 
            return_colors=False,
            trunc_margin=None, 
        )

    if method == 'integration':                        
        raise NotImplementedError("Integration method not implemented yet.")
        # sdf_function = partial(
        #     evaluate_cull_sdf_values,
        #     views=scene.getTrainCameras().copy(), 
        #     masks=None, 
        #     gaussians=gaussians, 
        #     pipeline=pipe, 
        #     background=background, 
        #     kernel_size=kernel_size, 
        #     return_colors=False, 
        #     isosurface_value=config["sdf_default_isosurface"], 
        #     transform_sdf_to_linear_space=config["transform_sdf_to_linear_space"], 
        #     min_occupancy_value=config["min_occupancy_value"],
        #     integrate_func=integrate_func,
        # )
        
    pivots_sdf = sdf_function(pivots)
    # pivots_sdf = pivots_sdf / pivots_sdf.abs().max()
    return pivots_sdf
