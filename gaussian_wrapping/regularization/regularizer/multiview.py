from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import random
import torch
from arguments import PipelineParams, ModelParams, OptimizationParams
from scene import Scene
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from regularization.multiview_gggs import (
    PatchMatch,
    PatchMatchFast,
    compute_nearest_cameras,
)
from utils.camera_utils import get_cameras_spatial_extent


def initialize_multiview_regularization(
    scene: Scene,
    pipe: PipelineParams,
    kernel_size: float,
    multiview_config: Dict[str, Any],
) -> Dict[str, Any]:  
    # Get nearest cameras
    nearest_cameras = compute_nearest_cameras(
        train_cameras=scene.getTrainCameras(),
        multi_view_max_angle=multiview_config["multi_view_max_angle"],
        multi_view_min_dis_relative=multiview_config["multi_view_min_dis_relative"],
        multi_view_max_dis_relative=multiview_config["multi_view_max_dis_relative"],
        multi_view_num=multiview_config["multi_view_num"],
    )
    
    # Initialize patchmatch object
    if multiview_config["use_fast_multiview"]:
        print(f"[INFO] Using fast multiview regularization.")
        patchmatch = PatchMatchFast(
            patch_size=multiview_config["multi_view_patch_size"],
            pixel_noise_th=multiview_config["multi_view_pixel_noise_th"],
            kernel_size=kernel_size,
            pipe=pipe,
            debug=False,
        )
    else:
        print(f"[INFO] Using slow multiview regularization.")
        patchmatch = PatchMatch(
            patch_size=multiview_config["multi_view_patch_size"],
            pixel_noise_th=multiview_config["multi_view_pixel_noise_th"],
            kernel_size=kernel_size,
            pipe=pipe,
            debug=False,
        )
    
    # Store scene radius
    scene_radius = get_cameras_spatial_extent(scene.getTrainCameras())['radius'].item()
    
    # Return state
    multiview_state = {
        "nearest_cameras": nearest_cameras,
        "patchmatch": patchmatch,
        "scene_radius": scene_radius,
    }
    return multiview_state


def compute_multiview_regularization(
    iteration: int,
    scene: Scene,
    render_pkg: Dict[str, torch.Tensor],
    viewpoint_cam: Camera,
    viewpoint_idx: int,
    gaussians: GaussianModel,
    render_func: Callable,
    pipe: PipelineParams,
    background: torch.Tensor,
    multiview_config: Dict[str, Any],
    multiview_state: Dict[str, Any],
    kernel_size: float,
    rasterizer: str,
):
    # Get device
    device = render_pkg["expected_depth"].device
    
    # Get scene radius
    scene_radius = multiview_state["scene_radius"]
    
    # Get znear
    znear = multiview_config["znear_relative"] * scene_radius
    
    # Message
    if iteration == multiview_config["start_multiview"]:
        print(f"[INFO] Starting multiview regularization at iteration {iteration}.")
        print(f"          > Znear: {znear}")
        print(f"          > Patch size: {multiview_config['multi_view_patch_size']}")
        print(f"          > Depth ratio: {multiview_config['depth_ratio']}")
        print(f"          > NCC weight: {multiview_config['multi_view_ncc_weight']}")
        print(f"          > Geo weight: {multiview_config['multi_view_geo_weight']}")
        print(f"          > Interpolation: {multiview_config['interpolation']}")
        if (
            multiview_config["use_fast_multiview"]
            and (rasterizer not in ["ours"])
        ):
            print(f"[WARNING] Switching to 'ours' rasterizer for fast multiview regularization.")
    
    # Get nearest camera
    nearest_id = multiview_state["nearest_cameras"][viewpoint_idx]["nearest_id"]
    nearest_cam = None if (len(nearest_id) == 0) else scene.getTrainCameras()[random.sample(nearest_id, 1)[0]]
    
    # If no nearest camera, set losses to 0
    if nearest_cam is None:
        geo_loss = torch.tensor(0.0, device=device)
        ncc_loss = torch.tensor(0.0, device=device)
    
    # If nearest camera exists, compute multiview losses
    else:
        # Get patchmatch object
        patchmatch = multiview_state["patchmatch"]
        
        # Compute multiview losses
        if multiview_config["use_fast_multiview"]:
            ncc_loss, geo_loss = patchmatch(
                gaussians=gaussians, 
                render_pkg=render_pkg, 
                viewpoint_cam=viewpoint_cam, 
                nearest_cam=nearest_cam, 
                iteration=iteration, 
                depth_normal=None,
                rasterizer=rasterizer,
            )
        else:
            # Render from nearest camera
            nearest_render_pkg = render_func(
                pc=gaussians,
                viewpoint_camera=nearest_cam,
                pipe=pipe,
                bg_color=background,
                kernel_size=kernel_size,
                require_depth=True,
            )
            
            ncc_loss, geo_loss = patchmatch(
                viewpoint_cam=viewpoint_cam, 
                nearest_cam=nearest_cam, 
                render_pkg=render_pkg, 
                nearest_render_pkg=nearest_render_pkg,
                depth_ratio=multiview_config["depth_ratio"],
                znear=znear,
                interpolation=multiview_config["interpolation"],
            )
        
    # Compute total multiview loss
    multiview_loss = (
        multiview_config["multi_view_ncc_weight"] * ncc_loss 
        + multiview_config["multi_view_geo_weight"] * geo_loss
    )
    
    # Return results
    multiview_render_pkg = {
        "multiview_loss": multiview_loss,
        "geo_loss": geo_loss,
        "ncc_loss": ncc_loss,
    }
    return multiview_render_pkg
