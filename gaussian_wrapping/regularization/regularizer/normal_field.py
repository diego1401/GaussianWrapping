from typing import Dict, Any, Tuple, Optional, List, Callable
import numpy as np
import torch
from argparse import Namespace
from arguments import PipelineParams
from scene import Scene
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render_depth
from regularization.normal_field import (
    get_pivots_from_normals,
    get_signed_distance_to_depthmap,
    get_gaussian_std_in_direction,
)
from densification.normal_error import compute_normal_error
from utils.geometry_utils import depth_to_normal, depth_to_normal_with_mask
from utils.general_utils import build_rotation


def initialize_normal_field(
    scene,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    normal_field_state = {}
    return normal_field_state


def compute_normal_field_regularization(
    iteration: int, 
    render_pkg: Dict[str, torch.Tensor],
    viewpoint_cam: Camera, 
    viewpoint_idx: int, 
    gaussians: GaussianModel, 
    scene: Scene, 
    pipe: PipelineParams, 
    background: torch.Tensor, 
    kernel_size: float, 
    config: Dict[str, Any],
    normal_field_state: Dict[str, Any], 
    render_func: Callable, 
    args: Namespace,
) -> Dict[str, Any]:
    # If first iteration, initialize normal features
    if iteration == config["start_iter"]:
        print("[INFO] Initializing normal features")
        gaussians.reset_normal_features()
    
    # Get Gaussian normals and pivots
    # FIXME: Should we normalize the normals here or not?
    gaussian_normals = gaussians.convert_features_to_normals(normalize=True)  # (N_gaussians, 3)
    pivots = get_pivots_from_normals(
        gaussians=gaussians, 
        normals=gaussian_normals, 
        std_factor=config["std_factor"],
        return_scales=False,
    )  # (N_gaussians, n_pivots_per_gaussian, 3)
    gaussian_scaling = gaussians.get_scaling_with_3D_filter
    
    # Render Gaussian normals
    normal_render_pkg = render_func(
        viewpoint_cam, gaussians, pipe, background,
        colors_precomp=gaussian_normals,
    )
    oriented_normal = normal_render_pkg["render"]  # (3, H, W)
    # oriented_normal = oriented_normal.permute(1, 2, 0) @ viewpoint_cam.world_view_transform[:3,:3]  # (H, W, 3)
    # oriented_normal = oriented_normal.permute(2, 0, 1)  # (3, H, W)
    
    # Align Gaussian normals with the rendered normals and/or depths
    # TODO
    if config["use_normal_field_alignment"]:
        normal_field_alignment_loss = torch.zeros(size=(), device=gaussians._xyz.device)
        
        assert (
            config["align_with_rendered_normals"]
            or config["align_with_rendered_median_depths"]
            or config["align_with_rendered_expected_depths"]
        ), "At least one of the alignment options must be enabled"
        
        view_to_world_transform = viewpoint_cam.world_view_transform[:3,:3].permute(-1, -2)
        
        if config["align_with_rendered_normals"]:
            rendered_normal = normal_render_pkg["normal"]  # (3, H, W)
            
            rendered_normal = (
                rendered_normal.permute(1, 2, 0) @ view_to_world_transform
            ).permute(2, 0, 1)  # (3, H, W)
            
            normal_field_alignment_loss = normal_field_alignment_loss + (
                1. - (oriented_normal * rendered_normal).sum(dim=0)
            ).mean()
        else:
            rendered_normal = torch.zeros_like(oriented_normal)
        
        if config["align_with_rendered_median_depths"]:
            median_depth = normal_render_pkg["median_depth"]  # (1, H, W)
            if args.mask_depth_normal:
                median_depth_normal, valid_depth_points = depth_to_normal_with_mask(
                    viewpoint_cam, 
                    median_depth
                )  # (3, H, W), (H, W)
            else:
                median_depth_normal = depth_to_normal(
                    viewpoint_cam,
                    median_depth,
                    None
                )  # (3, H, W)
            
            median_depth_normal = (
                median_depth_normal.permute(1, 2, 0) @ view_to_world_transform
            ).permute(2, 0, 1)  # (3, H, W)
            
            if args.mask_depth_normal:
                normal_error_map = 1. - (oriented_normal * median_depth_normal).sum(dim=0)  # (H, W)
                normal_field_alignment_loss = normal_field_alignment_loss + config["depth_ratio_for_alignment"] * (
                    torch.where(
                        valid_depth_points.squeeze(),  # (H, W)
                        normal_error_map,  # (H, W)
                        torch.zeros_like(normal_error_map),  # (H, W)
                    ).mean()
                )
            else:
                normal_field_alignment_loss = normal_field_alignment_loss + (
                    1. - (oriented_normal * median_depth_normal).sum(dim=0)
                ).mean() * config["depth_ratio_for_alignment"]
        else:
            median_depth_normal = torch.zeros_like(oriented_normal)
            
        if config["align_with_rendered_expected_depths"]:
            expected_depth = normal_render_pkg["expected_depth"]  # (1, H, W)
            if args.mask_depth_normal:
                expected_depth_normal, valid_depth_points = depth_to_normal_with_mask(
                    viewpoint_cam, 
                    expected_depth
                )  # (3, H, W), (H, W)
            else:
                expected_depth_normal = depth_to_normal(
                    viewpoint_cam,
                    expected_depth,
                    None
                )  # (3, H, W)
            
            expected_depth_normal = (
                expected_depth_normal.permute(1, 2, 0) @ view_to_world_transform
            ).permute(2, 0, 1)  # (3, H, W)
            
            if args.mask_depth_normal:
                normal_error_map = 1. - (oriented_normal * expected_depth_normal).sum(dim=0)  # (H, W)
                normal_field_alignment_loss = normal_field_alignment_loss + (1. - config["depth_ratio_for_alignment"]) * (
                    torch.where(
                        valid_depth_points.squeeze(),  # (H, W)
                        normal_error_map,  # (H, W)
                        torch.zeros_like(normal_error_map),  # (H, W)
                    ).mean()
                )
            else:                
                normal_field_alignment_loss = normal_field_alignment_loss + (
                    1. - (oriented_normal * expected_depth_normal).sum(dim=0)
                ).mean() * (1. - config["depth_ratio_for_alignment"])
        else:
            expected_depth_normal = torch.zeros_like(oriented_normal)
            
        normal_field_alignment_loss = normal_field_alignment_loss * config["normal_field_alignment_weight"]
    else:
        normal_field_alignment_loss = torch.zeros(size=(), device=gaussians._xyz.device)
    
    # Enforce pivots behind normals to be occluded
    if config["enforce_back_pivots_to_be_occluded"] or config["enforce_front_pivots_to_be_visible"]:
        pivot_idx_to_see = 0
        pivot_idx_to_occlude = 1
        n_pivots = pivots.shape[1]
        
        # Get indices of rendered Gaussians (with maximum contribution)
        if config["enforce_front_pivots_to_be_visible"]:
            with torch.no_grad():
                rendered_gaussian_idx = render_depth(
                    viewpoint_camera=viewpoint_cam, 
                    pc=gaussians, 
                    pipe=pipe, 
                    bg_color=background,
                    culling=None
                )["gidx"].unique()
        
        if (config["enforce_back_pivots_to_be_occluded"] and config["enforce_front_pivots_to_be_visible"]):
            _pivots_to_enforce = pivots.view(-1, 3)  # (N_gaussians * n_pivots, 3)
        
        elif config["enforce_front_pivots_to_be_visible"]:
            _pivots_to_enforce = pivots[:, pivot_idx_to_see, :]  # (N_gaussians, 3)
        
        elif config["enforce_back_pivots_to_be_occluded"]:
            _pivots_to_enforce = pivots[:, pivot_idx_to_occlude, :]  # (N_gaussians, 3)

        pivots_sdf_to_depth, _ = get_signed_distance_to_depthmap(
            depth=normal_render_pkg["median_depth"], 
            points=_pivots_to_enforce,
            camera=viewpoint_cam,
            interpolate_depth=True,
            interpolation_mode='bilinear',
            padding_mode='border',
            align_corners=True,
            znear=None,
            zfar=None,
        )  # (N_gaussians, 1)
        
        # All pivots behind normals should be occluded
        # Pivots in front of normals should be visible if Gaussians are visible
        if config["enforce_back_pivots_to_be_occluded"] and config["enforce_front_pivots_to_be_visible"]:
            pivots_sdf_to_depth = pivots_sdf_to_depth.view(gaussians._xyz.shape[0], n_pivots, 1)  # (N_gaussians, n_pivots, 1)
            positive_pivots_sdf_to_depth = pivots_sdf_to_depth[:, pivot_idx_to_see, :]  # (N_gaussians, 1)
            positive_pivots_sdf_to_depth = positive_pivots_sdf_to_depth[rendered_gaussian_idx]
            negative_pivots_sdf_to_depth = pivots_sdf_to_depth[:, pivot_idx_to_occlude, :]  # (N_gaussians, 1)
        elif config["enforce_front_pivots_to_be_visible"]:
            positive_pivots_sdf_to_depth = pivots_sdf_to_depth  # (N_gaussians, 1)
            positive_pivots_sdf_to_depth = positive_pivots_sdf_to_depth[rendered_gaussian_idx]
            negative_pivots_sdf_to_depth = None
        elif config["enforce_back_pivots_to_be_occluded"]:
            positive_pivots_sdf_to_depth = None
            negative_pivots_sdf_to_depth = pivots_sdf_to_depth  # (N_gaussians, 1)
        
        # Normalize SDF to depth by the minimum scale of the Gaussian
        # FIXME: Should we use the std in direction of the pivot instead?
        # FIXME: Should we use the min or the max scale?
        min_gaussian_scaling = gaussian_scaling.detach().min(dim=-1).values  # (N_gaussians,)
        
        if positive_pivots_sdf_to_depth is not None:
            positive_pivots_sdf_to_depth = (
                positive_pivots_sdf_to_depth.squeeze(-1)  # (N_gaussians,)
                / min_gaussian_scaling[rendered_gaussian_idx]  # (N_gaussians,)
            )
            front_pivots_visibility_loss = config["enforce_front_pivots_to_be_visible_weight"] * (
                - positive_pivots_sdf_to_depth.clamp_max(0.)
            ).sum() / min_gaussian_scaling.shape[0]
        else:
            front_pivots_visibility_loss = torch.zeros(size=(), device=gaussians._xyz.device)

        if negative_pivots_sdf_to_depth is not None:
            negative_pivots_sdf_to_depth = (
                negative_pivots_sdf_to_depth.squeeze(-1)  # (N_gaussians,)
                / min_gaussian_scaling  # (N_gaussians,)
            )
            back_pivots_occlusion_loss = config["enforce_back_pivots_to_be_occluded_weight"] * (
                negative_pivots_sdf_to_depth.clamp_min(0.)
            ).mean()
        else:
            back_pivots_occlusion_loss = torch.zeros(size=(), device=gaussians._xyz.device)
    else:
        front_pivots_visibility_loss = torch.zeros(size=(), device=gaussians._xyz.device)
        back_pivots_occlusion_loss = torch.zeros(size=(), device=gaussians._xyz.device)
    
    # Enforce Gaussian to flatten?
    if config["enforce_gaussian_flattening"]:
        gaussian_flattening_loss = config["enforce_gaussian_flattening_weight"] * (
            gaussian_scaling.min(dim=-1).values / gaussians.spatial_lr_scale
        ).mean()
    else:
        gaussian_flattening_loss = torch.zeros(size=(), device=gaussians._xyz.device)
    
    # Enforce SDF gradient to match Normal Field
    if config["use_sdf"]:
        raise NotImplementedError("SDF and normal field consistency enforcement is not implemented yet")
        if config["enforce_sdf_and_normal_field_consistency"]:
            pass
        pass
    else:
        sdf_and_normal_field_consistency_loss = torch.zeros(size=(), device=gaussians._xyz.device)
        
    normal_field_loss = (
        normal_field_alignment_loss
        + front_pivots_visibility_loss
        + back_pivots_occlusion_loss
        + gaussian_flattening_loss
        + sdf_and_normal_field_consistency_loss
    )
    
    # FIXME: Is it a problem to detach the losses here?
    if config["align_with_rendered_expected_depths"]:
        depth_normal_to_log = expected_depth_normal
    elif config["align_with_rendered_median_depths"]:
        depth_normal_to_log = median_depth_normal
    else:
        depth_normal_to_log = torch.zeros_like(oriented_normal)
    normal_field_render_pkg = {
        "normal_field_loss": normal_field_loss,
        "normal_field_alignment_loss": normal_field_alignment_loss.detach(),
        "front_pivots_visibility_loss": front_pivots_visibility_loss.detach(),
        "back_pivots_occlusion_loss": back_pivots_occlusion_loss.detach(),
        "gaussian_flattening_loss": gaussian_flattening_loss.detach(),
        "sdf_and_normal_field_consistency_loss": sdf_and_normal_field_consistency_loss.detach(),
        "render": oriented_normal,
        # "depth": normal_render_pkg["median_depth"],
        # "normals": normal_render_pkg["normal"],
        "depth": depth_normal_to_log,
        "normals": rendered_normal,
    }
    
    return normal_field_render_pkg


def reset_normal_field_state_at_next_iteration(
    normal_field_state: Dict[str, Any],
) -> Dict[str, Any]:
    return normal_field_state


@torch.no_grad()
def densify_normal_field(
    iteration: int, 
    gaussians: GaussianModel, 
    cameras: List[Camera],
    scene: Scene, 
    pipe: PipelineParams, 
    background: torch.Tensor, 
    kernel_size: float, 
    config: Dict[str, Any],
    normal_field_state: Dict[str, Any], 
    render_func: Callable, 
    args: Namespace,
    maintain_constant_volume: bool = False,
):
    # Get Gaussian normals
    gaussian_normals = gaussians.convert_features_to_normals()  # (N_gaussians, 3)
    gaussian_normals = torch.nn.functional.normalize(gaussian_normals, dim=-1)  # (N_gaussians, 3)
    
    # Compute normal errors
    normal_errors = compute_normal_error(
        gaussians=gaussians,
        cameras=cameras,
        render_func=render_func,
        pipe=pipe,
        background=background,
        method=config["densification_normalization_method"],  # "count" or "area" or "none"
        normal_to_use=config["densification_normal_to_use"],  # "rendered" or "median_depth" or "expected_depth"
    )  # (N_gaussians,)
    
    # Compute normal errors quantile
    normal_errors_quantile = torch.quantile(normal_errors, q=1. - config["densification_normal_errors_quantile"])
    
    # Densification mask
    densification_mask = normal_errors > normal_errors_quantile  # (N_gaussians,)

    # If N_max_gaussians is set, cap the number of new Gaussians
    if getattr(args, 'N_max_gaussians', None) is not None:
        n_current = gaussians._xyz.shape[0]
        n_allowed = args.N_max_gaussians - n_current
        if n_allowed <= 0:
            print("[WARNING] Maximum Number of Gaussians reached. Skipping Densification.")
            return  # Already at or above cap, skip densification entirely
        n_selected = densification_mask.sum().item()
        if n_selected > n_allowed:
            # Keep only the top n_allowed Gaussians by normal error
            candidate_indices = densification_mask.nonzero(as_tuple=True)[0]
            top_indices = candidate_indices[normal_errors[candidate_indices].topk(n_allowed).indices]
            densification_mask = torch.zeros_like(densification_mask)
            densification_mask[top_indices] = True
            print(f"[WARNING] Capping the number of gaussians to {args.N_max_gaussians}.")

    # Adjust scale of Gaussians to be densified. The idea is to divide the volume of the densified Gaussian by 2,
    # while taking into account the direction of the normal.
    if maintain_constant_volume:
        #   > First, we compute the local basis of the Gaussian
        local_basis = build_rotation(
            r=gaussians._rotation[densification_mask]  # (N_gaussians_to_densify, 3, n_vectors_in_basis)
        ).transpose(-1, -2)  # (N_gaussians_to_densify, n_vectors_in_basis, 3)
        
        #   > Then, we compute the projections of the normals on the local basis
        projections_on_local_basis = (
            gaussian_normals[densification_mask].unsqueeze(1)  # (N_gaussians_to_densify, 1, 3)
            * local_basis  # (N_gaussians_to_densify, n_vectors_in_basis, 3)
        ).sum(dim=-1)  # (N_gaussians_to_densify, n_vectors_in_basis)
        
        #   > We compute the logarithm of the adjustment factors
        log_adjustment_factors = np.log(1. / 2.) * projections_on_local_basis ** 2
        
        #   > Adjust the scaling of the Gaussians
        gaussians._scaling[densification_mask] = gaussians._scaling[densification_mask] + log_adjustment_factors
    
    # Compute xyz of cloned Gaussians as same xyz minus a small multiple of the normal
    new_xyz = gaussians._xyz[densification_mask]  # (N_new_gaussians, 3)
    new_normals = - gaussian_normals[densification_mask]  # (N_new_gaussians, 3)
    normal_stds = get_gaussian_std_in_direction(
        directions=new_normals.unsqueeze(1),  # (N_new_gaussians, 1, 3)
        gaussian_scaling=gaussians.get_scaling_with_3D_filter[densification_mask].detach(), 
        gaussian_rotation=gaussians._rotation[densification_mask].detach(),
        normalize_directions=False,
    )  # (N_gaussians, 1)
    # FIXME: What is the best factor to use here?
    delta = 0.1
    # delta = 1.0
    # delta = np.sqrt(3.)
    # new_xyz = new_xyz + 0.01 * normal_stds * new_normals
    # new_xyz = new_xyz + 1. * normal_stds * new_normals
    # new_xyz = new_xyz + 3. * normal_stds * new_normals
    # new_xyz = new_xyz + 0.05 * normal_stds * new_normals  # best so far?
    new_xyz = new_xyz + delta * normal_stds * new_normals
    
    # Compute normal features of cloned Gaussians to obtain the opposite normal
    new_gaussian_features = gaussians._gaussian_features[densification_mask]  # (N_new_gaussians, n_features)
    new_gaussian_features[:, -1:] = -new_gaussian_features[:, -1:]
    
    # Update xyz of densified Gaussians to be xyz plus a small multiple of the normal
    gaussians._xyz[densification_mask] = (
        gaussians._xyz[densification_mask]
        + delta * normal_stds * gaussian_normals[densification_mask]
    )
    
    # Densify Gaussians
    gaussians.densify_and_clone_from_mask(
        selected_pts_mask=densification_mask,
        new_xyz=new_xyz,
        new_gaussian_features=new_gaussian_features,
    )
    

@torch.no_grad()
def prune_non_maximal_gaussians(
    gaussians: GaussianModel,
    cameras: List[Camera],
    pipe: PipelineParams,
    background: torch.Tensor,
):
    is_maximal = torch.zeros(
        gaussians._xyz.shape[0],
        dtype=torch.bool,
        device=gaussians._xyz.device
    )
    
    for i_cam in range(len(cameras)):
        render_pkg = render_depth(
            viewpoint_camera=cameras[i_cam], 
            pc=gaussians, 
            pipe=pipe, 
            bg_color=background,
            culling=None
        )
        
        max_idx = render_pkg["gidx"].unique()
        is_maximal[max_idx] = True
        
    gaussians.prune_points(~is_maximal)