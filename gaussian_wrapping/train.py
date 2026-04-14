import os
import sys
import gc
import yaml
from functools import partial
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
SUBMODULES_DIR = os.path.join(ROOT_DIR, 'submodules')
sys.path.append(ROOT_DIR)
sys.path.append(SUBMODULES_DIR)
sys.path.append(os.path.join(SUBMODULES_DIR, 'Depth-Anything-V2'))

import torch
from random import randint
from utils.loss_utils import l1_loss, L1_loss_appearance, get_img_grad_weight
from fused_ssim import fused_ssim

from gaussian_renderer import network_gui
from gaussian_renderer import render_imp
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

import numpy as np
import time

from utils.geometry_utils import depth_to_normal, depth_to_normal_with_mask
from utils.log_utils import log_normal_field_training_progress
from regularization.regularizer.normal_field import (
    initialize_normal_field,
    compute_normal_field_regularization,
    reset_normal_field_state_at_next_iteration,
    densify_normal_field,
    prune_non_maximal_gaussians,
)
from regularization.regularizer.depth_order import (
    initialize_depth_order_supervision,
    compute_depth_order_regularization,
)
from regularization.regularizer.multiview import (
    initialize_multiview_regularization,
    compute_multiview_regularization,
)
from regularization.regularizer.mesh_in_the_loop import (
    initialize_mesh_in_the_loop_regularization,
    compute_mesh_in_the_loop_regularization,
    reset_milo_state_at_next_iteration,
)


def training(
    dataset, opt, pipe, 
    testing_iterations, saving_iterations, 
    checkpoint_iterations, checkpoint, 
    debug_from, args, 
    depth_order_config, normal_field_config, multiview_config, milo_config,
    log_interval,
):
    # ---Prepare logger--- 
    run = prepare_output_and_logger(dataset, args)
    
    # ---Initialize scene and Gaussians---
    first_iter = 0
    use_mip_filter = not args.disable_mip_filter
    
    if args.use_normal_field:
        if normal_field_config["use_smallest_axis"]:
            n_gaussian_features = 1
        else:
            n_gaussian_features = 4
    else:
        n_gaussian_features = 0
    
    n_pivots_per_gaussian = milo_config["n_pivots_per_gaussian"] if args.milo else 2
    
    gaussians = GaussianModel(
        sh_degree=dataset.sh_degree, 
        use_mip_filter=use_mip_filter, 
        learn_occupancy=True if args.milo else False,
        use_appearance_network=args.decoupled_appearance,
        n_gaussian_features=n_gaussian_features,
        n_pivots_per_gaussian=n_pivots_per_gaussian,
        use_radegs_densification=True,
        use_unbounded_opacity=dataset.use_unbounded_opacity,
        use_exposure_compensation=args.exposure_compensation,
    )
    scene = Scene(dataset, gaussians)
    
    if args.exposure_compensation:
        n_cameras = len(scene.getTrainCameras().copy())
        print(f"[INFO] Using exposure compensation for {n_cameras} cameras.")
        gaussians.initialize_exposure_compensation(num_cameras=n_cameras)
    
    gaussians.training_setup(opt)
    print(f"[INFO] Using 3D Mip Filter: {gaussians.use_mip_filter}")
    print(f"[INFO] Using learnable SDF: {gaussians.learn_occupancy}")

    if args.use_normal_field:
        print(f"[INFO] Using {n_gaussian_features} learnable Gaussian features.")

    if args.dense_gaussians:
        print("[INFO] Using dense Gaussians.")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        if args.use_normal_field:
            if first_iter > normal_field_config["start_iter"]:
                normal_field_config["start_iter"] = first_iter + 1
        if args.milo:
            if first_iter > milo_config["start_iter"]:
                milo_config["start_iter"] = first_iter + 1
        if args.multiview:
            if first_iter > multiview_config["start_multiview"]:
                multiview_config["start_multiview"] = first_iter + 1
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Initialize culling stats
    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    gaussians.init_culling(len(scene.getTrainCameras().copy()))
    
    # Initialize 3D Mip filter
    if use_mip_filter:
        gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())

    # Additional variables
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    all_viewpoints = None
    postfix_dict = {}
    ema_loss_for_log = 0.0
    ema_depth_normal_loss_for_log = 0.0
    
    # ---Prepare Normal Field Optimization---
    if args.use_normal_field:
        print("[INFO] Using Normal Field.")
        normal_field_state = initialize_normal_field(
            scene=scene,
            config=normal_field_config,
        )
        if normal_field_config["reset_normals_after_densification"]:
            print(f"[INFO] Normal features will be reset after densification.")
            print(f"        > Resetting normal directions: {normal_field_config['reset_normal_directions']}.")
            print(f"        > Resetting normal signs: {normal_field_config['reset_normal_signs']}.")
    ema_normal_field_alignment_loss_for_log = 0.0
    ema_front_pivots_visibility_loss_for_log = 0.0
    ema_back_pivots_occlusion_loss_for_log = 0.0
    ema_gaussian_flattening_loss_for_log = 0.0
    ema_sdf_and_normal_field_consistency_loss_for_log = 0.0
    
    # ---Prepare Multiview Regularization---
    if args.multiview:
        print("[INFO] Using multiview regularization.")
        multiview_state = initialize_multiview_regularization(
            scene=scene,
            pipe=pipe,
            kernel_size=0.0,
            multiview_config=multiview_config,
        )
    ema_multiview_loss_for_log = 0.0
    
    # ---Prepare Mesh-In-the-Loop Regularization---
    if args.milo:
        print(f"[INFO] Using mesh-in-the-loop regularization with {n_pivots_per_gaussian} pivots per Gaussian.")
        milo_state = initialize_mesh_in_the_loop_regularization(
            scene=scene,
            gaussians=gaussians,
            milo_config=milo_config,
        )
    ema_mesh_depth_loss_for_log = 0.0
    ema_mesh_normal_loss_for_log = 0.0
    ema_occupied_centers_loss_for_log = 0.0

    # ---Prepare Depth-Order Regularization---    
    if args.depth_order:
        print("[INFO] Using depth order regularization.")
        print(f"        > Using expected depth with depth_ratio {depth_order_config['depth_ratio']} for depth order regularization.")
        if depth_order_config["deactivate_depth_order_after"] > -1:
            print(f"        > Deactivating at iteration {depth_order_config['deactivate_depth_order_after']}.")
        depth_priors = initialize_depth_order_supervision(
            scene=scene,
            config=depth_order_config,
            device='cuda',
        )
    ema_depth_order_loss_for_log = 0.0
        
    # ---Log optimizable param groups---
    print(f"[INFO] Found {len(gaussians.optimizer.param_groups)} optimizable param groups:")
    n_total_params = 0
    for param in gaussians.optimizer.param_groups:
        name = param['name']
        n_params = len(param['params'])
        print(f"\n========== {name} ==========")
        print(f"Learning rate: {param['lr']}")
        print(f"Total number of param groups: {n_params}")
        for param_i in param['params']:
            print(f"   > Shape {param_i.shape}")
            n_total_params = n_total_params + param_i.numel()
    if gaussians.learn_occupancy:
        print(f"\n========== base_occupancy ==========")
        print(f"   > Not learnable")
        print(f"   > Shape {gaussians._base_occupancy.shape}")
    print(f"\nTotal number of optimizable parameters: {n_total_params}\n")
    
    # ---Start optimization loop---    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):   

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_imp(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # ---Update SH degree---
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ---Select random viewpoint---
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_idx_stack = list(range(len(viewpoint_stack)))
            all_viewpoints = scene.getTrainCameras().copy()

        _random_view_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_idx = viewpoint_idx_stack.pop(_random_view_idx)
        viewpoint_cam = viewpoint_stack.pop(_random_view_idx)

        # ---Render scene---
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
            
        reg_kick_on = iteration >= args.regularization_from_iter
        normal_field_kick_on = args.use_normal_field and (iteration >= normal_field_config["start_iter"])
        depth_order_kick_on = args.depth_order
        if args.depth_order and depth_order_config["deactivate_depth_order_after"] > -1:
            if iteration == depth_order_config["deactivate_depth_order_after"]:
                print(f"[INFO] Deactivating depth order regularization at iteration {iteration}.")
            depth_order_kick_on = depth_order_kick_on and (iteration < depth_order_config["deactivate_depth_order_after"])
        multiview_kick_on = args.multiview and (iteration >= multiview_config["start_multiview"])
        milo_kick_on = args.milo and (iteration >= milo_config["start_iter"])
        
        render_depth_in_forward_pass = (
            reg_kick_on 
            or normal_field_kick_on 
            or depth_order_kick_on
            or multiview_kick_on
            or milo_kick_on
        )
        
        # If depth-normal regularization or normal field regularization are active,
        # we use the rasterizer compatible with depth and normal rendering.
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, bg,
            require_coord=False, 
            require_depth=render_depth_in_forward_pass,
        )

        # ---Compute losses---
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], render_pkg["radii"]
        )
        gt_image = viewpoint_cam.original_image.cuda()
        if viewpoint_cam.gt_mask is not None:
            alpha_mask = viewpoint_cam.gt_mask.cuda()
            gt_image = gt_image * alpha_mask + bg.unsqueeze(-1).unsqueeze(-1) * (1.0 - alpha_mask)

        # Rendering loss
        if args.decoupled_appearance or args.exposure_compensation:
            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.uid)
        else:
            Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), padding="valid")
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        
        # Depth-Normal Consistency Regularization
        if reg_kick_on:
            if args.mask_depth_normal:
                reg_depth_ratio = 0.6
                
                # Blending that respects the median depthmap validity mask
                depth_blend = torch.where(
                    render_pkg["median_depth"] > 0,  # (1, H, W)
                    (1. - reg_depth_ratio) * render_pkg["expected_depth"] + reg_depth_ratio * render_pkg["median_depth"],  # (1, H, W)
                    render_pkg["median_depth"],  # (1, H, W)
                )
                
                depth_normal, valid_points = depth_to_normal_with_mask(viewpoint_cam, depth_blend)  # (3, H, W), (H, W)
                normal_error_map = 1 - torch.linalg.vecdot(render_pkg["normal"], depth_normal, dim=0)  # (H, W)
                depth_normal_loss = torch.where(valid_points.squeeze(), normal_error_map, torch.zeros_like(normal_error_map)).mean()
                depth_normal_loss = args.lambda_depth_normal * depth_normal_loss
                
            else:
                rendered_depth_to_normals: torch.Tensor = depth_to_normal(
                    viewpoint_cam, 
                    render_pkg["expected_depth"],  # 1, H, W
                    render_pkg["median_depth"],  # 1, H, W
                )  # 3, H, W or 2, 3, H, W
                rendered_normals: torch.Tensor = render_pkg["normal"]  # 3, H, W
            
                if rendered_depth_to_normals.ndim == 4:
                    # If shape is 2, 3, H, W
                    reg_depth_ratio = 0.6
                    normal_error_map = 1. - (rendered_normals[None] * rendered_depth_to_normals).sum(dim=1)  # 2, H, W
                    depth_normal_loss = args.lambda_depth_normal * (
                        (1. - reg_depth_ratio) * normal_error_map[0]  # (H, W)
                        + reg_depth_ratio * normal_error_map[1]  # (H, W)
                    )  # (H, W)
                else:
                    # If shape is 3, H, W
                    depth_normal_loss = args.lambda_depth_normal * (1 - (rendered_normals * rendered_depth_to_normals).sum(dim=0))  # (H, W)
            
            # Weight by image gradient as in PGSR
            if args.weight_by_img_grad:
                image_weight = get_img_grad_weight(img=gt_image)  # (H, W)
                image_weight = (1.0 - image_weight).clamp(min=0.0, max=1.0)  # (H, W)
                image_weight = image_weight ** 2  # (H, W)
                depth_normal_loss = depth_normal_loss * image_weight  # (H, W)
            
            depth_normal_loss = depth_normal_loss.mean()
            
            loss = loss + depth_normal_loss
            
        # Min scale regularization (from PGSR)
        if args.use_scale_loss:
            if visibility_filter.sum() > 0:
                min_scaling_loss = torch.sort(
                    gaussians.get_scaling_with_3D_filter[visibility_filter],  # (N_visible_gaussians, 3)
                    dim=-1
                ).values[..., 0]  # (N_visible_gaussians,)
                min_scaling_loss = args.scale_loss_weight * min_scaling_loss.mean()
                loss = loss + min_scaling_loss
            
        # Depth Order Regularization
        # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
        # > In the paper, MILo does not rely on any learned prior. 
        if depth_order_kick_on:
            if depth_order_config["depth_ratio"] < 1.:
                depth_for_depth_order = (
                    (1. - depth_order_config["depth_ratio"]) * render_pkg["expected_depth"]
                    + depth_order_config["depth_ratio"] * render_pkg["median_depth"]
                )
            else:
                depth_for_depth_order = render_pkg["median_depth"]
                
            depth_prior_loss, _, do_supervision_depth, lambda_depth_order = compute_depth_order_regularization(
                iteration=iteration,
                rendered_depth=depth_for_depth_order,
                depth_priors=depth_priors,
                viewpoint_idx=viewpoint_idx,
                gaussians=gaussians,
                config=depth_order_config,
            )
                
            loss = loss + depth_prior_loss
            depth_order_kick_on = lambda_depth_order > 0
            
        # Multiview Regularization
        if multiview_kick_on:
            
            if multiview_render is None:
                multiview_render_pkg = render_pkg
            else:
                multiview_render_pkg = multiview_render(
                    viewpoint_cam, gaussians, pipe, bg,
                    require_coord=False, 
                    require_depth=True,
                )
            
            multiview_render_pkg = compute_multiview_regularization(
                iteration=iteration,
                scene=scene,
                render_pkg=multiview_render_pkg,
                viewpoint_cam=viewpoint_cam,
                viewpoint_idx=viewpoint_idx,
                gaussians=gaussians,
                render_func=render,
                pipe=pipe,
                background=bg,
                multiview_config=multiview_config,
                multiview_state=multiview_state,
                kernel_size=0.0,
                rasterizer=args.rasterizer,
            )
            multiview_loss = multiview_render_pkg["multiview_loss"] * args.multiview_factor
            loss = loss + multiview_loss
        
        # Normal Field Regularization
        if normal_field_kick_on:
            if args.detach_gaussian_rendering:
                detached_render_pkg = {
                    "render": render_pkg["render"].detach(),
                    "median_depth": render_pkg["median_depth"].detach(),
                    "expected_depth": render_pkg["expected_depth"].detach(),
                    "normal": render_pkg["normal"].detach(),
                }
            
            normal_field_render_pkg = compute_normal_field_regularization(
                iteration=iteration,
                render_pkg=detached_render_pkg if args.detach_gaussian_rendering else render_pkg,
                viewpoint_cam=viewpoint_cam,
                viewpoint_idx=viewpoint_idx,
                gaussians=gaussians,
                scene=scene,
                pipe=pipe,
                background=bg,
                kernel_size=0.0,
                config=normal_field_config,
                normal_field_state=normal_field_state,
                render_func=partial(render, require_coord=False, require_depth=True),
                args=args,
            )
            normal_field_loss = normal_field_render_pkg["normal_field_loss"]
            loss = loss + normal_field_loss
            
            normal_field_alignment_loss = normal_field_render_pkg["normal_field_alignment_loss"]
            front_pivots_visibility_loss = normal_field_render_pkg["front_pivots_visibility_loss"]
            back_pivots_occlusion_loss = normal_field_render_pkg["back_pivots_occlusion_loss"]
            gaussian_flattening_loss = normal_field_render_pkg["gaussian_flattening_loss"]
            sdf_and_normal_field_consistency_loss = normal_field_render_pkg["sdf_and_normal_field_consistency_loss"]
            
        # Mesh-In-the-Loop Regularization
        if milo_kick_on:
            milo_pkg = compute_mesh_in_the_loop_regularization(
                iteration=iteration,
                train_cameras=all_viewpoints,
                viewpoint_cam=viewpoint_cam,
                viewpoint_idx=viewpoint_idx,
                render_pkg=render_pkg,
                gaussians=gaussians,
                pipe=pipe,
                background=bg,
                kernel_size=0.0,
                milo_config=milo_config,
                milo_state=milo_state,
                args=args,
            )
            milo_loss = milo_pkg["milo_loss"]
            loss = loss + milo_loss
            
            mesh_depth_loss = milo_pkg["mesh_depth_loss"]
            mesh_normal_loss = milo_pkg["mesh_normal_loss"]
            occupied_centers_loss = milo_pkg["occupied_centers_loss"]
        
        # ---Backward pass---
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # ---Logging---
            (
                postfix_dict,
                ema_loss_for_log, 
                ema_depth_normal_loss_for_log, 
                ema_normal_field_alignment_loss_for_log,
                ema_front_pivots_visibility_loss_for_log,
                ema_back_pivots_occlusion_loss_for_log,
                ema_gaussian_flattening_loss_for_log,
                ema_sdf_and_normal_field_consistency_loss_for_log,
                ema_depth_order_loss_for_log,
                ema_multiview_loss_for_log,
                ema_mesh_depth_loss_for_log,
                ema_mesh_normal_loss_for_log,
                ema_occupied_centers_loss_for_log,
            ) = log_normal_field_training_progress(
                args, iteration, log_interval, progress_bar, run,
                scene, gaussians, pipe, opt, bg,
                viewpoint_idx, viewpoint_cam, render_pkg, 
                normal_field_render_pkg if normal_field_kick_on else None, 
                milo_pkg if milo_kick_on else None,
                do_supervision_depth if depth_order_kick_on else None,
                reg_kick_on, normal_field_kick_on, depth_order_kick_on, multiview_kick_on, milo_kick_on,
                loss, depth_normal_loss if reg_kick_on else None, 
                normal_field_alignment_loss if normal_field_kick_on else None,
                front_pivots_visibility_loss if normal_field_kick_on else None,
                back_pivots_occlusion_loss if normal_field_kick_on else None,
                gaussian_flattening_loss if normal_field_kick_on else None,
                sdf_and_normal_field_consistency_loss if normal_field_kick_on else None,
                depth_prior_loss if depth_order_kick_on else None,
                multiview_loss if multiview_kick_on else None,
                mesh_depth_loss if milo_kick_on else None,
                mesh_normal_loss if milo_kick_on else None,
                occupied_centers_loss if milo_kick_on else None,
                normal_field_config if normal_field_kick_on else None, 
                milo_config if milo_kick_on else None,
                postfix_dict, ema_loss_for_log, ema_depth_normal_loss_for_log, 
                ema_normal_field_alignment_loss_for_log,
                ema_front_pivots_visibility_loss_for_log,
                ema_back_pivots_occlusion_loss_for_log,
                ema_gaussian_flattening_loss_for_log,
                ema_sdf_and_normal_field_consistency_loss_for_log,
                ema_depth_order_loss_for_log, 
                ema_multiview_loss_for_log,
                ema_mesh_depth_loss_for_log,
                ema_mesh_normal_loss_for_log,
                ema_occupied_centers_loss_for_log,
                testing_iterations, saving_iterations, render,
            )
            
            if iteration % 100 == 0:
                if dataset.use_unbounded_opacity:
                    _opacity = gaussians.get_opacity_with_3D_filter.detach()
                    _contribution = gaussians.get_contribution(viewpoint_cam).detach()
                    print(f"Min contribution: {_contribution.min()}, Max contribution: {_contribution.max()}")
                    print(f"Min opacity: {_opacity.min()}, Max opacity: {_opacity.max()}")

            # ---Densification---
            gaussians_have_changed = False
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats_radegs(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if not args.use_max_size_threshold:
                        size_threshold = None
                    gaussians.densify_and_prune_radegs(
                        opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold, 
                        use_abs_grad=args.use_abs_grad_for_densification,
                        viewpoint_cameras=scene.getTrainCameras().copy(),
                    )
                    gaussians_have_changed = True
                    if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras().copy()
                        )
                    else:
                        gaussians.reset_3D_filter()
                        
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    
            # ---Non-maximal pruning---
            if iteration in args.non_maximal_pruning_iterations:
                print(f"[INFO] Pruning non-maximal Gaussians at iteration {iteration+1}.")
                print(f"        > Number of Gaussians before pruning: {gaussians._xyz.shape[0]}.")
                prune_non_maximal_gaussians(
                    gaussians=gaussians,
                    cameras=scene.getTrainCameras().copy(),
                    pipe=pipe,
                    background=bg,
                )
                gaussians_have_changed = True
                print(f"        > Number of Gaussians after pruning: {gaussians._xyz.shape[0]}.")
                if use_mip_filter:
                    gaussians.compute_3D_filter(
                        cameras=scene.getTrainCameras().copy()
                    )
                
            # ---Normal Field Densification---
            if args.use_normal_field:
                cond_1 = (
                    normal_field_kick_on 
                    and normal_field_config["use_densification"]
                )
                cond_2 = (
                    (iteration+1 >= normal_field_config["start_iter_densification"])
                    and (iteration+1 <= normal_field_config["end_iter_densification"])
                )
                cond_3 = (
                    (iteration+1 - normal_field_config["start_iter_densification"]) % normal_field_config["densify_every_n_iterations"] == 0
                )
                if cond_1 and cond_2 and cond_3:
                    print(f"[INFO] Densifying normal field at iteration {iteration+1}.")
                    print(f"        > Using normalization method: {normal_field_config['densification_normalization_method']}.")
                    print(f"        > Using normal computed from: {normal_field_config['densification_normal_to_use']}.")
                    print(f"        > Using normal errors quantile: {normal_field_config['densification_normal_errors_quantile']}.")
                    print(f"        > Maintaining constant volume: {normal_field_config['maintain_constant_volume']}.")
                    print(f"        > Number of Gaussians before densification: {gaussians._xyz.shape[0]}.")
                    densify_normal_field(
                        iteration=iteration, 
                        gaussians=gaussians, 
                        cameras=scene.getTrainCameras().copy(),
                        scene=scene, 
                        pipe=pipe, 
                        background=bg, 
                        kernel_size=0.0, 
                        config=normal_field_config,
                        normal_field_state=normal_field_state, 
                        render_func=render, 
                        args=args,
                        maintain_constant_volume=normal_field_config["maintain_constant_volume"],
                    )
                    gaussians_have_changed = True
                    print(f"        > Number of Gaussians after densification: {gaussians._xyz.shape[0]}.")
                    if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras().copy()
                        )
                        
                    if normal_field_config["reset_normals_after_densification"]:
                        print(f"[INFO] Resetting normal features after densification.")
                        gaussians.reset_normal_features(
                            reset_directions=normal_field_config["reset_normal_directions"],
                            reset_signs=normal_field_config["reset_normal_signs"],
                        )
                        
            # ---Normal field pruning---
            if args.use_normal_field:
                cond_1 = (
                    normal_field_kick_on 
                    and normal_field_config["use_pruning"]
                )
                cond_2 = (
                    (iteration+1 >= normal_field_config["start_iter_pruning"])
                    and (iteration+1 <= normal_field_config["end_iter_pruning"])
                )
                cond_3 = (
                    (iteration+1 - normal_field_config["start_iter_pruning"]) % normal_field_config["prune_every_n_iterations"] == 0
                )
                if cond_1 and cond_2 and cond_3:
                    print(f"[INFO] Pruning non-maximal Gaussians at iteration {iteration+1}.")
                    print(f"        > Number of Gaussians before pruning: {gaussians._xyz.shape[0]}.")
                    prune_non_maximal_gaussians(
                        gaussians=gaussians,
                        cameras=scene.getTrainCameras().copy(),
                        pipe=pipe,
                        background=bg,
                    )
                    gaussians_have_changed = True
                    print(f"        > Number of Gaussians after pruning: {gaussians._xyz.shape[0]}.")
                    if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras().copy()
                        )

            # ---Reset Normal field state if Gaussians have changed---
            if normal_field_kick_on and gaussians_have_changed:
                normal_field_state = reset_normal_field_state_at_next_iteration(normal_field_state)
            
            # ---Reset MILO state if Gaussians have changed---
            if milo_kick_on and gaussians_have_changed:
                milo_state = reset_milo_state_at_next_iteration(milo_state)
            
            # ---Update 3D Mip Filter---
            if use_mip_filter and (iteration > opt.densify_until_iter) and (
                (iteration == args.warn_until_iter)
                or (iteration % args.update_mip_filter_every == 0)
            ):
                if iteration < opt.iterations - args.update_mip_filter_every:
                    gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())
                else:
                    print(f"[INFO] Skipping 3D Mip Filter update at iteration {iteration}")

            # ---Optimizer step---
            if iteration < opt.iterations:
                if gaussians.use_appearance_network or gaussians.use_exposure_compensation:
                    gaussians.optimizer.step()
                else:
                    visible = radii>0
                    gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none = True)

            # ---Save checkpoint---
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")  
                
        if iteration % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print('Num of Gaussians: %d'%(gaussians._xyz.shape[0]))
    
    if WANDB_FOUND:
        run.finish()
    
    return 


def prepare_output_and_logger(dataset, args):    
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))

    # Create WandB run       
    global WANDB_FOUND
    WANDB_FOUND = (
        WANDB_FOUND
        and (args.wandb_project is not None)
        and (args.wandb_entity is not None)
    )
    if WANDB_FOUND:
        run = wandb.init(
            name=args.wandb_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
        )
    else:
        run=None
        print("[INFO] WandB not found, skipping logging.")
    return run


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # ----- Usual arguments -----
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=-1)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # ----- Rasterization technique -----
    parser.add_argument("--rasterizer", type=str, default="radegs", choices=["radegs", "ours"])
    
    # ----- Normal Field -----
    parser.add_argument("--no_normal_field", action="store_true")
    parser.add_argument("--normal_field_config", type=str, default="default_regular_densification")
    # Gaussians management
    parser.add_argument("--dense_gaussians", action="store_true")
    parser.add_argument("--detach_gaussian_rendering", action="store_true")

    # ----- Densification and Simplification -----
    # > Inspired by Mini-Splatting2.
    # > Used for pruning, densification and Gaussian pivots selection.
    parser.add_argument("--N_max_gaussians", type=int, default=None,
        help="Cap Gaussian count during Normal Field Densification. If the next densification would exceed this, only the highest-error Gaussians are added up to the cap. None = no limit.")
    parser.add_argument("--warn_until_iter", type=int, default=3000)
    parser.add_argument("--non_maximal_pruning_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--use_max_size_threshold", action=BooleanOptionalAction, default=False)
    
    # ----- Depth-Normal consistency Regularization -----
    # > Inspired by 2DGS, GOF, RaDe-GS...
    parser.add_argument("--regularization_from_iter", type=int, default = 7_000)
    parser.add_argument("--lambda_depth_normal", type=float, default = 0.05)
    parser.add_argument("--mask_depth_normal", action="store_true")
    
    # ----- Multiview Regularization -----
    parser.add_argument("--multiview", action=BooleanOptionalAction, default=True)
    parser.add_argument("--multiview_config", type=str, default="fast")
    parser.add_argument("--multiview_rasterizer", type=str, default=None)
    parser.add_argument("--multiview_factor", type=float, default=1.0)
    
    # ----- Mesh-In-the-Loop Regularization -----
    parser.add_argument("--milo", action="store_true")
    parser.add_argument("--milo_config", type=str, default="default_regular_densification")
    
    # ----- Depth Order Regularization (Learned Prior) -----
    # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
    # > In the paper, MILo does not rely on any learned prior.
    parser.add_argument("--depth_order", action="store_true")
    parser.add_argument("--depth_order_config", type=str, default="default_regular_densification")

    # ----- 3D Mip Filter -----
    # > Inspired by Mip-Splatting.
    parser.add_argument("--disable_mip_filter", action="store_true", default=False)
    parser.add_argument("--update_mip_filter_every", type=int, default=100)

    # ----- Appearance Network for Exposure-aware loss -----
    # > Inspired by GOF.
    parser.add_argument("--decoupled_appearance", action="store_true")
    # > Inspired by PGSR.
    parser.add_argument("--exposure_compensation", action=BooleanOptionalAction, default=True)
    
    # ----- PGSR losses -----
    parser.add_argument("--use_scale_loss", action="store_true")
    parser.add_argument("--scale_loss_weight", type=float, default=100.0)
    parser.add_argument("--weight_by_img_grad", action="store_true")

    # ----- Logging -----
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    if not -1 in args.test_iterations:
        args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    args.use_normal_field = not args.no_normal_field
    
    if args.port == -1:
        args.port = np.random.randint(5000, 9000)
        print(f"Using random port: {args.port}")
        
    # Load multiview regularization config
    if args.multiview:
        # Get multiview config file
        multiview_config_file = os.path.join(BASE_DIR, "configs", "multiview", f"{args.multiview_config}.yaml")
        with open(multiview_config_file, "r") as f:
            multiview_config = yaml.safe_load(f)
        print(f"[INFO] Using multiview regularization with config: {args.multiview_config}")
    else:
        multiview_config = None
        
    # Load mesh-in-the-loop regularization config
    if args.milo:
        # Get mesh regularization config file
        milo_config_file = os.path.join(BASE_DIR, "configs", "mesh_in_the_loop", f"{args.milo_config}.yaml")
        with open(milo_config_file, "r") as f:
            milo_config = yaml.safe_load(f)
        print(f"[INFO] Using mesh-in-the-loop regularization with config: {args.milo_config}")
    else:
        milo_config = None
    
    # Load depth order regularization config (not used in MILo paper)
    if args.depth_order:
        # Get depth order config file
        depth_order_config_file = os.path.join(BASE_DIR, "configs", "depth_order", f"{args.depth_order_config}.yaml")
        with open(depth_order_config_file, "r") as f:
            depth_order_config = yaml.safe_load(f)
        print(f"[INFO] Using depth order regularization with config: {args.depth_order_config}")
    else:
        depth_order_config = None
        
    # Load mesh-in-the-loop regularization config
    if args.use_normal_field:
        # Get mesh regularization config file
        normal_field_config_file = os.path.join(BASE_DIR, "configs", "normal_field", f"{args.normal_field_config}.yaml")
        with open(normal_field_config_file, "r") as f:
            normal_field_config = yaml.safe_load(f)
        print(f"[INFO] Using normal field with config: {args.normal_field_config}")
    else:
        normal_field_config = None
    
    # Message for detach_gaussian_rendering
    if args.detach_gaussian_rendering:
        print(f"[INFO] Detaching Gaussian rendering for mesh regularization.")
    
    # Import rendering function
    print(f"[INFO] Using {args.rasterizer} as rasterizer.")
    if args.rasterizer == "radegs":
        from gaussian_renderer.radegs import render_radegs as render
        from gaussian_renderer.radegs import integrate_radegs as integrate
        args.use_abs_grad_for_densification = True
    elif args.rasterizer == "ours":
        from gaussian_renderer.ours import render_ours as render
        from gaussian_renderer.ours import integrate_ours as integrate
        args.use_abs_grad_for_densification = True
        args.mask_depth_normal = True
        print(f"[INFO] Using Ours rasterizer. Setting mask_depth_normal to True.")
    else:
        raise ValueError(f"Invalid rasterizer: {args.rasterizer}")
    
    if args.multiview_rasterizer == "ours":
        print(f"[INFO] Using Ours rasterizer for multiview regularization.")
        from gaussian_renderer.ours import render_ours as multiview_render
    elif args.multiview_rasterizer is None:
        print(f"[INFO] Using default rasterizer for multiview regularization.")
        multiview_render = None
    else:
        raise ValueError(f"Invalid multiview rasterizer: {args.multiview_rasterizer}")
        
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    torch.cuda.synchronize()
    time_start=time.time()
    
    training(
        lp.extract(args), op.extract(args), pp.extract(args), 
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args,
        depth_order_config,
        normal_field_config,
        multiview_config,
        milo_config,
        args.log_interval,
    )

    torch.cuda.synchronize()
    time_end=time.time()
    time_total=time_end-time_start
    print('Training time: %fs'%(time_total))

    time_txt_path=os.path.join(args.model_path, r'time.txt')
    with open(time_txt_path, 'w') as f:  
        f.write(str(time_total)) 

    # All done
    print("\nTraining complete.")
