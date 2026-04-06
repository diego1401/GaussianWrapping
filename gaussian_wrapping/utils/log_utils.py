import os
from typing import List, Union, Dict, Any
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from utils.geometry_utils import depth_to_normal
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, L1_loss_appearance
from lpipsPyTorch import lpips
from regularization.regularizer.snapping_points import write_snapping_mask
from scene.mesh import MeshRenderer, Meshes
try:
    import wandb
except ImportError:
    pass

def mesh_debug_plot(verts:torch.Tensor, faces:torch.Tensor, mesh_renderer:MeshRenderer, dataset:Any, iteration:int, cam_idx:int=32):
    mesh_render_pkg_debug = mesh_renderer(
        Meshes(verts=verts, faces=faces),
        cam_idx=cam_idx,
        return_depth=True,
        return_normals=True,
        use_antialiasing=False,
    )
    mesh_depth_debug = mesh_render_pkg_debug["depth"].squeeze()
    mesh_normal_debug = mesh_render_pkg_debug["normals"].squeeze()
    debug_plot_path = os.path.join(dataset.model_path, 'debug_plots')
    os.makedirs(debug_plot_path, exist_ok=True)
    debug_plot_path = os.path.join(debug_plot_path, f"mesh_depth_and_normal_iter_{iteration}.png")
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(mesh_depth_debug.cpu().numpy())
    plt.title("Mesh Depth")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow((1. - mesh_normal_debug.cpu().numpy()) / 2.)
    plt.title("Mesh Normal")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(debug_plot_path)
    plt.close()
    print(f"[DEBUG] saved example plot of mesh depth and normal to {debug_plot_path}")

def fix_normal_map(view, normal, normal_in_view_space=True):
    """_summary_

    Args:
        view (_type_): _description_
        normal (_type_): 

    Returns:
        _type_: _description_
    """
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
    rays_d = (intrins_inv @ points).reshape(3, H, W)
    
    if normal_in_view_space:
        normal_view = normal
    else:
        normal_view = normal.clone()
        if normal.shape[0] == 3:
            normal_view = normal_view.permute(1, 2, 0)
        normal_view = normal_view @ view.world_view_transform[:3,:3]
        if normal.shape[0] == 3:
            normal_view = normal_view.permute(2, 0, 1)
    
    if normal_view.shape[0] != 3:
        rays_d = rays_d.permute(1, 2, 0)
        dim_to_sum = -1
    else:
        dim_to_sum = 0
        
    return torch.sign((-rays_d * normal_view).sum(dim=dim_to_sum, keepdim=True)) * normal_view


def make_log_figure(
    images:List[torch.Tensor],
    nrows:Union[int, None]=None,
    ncols:Union[int, None]=None,
    titles:Union[List[str], None]=None,
    figsize:int=30,
    cmap:Union[str, List[str]]="Spectral",
    show_plot:bool=False,
    save_plot:bool=False,
    save_path:Union[str, None]=None,
    return_log_images:bool=False,
):
    log_images = []
    for image in images:
        image_to_add = image
        if (image.ndim == 3) and (image.shape[0] == 1 or image.shape[0] == 3):
            image_to_add = image.permute(1, 2, 0)
        log_images.append(image_to_add.cpu())
    
    n_images = len(log_images)
    if nrows is None and ncols is None:
        raise ValueError("nrows and ncols cannot be None at the same time")
    elif nrows is None:
        nrows = round(np.ceil(float(n_images) / float(ncols)))
    elif ncols is None:
        ncols = round(np.ceil(float(n_images) / float(nrows)))
    
    height, width = log_images[0].shape[:2]    
    plt.figure(figsize=(figsize, figsize * height / width * nrows / ncols))
    
    for i, log_image in enumerate(log_images):
        plt.subplot(nrows, ncols, i+1)
        if log_image.shape[-1] == 3:
            plt.imshow(log_image.clamp(min=0, max=1))
        else:
            if type(cmap) == list:
                plt.imshow(log_image, cmap=cmap[i])
            else:
                if log_image.dtype == torch.bool:
                    plt.imshow(log_image, cmap="gray")
                else:
                    plt.imshow(log_image, cmap=cmap)
            plt.colorbar()
        if titles is not None:
            plt.title(titles[i])
    
    if show_plot:
        plt.show()
    if save_plot and (save_path is not None):
        plt.savefig(save_path)
    plt.close()
    
    if return_log_images:
        log_images_dict = {}
        for i, log_image in enumerate(log_images):
            if titles is not None:
                log_images_dict[titles[i]] = log_image
            else:
                log_images_dict[f"image_{i}"] = log_image
        return log_images_dict


def training_report(iteration, l1_loss, testing_iterations, scene, renderFunc, renderArgs):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        # validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},)        

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims = []
                lpipss = []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    ssims.append(ssim(image, gt_image))
                    lpipss.append(lpips(image, gt_image, net_type='vgg'))                    


                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 

                ssims_test=torch.tensor(ssims).mean()
                lpipss_test=torch.tensor(lpipss).mean()

                print("\n[ITER {}] Evaluating {}: ".format(iteration, config['name']))
                print("  SSIM : {:>12.7f}".format(ssims_test.mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(psnr_test.mean(), ".5"))
                print("  LPIPS : {:>12.7f}".format(lpipss_test.mean(), ".5"))
                print("")

        torch.cuda.empty_cache()


# TODO: Logging function should be made clearer and more modular.
def log_training_progress(
    args, iteration:int, log_interval:int, progress_bar, run,
    # Objects
    scene, gaussians, pipe, opt, background,
    # Rendering results,
    viewpoint_idx, viewpoint_cam, render_pkg, mesh_render_pkg, do_supervision_depth,
    # Flags
    reg_kick_on:bool, mesh_kick_on:bool, depth_order_kick_on:bool,
    # Losses
    loss:torch.Tensor, depth_normal_loss:torch.Tensor, mesh_depth_loss:torch.Tensor,
    mesh_normal_loss:torch.Tensor, occupied_centers_loss:torch.Tensor, occupancy_labels_loss:torch.Tensor,
    non_zero_sdf_loss:torch.Tensor, angle_defect_loss:torch.Tensor, area_loss:torch.Tensor, 
    laplacian_smoothing_loss:torch.Tensor, normal_consistency_loss:torch.Tensor, 
    qem_loss:torch.Tensor,
    attract_pivots_along_mesh_edges_loss:torch.Tensor,
    depth_prior_loss:torch.Tensor,
    # Configs
    mesh_config:Dict[str, Any],
    # EMA losses for logging
    postfix_dict:Dict[str, Any], ema_loss_for_log:float, ema_depth_normal_loss_for_log:float,
    ema_mesh_depth_loss_for_log:float, ema_mesh_normal_loss_for_log:float,
    ema_occupied_centers_loss_for_log:float, ema_non_zero_sdf_loss_for_log:float, ema_occupancy_labels_loss_for_log:float, ema_angle_defect_loss_for_log:float, ema_area_loss_for_log:float, ema_laplacian_smoothing_loss_for_log:float, ema_normal_consistency_loss_for_log:float, ema_qem_loss_for_log:float,
    ema_attract_pivots_along_mesh_edges_loss_for_log:float,
    ema_depth_order_loss_for_log:float,
    # Additional arguments
    testing_iterations:List[int], saving_iterations:List[int], render_imp,
    # Snapping points
    snapping_mask,
):
    WANDB_FOUND = run is not None
    
    # ---Progress bar---
    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
    if reg_kick_on:
        ema_depth_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_depth_normal_loss_for_log
    if mesh_kick_on:
        ema_mesh_depth_loss_for_log = 0.4 * mesh_depth_loss.item() + 0.6 * ema_mesh_depth_loss_for_log
        ema_mesh_normal_loss_for_log = 0.4 * mesh_normal_loss.item() + 0.6 * ema_mesh_normal_loss_for_log
        if mesh_config["enforce_occupied"]:
            ema_occupied_centers_loss_for_log = 0.4 * occupied_centers_loss.item() + 0.6 * ema_occupied_centers_loss_for_log
        if mesh_config["use_occupancy_labels_loss"]:
            ema_occupancy_labels_loss_for_log = 0.4 * occupancy_labels_loss.item() + 0.6 * ema_occupancy_labels_loss_for_log
        if mesh_config["enforce_non_zero_sdf"]:
            ema_non_zero_sdf_loss_for_log = 0.4 * non_zero_sdf_loss.item() + 0.6 * ema_non_zero_sdf_loss_for_log
        if mesh_config["use_angle_defect_loss"]:
            ema_angle_defect_loss_for_log = 0.4 * angle_defect_loss.item() + 0.6 * ema_angle_defect_loss_for_log
        if mesh_config["use_area_loss"]:
            ema_area_loss_for_log = 0.4 * area_loss.item() + 0.6 * ema_area_loss_for_log
        if mesh_config["use_laplacian_smoothing_loss"]:
            ema_laplacian_smoothing_loss_for_log = 0.4 * laplacian_smoothing_loss.item() + 0.6 * ema_laplacian_smoothing_loss_for_log
        if mesh_config["use_normal_consistency_loss"]:
            ema_normal_consistency_loss_for_log = 0.4 * normal_consistency_loss.item() + 0.6 * ema_normal_consistency_loss_for_log
        if mesh_config["use_qem_loss"]:
            ema_qem_loss_for_log = 0.4 * qem_loss.item() + 0.6 * ema_qem_loss_for_log
        if mesh_config["attract_pivots_along_mesh_edges"]:
            ema_attract_pivots_along_mesh_edges_loss_for_log = 0.4 * attract_pivots_along_mesh_edges_loss.item() + 0.6 * ema_attract_pivots_along_mesh_edges_loss_for_log
    if depth_order_kick_on:
        ema_depth_order_loss_for_log = 0.4 * depth_prior_loss.item() + 0.6 * ema_depth_order_loss_for_log
    
    if iteration % 10 == 0:
        postfix_dict = {"Loss": f"{ema_loss_for_log:.{7}f}"}
        if reg_kick_on:
            postfix_dict["DNLoss"] = f"{ema_depth_normal_loss_for_log:.{7}f}"
        if depth_order_kick_on:
            postfix_dict["DOLoss"] = f"{ema_depth_order_loss_for_log:.{7}f}"
        if mesh_kick_on:
            postfix_dict["MDLoss"] = f"{ema_mesh_depth_loss_for_log:.{7}f}"
            postfix_dict["MNLoss"] = f"{ema_mesh_normal_loss_for_log:.{7}f}"
            if mesh_config["use_qem_loss"]:
                postfix_dict["QEMLoss"] = f"{ema_qem_loss_for_log:.{7}f}"
            if mesh_config["attract_pivots_along_mesh_edges"]:
                postfix_dict["AttractLoss"] = f"{ema_attract_pivots_along_mesh_edges_loss_for_log:.{7}f}"
            if mesh_config["enforce_occupied"]:
                postfix_dict["OccLoss"] = f"{ema_occupied_centers_loss_for_log:.{7}f}"
            if mesh_config["use_occupancy_labels_loss"]:
                postfix_dict["OccLabLoss"] = f"{ema_occupancy_labels_loss_for_log:.{7}f}"
            if mesh_config["enforce_non_zero_sdf"]:
                postfix_dict["NZSDFLoss"] = f"{ema_non_zero_sdf_loss_for_log:.{7}f}"
            if mesh_config["use_angle_defect_loss"]:
                postfix_dict["AngleDefectLoss"] = f"{ema_angle_defect_loss_for_log:.{7}f}"
            if mesh_config["use_area_loss"]:
                postfix_dict["AreaLoss"] = f"{ema_area_loss_for_log:.{7}f}"
            if mesh_config["use_laplacian_smoothing_loss"]:
                postfix_dict["LaplacianSmoothingLoss"] = f"{ema_laplacian_smoothing_loss_for_log:.{7}f}"
            if mesh_config["use_normal_consistency_loss"]:
                postfix_dict["NormalConsistencyLoss"] = f"{ema_normal_consistency_loss_for_log:.{7}f}"
        postfix_dict["N_Gauss"] = f"{gaussians._xyz.shape[0]}"
        progress_bar.set_postfix(postfix_dict)
        progress_bar.update(10)
        
    if iteration == opt.iterations:
        progress_bar.close()
    
    # ---Logging---
    if (log_interval is not None) and (
        (iteration % log_interval == 0) or (mesh_kick_on and iteration == mesh_config["start_iter"])
    ):
        images_to_log, titles_to_log = [], []
        
        if depth_order_kick_on:
            images_to_log.append(viewpoint_cam.original_image.cuda())
            titles_to_log.append(f"GT RGB {viewpoint_idx}")
            
            images_to_log.append(do_supervision_depth)
            titles_to_log.append(f"Supervision Depth {viewpoint_idx}")
            
            images_to_log.append((1. - depth_to_normal(viewpoint_cam, do_supervision_depth)) / 2.)
            titles_to_log.append(f"Supervision Depth to Normal {viewpoint_idx}")
        
        images_to_log.append(render_pkg["render"])
        titles_to_log.append(f"Rendered RGB {viewpoint_idx}")
        
        if reg_kick_on or mesh_kick_on or depth_order_kick_on:
            images_to_log.append(render_pkg["median_depth"])
            titles_to_log.append(f"Rendered Depth {viewpoint_idx}")
        
        if reg_kick_on or mesh_kick_on:
            images_to_log.append((1. - render_pkg["normal"]) / 2.)
            titles_to_log.append(f"Rendered Normals {viewpoint_idx}")
            
        if mesh_kick_on:
            images_to_log.append(torch.zeros_like(render_pkg["render"]))
            titles_to_log.append(f"Mesh RGB {viewpoint_idx}")

            images_to_log.append(
                torch.where(
                    mesh_render_pkg["depth"].detach() > 0,
                    mesh_render_pkg["depth"].detach(),
                    mesh_render_pkg["depth"].detach().max().item()
                )
            )
            titles_to_log.append(f"Mesh Depth {viewpoint_idx}")
            
            images_to_log.append(
                (1. - fix_normal_map(
                    viewpoint_cam, 
                    mesh_render_pkg["normals"].detach(),
                )) / 2.
            )
            titles_to_log.append(f"Mesh Normals {viewpoint_idx}")
        
        log_images_dict = make_log_figure(
            images=images_to_log, 
            titles=titles_to_log, 
            cmap='Spectral',
            ncols=3,
            figsize=30,
            show_plot=False,
            save_plot=not WANDB_FOUND,
            save_path=os.path.join(args.model_path, f"iter_{iteration}.png"),
            return_log_images=WANDB_FOUND,
        )
        
        if WANDB_FOUND:
            # Log metrics
            wandb_log_dict = {}
            for key, value in postfix_dict.items():
                try:
                    wandb_log_dict[key] = float(value)
                except:
                    wandb_log_dict[key] = value
            run.log(wandb_log_dict, step=iteration)
            
            # Log images
            wandb_log_images_dict = {}
            for log_img_name, log_img in log_images_dict.items():
                wandb_img_to_log = log_img.clone().detach().squeeze()
                # If grayscale, scale to 0-1 and apply colormap
                if wandb_img_to_log.ndim < 3:
                    wandb_img_to_log = (wandb_img_to_log - wandb_img_to_log.min()) / (wandb_img_to_log.max() - wandb_img_to_log.min())
                    cmap = plt.get_cmap('Spectral')
                    wandb_img_to_log = cmap(wandb_img_to_log)[..., :3]
                    wandb_img_to_log = torch.from_numpy(wandb_img_to_log).to(torch.float32)
                # If float, scale to 0-255
                if wandb_img_to_log.dtype == torch.float32:
                    wandb_img_to_log = (wandb_img_to_log.clamp(0., 1.) * 255).to(torch.uint8)
                # Convert to wandb image
                wandb_img_to_log = wandb_img_to_log.cpu().numpy()
                wandb_log_images_dict[log_img_name.replace(str(viewpoint_idx), "")] = wandb.Image(
                    wandb_img_to_log, caption=log_img_name
                )
            run.log(wandb_log_images_dict, step=iteration)

    # ---Report---
    training_report(iteration, l1_loss, testing_iterations, scene, render_imp, (pipe, background))
    if (iteration in saving_iterations):
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        scene.save(iteration)
        if snapping_mask is not None:
            write_snapping_mask(os.path.join(args.model_path, f"snapping_mask_iter_{iteration}.pt"), snapping_mask)

    return (
        postfix_dict,
        ema_loss_for_log, 
        ema_depth_normal_loss_for_log, 
        ema_mesh_depth_loss_for_log, ema_mesh_normal_loss_for_log, 
        ema_occupied_centers_loss_for_log, ema_non_zero_sdf_loss_for_log, ema_occupancy_labels_loss_for_log, ema_angle_defect_loss_for_log, ema_area_loss_for_log, ema_laplacian_smoothing_loss_for_log, ema_normal_consistency_loss_for_log, ema_qem_loss_for_log, 
        ema_attract_pivots_along_mesh_edges_loss_for_log,
        ema_depth_order_loss_for_log
    )
    
    
# TODO: Logging function should be made clearer and more modular.
def log_normal_field_training_progress(
    args, iteration:int, log_interval:int, progress_bar, run,
    # Objects
    scene, gaussians, pipe, opt, background,
    # Rendering results,
    viewpoint_idx, viewpoint_cam, render_pkg, normal_field_render_pkg, milo_pkg, do_supervision_depth,
    # Flags
    reg_kick_on:bool, normal_field_kick_on:bool, depth_order_kick_on:bool, multiview_kick_on:bool, milo_kick_on:bool,
    # Losses
    loss:torch.Tensor, depth_normal_loss:torch.Tensor, 
    normal_field_alignment_loss:torch.Tensor,
    front_pivots_visibility_loss:torch.Tensor,
    back_pivots_occlusion_loss:torch.Tensor, 
    gaussian_flattening_loss:torch.Tensor, 
    sdf_and_normal_field_consistency_loss:torch.Tensor,
    depth_prior_loss:torch.Tensor,
    multiview_loss:torch.Tensor,
    mesh_depth_loss:torch.Tensor,
    mesh_normal_loss:torch.Tensor,
    occupied_centers_loss:torch.Tensor,
    # Configs
    normal_field_config:Dict[str, Any],
    milo_config:Dict[str, Any],
    # EMA losses for logging
    postfix_dict:Dict[str, Any], ema_loss_for_log:float, 
    ema_depth_normal_loss_for_log:float,
    ema_normal_field_alignment_loss_for_log:float,
    ema_front_pivots_visibility_loss_for_log:float,
    ema_back_pivots_occlusion_loss_for_log:float,
    ema_gaussian_flattening_loss_for_log:float,
    ema_sdf_and_normal_field_consistency_loss_for_log:float,
    ema_depth_order_loss_for_log:float,
    ema_multiview_loss_for_log:float,
    ema_mesh_depth_loss_for_log:float,
    ema_mesh_normal_loss_for_log:float,
    ema_occupied_centers_loss_for_log:float,
    # Additional arguments
    testing_iterations:List[int], saving_iterations:List[int], render_imp,
):
    WANDB_FOUND = run is not None
    
    # ---Progress bar---
    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
    if reg_kick_on:
        ema_depth_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_depth_normal_loss_for_log
    if normal_field_kick_on:
        if normal_field_config["use_normal_field_alignment"]:
            ema_normal_field_alignment_loss_for_log = 0.4 * normal_field_alignment_loss.item() + 0.6 * ema_normal_field_alignment_loss_for_log
        if normal_field_config["enforce_front_pivots_to_be_visible"]:
            ema_front_pivots_visibility_loss_for_log = 0.4 * front_pivots_visibility_loss.item() + 0.6 * ema_front_pivots_visibility_loss_for_log
        if normal_field_config["enforce_back_pivots_to_be_occluded"]:
            ema_back_pivots_occlusion_loss_for_log = 0.4 * back_pivots_occlusion_loss.item() + 0.6 * ema_back_pivots_occlusion_loss_for_log
        if normal_field_config["enforce_gaussian_flattening"]:
            ema_gaussian_flattening_loss_for_log = 0.4 * gaussian_flattening_loss.item() + 0.6 * ema_gaussian_flattening_loss_for_log
        if normal_field_config["use_sdf"] and normal_field_config["enforce_sdf_and_normal_field_consistency"]:
            ema_sdf_and_normal_field_consistency_loss_for_log = 0.4 * sdf_and_normal_field_consistency_loss.item() + 0.6 * ema_sdf_and_normal_field_consistency_loss_for_log
    if depth_order_kick_on:
        ema_depth_order_loss_for_log = 0.4 * depth_prior_loss.item() + 0.6 * ema_depth_order_loss_for_log
    if multiview_kick_on:
        ema_multiview_loss_for_log = 0.4 * multiview_loss.item() + 0.6 * ema_multiview_loss_for_log
    if milo_kick_on:
        if milo_config["use_rgb_loss"]:
            raise NotImplementedError("RGB loss is not implemented")
        if milo_config["use_depth_loss"]:
            ema_mesh_depth_loss_for_log = 0.4 * mesh_depth_loss.item() + 0.6 * ema_mesh_depth_loss_for_log
        if milo_config["use_normal_loss"]:
            ema_mesh_normal_loss_for_log = 0.4 * mesh_normal_loss.item() + 0.6 * ema_mesh_normal_loss_for_log
        if milo_config["enforce_occupied_centers"]:
            ema_occupied_centers_loss_for_log = 0.4 * occupied_centers_loss.item() + 0.6 * ema_occupied_centers_loss_for_log
    
    if iteration % 10 == 0:
        postfix_dict = {"Loss": f"{ema_loss_for_log:.{7}f}"}
        if reg_kick_on:
            postfix_dict["DNLoss"] = f"{ema_depth_normal_loss_for_log:.{7}f}"
        if depth_order_kick_on:
            postfix_dict["DOLoss"] = f"{ema_depth_order_loss_for_log:.{7}f}"
        if multiview_kick_on:
            postfix_dict["MVLoss"] = f"{ema_multiview_loss_for_log:.{7}f}"
        if normal_field_kick_on:
            postfix_dict["NFLoss"] = f"{ema_normal_field_alignment_loss_for_log:.{7}f}"
            if normal_field_config["enforce_front_pivots_to_be_visible"]:
                postfix_dict["VisLoss"] = f"{ema_front_pivots_visibility_loss_for_log:.{7}f}"
            if normal_field_config["enforce_back_pivots_to_be_occluded"]:
                postfix_dict["OcclLoss"] = f"{ema_back_pivots_occlusion_loss_for_log:.{7}f}"
            if normal_field_config["enforce_gaussian_flattening"]:
                postfix_dict["FlatLoss"] = f"{ema_gaussian_flattening_loss_for_log:.{7}f}"
            if normal_field_config["use_sdf"] and normal_field_config["enforce_sdf_and_normal_field_consistency"]:
                postfix_dict["SDFLoss"] = f"{ema_sdf_and_normal_field_consistency_loss_for_log:.{7}f}"
        if milo_kick_on:
            if milo_config["use_rgb_loss"]:
                raise NotImplementedError("RGB loss is not implemented")
            if milo_config["use_depth_loss"]:
                postfix_dict["MiDLoss"] = f"{ema_mesh_depth_loss_for_log:.{7}f}"
            if milo_config["use_normal_loss"]:
                postfix_dict["MiNLoss"] = f"{ema_mesh_normal_loss_for_log:.{7}f}"
            if milo_config["enforce_occupied_centers"]:
                postfix_dict["MiOccLoss"] = f"{ema_occupied_centers_loss_for_log:.{7}f}"
        postfix_dict["N_Gauss"] = f"{gaussians._xyz.shape[0]}"
        progress_bar.set_postfix(postfix_dict)
        progress_bar.update(10)
        
    if iteration == opt.iterations:
        progress_bar.close()
    
    # ---Logging---
    if (log_interval is not None) and (
        (iteration % log_interval == 0) 
        or (normal_field_kick_on and iteration == normal_field_config["start_iter"])
        or (milo_kick_on and iteration == milo_config["start_iter"])
    ):
        images_to_log, titles_to_log = [], []
        
        if depth_order_kick_on:
            images_to_log.append(viewpoint_cam.original_image.cuda())
            titles_to_log.append(f"GT RGB {viewpoint_idx}")
            
            images_to_log.append(do_supervision_depth)
            titles_to_log.append(f"Supervision Depth {viewpoint_idx}")
            
            images_to_log.append((1. - depth_to_normal(viewpoint_cam, do_supervision_depth)) / 2.)
            titles_to_log.append(f"Supervision Depth to Normal {viewpoint_idx}")
        
        images_to_log.append(render_pkg["render"])
        titles_to_log.append(f"Rendered RGB {viewpoint_idx}")
        
        if reg_kick_on or normal_field_kick_on or depth_order_kick_on:
            images_to_log.append(render_pkg["median_depth"])
            titles_to_log.append(f"Rendered Depth {viewpoint_idx}")
        
        if reg_kick_on or normal_field_kick_on:
            images_to_log.append((1. - render_pkg["normal"]) / 2.)
            titles_to_log.append(f"Rendered Normals {viewpoint_idx}")
            
        if normal_field_kick_on:
            images_to_log.append(
                (1. - normal_field_render_pkg["render"].detach()) / 2.
            )
            
            titles_to_log.append(f"Rendered Normal Field {viewpoint_idx}")

            # images_to_log.append(
            #     torch.where(
            #         normal_field_render_pkg["depth"].detach() > 0,
            #         normal_field_render_pkg["depth"].detach(),
            #         normal_field_render_pkg["depth"].detach().max().item()
            #     )
            # )
            images_to_log.append(
                (1. - normal_field_render_pkg["depth"].detach()) / 2.
            )
            titles_to_log.append(f"NF Depth {viewpoint_idx}")
            
            images_to_log.append(
                (1. - normal_field_render_pkg["normals"].detach()) / 2.
            )
            titles_to_log.append(f"NF Normals {viewpoint_idx}")
            
        if milo_kick_on:
            if milo_config["use_rgb_loss"]:
                raise NotImplementedError("RGB loss is not implemented")
                images_to_log.append(milo_pkg["mesh_rgb"])
                titles_to_log.append(f"Mesh RGB {viewpoint_idx}")
            if milo_config["use_depth_loss"]:
                images_to_log.append(milo_pkg["mesh_depth"])
                titles_to_log.append(f"Mesh Depth {viewpoint_idx}")
            if milo_config["use_normal_loss"]:                
                images_to_log.append(
                    (1. - fix_normal_map(viewpoint_cam, milo_pkg["mesh_normal"])) / 2.
                )
                titles_to_log.append(f"Mesh Normals {viewpoint_idx}")
        
        log_images_dict = make_log_figure(
            images=images_to_log, 
            titles=titles_to_log, 
            cmap='Spectral',
            ncols=3,
            figsize=30,
            show_plot=False,
            save_plot=not WANDB_FOUND,
            save_path=os.path.join(args.model_path, f"iter_{iteration}.png"),
            return_log_images=WANDB_FOUND,
        )
        
        if WANDB_FOUND:
            # Log metrics
            wandb_log_dict = {}
            for key, value in postfix_dict.items():
                try:
                    wandb_log_dict[key] = float(value)
                except:
                    wandb_log_dict[key] = value
            run.log(wandb_log_dict, step=iteration)
            
            # Log images
            wandb_log_images_dict = {}
            for log_img_name, log_img in log_images_dict.items():
                wandb_img_to_log = log_img.clone().detach().squeeze()
                # If grayscale, scale to 0-1 and apply colormap
                if wandb_img_to_log.ndim < 3:
                    wandb_img_to_log = (wandb_img_to_log - wandb_img_to_log.min()) / (wandb_img_to_log.max() - wandb_img_to_log.min())
                    cmap = plt.get_cmap('Spectral')
                    wandb_img_to_log = cmap(wandb_img_to_log)[..., :3]
                    wandb_img_to_log = torch.from_numpy(wandb_img_to_log).to(torch.float32)
                # If float, scale to 0-255
                if wandb_img_to_log.dtype == torch.float32:
                    wandb_img_to_log = (wandb_img_to_log.clamp(0., 1.) * 255).to(torch.uint8)
                # Convert to wandb image
                wandb_img_to_log = wandb_img_to_log.cpu().numpy()
                wandb_log_images_dict[log_img_name.replace(str(viewpoint_idx), "")] = wandb.Image(
                    wandb_img_to_log, caption=log_img_name
                )
            run.log(wandb_log_images_dict, step=iteration)

    # ---Report---
    training_report(iteration, l1_loss, testing_iterations, scene, render_imp, (pipe, background))
    if (iteration in saving_iterations):
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        scene.save(iteration)

    return (
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
    )