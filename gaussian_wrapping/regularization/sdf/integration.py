from typing import List, Tuple, Union, Callable
import numpy as np
import torch
from scene.cameras import Camera
from arguments import PipelineParams
from scene.gaussian_model import GaussianModel
from tqdm import tqdm
import gc


@torch.no_grad()
def evaluate_sdf_values(points, views, gaussians, pipeline, background, kernel_size, isosurface_value=0.5, integrate_func:Callable=None):
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    if integrate_func is None:
        raise ValueError("integrate_func must be provided.")
    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            ret = integrate_func(points, view, gaussians, pipeline, background, kernel_size=kernel_size)
            alpha_integrated = ret["alpha_integrated"]
            final_alpha = torch.min(final_alpha, alpha_integrated)
        sdf = isosurface_value - final_alpha
    return sdf


@torch.no_grad()
def evaluate_cull_sdf_values(
    points, views, masks, gaussians, pipeline, background, kernel_size, return_colors=False, 
    isosurface_value:Union[float, torch.Tensor]=0.5, 
    transform_sdf_to_linear_space:bool=False, 
    min_occupancy_value:float=1e-10,
    integrate_func:Callable=None,
):    
    if integrate_func is None:
        raise ValueError("integrate_func must be provided.")

    # final_sdf = torch.zeros((points.shape[0]), dtype=torch.float32, device="cuda")
    final_sdf = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    weight = torch.zeros((points.shape[0]), dtype=torch.int32, device="cuda")
    if return_colors:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")    
    with torch.no_grad():
        for cam_id, view in enumerate(tqdm(views, desc="Rendering progress")):
            
            if cam_id % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            ret = integrate_func(points, view, gaussians, pipeline, background, kernel_size)
            alpha_integrated = ret["alpha_integrated"]
            point_coordinate = ret["point_coordinate"]
            if point_coordinate is not None:
                point_coordinate[:,0] = (point_coordinate[:,0]*2+1)/(views[cam_id].image_width-1) - 1
                point_coordinate[:,1] = (point_coordinate[:,1]*2+1)/(views[cam_id].image_height-1) - 1
                rendered_mask = ret["render"][7]
                mask = rendered_mask[None]
                if not view.gt_mask is None:
                    mask = mask * view.gt_mask
                if not masks is None:
                    mask = mask * masks[cam_id]
                valid_point_prob = torch.nn.functional.grid_sample(mask.type(torch.float32)[None],point_coordinate[None,None],padding_mode='zeros',align_corners=False)
                valid_point_prob = valid_point_prob[0,0,0]
                valid_point = valid_point_prob>0.5
            else:
                valid_point = torch.ones_like(alpha_integrated, dtype=torch.bool)

            if return_colors:
                color_integrated = ret["color_integrated"]
                final_color = torch.where(
                    valid_point.reshape(-1, 1) * (alpha_integrated < final_sdf).reshape(-1, 1), 
                    color_integrated, 
                    final_color,
                )
            final_sdf = torch.where(valid_point, torch.min(alpha_integrated, final_sdf), final_sdf)
            weight = torch.where(valid_point, weight+1, weight)
        
        # Old
        # final_sdf = torch.where(weight > 0, isosurface_value - final_sdf, -100)
        
        # New
        if not isinstance(isosurface_value, torch.Tensor):
            isosurface_value = torch.tensor(isosurface_value, device=final_sdf.device)
        else:
            isosurface_value = isosurface_value.squeeze()
        
        if transform_sdf_to_linear_space:
            print("Transforming SDF values to linear space...")
            final_sdf = (
                torch.sqrt(-2. * torch.log(final_sdf.clamp(min=min_occupancy_value))) 
                - torch.sqrt(-2. * torch.log(isosurface_value))
            )
        else:
            final_sdf = isosurface_value - final_sdf
        
        final_sdf = torch.where(weight > 0, final_sdf, -100)
    
    if return_colors:
        return final_sdf, final_color
    return final_sdf


@torch.no_grad()
def evaluate_cull_sdf_values_chunk(
    points:torch.Tensor, 
    chunk_size:int,
    **kwargs
):
    n_chunks = (len(points) + chunk_size - 1) // chunk_size
    sdf_values = torch.zeros((len(points)), dtype=torch.float32, device=points.device)
    if "return_colors" in kwargs and kwargs["return_colors"]:
        _return_colors = True
        color_values = torch.zeros((len(points), 3), dtype=torch.float32, device=points.device)
    else:
        _return_colors = False
    
    torch.cuda.empty_cache()
    gc.collect()
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(points))
        points_chunk = points[start_idx:end_idx]
        output = evaluate_cull_sdf_values(points_chunk, **kwargs)
        
        if isinstance(output, tuple):
            sdf_values[start_idx:end_idx] = output[0]
            color_values[start_idx:end_idx] = output[1]
        else:
            sdf_values[start_idx:end_idx] = output
            
        torch.cuda.empty_cache()
        gc.collect()
    if _return_colors:
        return sdf_values, color_values
    return sdf_values
