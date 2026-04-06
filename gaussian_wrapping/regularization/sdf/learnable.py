from typing import List, Tuple, Callable
import numpy as np
import torch
from arguments import PipelineParams
from scene.mesh import Meshes, MeshRenderer
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.geometry_utils import is_in_view_frustum
from utils.tetmesh import marching_tetrahedra
from utils.geometry_utils import unflatten_voronoi_features
from regularization.sdf.depth_fusion import evaluate_sdf_values
import gc


def refine_intersections_with_binary_search(
    end_points:torch.Tensor,
    end_sdf:torch.Tensor,
    sdf_function:Callable,
    n_binary_steps:int,
) -> torch.Tensor:
    """
    Refine the intersected isosurface points with binary search.
    
    Args:
        end_points (torch.Tensor): The end points. (N_verts, 2, 3)
        end_sdf (torch.Tensor): The SDF values at the end points. (N_verts, 2, 1)
        sdf_function (Callable): The SDF function. Takes a tensor of points and returns the SDF values.
        n_binary_steps (int): The number of binary steps.
        
    Returns:
        refined_points (torch.Tensor): The refined points. (N_verts, 3)
    """
    
    left_points = end_points[:, 0, :].clone()  # (N_verts, 3)
    right_points = end_points[:, 1, :].clone()  # (N_verts, 3)
    left_sdf = end_sdf[:, 0, :].clone()  # (N_verts, 1)
    right_sdf = end_sdf[:, 1, :].clone()  # (N_verts, 1)
    points = (left_points + right_points) / 2  # (N_verts, 3)

    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        
        mid_sdf = sdf_function(mid_points)
        mid_sdf = mid_sdf.unsqueeze(-1)
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
        
        points = (left_points + right_points) / 2  # (N_verts, 3)

        torch.cuda.empty_cache()
        gc.collect()
        
    return points


def linearize_sdf_values(
    sdf_values:torch.Tensor,
    end_points:torch.Tensor,
    end_sdf:torch.Tensor,
    end_idx:torch.Tensor,
    verts:torch.Tensor,
    min_shift_length:float=1e-8,
) -> torch.Tensor:
    """
    Linearize the SDF values.
    
    Args:
        sdf_values (torch.Tensor): The SDF values. (N_voronoi, )
        end_points (torch.Tensor): The end points. (N_verts, 2, 3)
        end_sdf (torch.Tensor): The SDF values at the end points. (N_verts, 2, 1)
        end_idx (torch.Tensor): The indices of the end points. (N_verts, 2)
        verts (torch.Tensor): The vertices. (N_verts, 3)
        
    Returns:
        sdf_values (torch.Tensor): The linearized SDF values. (N_voronoi, 9)
    """
    
    shifts = (end_points[:, 0, :] - end_points[:, 1, :]).norm(dim=-1).clamp(min=min_shift_length)  # (N_verts, )
    factors = (end_sdf[:, 0].abs() + end_sdf[:, 1].abs()).squeeze()  # (N_verts, )
    
    sdf_0 = end_sdf[:, 0].sign().squeeze() * (verts - end_points[:, 0, :]).norm(dim=-1) / shifts * factors  # (N_verts, )
    sdf_1 = end_sdf[:, 1].sign().squeeze() * (verts - end_points[:, 1, :]).norm(dim=-1) / shifts * factors  # (N_verts, )
    sdfs = torch.cat([sdf_0[:, None], sdf_1[:, None]], dim=-1)  # (N_verts, 2)
    
    new_sdf_values = torch.zeros_like(sdf_values)
    sdf_counts = torch.zeros_like(sdf_values)
    
    new_sdf_values.index_add_(
        dim=0,
        index=end_idx.flatten().long(),
        source=sdfs.flatten()
    )
    sdf_counts.index_add_(
        dim=0,
        index=end_idx.flatten().long(),
        source=torch.ones_like(sdfs.flatten())
    )
    
    new_sdf_values = torch.where(
        sdf_counts > 0,
        new_sdf_values / sdf_counts,
        sdf_values
    )
    
    return new_sdf_values


def apply_iterative_linearization_to_sdf(
    initial_sdf:torch.Tensor,
    isosurface_verts:torch.Tensor,
    end_points:torch.Tensor,
    end_idx:torch.Tensor,
    n_steps:int=500,
    enforce_std:float=None,
) -> torch.Tensor:
    """
    Starting from initial SDF values, progressively linearize the SDF values
    to make them match a set of isosurface vertices after applying Marching Tetrahedra.

    Args:
        initial_sdf (torch.Tensor): The initial SDF values. (N_voronoi, )
        isosurface_verts (torch.Tensor): The intersected isosurface vertices along the edges. (N_edges, 3)
        end_points (torch.Tensor): The edge endpoints. (N_edges, 2, 3)
        end_idx (torch.Tensor): The indices in [0, N_voronoi) corresponding to the edge endpoints. (N_edges, 2)
        n_steps (int, optional): The number of steps. Defaults to 500.
        enforce_std (float, optional): The standard deviation to enforce. Defaults to None.
    
    Returns:
        linearized_sdf (torch.Tensor): The linearized SDF values in range [-1, 1]. (N_voronoi, )
    """
    linearized_sdf = initial_sdf.clone()
    
    for _ in range(n_steps):
        linearized_sdf = linearize_sdf_values(
            sdf_values=linearized_sdf,
            end_points=end_points,
            end_sdf=linearized_sdf[end_idx].unsqueeze(-1),
            end_idx=end_idx,
            verts=isosurface_verts,
            min_shift_length=1e-8
        )
        # linearized_sdf = linearized_sdf / linearized_sdf.abs().max()
        # linearized_sdf = linearized_sdf / linearized_sdf.std()
        if enforce_std is not None:
            linearized_sdf = (enforce_std * linearized_sdf / (linearized_sdf.std())).clamp(min=-1, max=1)
        else:
            linearized_sdf = linearized_sdf / linearized_sdf.abs().max()
    
    if n_steps == 0:
        if enforce_std is not None:
            linearized_sdf = (enforce_std * linearized_sdf / (linearized_sdf.std())).clamp(min=-1, max=1)
        else:
            linearized_sdf = linearized_sdf / linearized_sdf.abs().max()
    
    return linearized_sdf


@torch.no_grad()
def compute_initial_sdf_with_binary_search(
    voronoi_points:torch.Tensor,
    voronoi_scales:torch.Tensor,
    delaunay_tets:torch.Tensor,
    sdf_function:Callable,
    n_binary_steps:int,
    n_linearization_steps:int,
    enforce_std:float=None,
) -> torch.Tensor:
    """
    Compute initial SDF values with binary search and linearization.
    
    Args:
        voronoi_points (torch.Tensor): The voronoi points. (N_voronoi, 3)
        voronoi_scales (torch.Tensor): The voronoi scales. (N_voronoi, 1)
        delaunay_tets (torch.Tensor): The delaunay tets. (N_tets, 4)
        sdf_function (Callable): The SDF function.
        n_binary_steps (int): The number of binary steps.
        n_linearization_steps (int): The number of linearization steps.
    
    Returns:
        linearized_sdf (torch.Tensor): The linearized SDF values. (N_voronoi, )
    """
    
    # Compute initial SDF values
    voronoi_sdf = sdf_function(voronoi_points)  # (N_voronoi, )
    
    # Refine the initial SDF values with binary search
    if n_binary_steps > 0:
        # Initial Marching Tetrahedra
        verts_list, scale_list, faces_list, interp_v = marching_tetrahedra(
            vertices=voronoi_points[None],
            tets=delaunay_tets,
            sdf=voronoi_sdf.reshape(1, -1),
            scales=voronoi_scales[None]
        )
        end_points, end_sdf = verts_list[0]  # (N_verts, 2, 3) and (N_verts, 2, 1)
        end_scales = scale_list[0]  # (N_verts, 2, 1)
        end_idx = interp_v[0]  # (N_verts, 2)
        
        refined_verts = refine_intersections_with_binary_search(
            end_points=end_points,
            end_sdf=end_sdf,
            sdf_function=sdf_function,
            n_binary_steps=n_binary_steps,
        )
    # If no binary search, just return the initial SDF values
    else:
        n_linearization_steps = 0
        # norm_sdf = end_sdf.abs() / end_sdf.abs().sum(dim=1, keepdim=True)
        # refined_verts = end_points[:, 0, :] * norm_sdf[:, 1, :] + end_points[:, 1, :] * norm_sdf[:, 0, :]
        end_points = None
        end_idx = None
        refined_verts = None
    
    # Apply iterative linearization to the SDF values
    linearized_sdf = apply_iterative_linearization_to_sdf(
        initial_sdf=voronoi_sdf,
        isosurface_verts=refined_verts,
        end_points=end_points,
        end_idx=end_idx,
        n_steps=n_linearization_steps,
        enforce_std=enforce_std,
    )
    
    return linearized_sdf


def convert_sdf_to_occupancy(
    sdf:torch.Tensor,
):
    return - sdf * 0.99 / 2. + 0.5  # Between 0.005 and 0.995


def convert_occupancy_to_sdf(
    occupancy:torch.Tensor,
):
    return - (occupancy - 0.5) * 2. / 0.99  # Between -1 and 1
