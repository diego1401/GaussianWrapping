from typing import List, Callable
import torch
from arguments import PipelineParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.mesh import MeshRenderer, MeshRasterizer, Meshes
from functional.mesh import frustum_cull_mesh
from gaussian_renderer import (
    render_depth,
    render_simp,
)

@torch.no_grad()
def compute_depth_error_between_gaussian_and_mesh(
    gaussians: GaussianModel,
    mesh: Meshes,
    cameras: List[Camera],
    render_func: Callable,
    pipe: PipelineParams,
    background: torch.Tensor = torch.zeros(3, device="cuda"),
    method: str ="count",  # "count" or "area" or "none"
) -> torch.Tensor:
    """
    Compute, for each Gaussian, the average depth error between the Gaussian rendering and the mesh rendering.

    Args:
        gaussians (GaussianModel): The Gaussian model.
        mesh (Meshes): The mesh.
        cameras (List[Camera]): The cameras.
        render_func (Callable): The rendering function.
        pipe (PipelineParams): The pipeline parameters.
        background (torch.Tensor, optional): The background color. Defaults to torch.zeros(3, device="cuda").
        method (str, optional): The method to use for normalization. Defaults to "count".

    Raises:
        ValueError: If the method is not "count", "area" or "none".

    Returns:
        torch.Tensor: The average depth error between the Gaussian rendering and the mesh rendering. Has shape (N_gaussians,).
    """
    
    gaussian_errors = torch.zeros_like(gaussians._xyz[:, 0])
    
    # Get mesh renderer
    mesh_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras)
    )
    
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
        
        # Get Gaussian Splatting rendering
        gaussian_render_pkg = render_func(
            viewpoint_camera=cameras[i_img], 
            pc=gaussians, 
            pipe=pipe, 
            bg_color=background, 
            require_coord=False, 
            require_depth=True
        )
        gaussians_depth = gaussian_render_pkg["median_depth"]
        
        # Get mesh rendering
        mesh_render_pkg = mesh_renderer(
            mesh=frustum_cull_mesh(mesh, cameras[i_img]),
            cam_idx=i_img,
            return_depth=True,
            return_normals=True,
            use_antialiasing=True,
            return_pix_to_face=True,
            check_errors=True,
        )
        mesh_depth = mesh_render_pkg["depth"].squeeze()
        # mesh_pix_to_face = mesh_render_pkg["pix_to_face"].squeeze()
        
        # Compute depth error
        depth_error = (mesh_depth.unsqueeze(0) - gaussians_depth).clamp_min(0.) / gaussians_depth
        
        # Compute per-Gaussian error
        gaussian_errors_i = torch.zeros_like(gaussians_proj_area, dtype=torch.float32)
        gaussian_errors_i.index_add_(0, msv2_idx.flatten(), depth_error.flatten())
        
        # If count, we normalize by the number of pixels in which the Gaussian is visible
        if method == "count":
            gaussian_count = torch.zeros_like(gaussians_proj_area, dtype=torch.float32)
            gaussian_count.index_add_(0, msv2_idx.flatten(), torch.ones_like(depth_error.flatten(), dtype=torch.float32))
            
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
    
    return gaussian_errors / len(cameras)