from typing import List, Optional
import torch
from arguments import PipelineParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer.sof import (
    GlobalSortOrder,
    default_splat_args,
    evaluate_vacancy_sof,
)
from regularization.sdf.depth_fusion import evaluate_mesh_colors_all_vertices
from extraction.mesh import extract_mesh
from utils.camera_utils import get_cameras_spatial_extent


@torch.no_grad()
def compute_vacancy_sof(
    points: torch.Tensor, 
    cameras: List[Camera], 
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    background: torch.Tensor,
    kernel_size: float,
    frustum_near: Optional[float]=None,
    frustum_far: Optional[float]=None,
):
    # Compute default frustum parameters if not provided
    if frustum_near is None or frustum_far is None:
        scene_radius = get_cameras_spatial_extent(cameras=cameras)['radius'].item()
        standard_scale = 6.
        if frustum_near is None:
            frustum_near = 0.02 * scene_radius / standard_scale
        if frustum_far is None:
            frustum_far = 1e6 * scene_radius / standard_scale
    
    # Define splat arguments
    splat_args = default_splat_args()
    splat_args.sort_settings.sort_order = GlobalSortOrder.MIN_Z_BOUNDING
    splat_args.meshing_settings.alpha_early_stop = False
    splat_args.meshing_settings.transmittance_threshold = 0.5
    
    # Evaluate vacancy
    vacancy = evaluate_vacancy_sof(
        points=points.view(-1, 3),
        views=cameras,
        gaussians=gaussians,
        pipeline=pipeline,
        background=background,
        kernel_size=kernel_size,
        splat_args=splat_args,
        znear=frustum_near,
        zfar=frustum_far,
    )
    
    return vacancy.view(-1)


@torch.no_grad()
def compute_occupancy_sof(
    points: torch.Tensor, 
    cameras: List[Camera], 
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    background: torch.Tensor,
    kernel_size: float,
    frustum_near: Optional[float]=None,
    frustum_far: Optional[float]=None,
):
    # Occupancy is the complement of the vacancy
    return 1. - compute_vacancy_sof(
        points=points,
        cameras=cameras,
        gaussians=gaussians,
        pipeline=pipeline,
        background=background,
        kernel_size=kernel_size,
        frustum_near=frustum_near,
        frustum_far=frustum_far,
    )


@torch.no_grad()
def initialize_pivots_colors(
    cameras: List[Camera],
    delaunay_tets: torch.Tensor,
    pivots: torch.Tensor,
    pivots_sdf: torch.Tensor,
    pivots_scale: torch.Tensor,
    filter_large_edges: bool=False,
    collapse_large_edges: bool=False,
    sdf_sh: Optional[torch.Tensor]=None,
    mtet_on_cpu: bool=False,
) -> torch.Tensor:
    
    # Extract mesh
    mesh = extract_mesh(
        delaunay_tets=delaunay_tets,
        pivots=pivots.view(-1, 3),
        pivots_sdf=pivots_sdf.view(-1),
        pivots_colors=pivots_colors.view(-1, 3),
        pivots_scale=pivots_scale.view(-1),
        filter_large_edges=filter_large_edges,
        collapse_large_edges=collapse_large_edges,
        return_details=False,
        sdf_sh=sdf_sh,
        mtet_on_cpu=mtet_on_cpu,
    )
    
    # Get pivots colors with depth fusion
    pivots_colors = evaluate_mesh_colors_all_vertices(
        views=cameras, 
        mesh=mesh,
        masks=None,
        use_scalable_renderer=True,
        override_points=pivots.view(-1, 3),
    ).view(*pivots.shape)  # (N_gaussians, n_pivots, 3)
    
    return pivots_colors


def convert_sdf_to_occupancy(
    sdf:torch.Tensor,
):
    return - sdf * 0.99 / 2. + 0.5  # Between 0.005 and 0.995


def convert_occupancy_to_sdf(
    occupancy:torch.Tensor,
):
    return - (occupancy - 0.5) * 2. / 0.99  # Between -1 and 1