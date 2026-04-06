from typing import Optional, List
import numpy as np
import torch
from einops import einsum
import math
from diff_gaussian_rasterization_sof import (
    GaussianRasterizationSettings, 
    ExtendedSettings, 
    GaussianRasterizer, 
    DebugVisualization, 
    SortMode, 
    GlobalSortOrder, 
    DebugVisualizationType,
)
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import focal2fov, fov2focal
from utils.geometry_utils import transform_points_world_to_view
from tqdm import tqdm


@torch.no_grad()
def get_frustum_mask_batched(points: torch.Tensor, cameras: List[Camera], near: float = 0.02, far: float = 1e6):
    
    N = 200_000
    
    mask = torch.empty(0, device='cuda', dtype=torch.bool)
    number_of_batches = np.ceil(len(points)/N).astype(int)
    for i in range(number_of_batches):        
        mask = torch.cat((mask, get_frustum_mask(points[N*i: N * (i+1)], cameras, near, far)))
    return mask
    
@torch.no_grad()
def get_frustum_mask(points: torch.Tensor, cameras: List[Camera], near: float = 0.02, far: float = 1e6):
    H, W = cameras[0].image_height, cameras[0].image_width

    intrinsics = torch.stack(
        [
            torch.Tensor(
                [[fov2focal(cam.FoVx, cam.image_width), 0, W / 2],
                 [0, fov2focal(cam.FoVy, cam.image_height), H / 2],
                 [0, 0, 1]]
            ) for cam in cameras
        ], 
        dim=0
    ).to(points.device)

    # full_proj_matrices: (n_view, 4, 4)
    view_matrices = torch.stack(
        [cam.world_view_transform for cam in cameras], dim=0
    ).transpose(1, 2)

    ones = torch.ones_like(points[:, 0]).unsqueeze(-1)
    # homo_points: (N, 4)
    homo_points = torch.cat([points, ones], dim=-1)

    # uv_points: (n_view, N, 4, 4)
    # Apply batch matrix multiplication to get uv_points for all cameras
    view_points = einsum(view_matrices, homo_points, "n_view b c, N c -> n_view N b")
    view_points = view_points[:, :, :3]

    uv_points = einsum(intrinsics, view_points, "n_view b c, n_view N c -> n_view N b")

    z = uv_points[:, :, -1:]
    uv_points = uv_points[:, :, :2] / z
    u, v = uv_points[:, :, 0], uv_points[:, :, 1]

    # Optionally, we can apply near-far culling
    # Apply near-far culling
    depth = view_points[:, :, -1]
    cull_near_fars = (depth >= near) & (depth <= far)

    # Apply frustum mask
    mask = torch.any(cull_near_fars & (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1), dim=0)
    return mask


# Defines default SOF settings based on the config file from the SOF repository
def default_splat_args() -> ExtendedSettings:
    splat_args = ExtendedSettings()
        
    splat_args.culling_settings.hierarchical_4x4_culling = False
    splat_args.culling_settings.rect_bounding = True
    splat_args.culling_settings.tight_opacity_bounding = True
    splat_args.culling_settings.tile_based_culling = False

    splat_args.load_balancing = True
    splat_args.proper_ewa_scaling = False
    splat_args.exact_depth = True

    splat_args.sort_settings.queue_sizes.per_pixel = 4
    splat_args.sort_settings.queue_sizes.tile_2x2 = 8
    splat_args.sort_settings.queue_sizes.tile_4x4 = 64
    splat_args.sort_settings.sort_mode = SortMode(3)
    splat_args.sort_settings.sort_order = GlobalSortOrder(3)
    
    return splat_args


def get_full_proj_transform_inverse(viewpoint_camera: Camera) -> torch.Tensor:
    return (
        viewpoint_camera.world_view_transform.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))
    ).squeeze(0).inverse()


def render_sof(
    viewpoint_camera : Camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size=0.0, scaling_modifier = 1.0, 
    require_coord : bool = False, require_depth : bool = True,
    colors_precomp : Optional[torch.Tensor]=None,
    splat_args: ExtendedSettings = None, 
    debugVis : DebugVisualization = DebugVisualization(),
    render_opacity : bool = False
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    if splat_args is None:
        # Default settings
        splat_args = default_splat_args()
        splat_args.render_opacity = render_opacity

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        inv_viewprojmatrix=get_full_proj_transform_inverse(viewpoint_camera=viewpoint_camera),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        settings=splat_args,
        debug_data=debugVis,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    # TODO: Add 3D filter
    if pc.use_unbounded_opacity:
        scales = pc.get_scaling_with_3D_filter
        opacity = pc.get_contribution(viewpoint_camera=viewpoint_camera)
    else:
        scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    # scales, opacity = pc.get_scaling, pc.get_opacity
    
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    if colors_precomp is None:
        shs = pc.get_features
    else:
        shs = None
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendering, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp = None,
        filter_3d = None,
        )
    
    # RGB
    rendered_image = rendering[:3, :, :]
    
    # Depth
    rendered_expected_depth = rendering[6:7, :, :]
    rendered_median_depth = rendering[6:7, :, :]
    
    # Normal
    rendered_normal = rendering[3:6, :, :]
    rendered_normal = torch.nn.functional.normalize(rendered_normal, p=2, dim=0)
    
    # Occupancy field
    occ_field = rendering[7:8, :, :]
    
    # Distortion map
    distortion_map = rendering[8:9, :, :]
    
    # Extent loss
    extent_loss = rendering[9:10, :, :]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": None,
            "expected_coord": None,
            "median_coord": None,
            "expected_depth": rendered_expected_depth,
            "median_depth": rendered_median_depth,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "normal":rendered_normal,
            "occ_field": occ_field,
            "distortion_map": distortion_map,
            "extent_loss": extent_loss,
            }
    
    
def integrate_sof(
    points3D, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
    kernel_size : float, scaling_modifier = 1.0, override_color = None,
    subpixel_offset=None, splat_args=None,
    alpha:torch.Tensor = None,
):
    
    assert alpha is not None, "Alpha is required for integration"
    
     # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")
        
    if splat_args is None:
        splat_args = default_splat_args()
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        inv_viewprojmatrix=get_full_proj_transform_inverse(viewpoint_camera=viewpoint_camera),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        settings=splat_args,
        debug_data=DebugVisualization(),
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means3D = pc.get_xyz
    rotations = pc.get_rotation
    if pc.use_unbounded_opacity:
        scales = pc.get_scaling_with_3D_filter
        opacity = pc.get_contribution(viewpoint_camera=viewpoint_camera)
    else:
        scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    
    colors_precomp = override_color
    if colors_precomp is None:
        shs = pc.get_features
    else:
        shs = None
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, color_integrated, radii = rasterizer.integrate(
        points3D = points3D,
        means3D = means3D,
        opacities = opacity,
        alpha = alpha,
        shs = shs,
        colors_precomp = colors_precomp,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        view2gaussian_precomp = None
    )
    
    return {"render": rendered_image,
            # "alpha_integrated": alpha,
            "color_integrated": color_integrated,
            "point_coordinate": None,
            "point_sdf": None,
            "viewspace_points": None,
            "visibility_filter" : radii > 0,
            "radii": radii}
    
    
@torch.no_grad()
def compute_transmittance(
    points3D, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
    kernel_size : float, scaling_modifier = 1.0,
    subpixel_offset=None, splat_args=None,
):  
     # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")
        
    if splat_args is None:
        splat_args = default_splat_args()
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        inv_viewprojmatrix=get_full_proj_transform_inverse(viewpoint_camera=viewpoint_camera),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        settings=splat_args,
        debug_data=DebugVisualization(),
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means3D = pc.get_xyz
    rotations = pc.get_rotation
    if pc.use_unbounded_opacity:
        scales = pc.get_scaling_with_3D_filter
        opacity = pc.get_contribution(viewpoint_camera=viewpoint_camera)
    else:
        scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    transmittance, radii = rasterizer.compute_transmittance(
        points3D = points3D,
        means3D = means3D,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        view2gaussian_precomp = None
    )
    
    return {
        "transmittance": transmittance,
        "visibility_filter" : radii > 0,
        "radii": radii
    }
    
    
@torch.no_grad()
def evaluate_occupancy_field_sof_frustum(points, views, gaussians, pipeline, background, kernel_size, splat_args: ExtendedSettings, exact_alpha_eval = False):
    alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    
    if splat_args.meshing_settings.return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")
    
    with torch.no_grad():       
        for _, view in enumerate(tqdm(views, desc="Meshing progress")):
            # print("\nPoints shape: ", points.shape)
            # print("Points dtype: ", points.dtype)
            # print("Gaussians xyz shape: ", gaussians.get_xyz.shape)
            # print("Gaussians xyz dtype: ", gaussians.get_xyz.dtype)
            # print("alpha shape: ", alpha.shape)
            # print("alpha dtype: ", alpha.dtype)
            # print("final_alpha shape: ", final_alpha.shape)
            # print("final_alpha dtype: ", final_alpha.dtype)            
            # print('Alpha min: ', alpha.min(), 'Alpha max: ', alpha.max(), 'Alpha mean: ', alpha.mean())
            
            ret = integrate_sof(
                points3D=points, 
                viewpoint_camera=view, 
                pc=gaussians, 
                pipe=pipeline, 
                bg_color=background, 
                kernel_size=kernel_size, 
                scaling_modifier = 1.0, 
                override_color = None,
                subpixel_offset=None, 
                splat_args=splat_args,
                alpha=alpha,
            )

            if splat_args.meshing_settings.return_color:
                color_integrated = ret["color_integrated"]
                final_color = torch.where((alpha < final_alpha).reshape(-1, 1), color_integrated, final_color)
                
            if exact_alpha_eval:
                final_alpha = torch.min(final_alpha, alpha)
                alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda") * ~(alpha == 0)
            
            # print("alpha shape: ", alpha.shape)
            # print("alpha dtype: ", alpha.dtype)
            # print("final_alpha shape: ", final_alpha.shape)
            # print("final_alpha dtype: ", final_alpha.dtype)
            # print('Alpha min: ', alpha.min(), 'Alpha max: ', alpha.max(), 'Alpha mean: ', alpha.mean())
            
        # if we have exact_eval, the result is in final_alpha, else it is in alpha
        if exact_alpha_eval:
            alpha = final_alpha + 0.
        else:
            alpha = alpha + 0.

    if splat_args.meshing_settings.return_color:
        return alpha, final_color
    return alpha, None


@torch.no_grad()
def evaluate_occupancy_field_sof(points, views, gaussians, pipeline, background, kernel_size, splat_args: ExtendedSettings, exact_alpha_eval = False):
    alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    vertex_mask = get_frustum_mask_batched(points, views, 0.02, 1e6)
    
    alpha[vertex_mask], _ = evaluate_occupancy_field_sof_frustum(
        points[vertex_mask], views, gaussians, pipeline, background, kernel_size, splat_args, exact_alpha_eval
    )
    
    return alpha, None

@torch.no_grad()
def evaluate_vacancy_sof(
    points, 
    views, 
    gaussians, 
    pipeline, 
    background, 
    kernel_size, 
    splat_args: ExtendedSettings, 
    znear = 0.02,
    zfar = 1e6,
):
    """
    Evaluate a lower bound of the vacancy field of the scene.

    Args:
        points (torch.Tensor): Points to evaluate the vacancy field at. (N, 3)
        views (List[Camera]): Views to evaluate the vacancy field at.
        gaussians (GaussianModel): Gaussian model to evaluate the vacancy field at.
        pipeline (_type_): _description_
        background (_type_): _description_
        kernel_size (_type_): _description_
        splat_args (ExtendedSettings): _description_
        znear (float, optional): _description_. Defaults to 0.02.
        zfar (_type_, optional): _description_. Defaults to 1e6.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    vacancy = torch.zeros((points.shape[0]), dtype=torch.float32, device="cuda")
    
    for _, view in enumerate(tqdm(views, desc="Vacancy evaluation progress")):
        frustum_mask = get_frustum_mask(points=points, cameras=[view], near=znear, far=zfar)
        
        ret = compute_transmittance(
            points3D=points[frustum_mask],  # (N_frustum, 3)
            viewpoint_camera=view, 
            pc=gaussians, 
            pipe=pipeline, 
            bg_color=background, 
            kernel_size=kernel_size, 
            scaling_modifier=1.0, 
            subpixel_offset=None, 
            splat_args=splat_args, 
        )
        
        vacancy[frustum_mask] = torch.maximum(
            vacancy[frustum_mask],  # (N_frustum,)
            ret["transmittance"],  # (N_frustum,)
        )
        
    return vacancy


@torch.no_grad()
def evaluate_vacancy_sof_fast(
    points, 
    views, 
    gaussians, 
    pipeline, 
    background, 
    kernel_size, 
    splat_args: ExtendedSettings, 
    znear = 0.02,
    zfar = 1e6,
    permute_views = True,
):
    """
    Evaluate if the vacancy at a point is greater than a threshold.
    Returns a boolean mask indicating which points have a vacancy value greater than the threshold in splat_args.meshing_settings.transmittance_threshold.

    Args:
        points (torch.Tensor): Points to evaluate the vacancy field at. (N, 3)
        views (List[Camera]): Views to evaluate the vacancy field at.
        gaussians (GaussianModel): Gaussian model to evaluate the vacancy field at.
        pipeline (_type_): _description_
        background (_type_): _description_
        kernel_size (_type_): _description_
        splat_args (ExtendedSettings): _description_
        znear (float, optional): _description_. Defaults to 0.02.
        zfar (_type_, optional): _description_. Defaults to 1e6.
        permute_views (bool, optional): Whether to permute the list of views. Defaults to True.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: Boolean mask indicating which points have a vacancy value greater than the threshold. Shape (N,).
    """
    vacancy = torch.zeros((points.shape[0]), dtype=torch.bool, device="cuda")
    update_idx = torch.arange(points.shape[0], device="cuda")
    
    # Permute the list of views
    if permute_views:
        permuted_indices = np.random.permutation(len(views))
        views_to_use = [views[i] for i in permuted_indices]
    else:
        views_to_use = views
    
    for _, view in enumerate(tqdm(views_to_use, desc="Vacancy evaluation progress")):
        # Get current remaining points to update
        N_current = update_idx.shape[0]
        current_points = points[update_idx]  # (N_current, 3)
        
        # Get frustum mask for current view
        frustum_mask = get_frustum_mask(points=current_points, cameras=[view], near=znear, far=zfar)  # (N_current,)
        
        # Compute transmittance of points that are in the frustum
        ret = compute_transmittance(
            points3D=current_points[frustum_mask],  # (N_frustum, 3)
            viewpoint_camera=view, 
            pc=gaussians, 
            pipe=pipeline, 
            bg_color=background, 
            kernel_size=kernel_size, 
            scaling_modifier=1.0, 
            subpixel_offset=None, 
            splat_args=splat_args, 
        )
        
        # Compute mask of points that pass the threshold
        passing_mask = torch.zeros((N_current), dtype=torch.bool, device="cuda")  # (N_current,)
        passing_mask[frustum_mask] = ret["transmittance"] > splat_args.meshing_settings.transmittance_threshold  # (N_frustum,)
        
        # Get indices of points that pass the threshold
        pass_idx = update_idx[passing_mask]  # (N_pass,)
        
        # Update vacancy of points that pass the threshold
        vacancy[pass_idx] = True  # (N_pass,)
        
        # Update update_idx to only include points that did not pass the threshold yet
        update_idx = update_idx[~passing_mask]  # (N_current - N_pass,)
        
    return vacancy


def integrate_sof_(points3D, alpha, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size: float, scaling_modifier = 1.0, override_color = None, subpixel_offset=None, splat_args=None):
    """
    integrate Gaussians to the points, we also render the image for visual comparison. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # try:
    #     screenspace_points.retain_grad()
    # except:
    #     pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        inv_viewprojmatrix=get_full_proj_transform_inverse(viewpoint_camera=viewpoint_camera),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        settings=splat_args,
        debug_data=DebugVisualization(),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    # no 2D means required here
    
    scales = None
    opacity = None
    # TODO: add 3d filter in cuda here as well
    scales = pc.get_scaling_with_3D_filter
    opacity = pc.get_opacity_with_3D_filter


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    rotations = pc.get_rotation

    view2gaussian_precomp = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, color_integrated, radii = rasterizer.integrate(
            points3D = points3D,
            means3D = means3D,
            opacities = opacity,
            alpha = alpha,
            shs = shs,
            colors_precomp = colors_precomp,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            view2gaussian_precomp=view2gaussian_precomp)

    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "color_integrated": color_integrated,
            "viewspace_points": None,
            "radii": radii}
    
    
@torch.no_grad()
def evaluate_alpha_sof_proto(points, views, gaussians, pipeline, background, kernel_size, splat_args: ExtendedSettings, exact_alpha_eval = False, model_path: str = None):
    alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    
    if splat_args.meshing_settings.return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")
    
    import torchvision
    import os
    
    with torch.no_grad():       
        for _, view in enumerate(tqdm(views, desc="Meshing progress")):
            ret = integrate_sof_(points, alpha, view, gaussians, pipeline, background, kernel_size=kernel_size, splat_args=splat_args)

            if splat_args.meshing_settings.return_color:
                color_integrated = ret["color_integrated"]
                final_color = torch.where((alpha < final_alpha).reshape(-1, 1), color_integrated, final_color)
                
            if exact_alpha_eval:
                final_alpha = torch.min(final_alpha, alpha)
                alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda") * ~(alpha == 0)
            
        # if we have exact_eval, the result is in final_alpha, else it is in alpha
        if exact_alpha_eval:
            alpha = 1 - final_alpha
        else:
            alpha = 1 - alpha

    if splat_args.meshing_settings.return_color:
        return alpha, final_color
    return alpha, None


@torch.no_grad()
def evaluate_alpha_sof_(points, views, gaussians, pipeline, background, kernel_size, splat_args: ExtendedSettings, exact_alpha_eval = False, model_path: str = None):
    alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    vertex_mask = get_frustum_mask_batched(points, views, 0.02, 1e6)
    
    points_to_integrate = points[vertex_mask]
    
    alpha_to_integrate, _ = evaluate_alpha_sof_proto(points_to_integrate, views, gaussians, pipeline, background, kernel_size, splat_args, exact_alpha_eval, model_path)
    alpha[vertex_mask] = alpha_to_integrate
    
    return alpha, None