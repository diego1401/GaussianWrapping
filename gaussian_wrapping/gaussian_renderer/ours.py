from typing import Optional
import math
import torch
from diff_gaussian_rasterization_ours import GaussianRasterizationSettings, GaussianRasterizer
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel


def render_ours(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    kernel_size = 0.0, 
    scaling_modifier = 1.0, 
    require_coord : bool = False,
    require_depth : bool = True,
    colors_precomp : Optional[torch.Tensor]=None,
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
    
    active_sg_degree = 0

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sg_degree=active_sg_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        require_depth = require_depth,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    # TODO: Add 3D filter
    if pc.use_unbounded_opacity:
        scales = pc.get_scaling_with_3D_filter
        opacity = pc.get_contribution(viewpoint_camera=viewpoint_camera)
    else:
        scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_features if colors_precomp is None else None
    
    # sg_axis = pc.get_sg_axis
    # sg_sharpness = pc.get_sg_sharpness
    # sg_color = pc.get_sg_color
    
    n_gaussians = pc._xyz.shape[0]
    sg_axis = torch.zeros(n_gaussians, active_sg_degree, 3, device=pc._xyz.device)
    sg_sharpness = torch.zeros(n_gaussians, active_sg_degree, device=pc._xyz.device)
    sg_color = torch.zeros(n_gaussians, active_sg_degree, 3, device=pc._xyz.device)

    rendered_image, radii, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        sg_axis = sg_axis,
        sg_sharpness = sg_sharpness,
        sg_color = sg_color,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": None,
            "expected_coord": None,
            "median_coord": None,
            "expected_depth": rendered_median_depth,
            "median_depth": rendered_median_depth,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "normal":rendered_normal,
            }


# integration is adopted from GOF for marching tetrahedra https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/gaussian_renderer/__init__.py
def integrate_ours(points3D, viewpoint_camera, pc : GaussianModel, pipe, kernel_size : float, scaling_modifier = 1.0):
    """
    integrate Gaussians to the points
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    active_sg_degree = 0
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = kernel_size,
        bg=None,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sg_degree=active_sg_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        require_depth = True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    depth_plane_precomp = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    alpha_integrated, inside = rasterizer.integrate(
        points3D = points3D,
        means3D = means3D,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=depth_plane_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"alpha_integrated": alpha_integrated,
            "inside": inside}
    
    
def sample_depth_with_ours(points3D, viewpoint_camera: Camera, pc : GaussianModel, pipe : torch.Tensor, kernel_size : float, scaling_modifier = 1.0):

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    active_sg_degree = 0

        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = kernel_size,
        bg=0,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sg_degree=active_sg_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        require_depth = True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    depth, inside = rasterizer.sample_depth(
        points3D = points3D,
        means3D = means3D,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"sampled_depth": depth,
            "inside": inside}
    
    
@torch.no_grad()
def sample_depth_and_normal_in_view(
    points: torch.Tensor, 
    viewpoint_camera: Camera, 
    gaussians: GaussianModel, 
    pipe, 
    background: torch.Tensor, 
    kernel_size: float=0.0, 
    scaling_modifier: float=1.0
):
    from utils.geometry_utils import sample_depth_normal, depth_to_normal_with_mask
    
    # Get depth and normal maps
    render_pkg = render_ours(
        viewpoint_camera=viewpoint_camera,
        pc=gaussians,
        pipe=pipe,
        bg_color=background,
        kernel_size=kernel_size,
        scaling_modifier=scaling_modifier,
    )
    depth = render_pkg["median_depth"]  # (1, H, W)
    normal, _ = depth_to_normal_with_mask(viewpoint_camera, depth)  # (3, H, W), (1, H, W)
    
    # Sample depth and normal for each point
    return sample_depth_normal(viewpoint_camera, depth, normal, points)