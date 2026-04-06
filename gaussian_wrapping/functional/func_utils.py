# A large portion of this script is adapted from MiniSplatting2.
# Link: https://github.com/fatPeter/mini-splatting2
#
from typing import Union, Dict
import math
import torch
from diff_gaussian_rasterization_ms import GaussianRasterizationSettings as MiniSplatting2RasterizationSettings
from diff_gaussian_rasterization_ms import GaussianRasterizer as MiniSplatting2Rasterizer
from scene.cameras import Camera


def _init_cdf_mask(
    importance:torch.Tensor, 
    thres:float=1.0
) -> torch.Tensor:
    importance = importance.flatten()   
    if thres!=1.0:
        percent_sum = thres
        vals,idx = torch.sort(importance+(1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]

        non_prune_mask = importance>split_val_nonprune 
    else: 
        non_prune_mask = torch.ones_like(importance).bool()
        
    return non_prune_mask


def _render_simp(
    viewpoint_camera:Camera, 
    means:torch.Tensor,
    opacities:torch.Tensor,
    scales:torch.Tensor,
    rotations:torch.Tensor,
    bg_color:torch.Tensor,
    dc:Union[torch.Tensor, None]=None,
    shs:Union[torch.Tensor, None]=None,
    override_color:Union[torch.Tensor, None]=None, 
    scaling_modifier:float=1.0, 
    culling:Union[torch.Tensor, None]=None
) -> Dict[str, torch.Tensor]:
    """
    Background tensor (bg_color) must be on GPU!
    """
    
    assert (dc is not None and shs is not None) or (override_color is not None)
    if override_color is not None:
        active_sh_degree = 0
    else:
        active_sh_degree = int(math.sqrt(shs.shape[1] + 1) - 1)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means, dtype=means.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = MiniSplatting2RasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = MiniSplatting2Rasterizer(raster_settings=raster_settings)
    means2D = screenspace_points

    if culling==None:
        culling=torch.zeros(means.shape[0], dtype=torch.bool, device='cuda')

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    (
        rendered_image, radii, 
        accum_weights_ptr, accum_weights_count, accum_max_count
    )  = rasterizer.render_simp(
        means3D=means,
        means2D=means2D,
        dc=dc,
        shs=shs,
        culling=culling,
        colors_precomp=override_color,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # The Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "accum_weights": accum_weights_ptr,
            "area_proj": accum_weights_count,
            "area_max": accum_max_count,
        }