from typing import Optional, Callable, Tuple, List
import torch
import trimesh
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from regularization.normal_field import get_gaussian_std_in_direction
from utils.general_utils import build_rotation


def get_intersecting_pivots_from_normals(
    n_pivots: int = 2,
    gaussians: Optional[GaussianModel]=None,
    normals: Optional[torch.Tensor]=None,
    std_factor: float = 3.0,
    xyz: Optional[torch.Tensor]=None,
    scaling: Optional[torch.Tensor]=None,
    rotation: Optional[torch.Tensor]=None,
    use_smallest_axis_as_normal: bool = False,
    sdf_function: Optional[Callable]=None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Get pivots from Gaussians based on their learned normals.
    The learned normal can be replaced by the smallest axis of the Gaussian by setting use_smallest_axis_as_normal to True.
    
    An sdf_function can be provided to only return the pivots for which the surface crosses between the pivots and the center point.
    This drastically reduces the number of pivots to be considered for the mesh extraction, but can hurt the quality of the mesh.

    Args:
        n_pivots (int, optional): _description_. Defaults to 2.
        gaussians (Optional[GaussianModel], optional): _description_. Defaults to None.
        normals (Optional[torch.Tensor], optional): _description_. Defaults to None.
        std_factor (float, optional): _description_. Defaults to 3.0.
        xyz (Optional[torch.Tensor], optional): _description_. Defaults to None.
        scaling (Optional[torch.Tensor], optional): _description_. Defaults to None.
        rotation (Optional[torch.Tensor], optional): _description_. Defaults to None.
        use_smallest_axis_as_normal (bool, optional): _description_. Defaults to False.
        sdf_function (Optional[Callable], optional): _description_. Defaults to None.

    Raises:
        ValueError: If gaussians or xyz, scaling and rotation are not provided.
        ValueError: If n_pivots is not in [2, 3, 7, 9].

    Returns:
        torch.Tensor: A tuple containing the pivots, the pivot scales, and the SDF values for the pivots.
            - pivots (torch.Tensor): The pivots, of shape (N_gaussians, n_pivots, 3).
            - pivot_scales (torch.Tensor): The pivot scales, of shape (N_gaussians, n_pivots, 1).
            - pivot_sdf (Optional[torch.Tensor]): The SDF values for the pivots, of shape (N_gaussians, n_pivots). Only returned if sdf_function is provided.
    """
    assert (
        gaussians is not None
        or (xyz is not None and scaling is not None and rotation is not None and normals is not None)
    ), "Either gaussians or xyz, scaling and rotation must be provided"

    assert n_pivots in [2, 3, 7, 9], f"Invalid number of pivots: {n_pivots}"
    
    if xyz is None:
        xyz = gaussians.get_xyz  # (N_gaussians, 3)
    if scaling is None:
        scaling = gaussians.get_scaling_with_3D_filter  # (N_gaussians, 3)
    if rotation is None:
        rotation = gaussians.get_rotation  # (N_gaussians, 4)
        
    if use_smallest_axis_as_normal:
        #   > Compute main Gaussian axes
        rots = build_rotation(rotation)  # (N_gaussians, 3, n_axes)
        axes = rots.transpose(-1, -2)  # (N_gaussians, n_axes, 3)
        
        min_scale_idx = torch.argmin(scaling, dim=-1, keepdim=True)  # (N_gaussians, 1)
        
        normals = torch.gather(
            input=axes,  # (N_gaussians, n_axes, 3)
            index=min_scale_idx.unsqueeze(-1).repeat(1, 1, 3),  # (N_gaussians, 1, 3)
            dim=1
        )  # (N_gaussians, 1, 3)
        normals = normals.squeeze(1)  # (N_gaussians, 3)
    else:
        if normals is None:
            normals = gaussians.convert_features_to_normals()  # (N_gaussians, 3)
    
    n_pivots_no_center = n_pivots - 1
    
    # Normalize the normals
    normals = torch.nn.functional.normalize(normals, dim=-1)  # (N_gaussians, 3)
    
    # If using 2 or 3 pivots, we just need the normal
    if n_pivots in [2, 3]:
        # Get the standard deviation of the Gaussian in the direction of the normal
        normal_stds = get_gaussian_std_in_direction(
            directions=normals.unsqueeze(1),  # (N_gaussians, 1, 3)
            gaussians=None,
            gaussian_scaling=scaling,
            gaussian_rotation=rotation, 
            normalize_directions=False,
        )  # (N_gaussians, 1)
    
    # If using 7 or 9 pivots, we need to compute an orthonormal basis relying on the learned normal
    elif n_pivots in [7, 9]:
        # Compute an orthonormal basis relying on the learned normal
        #   > Compute main Gaussian axes
        rots = build_rotation(rotation)  # (N_gaussians, 3, n_axes)
        axes = rots.transpose(-1, -2)  # (N_gaussians, n_axes, 3)
        
        #   > Project axes onto the normal
        axes_projections = (
            normals.unsqueeze(1)  # (N_gaussians, 1, 3)
            * axes  # (N_gaussians, n_axes, 3)
        ).sum(dim=-1).abs()  # (N_gaussians, n_axes)
        
        # Get the two axes with the smallest absolute projection
        two_smallest_axes_idx = torch.argsort(axes_projections, dim=-1)[:, :2]  # (N_gaussians, 2)
        two_smallest_axes = torch.gather(
            input=axes,  # (N_gaussians, n_axes, 3)
            index=two_smallest_axes_idx.unsqueeze(-1).repeat(1, 1, 3),  # (N_gaussians, 2, 3)
            dim=1
        )  # (N_gaussians, 2, 3)
        
        # Use normal as first axis and apply Gram-Schmidt orthonormalization to the two remaining axes
        new_axes = torch.zeros_like(axes)  # (N_gaussians, n_axes, 3)
        new_axes[:, 0, :] = normals  # (N_gaussians, 3)
        new_axes[:, 1, :] = torch.nn.functional.normalize(
            two_smallest_axes[:, 1, :]  # (N_gaussians, 3)
            - (two_smallest_axes[:, 1, :] * new_axes[:, 0, :]).sum(dim=-1, keepdim=True) * new_axes[:, 0, :],  # (N_gaussians, 3)
            dim=-1,
        )  # (N_gaussians, 3)
        new_axes[:, 2, :] = torch.nn.functional.normalize(
            torch.cross(
                new_axes[:, 0, :],  # (N_gaussians, 3)
                new_axes[:, 1, :],  # (N_gaussians, 3)
                dim=-1,
            ),
            dim=-1,
        )  # (N_gaussians, 3)
        
        # Get the standard deviation of the awes
        axes_stds = get_gaussian_std_in_direction(
            directions=new_axes,  # (N_gaussians, n_axes, 3)
            gaussians=None,
            gaussian_scaling=scaling,
            gaussian_rotation=rotation, 
            normalize_directions=False,
        ).unsqueeze(-1)  # (N_gaussians, n_axes, 1)
    
    pivots = torch.zeros(xyz.shape[0], n_pivots_no_center, 3, device=xyz.device)
    
    # Apart from the center point, we use:
    if (n_pivots_no_center == 1):
        # A pivot in front of the Gaussian in the direction of the normal
        pivots[:, 0, :] = xyz + std_factor * normal_stds * normals
        
    elif n_pivots_no_center == 2:
        # Two pivots, one in front and one behind the Gaussian in the direction of the normal
        pivots[:, 0, :] = xyz + std_factor * normal_stds * normals    
        pivots[:, 1, :] = xyz - std_factor * normal_stds * normals
        
    elif n_pivots_no_center == 6:
        # Six pivots, two in each direction of the orthogonal basis
        pivots[:, 0, :] = xyz + std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :]
        pivots[:, 1, :] = xyz - std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :]
        
        pivots[:, 2, :] = xyz + std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :]
        pivots[:, 3, :] = xyz - std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :]
        
        pivots[:, 4, :] = xyz + std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        pivots[:, 5, :] = xyz - std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        
    elif n_pivots_no_center == 8:
        # Eight pivots, one for each corner of the box aligned with the orthogonal basis
        pivots[:, 0, :] = (
            xyz 
            + std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            + std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            + std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
        pivots[:, 1, :] = (
            xyz 
            + std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            + std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            - std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
        pivots[:, 2, :] = (
            xyz 
            + std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            - std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            + std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
        pivots[:, 3, :] = (
            xyz 
            + std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            - std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            - std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
        pivots[:, 4, :] = (
            xyz 
            - std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            + std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            + std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
        pivots[:, 5, :] = (
            xyz 
            - std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            + std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            - std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
        pivots[:, 6, :] = (
            xyz 
            - std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            - std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            + std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
        pivots[:, 7, :] = (
            xyz 
            - std_factor * axes_stds[:, 0, :] * new_axes[:, 0, :] 
            - std_factor * axes_stds[:, 1, :] * new_axes[:, 1, :] 
            - std_factor * axes_stds[:, 2, :] * new_axes[:, 2, :]
        )
        
    else:
        raise ValueError(f"Invalid number of pivots: {n_pivots}")
    
    # Add the center point of the Gaussian to the pivots
    pivots = torch.cat(
        [
            pivots,  # (N_gaussians, n_pivots_no_center, 3)
            xyz.unsqueeze(1),  # (N_gaussians, 1, 3)
        ], 
        dim=1
    )  # (N_gaussians, n_pivots, 3)
    
    # Get pivot scales
    pivot_scales = 3. * scaling.detach().max(dim=-1, keepdim=True).values.unsqueeze(1).repeat(1, n_pivots, 1)  # (N_gaussians, n_pivots, 1)
    
    # If an SDF function is provided, we only return the pivots for which the surface crosses between the pivots and the center point
    if sdf_function is not None:
        with torch.no_grad():
            sdf = sdf_function(pivots.reshape(-1, 3)).reshape(-1, n_pivots)  # (N_gaussians, n_pivots)
        intersection_mask = sdf[:, :-1] * sdf[:, -1:] < 0.0  # (N_gaussians, n_pivots - 1)
        
        pivots = torch.cat(
            [
                pivots[:, :-1, :][intersection_mask],
                pivots[:, -1, :][intersection_mask.any(dim=-1)]
            ],
            dim=0,
        )
        
        pivot_scales = torch.cat(
            [
                pivot_scales[:, :-1, :][intersection_mask],
                pivot_scales[:, -1, :][intersection_mask.any(dim=-1)]
            ],
            dim=0,
        )
        
        pivot_sdf = torch.cat(
            [
                sdf[:, :-1][intersection_mask],
                sdf[:, -1][intersection_mask.any(dim=-1)]
            ],
            dim=0,
        )
        
        return pivots, pivot_scales, pivot_sdf  # (N_gaussians, n_pivots, 3), (N_gaussians, n_pivots, 1), (N_gaussians, n_pivots)
    
    return pivots, pivot_scales  # (N_gaussians, n_pivots, 3), (N_gaussians, n_pivots, 1)


@torch.no_grad()
def compute_pivots_scores(
    pivots_directions: torch.Tensor, 
    cameras: List[Camera],
    gaussians: GaussianModel, 
    pipe, 
    background: torch.Tensor, 
    kernel_size=0.0, 
    scaling_modifier=1.0
):
    from gaussian_renderer.ours import (
        integrate_ours,
        sample_depth_and_normal_in_view,
    )
    
    N, P, _ = pivots_directions.shape
    scores = torch.zeros(N, P, device=pivots_directions.device)
    
    for camera in cameras:
        # Get Gaussian centers
        gaussian_centers = gaussians.get_xyz
        
        # Get Gaussian transmittance
        integrate_pkg = integrate_ours(
            points3D = gaussian_centers,
            viewpoint_camera=camera,
            pc=gaussians,
            pipe=pipe,
            kernel_size=kernel_size,
            scaling_modifier=scaling_modifier,
        )
        gaussian_T = 1. - integrate_pkg["alpha_integrated"]  # (N,)
        
        # Sample depth and normal
        sample_pkg = sample_depth_and_normal_in_view(
            gaussian_centers,  # (N, 3)
            camera, 
            gaussians, 
            pipe, 
            background, 
        )
        valid_points = sample_pkg["valid"]  # (N,)
        # sampled_depth = sample_pkg["sampled_depth"]  # (N, 3)
        sampled_normal = sample_pkg["sampled_normal"]  # (N, 3)
        
        # Compute dot products
        dot_products = (
            pivots_directions  # (N, P, 3)
            * sampled_normal.unsqueeze(1)  # (N, 1, 3)
        ).sum(dim=-1).clamp_min(0.)  # (N, P)

        dot_products[~valid_points] = 0.  # (N, P)
        
        # Get weight updates
        scores = scores + gaussian_T.unsqueeze(1) * dot_products  # (N, P)
    
    return scores / len(cameras)


def get_pivots_by_scores(
    gaussians: GaussianModel,
    cameras: List[Camera],
    pipe,
    background: torch.Tensor,
    score_ratio_threshold: float=0.75,
    kernel_size: float=0.0,
    scaling_modifier: float=1.0,
):
    """
    Get pivots by scores.

    Args:
        gaussians (GaussianModel): _description_
        cameras (List[Camera]): _description_
        pivots_scores (torch.Tensor): The scores of the 
    """
    # Compute normals
    gaussian_normals = gaussians.convert_features_to_normals(normalize=True)  # (N_gaussians, 3)
    gaussian_normals = torch.nn.functional.normalize(gaussian_normals, dim=-1)  # (N_gaussians, 3)
    
    # Get pivots
    multi_pivots, _ = get_intersecting_pivots_from_normals(n_pivots=7, gaussians=gaussians)  # (N_gaussians, 7, 3)
    
    # Get directions of non-center pivots
    pivots_directions = torch.nn.functional.normalize(multi_pivots[:, :-1] - multi_pivots[:, -1:], dim=-1)  # (N_gaussians, 6, 3)
    
    # We remove the pivot opposite to the normal
    pivots_directions = torch.cat(
        [
            pivots_directions[:, 0:1, :],
            pivots_directions[:, 2:, :],
        ],
        dim=1,
    )  # (N_gaussians, 5, 3)
    
    # Compute pivots scores
    pivots_scores = compute_pivots_scores(
        pivots_directions,
        cameras,
        gaussians,
        pipe,
        background,
        kernel_size=kernel_size,
        scaling_modifier=scaling_modifier,
    )  # (N_gaussians, 5)
    
    pivots_scores_sum = pivots_scores.sum(dim=-1, keepdim=True)  # (N_gaussians, 1)
    default_scores = torch.zeros_like(pivots_scores)  # (N_gaussians, 5)
    default_scores[..., 0] = 1.  # (N_gaussians,)

    pivots_scores = torch.where(
        pivots_scores_sum > 0.,
        pivots_scores / pivots_scores_sum,
        default_scores,
    )  # (N_gaussians, 5)
    
    # Get maximum scores
    pivots_max_scores = pivots_scores.max(dim=-1, keepdim=True).values  # (N_gaussians, 1)

    # Get valid pivots mask by ratio threshold
    pivots_mask = pivots_scores > score_ratio_threshold * pivots_max_scores  # (N_gaussians, 5)

    # Gaussian centers
    center_pivots = multi_pivots[:, -1]  # (N_gaussians, 3)

    # Other pivots, without the opposite to the normal
    other_pivots = torch.cat(
        [
            multi_pivots[:, 0:1, :],
            multi_pivots[:, 2:-1, :],
        ],
        dim=1,
    )  # (N_gaussians, 5, 3)

    # Keep only pivots with high score
    other_pivots = other_pivots[pivots_mask]  # (M, 3)

    # All pivots
    pivots = torch.cat([other_pivots, center_pivots], dim=0)  # (M+N, 3)
    
    return pivots


def sample_gaussian_points(n_samples_per_gaussian, gaussians: GaussianModel, sample_radius: float=1.0, fixed_radius: float=None):
    samples = torch.randn(gaussians._xyz.shape[0], n_samples_per_gaussian, 3, device=gaussians._xyz.device)  # (N, P, 3)
    
    if fixed_radius is not None:
        sample_radius = fixed_radius
        samples = torch.nn.functional.normalize(samples, dim=-1)  # Norm 1
    
    rotations = build_rotation(gaussians.get_rotation)  # (N, 3, 3)
    samples = (
        gaussians.get_xyz.unsqueeze(1)  # (N, 1, 3)
        + torch.bmm(
            rotations,  # (N, 3, 3)
            sample_radius * gaussians.get_scaling_with_3D_filter.unsqueeze(-1) * samples.transpose(-1, -2)  # (N, 3, P)
        ).transpose(-1, -2)  # (N, P, 3)
    )
    return samples


def sample_random_pivots(
    n_pivots_per_gaussian, 
    gaussians: GaussianModel, 
    sample_radius: float=1.0,
    fixed_radius: float=None,
    sdf_function: Optional[Callable]=None,
):
    assert n_pivots_per_gaussian > 1, "n_pivots_per_gaussian must be greater than 1"
    P = n_pivots_per_gaussian
    n_samples_per_gaussian = P - 1
    
    # Sample random points
    samples = sample_gaussian_points(n_samples_per_gaussian, gaussians, sample_radius=sample_radius, fixed_radius=fixed_radius)  # (N, P-1, 3)
    
    # Concatenate with Gaussian centers
    pivots = torch.cat([samples, gaussians.get_xyz.unsqueeze(1)], dim=1)  # (N, P, 3)
    
    if sdf_function is not None:
        # Get SDF
        pivots_sdf = sdf_function(pivots.view(-1, 3)).view(-1, P)  # (N, P)
        
        # Check wich pivots has a different sign than the center
        different_sign = (pivots_sdf[:, 0:P-1] * pivots_sdf[:, P-1:P]) < 0.  # (N, P-1)
        
        # Check wich gaussian has at least one pivot with a different sign
        gaussian_has_different_sign = different_sign.any(dim=-1)  # (N,)
        
        # Select only the pivots with a different sign than the center
        selected_pivots = torch.cat(
            [
                pivots[:, 0:P-1][different_sign],  # (N_diff, 3)
                # pivots[:, P-1][gaussian_has_different_sign],
                pivots[:, P-1],  # (N, 3)
            ],
            dim=0,
        )
        selected_sdf = torch.cat(
            [
                pivots_sdf[:, 0:P-1][different_sign],
                # pivots_sdf[:, P-1][gaussian_has_different_sign],
                pivots_sdf[:, P-1],
            ],
            dim=0,
        )
        return selected_pivots, selected_sdf  # (M, 3), (M,)
    
    return pivots


@torch.no_grad()
def get_searched_pivots(
    gaussians: GaussianModel,
    search_iter: int=10,
    sdf_function: Callable=None,
    std_factor: float=3.33,
    step_size: float=1.0,
):
    # Get pivots
    pivots, _ = get_intersecting_pivots_from_normals(n_pivots=2, gaussians=gaussians, std_factor=std_factor)  # (N, 2, 3)
    center_pivots = pivots[:, 1]  # (N, 3)
    normal_pivots = pivots[:, 0]  # (N, 3)
    normal_rays = (normal_pivots - center_pivots) / std_factor  # (N, 3)
    
    # Get normals
    gaussian_normals = gaussians.convert_features_to_normals(normalize=True)  # (N, 3)
    gaussian_normals = torch.nn.functional.normalize(gaussian_normals, dim=-1)  # (N, 3)
    
    # Get the standard deviation of the Gaussian in the direction of the normal
    # normal_stds = get_gaussian_std_in_direction(
    #     directions=gaussian_normals.unsqueeze(1),  # (N, 1, 3)
    #     gaussians=None,
    #     gaussian_scaling=gaussians.get_scaling_with_3D_filter,
    #     gaussian_rotation=gaussians.get_rotation, 
    #     normalize_directions=False,
    # )  # (N, 1)
    
    # Get SDF
    pivots_sdf = sdf_function(pivots.view(-1, 3)).view(-1, 2)  # (N, 2)
    center_sdf = pivots_sdf[:, 1]  # (N,)
    normal_sdf = pivots_sdf[:, 0]  # (N,)
    
    # Define normal multiples
    ray_multiples = torch.ones_like(normal_rays[:, 0:1]) * std_factor  # (N, 1)
    
    # Get the pivots to search
    center_is_occupied = center_sdf <= 0.  # (N,)
    same_sign = (center_sdf * normal_sdf) > 0.  # (N,)
    
    for i_search in range(search_iter):
        # Get the pivots to search
        search_mask = center_is_occupied & same_sign  # (N,)
        
        # Get new values
        center_pivots_to_search = center_pivots[search_mask]  # (M, 3)
        normal_rays_to_search = normal_rays[search_mask]  # (M, 3)
        ray_multiples_to_search = ray_multiples[search_mask] + step_size  # (M, 1)
        normal_pivots_to_search = center_pivots_to_search + normal_rays_to_search * ray_multiples_to_search  # (M, 3)
        
        # Update ray multiple
        ray_multiples[search_mask] = ray_multiples_to_search  # (M, 1)
        # normal_pivots[search_mask] = normal_pivots_to_search  # (M, 3)
        
        # Get the SDFs
        if False:
            all_pivots_to_search = torch.cat(
                [
                    normal_pivots_to_search.unsqueeze(1),
                    center_pivots_to_search.unsqueeze(1),
                ],
                dim=1,
            )  # (M, 2, 3)
            all_pivots_to_search_sdf = sdf_function(all_pivots_to_search.view(-1, 3)).view(-1, 2)  # (M, 2)
            
            # Update sdf
            normal_sdf[search_mask] = all_pivots_to_search_sdf[:, 0]  # (M,)
            
            # Update same_sign
            same_sign[search_mask] = (all_pivots_to_search_sdf[:, 0] * all_pivots_to_search_sdf[:, 1]) > 0.  # (M,)
        else:
            # Get the SDFs
            normal_pivots_to_search_sdf = sdf_function(normal_pivots_to_search)  # (M,)
            center_pivots_to_search_sdf = center_sdf[search_mask]  # (M,)
            
            # Update sdf
            normal_sdf[search_mask] = normal_pivots_to_search_sdf  # (M,)
            
            # Update same_sign
            same_sign[search_mask] = (normal_pivots_to_search_sdf * center_pivots_to_search_sdf) > 0.  # (M,)
        
    normal_pivots = center_pivots + normal_rays * ray_multiples  # (N, 3)
    all_pivots = torch.cat(
        [
            normal_pivots.unsqueeze(1),  # (N, 1, 3)
            center_pivots.unsqueeze(1),  # (N, 1, 3)
        ], 
        dim=1
    )  # (N, 2, 3)
    all_pivots_sdf = torch.cat(
        [
            normal_sdf.unsqueeze(1),  # (N, 1)
            center_sdf.unsqueeze(1),  # (N, 1)
        ], 
        dim=1
    )  # (N, 2)
        
    return all_pivots, all_pivots_sdf  # (N, 2, 3), (N, 2)