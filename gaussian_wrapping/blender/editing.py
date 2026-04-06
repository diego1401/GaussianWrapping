import math
import torch
import torch.nn as nn
from torch.nn.functional import normalize as torch_normalize
from typing import List
from scene.mesh import Meshes
from scene.gaussian_model import GaussianModel
from utils.general_utils import (
    build_scaling_rotation, 
    inverse_sigmoid
)
from utils.sh_utils import RGB2SH, eval_sh
from blender.blender_utils import (
    get_knn_index,
    find_affine_transform,
    transform_points,
    orthogonalize_basis,
    matrix_to_quaternion,
)


@torch.no_grad()
def bind_gaussians_to_mesh(
    means:torch.Tensor,
    initial_mesh:Meshes,
    n_verts_per_gaussian:int=8,
    bind_to_triangles:bool=False,
) -> torch.Tensor :
    assert n_verts_per_gaussian > 3, "n_verts_per_gaussian must be greater than 3"
    
    if bind_to_triangles:
        # Binds each Gaussian to the closest triangle based on barycenters
        faces_barycenters = initial_mesh.verts[initial_mesh.faces].mean(dim=1)  # (n_faces, 3)
        gaussian_to_vert_idx = get_knn_index(
            points=means, 
            points2=faces_barycenters, 
            k=1,
        )  # (n_gaussians, 1)
        
    else:
        # Binds each Gaussian to the closest vertices
        gaussian_to_vert_idx = get_knn_index(
            points=means, 
            points2=initial_mesh.verts, 
            k=n_verts_per_gaussian,
        )  # (n_gaussians, n_verts_per_gaussian)
    
    return gaussian_to_vert_idx


def apply_poses_to_meshes(
    meshes:List[Meshes], 
    i_frame:int, 
    package:dict,
) -> List[Meshes]:
    n_frames = len(package['camera']['lens'])
    bone_to_vertices = package['bone_to_vertices']
    bone_to_vertex_weights = package['bone_to_vertex_weights']
    
    posed_meshes = []
    
    for i_mesh, mesh_dict in enumerate(package['bones']):
        # Initialize posed mesh with original vertices
        posed_mesh = Meshes(
            verts=meshes[i_mesh].verts.clone(),
            faces=meshes[i_mesh].faces,
            verts_colors=meshes[i_mesh].verts_colors,
        )
        
        # If the mesh is animated, apply the poses
        if mesh_dict:
            vertex_groups_idx = bone_to_vertices[i_mesh]
            vertex_groups_weights = bone_to_vertex_weights[i_mesh]
            
            # Tpose/restpose points
            tpose_points = mesh_dict['vertex']['tpose_points'].to(posed_mesh.verts.device)
            
            # Posed points
            new_points = torch.zeros_like(tpose_points)
            
            # Go from Tpose/restpose to animated pose
            for vertex_group, vertex_group_idx in vertex_groups_idx.items():
                if len(vertex_group_idx) > 0:
                    # Build bone transform
                    bone_transform = mesh_dict['armature']['pose_bones'][vertex_group][i_frame % n_frames].transpose(-1, -2)
                    reset_transform = mesh_dict['armature']['rest_bones'][vertex_group].inverse().transpose(-1, -2)
                    
                    # Transform points
                    weights = vertex_groups_weights[vertex_group]
                    new_points[vertex_group_idx] += (
                        weights[..., None] 
                        * transform_points(
                            X=transform_points(
                                X=tpose_points[vertex_group_idx],
                                M=reset_transform, 
                            ),
                            M=bone_transform,
                        )
                    )
            posed_mesh.verts = new_points
        posed_meshes.append(posed_mesh)
    return posed_meshes


def update_gaussian_parameters(
    means,
    scales,
    quaternions,
    L, T,
    remove_gaussians_with_negative_determinant:bool=True,
):
    """Return the new Gaussian parameters after applying a 3D affine transform.

    Args:
        means (torch.Tensor): (..., 3)
        scales (torch.Tensor): (..., 3)
        quaternions (torch.Tensor): (..., 4)
        L (torch.Tensor): (..., 3, 3)
        T (torch.Tensor): (..., 3)

    Returns:
        new_means (torch.Tensor): (..., 3)
        new_scales (torch.Tensor): (..., 3)
        new_quaternions (torch.Tensor): (..., 4)
    """
    # Compute principal axes of initial Gaussians
    gaussian_axes = build_scaling_rotation(scales, quaternions).transpose(-2, -1)  # (..., 3, 3)
    gaussian_points = torch.cat(
        [
            means[..., None, :],  # (..., 1, 3)
            means[..., None, :] + gaussian_axes,  # (..., 3, 3)
        ],
        dim=-2,
    )  # (..., 4, 3)
    
    # Transform Gaussian axes
    new_gaussian_points = transform_points(X=gaussian_points, L=L, T=T)  # (..., 4, 3)
    new_gaussian_axes = new_gaussian_points[..., 1:, :] - new_gaussian_points[..., 0:1, :]  # (..., 3, 3)
    
    # Orthogonalize the transformed axes
    new_gaussian_axes = orthogonalize_basis(new_gaussian_axes)  # (..., 3, 3)
    
    # We want rotations; If determinant is negative, we need to flip axes
    negative_det_mask = torch.det(new_gaussian_axes) <= 0.
    new_gaussian_axes = torch.where(
        negative_det_mask[..., None, None],
        -new_gaussian_axes,
        new_gaussian_axes,
    )  # (..., 3, 3)
    

    # Compute transformed Gaussian parameters from transformed axes 
    new_means = new_gaussian_points[..., 0, :]  # (..., 3)
    new_scales = new_gaussian_axes.norm(dim=-1)  # (..., 3)
    
    # If determinant is negative, Gaussians are not reliable.
    # We set their scales to 0.
    if remove_gaussians_with_negative_determinant:
        new_scales = torch.where(negative_det_mask[:, None], 0., new_scales)
    
    new_R = torch_normalize(new_gaussian_axes, dim=-1).transpose(-2, -1)  # (..., 3, 3)
    new_quaternions = matrix_to_quaternion(new_R)
    
    return new_means, new_scales, new_quaternions


def apply_mesh_edits_to_gaussians(
    initial_mesh:Meshes,
    edited_mesh:Meshes,
    gaussian_to_vert_idx:torch.Tensor,
    means:torch.Tensor,
    scales:torch.Tensor,
    rotations:torch.Tensor,
    return_LT:bool=False,
    bound_to_triangles:bool=False,
):
    # Compute the local affine transforms
    if bound_to_triangles:
        n_gaussians = gaussian_to_vert_idx.shape[0]
        gaussian_to_face_idx = gaussian_to_vert_idx.view(n_gaussians)
        faces = initial_mesh.faces[gaussian_to_face_idx]  # (n_gaussians, 3)
        
        # Get triangle vertices
        X = initial_mesh.verts[faces]  # (n_gaussians, 3, 3)
        Y = edited_mesh.verts[faces]  # (n_gaussians, 3, 3)
        
        # Get median scaling ratio
        # TODO: Replace this by the ratio of triangle sizes
        initial_mesh_scale = (initial_mesh.verts - initial_mesh.verts.mean(dim=0, keepdim=True)).norm(dim=1).median()
        edited_mesh_scale = (edited_mesh.verts - edited_mesh.verts.mean(dim=0, keepdim=True)).norm(dim=1).median()
        avg_scaling_ratio = edited_mesh_scale / initial_mesh_scale.clamp(min=1e-8)
        
        # Get triangle normals
        X_normals = initial_mesh.face_normals[gaussian_to_face_idx]  # (n_gaussians, 3)
        Y_normals = edited_mesh.face_normals[gaussian_to_face_idx]  # (n_gaussians, 3)
        
        # Get points
        X = torch.cat([X, X[:, 0:1, :] + X_normals[:, None, :]], dim=1)  # (n_gaussians, 4, 3)
        Y = torch.cat([Y, Y[:, 0:1, :] + avg_scaling_ratio * Y_normals[:, None, :]], dim=1)  # (n_gaussians, 4, 3)
        
        _, L, T = find_affine_transform(X=X, Y=Y)
    
    else:
        _, L, T = find_affine_transform(
            X=initial_mesh.verts[gaussian_to_vert_idx],  # (n_gaussians, n_verts_per_gaussian, 3)
            Y=edited_mesh.verts[gaussian_to_vert_idx],  # (n_gaussians, n_verts_per_gaussian, 3)
        )

    # Update the Gaussian parameters by applying local affine transforms
    new_means, new_scales, new_rotations = update_gaussian_parameters(
        means=means,
        scales=scales,
        quaternions=rotations,
        L=L, T=T,
    )
    transformed_params = {
        "means": new_means,
        "scales": new_scales,
        "quaternions": new_rotations,
    }
    if return_LT:
        transformed_params["L"] = L
        transformed_params["T"] = T
    return transformed_params


def get_edited_gaussians_parameters_from_scene_state(
    scene_state,
    i_frame,
    package,
    return_LT:bool=False,
    filter_big_gaussians_with_th:float=-1.,
    clamp_big_gaussians_with_th:float=-1.,
    filter_distant_gaussians_with_th:float=-1.,
    verbose:bool=False,
):
    edited_meshes = scene_state['edited_meshes']
    initial_meshes = scene_state['initial_meshes']
    means = scene_state['means']
    scales = scene_state['scales']
    rotations = scene_state['rotations']
    features = scene_state['features']
    opacities = scene_state['opacities']
    gausstovert_idx = scene_state['gausstovert_idx']
    
    device = means[0].device

    # Apply pose to edited meshes
    posed_meshes = apply_poses_to_meshes(
        meshes=edited_meshes, 
        i_frame=i_frame, 
        package=package,
    )

    # Propagate edit to Gaussians
    edited_means = torch.zeros(0, 3, dtype=torch.float32, device=device)
    edited_scales = torch.zeros(0, 3, dtype=torch.float32, device=device)
    edited_rotations = torch.zeros(0, 4, dtype=torch.float32, device=device)
    edited_features = torch.zeros(0, *features[0].shape[1:], dtype=torch.float32, device=device)
    edited_opacities = torch.zeros(0, 1, dtype=torch.float32, device=device)
    if return_LT:
        edited_L = torch.zeros(0, 3, 3, dtype=torch.float32, device=device)
        edited_T = torch.zeros(0, 3, dtype=torch.float32, device=device)

    for i_mesh in range(len(edited_meshes)):
        transformed_params = apply_mesh_edits_to_gaussians(
            initial_mesh=initial_meshes[i_mesh],
            edited_mesh=posed_meshes[i_mesh],
            gaussian_to_vert_idx=gausstovert_idx[i_mesh],
            means=means[i_mesh],
            scales=scales[i_mesh],
            rotations=rotations[i_mesh],
            return_LT=return_LT,
            bound_to_triangles=scene_state['bound_to_triangles'],
        )
            
        if filter_big_gaussians_with_th > 0 or clamp_big_gaussians_with_th > 0:
            initial_mesh_scale = (initial_meshes[i_mesh].verts - initial_meshes[i_mesh].verts.mean(dim=0, keepdim=True)).norm(dim=1).mean()
            edited_mesh_scale = (posed_meshes[i_mesh].verts - posed_meshes[i_mesh].verts.mean(dim=0, keepdim=True)).norm(dim=1).mean()
            avg_scaling_ratio = edited_mesh_scale / initial_mesh_scale.clamp(min=1e-8)
            expected_scales = scales[i_mesh] * avg_scaling_ratio
            
            if filter_big_gaussians_with_th:
                big_gaussians_mask = (
                    transformed_params["scales"] > expected_scales * filter_big_gaussians_with_th
                ).any(dim=1)  # (n_gaussians, )
                transformed_params["scales"] = torch.where(big_gaussians_mask[:, None], 0., transformed_params["scales"])
                if verbose:
                    print(f"[INFO] Filtered {big_gaussians_mask.sum()}={big_gaussians_mask.sum()/transformed_params['scales'].shape[0]*100:.2f}% big Gaussians.")
            
            if clamp_big_gaussians_with_th:
                transformed_params["scales"] = torch.clamp_max(
                    input=transformed_params["scales"],
                    max= expected_scales * clamp_big_gaussians_with_th,
                )
                
        if filter_distant_gaussians_with_th > 0:
            if scene_state['bound_to_triangles']:
                gaussian_to_face_idx = gausstovert_idx[i_mesh].view(-1)
                faces = initial_meshes[i_mesh].faces[gaussian_to_face_idx]  # (n_gaussians, 3)
                
                # Get length of triangle biggest side
                faces_verts = initial_meshes[i_mesh].verts[faces]  # (n_gaussians, 3, 3)
                faces_sides = faces_verts - faces_verts.roll(shifts=1, dims=1)  # (n_gaussians, 3, 3)
                faces_sizes = faces_sides.norm(dim=1).max(dim=1).values  # (n_gaussians, )
                
                # Get distance of Gaussians to their closest triangle
                face_barycenters = faces_verts.mean(dim=1)  # (n_gaussians, 3)
                gaussian_to_face_dist = (means[i_mesh] - face_barycenters).norm(dim=1)  # (n_gaussians, )
                
                # Filter Gaussians
                distant_gaussians_mask = gaussian_to_face_dist > faces_sizes * filter_distant_gaussians_with_th  # (n_gaussians, )
                transformed_params["scales"] = torch.where(distant_gaussians_mask[:, None], 0., transformed_params["scales"])
            else:
                # Get spatial extent of neighborhoods for each Gaussians
                neighbor_verts = initial_meshes[i_mesh].verts[gausstovert_idx[i_mesh]]  # (n_gaussians, n_neighbors, 3)
                neighborhood_centers = neighbor_verts.mean(dim=1)  # (n_gaussians, 3)
                neighborhood_sizes = 2. * (neighbor_verts - neighborhood_centers.unsqueeze(1)).norm(dim=-1).max(dim=1).values  # (n_gaussians, )
                
                # Get max distance of Gaussians to their neighbors
                neighborhood_to_gaussian_dist = (neighborhood_centers - means[i_mesh]).norm(dim=1)  # (n_gaussians, )
                
                # Filter Gaussians
                distant_gaussians_mask = neighborhood_to_gaussian_dist > neighborhood_sizes * filter_distant_gaussians_with_th  # (n_gaussians, )
                transformed_params["scales"] = torch.where(distant_gaussians_mask[:, None], 0., transformed_params["scales"])

            if verbose:
                print(f"[INFO] Filtered {distant_gaussians_mask.sum()}={distant_gaussians_mask.sum()/transformed_params['scales'].shape[0]*100:.2f}% distant Gaussians.")
        
        edited_means = torch.cat([edited_means, transformed_params["means"]], dim=0)
        edited_scales = torch.cat([edited_scales, transformed_params["scales"]], dim=0)
        edited_rotations = torch.cat([edited_rotations, transformed_params["quaternions"]], dim=0)
        edited_features = torch.cat([edited_features, features[i_mesh]], dim=0)
        edited_opacities = torch.cat([edited_opacities, opacities[i_mesh]], dim=0)
        if return_LT:
            edited_L = torch.cat([edited_L, transformed_params["L"]], dim=0)
            edited_T = torch.cat([edited_T, transformed_params["T"]], dim=0)    
    
    gaussian_parameters = {
        'means': edited_means,
        'scales': edited_scales,
        'rotations': edited_rotations,
        'features': edited_features,
        'opacities': edited_opacities,
    }
    if return_LT:
        gaussian_parameters["L"] = edited_L
        gaussian_parameters["T"] = edited_T
        
    return gaussian_parameters


# Please change this function if you use a GaussianModel class 
# that follows a different convention for the parameters or a different appearance model.
def get_edited_gaussians_from_scene_state(
    scene_state,
    i_frame,
    package,
    camera,
    filter_big_gaussians_with_th:float=-1.,
    clamp_big_gaussians_with_th:float=-1.,
    filter_distant_gaussians_with_th:float=-1.,
    out_of_bounds_threshold:float=1000.,
    verbose:bool=False,
):
    # Get edited Gaussian parameters
    gaussian_parameters = get_edited_gaussians_parameters_from_scene_state(
        scene_state=scene_state,
        i_frame=i_frame,
        package=package,
        return_LT=True,
        filter_big_gaussians_with_th=filter_big_gaussians_with_th,
        clamp_big_gaussians_with_th=clamp_big_gaussians_with_th,
        filter_distant_gaussians_with_th=filter_distant_gaussians_with_th,
        verbose=verbose,
    )
    device = gaussian_parameters['means'].device
    
    # Get SH degree
    n_gaussians = gaussian_parameters['means'].shape[0]
    sh_degree = int(math.sqrt(gaussian_parameters['features'].shape[1]) - 1)
    
    # We compute colors by using SHs, but first we apply the inverse 
    # of the local affine transforms to the view directions.
    # This allows for computing correct colors for edited Gaussians
    shs_view = gaussian_parameters['features'].transpose(1, 2).view(-1, 3, (sh_degree+1)**2)
    dir_pp = (gaussian_parameters['means'] - camera.camera_center.repeat(n_gaussians, 1))  # (n_gaussians, 3)
    dir_pp =  (gaussian_parameters['L'] @ dir_pp.unsqueeze(-1)).squeeze(-1)  # (n_gaussians, 3)
    dir_pp_normalized = torch.nn.functional.normalize(dir_pp, dim=1)  # (n_gaussians, 3) 
    sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    
    # Create GaussianModel
    gaussians = GaussianModel(sh_degree=0)
    gaussians._xyz = gaussian_parameters['means']
    gaussians._features_dc = RGB2SH(colors_precomp).unsqueeze(-2)
    gaussians._features_rest = torch.zeros(n_gaussians, 0, 3, device=device)
    gaussians._scaling = torch.log(gaussian_parameters['scales'])
    gaussians._rotation = gaussian_parameters['rotations']
    gaussians._opacity = inverse_sigmoid(gaussian_parameters['opacities'])
    gaussians.max_radii2D = torch.zeros((n_gaussians), device="cuda")
    
    # Handling out-of-bounds Gaussians
    oob_mask = (
        (gaussians._xyz - scene_state['camera_center']).norm(dim=-1) 
        > scene_state['camera_radius'] * out_of_bounds_threshold
    )
    gaussians._xyz[oob_mask] = 0.
    gaussians._scaling[oob_mask] = -100_000.

    # Handling very small scaling values
    gaussians._scaling = gaussians._scaling.clamp(min=-100_000.)
    
    # Handling nan rotations
    nan_mask = gaussians._rotation.isnan().any(dim=-1)
    gaussians._rotation[nan_mask] = torch.tensor([[1., 0., 0., 0.]], device=gaussians._rotation.device)
    
    return gaussians


# Please change this class if you use a Pipeline class 
# that follows a different convention.
class EditedScenePipelineParams():
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False