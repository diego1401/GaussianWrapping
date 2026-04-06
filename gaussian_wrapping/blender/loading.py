import json
import numpy as np
import torch
import open3d as o3d
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from scene.mesh import Meshes
from utils.camera_utils import get_cameras_spatial_extent
from utils.graphics_utils import focal2fov, fov2focal
from blender.editing import bind_gaussians_to_mesh
from blender.blender_utils import transform_points


# Please change this function if you use a GaussianModel class 
# that follows a different convention for the parameters.
def load_model_from_path(ply_path):
    gaussians = GaussianModel(sh_degree=None)
    gaussians.load_ply(ply_path)
    return gaussians


# Please change this function if you use a GaussianModel class 
# that follows a different convention for the parameters.
# You should simply preserve the same keys and shapes.
def get_parameters_from_model(model:GaussianModel):
    """Returns the parameters of Gaussians.

    Args:
        model (GaussianModel): The GaussianModel object.

    Returns:
        dict: A dictionary containing the parameters of Gaussians.
            The keys, values are:
            'means': (n_gaussians, 3)
            'scales': (n_gaussians, 3)
            'rotations': (n_gaussians, 4)
            'opacities': (n_gaussians, 1)
            'features': (n_gaussians, n_features, 3)
    """
    return {
        'means': model.get_xyz,
        'scales': model.get_scaling_with_3D_filter,
        'rotations': model.get_rotation,
        'opacities': model.get_opacity_with_3D_filter,
        'features': model.get_features,
    }


def load_blender_package(package_path, device='cuda'):
    # Load package
    package = json.load(open(package_path))
    # Convert lists into tensors
    for key, object in package.items():
        if type(object) is dict:
            for sub_key, sub_object in object.items():
                if type(sub_object) is list:
                    object[sub_key] = torch.tensor(sub_object)
        elif type(object) is list:
            for element in object:
                if element:
                    for sub_key, sub_object in element.items():
                        if type(sub_object) is list:
                            element[sub_key] = torch.tensor(sub_object)
    # Process bones
    bone_to_vertices = []
    bone_to_vertex_weights = []
    for i_mesh, mesh_dict in enumerate(package['bones']):
        if mesh_dict:
            vertex_dict = mesh_dict['vertex']
            armature_dict = mesh_dict['armature']
            
            # Per vertex info
            vertex_dict['matrix_world'] = torch.Tensor(vertex_dict['matrix_world']).to(device)
            vertex_dict['tpose_points'] = torch.Tensor(vertex_dict['tpose_points']).to(device)
            
            # Per bone info
            armature_dict['matrix_world'] = torch.Tensor(armature_dict['matrix_world']).to(device)
            for key, val in armature_dict['rest_bones'].items():
                armature_dict['rest_bones'][key] = torch.Tensor(val).to(device)
            for key, val in armature_dict['pose_bones'].items():
                armature_dict['pose_bones'][key] = torch.Tensor(val).to(device)
                
            # Build mapping from bone name to corresponding vertices
            vertex_groups_idx = {}
            vertex_groups_weights = {}
            
            # > For each bone of the current armature, we initialize an empty list
            for bone_name in armature_dict['rest_bones']:
                vertex_groups_idx[bone_name] = []
                vertex_groups_weights[bone_name] = []
                
            # > For each vertex, we add the vertex index to the corresponding bone lists
            for i in range(len(vertex_dict['groups'])):
                groups_in_which_vertex_appears = []
                weights_of_the_vertex_in_those_groups = []

                # We start by filtering out the groups that are not part of the current armature.
                # This is necessary for accurately normalizing the weights.
                for j_group, group in enumerate(vertex_dict['groups'][i]):
                    if group in vertex_groups_idx:
                        groups_in_which_vertex_appears.append(group)
                        weights_of_the_vertex_in_those_groups.append(vertex_dict['weights'][i][j_group])
                
                # We normalize the weights
                normalize_weights = True
                if normalize_weights:
                    sum_of_weights = np.sum(weights_of_the_vertex_in_those_groups)
                    weights_of_the_vertex_in_those_groups = [w / sum_of_weights for w in weights_of_the_vertex_in_those_groups]
                
                # We add the vertex index and the associated weight to the corresponding bone lists
                for j_group, group in enumerate(groups_in_which_vertex_appears):
                    # For safety, we check that the group belongs to the current armature, used for rendering.
                    # Indeed, for editing purposes, one might want to use multiple armatures in the Blender scene, 
                    # but only one (as expected) for the final rendering.
                    if group in vertex_groups_idx:
                        vertex_groups_idx[group].append(i)
                        vertex_groups_weights[group].append(weights_of_the_vertex_in_those_groups[j_group])

            # > Convert the lists to tensors
            for bone_name in vertex_groups_idx:
                if len(vertex_groups_idx[bone_name]) > 0:
                    vertex_groups_idx[bone_name] = torch.tensor(vertex_groups_idx[bone_name], dtype=torch.long, device=device)
                    vertex_groups_weights[bone_name] = torch.tensor(vertex_groups_weights[bone_name], device=device)

            bone_to_vertices.append(vertex_groups_idx)
            bone_to_vertex_weights.append(vertex_groups_weights)
        
        else:
            bone_to_vertices.append(None)
            bone_to_vertex_weights.append(None)
            
    package['bone_to_vertices'] = bone_to_vertices
    package['bone_to_vertex_weights'] = bone_to_vertex_weights
    
    return package


def load_cameras_from_blender_package(package, device="cuda"):
    matrix_world = package['camera']['matrix_world'].to(device)
    angle = package['camera']['angle']
    znear = package['camera']['clip_start']
    zfar = package['camera']['clip_end']
    
    if not 'image_height' in package['camera']:
        print('[WARNING] Image size not found in the package. Using default value 1920 x 1080.')
        height, width = 1080, 1920
    else:
        height, width = package['camera']['image_height'], package['camera']['image_width']

    cameras = []
    for i_cam in range(len(angle)):
        c2w = matrix_world[i_cam]
        c2w[:3, 1:3] *= -1  # Blender to COLMAP convention
        w2c = c2w.inverse()
        R, T = w2c[:3, :3].transpose(-1, -2), w2c[:3, 3]  # R is stored transposed due to 'glm' in CUDA code
        
        fov = angle[i_cam].item()
        
        if width > height:
            fov_x = fov
            fov_y = focal2fov(fov2focal(fov_x, width), height)
        else:
            fov_y = fov
            fov_x = focal2fov(fov2focal(fov_y, height), width)
        
        camera = Camera(
            colmap_id=str(i_cam), 
            R=R.cpu().numpy(), 
            T=T.cpu().numpy(), 
            FoVx=fov_x, 
            FoVy=fov_y, 
            image=torch.empty(3, height, width),
            gt_alpha_mask=None,
            image_name=f"frame_{i_cam}", 
            uid=i_cam,
            data_device=device,
        )
        cameras.append(camera)
    
    return cameras


def load_models_from_blender_package(package):
    models = {}
    models_paths = []

    for mesh in package['meshes']:
        model_path = mesh['checkpoint_name']
        if not model_path in models_paths:
            models_paths.append(model_path)

    for _, model_path in enumerate(models_paths):        
        print(f'\nLoading Gaussians: {model_path}')
        model = load_model_from_path(model_path)
        models[model_path] = model
    
    return models, models_paths


def load_initial_meshes_from_blender_package(package, device='cuda'):
    initial_meshes = {}
    initial_meshes_path = []
    mesh_models_paths = []
    
    for mesh in package['meshes']:
        mesh_path = mesh['mesh_name']
        model_path = mesh['checkpoint_name']
        if not mesh_path in initial_meshes_path:
            initial_meshes_path.append(mesh_path)
            mesh_models_paths.append(model_path)
            
    for _, mesh_path in enumerate(initial_meshes_path):
        print(f'\nLoading mesh: {mesh_path}')
        o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
        verts = torch.from_numpy(np.asarray(o3d_mesh.vertices)).float().to(device)
        faces = torch.from_numpy(np.asarray(o3d_mesh.triangles)).to(device)
        verts_colors = torch.from_numpy(np.asarray(o3d_mesh.vertex_colors)).float().to(device)
        mesh = Meshes(verts=verts, faces=faces, verts_colors=verts_colors)
        initial_meshes[mesh_path] = mesh
        
    return initial_meshes, initial_meshes_path, mesh_models_paths


def load_scene_state_from_blender_package(package, n_verts_per_gaussian=4, bind_to_triangles=False, device='cuda'):
    # Load cameras
    render_cameras = load_cameras_from_blender_package(package, device=device)
    cameras_spatial_extent = get_cameras_spatial_extent(render_cameras)
    camera_radius = cameras_spatial_extent["radius"]
    camera_center = cameras_spatial_extent["avg_cam_center"]
    
    # Load the Gaussian models
    models, _ = load_models_from_blender_package(package)
    
    # Load the initial meshes
    initial_meshes, meshes_paths, mesh_models_paths = load_initial_meshes_from_blender_package(package, device=device)
    
    # First, we bind each initial full mesh to a Gaussian model
    gaussian_to_vert_indices = {}
    with torch.no_grad():
        for i_mesh in range(len(meshes_paths)):
            gaussian_to_vert_indices[meshes_paths[i_mesh]] = bind_gaussians_to_mesh(
                means=get_parameters_from_model(models[mesh_models_paths[i_mesh]])['means'],
                initial_mesh=initial_meshes[meshes_paths[i_mesh]],
                n_verts_per_gaussian=n_verts_per_gaussian,
                bind_to_triangles=bind_to_triangles,
            )
    
    # Then, we use the initial bindings to identify which Gaussians should be kept,
    # and we re-bind each edited mesh to a subset of Gaussians.
    # Several edited meshes could come from the same initial mesh;
    # Hence, we do not iterate over meshes_paths, but over the list of edited meshes in the Blender file.
    edited_meshes_initial = []
    edited_meshes = []
    edited_meshes_means = []
    edited_meshes_scales = []
    edited_meshes_rotations = []
    edited_meshes_opacities = []
    edited_meshes_features = []
    edited_meshes_gausstovert_idx = []
    
    for i_mesh, mesh in enumerate(package['meshes']):
        mesh_path = mesh['mesh_name']
        model_path = mesh['checkpoint_name']
        
        # Get the full initial mesh, the Gaussian model, and the binding of the initial mesh to the Gaussians
        initial_mesh_i = initial_meshes[mesh_path]
        gaussian_params_i = get_parameters_from_model(models[model_path])
        gaussian_to_vert_idx_i = gaussian_to_vert_indices[mesh_path]
        
        # Build the edited mesh object
        vert_idx_i = mesh['idx']  # (n_verts_edited, )
        vert_mask_i = torch.zeros(initial_mesh_i.verts.shape[0], dtype=torch.bool, device=initial_mesh_i.verts.device)  # (n_verts_initial, )
        vert_mask_i[vert_idx_i] = True
        initial_submesh_i = initial_mesh_i.submesh(vert_mask=vert_mask_i)
        edited_submesh_i = Meshes(
            verts=transform_points(
                X=mesh['xyz'].to(initial_mesh_i.verts.device),
                M=package['meshes'][i_mesh]['matrix_world'].transpose(-1, -2).to(initial_mesh_i.verts.device),
            ),
            faces=initial_submesh_i.faces, 
            verts_colors=initial_submesh_i.verts_colors,
        )

        # Check which Gaussians should be bound to the edited mesh
        if bind_to_triangles:
            faces_mask_i = vert_mask_i[initial_mesh_i.faces].all(dim=1)  # (n_faces, )
            gaussian_mask_i = faces_mask_i[gaussian_to_vert_idx_i].squeeze(1)  # (n_gaussians, )
        else:
            gaussian_mask_i = vert_mask_i[gaussian_to_vert_idx_i]  # (n_gaussians, n_verts_per_gaussian)            
            if False:
                #  > Option 1: Keep only Gaussians for which at least half of the bound vertices belong to the submesh
                gaussian_mask_i = gaussian_mask_i.sum(dim=1).float() >= 1/2 * gaussian_mask_i.shape[1]  # (n_gaussians, )
            elif False:
                #  > Option 2: Keep only Gaussians for which the closest vertex to the mean is in the submesh
                gaussian_mask_i = gaussian_mask_i[:, 0]  # (n_gaussians, )
            else:
                #  > Option 3: Keep only Gaussians for which the n/2 closest vertices to the mean are in the submesh
                gaussian_mask_i = gaussian_mask_i[:, :gaussian_mask_i.shape[1]//2].all(dim=1)  # (n_gaussians, )
        
        # Re-bind the selected Gaussians to the submesh
        edited_gausstovert_idx_i = bind_gaussians_to_mesh(
            means=gaussian_params_i['means'][gaussian_mask_i],
            initial_mesh=initial_submesh_i,
            n_verts_per_gaussian=n_verts_per_gaussian,
            bind_to_triangles=bind_to_triangles,
        )
        
        # Add the edited mesh and the corresponding Gaussians to the list
        edited_meshes_initial.append(initial_submesh_i)
        edited_meshes.append(edited_submesh_i)
        edited_meshes_means.append(gaussian_params_i['means'][gaussian_mask_i])
        edited_meshes_scales.append(gaussian_params_i['scales'][gaussian_mask_i])
        edited_meshes_rotations.append(gaussian_params_i['rotations'][gaussian_mask_i])
        edited_meshes_opacities.append(gaussian_params_i['opacities'][gaussian_mask_i])
        edited_meshes_features.append(gaussian_params_i['features'][gaussian_mask_i])
        edited_meshes_gausstovert_idx.append(edited_gausstovert_idx_i)
        
    scene_state = {
        'initial_meshes': edited_meshes_initial,
        'edited_meshes': edited_meshes,
        'means': edited_meshes_means,
        'scales': edited_meshes_scales,
        'rotations': edited_meshes_rotations,
        'opacities': edited_meshes_opacities,
        'features': edited_meshes_features,
        'gausstovert_idx': edited_meshes_gausstovert_idx,
        'bound_to_triangles': bind_to_triangles,
        'render_cameras': render_cameras,
        'camera_radius': camera_radius,
        'camera_center': camera_center,
    }
    return scene_state