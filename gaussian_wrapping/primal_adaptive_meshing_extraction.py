import numpy as np
import torch
import os
import sys
from typing import Callable, List, Dict, Any
from argparse import ArgumentParser
import open3d as o3d
import trimesh
import copy
from tqdm import tqdm
from arguments import ModelParams, PipelineParams, get_combined_args
# Scene imports
from scene.cameras import Camera
from scene.mesh import Meshes
from scene import Scene, GaussianModel

# Utils impots
from utils.camera_utils import get_cameras_spatial_extent
from utils.general_utils import safe_state
from utils.primal_adaptive_meshing_utils import (
    MeshFromDelaunay,
    GaussianVectorField,
    sample_mesh_proportional_to_camera,
    plot_histogram
)

# Loading imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "TNTUniScanEvals"))
from eval.TNTUniScanEvals.uniform_sampling_eval import (
    load_gt_pcd_and_transform, 
    sample_surface
)

# Extraction and processing imports
from pivot_based_mesh_extraction import (
    evaluation_validation,
    post_process_mesh
)

'''
Global Occupancy Function
'''

def compute_global_occupancy(points_input, views, gaussians, pipeline, kernel_size, integrate_func, chunk_size=1_000_000, verbose_views=False, device="cuda"):
    all_occupancy_values = []
    input_is_numpy = False
    if isinstance(points_input, np.ndarray):
        input_is_numpy = True
        points = torch.from_numpy(points_input)
    else:
        points = points_input
    
    has_inside_key = False
    for point_chunk in torch.chunk(points, points.shape[0] // chunk_size + 1):
        point_chunk = point_chunk.to(device)
        final_weight_chunk = torch.ones(point_chunk.shape[0], dtype=torch.float32, device=device)
        any_valid_chunk = torch.zeros(point_chunk.shape[0], dtype=torch.bool, device=device)
        for view in tqdm(views, desc="Rendering progress") if verbose_views else views:
            ret = integrate_func(point_chunk, view, gaussians, pipeline, kernel_size)
            if "inside" in ret:
                has_inside_key = True
                valid_points = evaluation_validation(view, point_chunk, ret["inside"])
                any_valid_chunk = torch.logical_or(any_valid_chunk, valid_points)
                final_weight_chunk = torch.where(
                    valid_points,
                    torch.min(ret["alpha_integrated"], final_weight_chunk),
                    final_weight_chunk,
                )
            else:
                has_inside_key = False
                final_weight_chunk = torch.min(final_weight_chunk, ret["alpha_integrated"])

        if has_inside_key:
            final_weight_chunk[torch.logical_not(any_valid_chunk)] = 0
        all_occupancy_values.append(final_weight_chunk.cpu())
    
    occupancies = torch.cat(all_occupancy_values, dim=0)
    if input_is_numpy:
        occupancies = occupancies.detach().cpu().numpy()
    return occupancies

'''
Auxiliary Loading Functions
'''

def export_mesh(nv: np.ndarray, nf: np.ndarray, name: str, args):
    assert name.endswith(".ply"), "Mesh must be saved as PLY file"
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(nv, dtype=np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(nf, dtype=np.int32))
    if args.post_process:
        mesh = post_process_mesh(o3d_mesh, 1)
    else:
        mesh = o3d_mesh
    o3d.io.write_triangle_mesh(name, mesh)

def create_global_occupancy_func(args: ArgumentParser, return_details=False):
    # Setup Rasterizer
    print(f"[INFO] Using {args.rasterizer} as rasterizer.")
    if args.rasterizer == "ours":
        from gaussian_renderer.ours import integrate_ours as integrate_func
    else:
        raise ValueError(f"Always use the Ours rasterizer to apply Primal Adaptive Meshing.")

    gaussians = GaussianModel(
        args.sh_degree,
        learn_occupancy=True, 
        n_gaussian_features=4, 
        use_unbounded_opacity=args.use_unbounded_opacity,
        use_appearance_network=False,
        use_mip_filter=not args.disable_mip_filter,
    )
    
    print(f"[INFO] Loading Gaussian Model at iteration {args.iteration}")
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    
    # Get views
    views = scene.getTrainCameras()
    print(f"[INFO] Loaded {len(views)} views.")
    
    # Pipeline
    pipeline = PipelineParams(ArgumentParser()).extract(args)
    
    # Kernel size
    kernel_size = args.kernel_size if hasattr(args, 'kernel_size') else 0.0

    global_occupancy_func = lambda query_points: compute_global_occupancy(
        query_points, 
        views, 
        gaussians, 
        pipeline, 
        kernel_size, 
        integrate_func,
        device="cuda"
    )
    vector_field = GaussianVectorField(
        gaussians, 
        nearest_neighbors_type="kdtree", 
        k_neighbors=args.n_neighbors_vector_field
    )
    grad_occupancy_func = lambda query_points: vector_field.get_vector_field_quantities(
            query_points=query_points,
            gaussians=gaussians,
            k_neighbors=args.n_neighbors_vector_field,
            compute_curl=False,
            compute_normal_field=True,
        )["normal_field"] # NOTE: Non normalized normal field is grad log v
    
    if return_details: 
        ret_pckg = {"scene_radius": get_cameras_spatial_extent(cameras=views)['radius'].item(),
                    "cameras": views}
        return global_occupancy_func, grad_occupancy_func, ret_pckg
    return global_occupancy_func, grad_occupancy_func

def load_crop_box_and_transform(args):
    """
    Loads the crop volume and trajectory transform.
    """
    scene_name = os.path.basename(os.path.normpath(args.source_path))
    traj_path = os.path.join(args.source_path, scene_name + "_COLMAP_SfM.log")
    _, trajectory_transform = load_gt_pcd_and_transform(args.source_path, scene_name, traj_path)

    crop_json_path = os.path.join(args.source_path, scene_name + ".json")
    if not os.path.exists(crop_json_path):
        raise FileNotFoundError(f"Crop file not found: {crop_json_path}")

    print(f"[INFO] Loading crop volume from {crop_json_path}")
    vol = o3d.visualization.read_selection_polygon_volume(crop_json_path)

    return vol, trajectory_transform

def load_blender_crop_volume(json_path: str):
    """
    Loads a convex-hull bounding volume exported from the GW Blender add-on.
    Returns a scipy.spatial.Delaunay object (used for inside-hull tests) and an identity transform.
    """
    import json as _json
    from scipy.spatial import Delaunay

    print(f"[INFO] Loading Blender bounding volume from {json_path}")
    with open(json_path, "r") as f:
        data = _json.load(f)

    if data.get("class_name") != "GaussianWrappingBoundingVolume":
        raise ValueError(
            f"Expected class_name 'GaussianWrappingBoundingVolume', got '{data.get('class_name')}'. "
            "Make sure the JSON was exported with the GW Bounding Volume Blender add-on."
        )

    vertices = np.array(data["vertices"], dtype=np.float64)
    hull = Delaunay(vertices)
    return hull, np.identity(4)  # Blender world == model world

'''
Auxiliary Mesh Functions
'''

def mesh_from_points_and_occupancy(points: np.ndarray, args: ArgumentParser, global_occupancy_func: Callable):
    # Initialize MeshFromDelaunay
    print("[INFO] Initializing Delaunay Triangulation...")
    meshfromdelaunay = MeshFromDelaunay(points, add_corners=False)
    
    if args.p_per_tet == 1:
        query_points = meshfromdelaunay.barycenters
    else:
        query_points = meshfromdelaunay.sample_random_tet_points(args.p_per_tet)
        query_points = query_points.reshape(-1, 3)
    
    torch_query_points = torch.tensor(query_points, dtype=torch.float32).to('cuda')
    
    # Compute occupancy
    print(f"[INFO] Computing occupancy on {torch_query_points.shape[0]} points...")
    occupancy = global_occupancy_func(torch_query_points)
    
    if args.p_per_tet > 1:
        # Reshape back to (p_per_tet, n_tets) and mean over p_per_tet
        occupancy = occupancy.reshape(args.p_per_tet, -1).mean(0)
    
    # Determine inside/outside based on occupancy threshold
    # occupancy > 0.5 means inside/occupied
    occupancy_np = occupancy.cpu().detach().numpy()
    vacancy = 1.0 - occupancy_np
    meshfromdelaunay.tet_colors = (vacancy < 0.5)

    # Extract surface
    print("[INFO] Extracting surface...")
    surface_data = meshfromdelaunay.get_surface()
    surface_verts = surface_data[0]
    surface_faces = surface_data[1]
    
    return surface_verts, surface_faces

def sample_points_from_mesh_robust(mesh_cropped: Meshes, num_points: int, transform: np.ndarray, crop_volume: o3d.visualization.SelectionPolygonVolume, args: ArgumentParser, cameras: List[Camera], face_probs=None):
    # Sample points AND normals
    if args.mesh_sampling_method == "surface_even":
        print(f"[INFO] Sampling {num_points} points from cropped mesh using surface_even method...")
        sampled_vertices, sampled_normals = sample_surface(mesh_cropped, surface_samples=num_points)
        face_probs = None
    elif args.mesh_sampling_method == "proportional_to_camera":
        print(f"[INFO] Sampling {num_points} points from cropped mesh using proportional_to_camera method...")
        sampled_vertices, sampled_normals, face_probs = sample_mesh_proportional_to_camera(mesh_cropped, cameras, num_points, face_probs=face_probs)
    else:
        raise ValueError(f"Invalid mesh sampling method: {args.mesh_sampling_method}")

    # Transform points and normals back to world space
    # Inverse transform
    inv_transform = np.linalg.inv(transform)

    # Apply inverse transform to points
    # Points: (N, 3)
    # Add homogeneous coord
    ones = np.ones((sampled_vertices.shape[0], 1))
    vertices_hom = np.hstack((sampled_vertices, ones))
    vertices_world = (inv_transform @ vertices_hom.T).T[:, :3]

    # Apply inverse transform rotation to normals
    # Normals: (N, 3)
    inv_rotation = inv_transform[:3, :3]
    normals_world = (inv_rotation @ sampled_normals.T).T

    # Renormalize normals
    normals_norms = np.linalg.norm(normals_world, axis=1, keepdims=True)
    normals_world = normals_world / (normals_norms + 1e-8)

    return vertices_world, normals_world, face_probs

'''
Refining and Filtering
'''

def gradient_descent_refinement(points: np.ndarray, occupancy_function: Callable, grad_occupancy_function: Callable, args: ArgumentParser, chunk_size: int = 1_000_000, device: str = "cuda"):
    print(f"[INFO] Refining points for {args.n_steps} steps...")
    xi = torch.from_numpy(points).float()

    for i in tqdm(range(args.n_steps), desc="Gradient Descent Steps"): # we use this as number of gradient descent steps
        xi_new_chunks = []
            
        for xi_chunk in torch.chunk(xi, xi.shape[0] // chunk_size + 1):
            xi_chunk = xi_chunk.to(device) # (N, 3)
            grad = grad_occupancy_function(xi_chunk) # (N, 3) # NOTE: Gradient log v
            occupancy_values = occupancy_function(xi_chunk).to(device).unsqueeze(-1)
            alpha = (0.5 - occupancy_values) / (grad.norm(dim=-1, keepdim=True)**2 + 1e-6) # (N, 1)
            new_xi_chunk = xi_chunk - alpha.clamp(min=-1.0, max=1.0) * grad # (N, 3)
            xi_new_chunks.append(new_xi_chunk.clone().cpu())
        xi = torch.cat(xi_new_chunks, dim=0) # (N, 3)
        
        if args.plot_vacancy_histogram:
            plot_histogram(occupancy_values.cpu().numpy(), filename=os.path.join(args.model_path, f"histogram_occupancy_values_{i}.png"), title="Histogram of Occupancy Values")

    normal_chunks, norm_chunks = [], []
    for xi_chunk in torch.chunk(xi, xi.shape[0] // chunk_size + 1):
        n = grad_occupancy_function(xi_chunk.to(device))
        norms = n.norm(dim=-1, keepdim=True)
        normal_chunks.append((n / (norms + 1e-8)).cpu())
        norm_chunks.append(norms.cpu())
    refined_normals = torch.cat(normal_chunks, dim=0).contiguous().numpy()
    vector_field_norms = torch.cat(norm_chunks, dim=0)
    refined_points = xi.contiguous().numpy()

    return refined_points, refined_normals, vector_field_norms

def filter_points_by_occupancy(points: np.ndarray, global_occupancy_func: Callable, threshold=0.1):
    if points.shape[0] == 0:
        return points, np.zeros(points.shape[0], dtype=bool)
        
    points_torch = torch.from_numpy(points).float().to("cuda")
    occupancy = global_occupancy_func(points_torch)
    vacancy = 1.0 - occupancy
    
    mask = (vacancy - 0.5).abs() <= threshold
    
    filtered_points = points[mask.cpu().numpy()]
    
    return filtered_points, mask.cpu().numpy()

'''
Candidate Generation Functions
'''

def load_mesh(args: ArgumentParser, scene_pckg: Dict[str, Any]):
    '''
    Generates candidate points from a mesh input.
    '''
    # Load mesh
    print(f"[INFO] Loading mesh from {args.input_mesh}")
    if not os.path.exists(args.input_mesh):
        raise FileNotFoundError(f"Input mesh not found: {args.input_mesh}")
    
    mesh = trimesh.load(args.input_mesh)

    if args.bounding_box_method == "ground_truth":
        assert args.bounding_box_scaling == 1.0, "Can only adjust bounding box for scene BB."
        # Load cropping bounding box and transform
        crop_volume, transform = load_crop_box_and_transform(args)

    elif args.bounding_box_method == "scene":
        # Define the extent of the box based on the radius
        # For a center at zero, min is -radius and max is +radius
        scene_radius_scaled = scene_pckg["scene_radius"] * args.bounding_box_scaling
        min_bound = np.array([-scene_radius_scaled, -scene_radius_scaled, -scene_radius_scaled])
        max_bound = np.array([scene_radius_scaled, scene_radius_scaled, scene_radius_scaled])

        # Create an AxisAlignedBoundingBox
        crop_volume = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        # Create an identity transform (4x4 matrix)
        transform = np.identity(4)

    elif args.bounding_box_method == "blender":
        assert args.bounding_box_file is not None, \
            "--bounding_box_file is required when --bounding_box_method=blender"
        crop_volume, transform = load_blender_crop_volume(args.bounding_box_file)

    else:
        raise ValueError(f"Bounding Box Method: [{args.bounding_box_method}] does not exist")
    
    # Transform mesh to alignment space
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    
    # Apply transform (World -> GT/Crop Space)
    mesh_transformed = copy.deepcopy(mesh_o3d).transform(transform)
    
    # Crop mesh
    if isinstance(crop_volume, o3d.geometry.AxisAlignedBoundingBox) or \
       isinstance(crop_volume, o3d.geometry.OrientedBoundingBox):
        # Standard Bounding Boxes: The mesh handles the operation
        mesh_cropped = mesh_transformed.crop(crop_volume)

    elif isinstance(crop_volume, o3d.visualization.SelectionPolygonVolume):
        # Polygon Volumes: The volume object handles the operation
        mesh_cropped = crop_volume.crop_triangle_mesh(mesh_transformed)

    else:
        # Blender convex-hull volume: scipy.spatial.Delaunay inside test
        from scipy.spatial import Delaunay
        if not isinstance(crop_volume, Delaunay):
            raise TypeError(f"Unsupported crop_volume type: {type(crop_volume)}. "
                            "Expected AxisAlignedBoundingBox, SelectionPolygonVolume, or Delaunay.")

        verts_np = np.asarray(mesh_transformed.vertices, dtype=np.float64)
        inside_mask = crop_volume.find_simplex(verts_np) >= 0

        faces_np = np.asarray(mesh_transformed.triangles)
        # Keep faces where ALL three vertices are inside the hull
        face_mask = inside_mask[faces_np].all(axis=1)

        mesh_cropped = copy.deepcopy(mesh_transformed)
        mesh_cropped.remove_triangles_by_mask(~face_mask)
        mesh_cropped.remove_unreferenced_vertices()
    
    if len(mesh_cropped.vertices) == 0:
        print("[WARNING] Cropped mesh is empty!")
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    mesh = Meshes(verts=torch.as_tensor(np.asarray(mesh_cropped.vertices), dtype=torch.float32).to(device="cuda"), 
                  faces=torch.as_tensor(np.asarray(mesh_cropped.triangles), dtype=torch.int32).to(device="cuda"))

    return mesh, crop_volume, transform

def refine_and_filter_points(points: np.ndarray, global_occupancy_func: Callable, grad_occupancy_function: Callable, args: ArgumentParser):
    ret_pckg = {}
    # Refinement procedure
    refined_points, refined_normal, vector_field_norms = gradient_descent_refinement(points, global_occupancy_func, grad_occupancy_function, args)
    
    # Filter points with global occupancy function
    print(f"[INFO] Filtering points with vacancy threshold {args.vacancy_threshold}...")
    n_points_before_filtering = refined_points.shape[0]
    final_points, mask = filter_points_by_occupancy(refined_points, global_occupancy_func, threshold=args.vacancy_threshold)
    n_points_after_filtering = final_points.shape[0]
    print(f"[INFO] Filtered {n_points_before_filtering - n_points_after_filtering} points out of {n_points_before_filtering}")
    ret_pckg["final_points"] = final_points
    ret_pckg["refined_normal"] = refined_normal[mask]
    ret_pckg["vector_field_norms"] = vector_field_norms[mask]
    return ret_pckg

def get_candidate_points(args: ArgumentParser, global_occupancy_func: Callable, grad_occupancy_function: Callable = None, scene_pckg: Dict[str, Any] = None):
    mesh, crop_volume, transform = load_mesh(args, scene_pckg=scene_pckg)
    
    refine_filter = lambda p: refine_and_filter_points(p, global_occupancy_func, grad_occupancy_function, args)["final_points"]
    if args.no_force_use_all_points:
        sampled_points, _, face_probs = sample_points_from_mesh_robust(mesh, args.max_points, transform, crop_volume, args, scene_pckg["cameras"])
        points = refine_filter(sampled_points)
    else:
        # Sample oversampling_factor * max_points upfront to minimise the number of
        # expensive occupancy passes (each pass iterates over all views).
        n_sample = args.max_points * args.oversampling_factor
        sampled_points, _, face_probs = sample_points_from_mesh_robust(mesh, n_sample, transform, crop_volume, args, scene_pckg["cameras"])
        all_points = []
        points_to_extract = args.max_points
        i = 0
        while points_to_extract > 0:
            print(f"[INFO] Extracting {points_to_extract} points from iteration {i}...")
            curr_points = refine_filter(sampled_points)
            all_points.append(curr_points)
            points_to_extract -= curr_points.shape[0]
            i += 1
            if points_to_extract > 0:
                n_sample = points_to_extract * args.oversampling_factor
                sampled_points, _, face_probs = sample_points_from_mesh_robust(mesh, n_sample, transform, crop_volume, args, scene_pckg["cameras"], face_probs=face_probs)
        points = np.concatenate(all_points, axis=0)[:args.max_points]

    return points

'''
Main Procedure
'''

def main(args):
    # Initialize system state
    safe_state(args.quiet)
    
    # Create global occupancy function
    occupancy_func, grad_occupancy_func, ret_pckg = create_global_occupancy_func(args, return_details=True)

    # Get candidate points
    points = get_candidate_points(args, occupancy_func, grad_occupancy_function=grad_occupancy_func, scene_pckg=ret_pckg)

    if args.save_candidate_points:
        print(f"[INFO] Saving candidate points to {args.output_mesh.replace('.ply', '_candidate_points.ply')}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(args.output_mesh.replace(".ply", "_candidate_points.ply"), pcd)

    # Extract mesh
    surface_verts, surface_faces = mesh_from_points_and_occupancy(points, args, occupancy_func)
    
    # Save mesh
    print(f"[INFO] Saving mesh to {args.output_mesh}")
    export_mesh(surface_verts, surface_faces, args.output_mesh, args)

if __name__ == "__main__":
    parser = ArgumentParser(description="PoNQ Mesh Extraction from Point Cloud")
    
    # Model parameters
    ModelParams(parser)
    PipelineParams(parser)
    
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--rasterizer", default="ours", type=str, choices=["ours"])
    parser.add_argument("--disable_mip_filter", action="store_true")
    
    parser.add_argument("--input_mesh", type=str, help="Path to input mesh (ply/obj)", required=True)
    parser.add_argument("--output_mesh", type=str, required=True, help="Path to output mesh (ply/obj)")
    parser.add_argument("--max_points", type=int, default=1000000, help="Max points to process")
    parser.add_argument("--p_per_tet", type=int, default=10, help="Points per tet for occupancy check")
    parser.add_argument("--n_steps", default=10, type=int, help="Number of refinement steps")
    parser.add_argument("--vacancy_threshold", default=0.1, type=float, help="Vacancy threshold for filtering")
    parser.add_argument("--save_candidate_points", action="store_true", help="Save candidate points to ply file")
    parser.add_argument("--post_process", action="store_true", help="Post process mesh")
    parser.add_argument("--bounding_box_method", default="scene", type=str, choices=["ground_truth", "scene", "blender"])
    parser.add_argument("--bounding_box_scaling", default=1.0, type=float)
    parser.add_argument("--bounding_box_file", default=None, type=str,
                        help="Path to bounding volume JSON exported from the GW Blender add-on (required for --bounding_box_method=blender)")
    parser.add_argument("--n_neighbors_vector_field", type=int, default=32, help="Number of neighbors for grad occupancy computation")
    parser.add_argument("--mesh_sampling_method", default="proportional_to_camera", type=str, choices=["surface_even", "proportional_to_camera"])
    parser.add_argument("--plot_vacancy_histogram", action="store_true", help="Plot histogram of vacancy values")
    parser.add_argument("--no_force_use_all_points", action="store_true", help="Disable resampling to guarantee max_points valid points after vacancy filtering.")
    parser.add_argument("--oversampling_factor", type=int, default=2, help="Sample this multiple of max_points upfront to reduce the number of occupancy passes. This is mostly useful when max_points is low.")
    
    args = get_combined_args(parser)
    
    # Initialize RNG
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    with torch.no_grad():
        main(args)