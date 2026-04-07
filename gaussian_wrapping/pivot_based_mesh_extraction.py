#adopted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/extract_mesh.py
from typing import List
import numpy as np
import torch
from functools import partial
from scene import Scene
import os
import random
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import trimesh
from scene.cameras import Camera
from scene.mesh import return_delaunay_tets
from gaussian_renderer.sof import (
    default_splat_args, 
    GlobalSortOrder,
    evaluate_vacancy_sof_fast,
)
from gaussian_renderer.ours import integrate_ours
from regularization.sdf.learnable import refine_intersections_with_binary_search
from regularization.sdf.depth_fusion import evaluate_mesh_colors_all_vertices
from extraction.pivots import get_intersecting_pivots_from_normals, get_pivots_by_scores, sample_random_pivots, get_searched_pivots
from extraction.mesh import extract_mesh, compute_isosurface_value_from_depth
from utils.camera_utils import get_cameras_spatial_extent
from utils.geometry_utils import transform_points_world_to_view
import time
from tqdm import tqdm
import gc
import open3d as o3d


def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy

    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


@torch.no_grad()
def evaluation_validation(view, points, inside):
    if view.gt_mask is None:
        return inside

    points_cam = points @ view.R + view.T
    pts2d = points_cam[:, :2] / points_cam[:, 2:]
    pts2d = torch.addcmul(
        pts2d.new_tensor(
            [
                (view.Cx * 2.0 + 1.0) / view.image_width - 1.0,
                (view.Cy * 2.0 + 1.0) / view.image_height - 1.0,
            ]
        ),
        pts2d.new_tensor([view.Fx * 2.0 / view.image_width, view.Fy * 2.0 / view.image_height]),
        pts2d,
    )
    sampled_mask = torch.nn.functional.grid_sample(view.gt_mask[None].cuda(), pts2d[None, None], align_corners=True)
    return (sampled_mask.squeeze() > 0.5) & inside


@torch.no_grad()
def compute_valid_mask_single_view(
    fov_camera: Camera, points: torch.Tensor, znear=0.1,
) -> torch.Tensor:
    # Get parameters
    points_shape = points.shape
    Fx = fov_camera.Fx
    Fy = fov_camera.Fy
    Cx = fov_camera.Cx
    Cy = fov_camera.Cy
    H = fov_camera.image_height
    W = fov_camera.image_width
    
    # Transform points to camera space
    points_in_camera_space = transform_points_world_to_view(
        points=points.view(1, -1, 3),
        cameras=[fov_camera],
    ).squeeze(0)  # (N, 3)
    
    # Compute point projections
    pts_projections = torch.stack(
        [
            points_in_camera_space[:,0] * Fx / points_in_camera_space[:,2] + Cx,
            points_in_camera_space[:,1] * Fy / points_in_camera_space[:,2] + Cy
        ],
        -1
    ).float()
    
    # Compute frustum mask
    mask = (
        (pts_projections[:, 0] > 0) 
        & (pts_projections[:, 0] < W) 
        & (pts_projections[:, 1] > 0) 
        & (pts_projections[:, 1] < H) 
        & (points_in_camera_space[:,2] > znear)
    )
    
    return mask.view(points_shape[:-1])


@torch.no_grad()
def compute_valid_mask(points, views):
    any_valid = []
    chunk_size = 20_000_000
    
    # Get scene radius
    scene_radius = get_cameras_spatial_extent(cameras=views)['radius'].item()
    znear = 0.02 * scene_radius
    
    for point_chunk in torch.chunk(points, points.shape[0] // chunk_size + 1):
        # Initialize valid mask as False
        any_valid_chunk = torch.zeros(point_chunk.shape[0], dtype=torch.bool, device="cuda")

        # Iterate over views
        for view in tqdm(views, desc="Rendering progress"):
            # Compute frustum mask for single view
            inside = compute_valid_mask_single_view(fov_camera=view, points=point_chunk, znear=znear).view(-1)
            assert inside.shape == any_valid_chunk.shape
            
            # Combine with GT mask if available
            valid_points = evaluation_validation(view, point_chunk, inside)
            
            # Update valid mask
            any_valid_chunk = torch.logical_or(any_valid_chunk, valid_points)

        any_valid.append(any_valid_chunk)

    return torch.cat(any_valid)


@torch.no_grad()
def marching_tetrahedra_with_binary_search(
    model_path: str, 
    views: List[Camera], 
    scene: Scene, 
    gaussians: GaussianModel, 
    pipeline: PipelineParams, 
    background: torch.Tensor, 
    kernel_size: float, 
    args,
):  
    if args.dtype == "int32":
        index_dtype = torch.int32
    elif args.dtype == "int64":
        index_dtype = torch.int64
    else:
        raise ValueError(f"Invalid dtype: {args.dtype}")
    print(f"[INFO] Using {index_dtype} for indexing.")
    
    # Get scene spatial extent
    scene_radius = get_cameras_spatial_extent(cameras=views)['radius'].item()
    print(f"[INFO] Scene radius: {scene_radius}")
    
    # Define frustum parameters
    apply_frustum_culling = True
    standard_scale = 6.
    frustum_near = 0.02 * scene_radius / standard_scale
    frustum_far = 1e6 * scene_radius / standard_scale
    if apply_frustum_culling:
        print(f"[INFO] Using frustum culling with znear={frustum_near} and zfar={frustum_far}")

    if args.sdf_mode == "exact_computation":        
        transmittance_threshold = 0.5 + args.isosurface_value
        print(f"[INFO] Using transmittance threshold: {transmittance_threshold}")
        args.isosurface_value = 0.0
        
        @torch.no_grad()
        def sdf_function(points):
            splat_args = default_splat_args()
            # splat_args = ExtendedSettings()
            splat_args.sort_settings.sort_order = GlobalSortOrder.MIN_Z_BOUNDING
            splat_args.meshing_settings.alpha_early_stop = True
            splat_args.meshing_settings.transmittance_threshold = transmittance_threshold
            
            is_vacant = evaluate_vacancy_sof_fast(
                points=points,
                views=views, 
                gaussians=gaussians, 
                pipeline=pipeline, 
                background=background, 
                kernel_size=kernel_size, 
                splat_args=splat_args, 
                znear=frustum_near,
                zfar=frustum_far,
                permute_views=True,
            )  # (N,)
            
            occupancy = 1. - is_vacant.float()  # (N,)
            return 0.5 - occupancy.view(-1)  # (N,)    
    elif args.sdf_mode == "ours":
        @torch.no_grad()
        def evaluation_validation(view, points, inside):
            if view.gt_mask is None:
                return inside

            points_cam = points @ view.R + view.T
            pts2d = points_cam[:, :2] / points_cam[:, 2:]
            pts2d = torch.addcmul(
                pts2d.new_tensor(
                    [
                        (view.Cx * 2.0 + 1.0) / view.image_width - 1.0,
                        (view.Cy * 2.0 + 1.0) / view.image_height - 1.0,
                    ]
                ),
                pts2d.new_tensor([view.Fx * 2.0 / view.image_width, view.Fy * 2.0 / view.image_height]),
                pts2d,
            )
            sampled_mask = torch.nn.functional.grid_sample(view.gt_mask[None].cuda(), pts2d[None, None], align_corners=True)
            return (sampled_mask.squeeze() > 0.5) & inside
        
        @torch.no_grad()
        def sdf_function(points):
            final_sdf = []
            any_valid = []
            chunk_size = 20_000_000
            for point_chunk in torch.chunk(points, points.shape[0] // chunk_size + 1):
                final_weight_chunk = torch.ones(point_chunk.shape[0], dtype=torch.float32, device="cuda")
                any_valid_chunk = torch.zeros(point_chunk.shape[0], dtype=torch.bool, device="cuda")
                for view in tqdm(views, desc="Rendering progress"):
                    ret = integrate_ours(point_chunk, view, gaussians, pipeline, kernel_size)
                    valid_points = evaluation_validation(view, point_chunk, ret["inside"])
                    any_valid_chunk = torch.logical_or(any_valid_chunk, valid_points)
                    final_weight_chunk = torch.where(
                        valid_points,
                        torch.min(ret["alpha_integrated"], final_weight_chunk),
                        final_weight_chunk,
                    )
                final_weight_chunk[torch.logical_not(any_valid_chunk)] = 0
                final_sdf_chunk = 0.5 - final_weight_chunk
                final_sdf.append(final_sdf_chunk)
                any_valid.append(any_valid_chunk)
            # return torch.cat(final_sdf), torch.cat(any_valid)
            return torch.cat(final_sdf).view(-1)  # (N,)  
    else:
        raise ValueError(f"Invalid sdf mode: {args.sdf_mode}")
    
    # Compute best isosurface value using depth points
    if args.compute_automatically_isosurface_value:
        print("[INFO] Computing isosurface value automatically...")
        
        # Compute depth maps
        depth_maps = []
        for i_cam in tqdm(range(len(views)), desc="Computing depth maps"):
            depth_i = render(
                viewpoint_camera=views[i_cam],
                pc=gaussians,
                pipe=pipeline,
                bg_color=background,
            )['expected_depth'].cpu()
            depth_maps.append(depth_i)
        
        # Compute isosurface value from depth points
        sdf_isosurface_value = compute_isosurface_value_from_depth(
            cameras=views,
            depth=depth_maps,
            sdf_function=sdf_function,
            n_depth_points=1_000_000,
            reduction="median",
        )
        
        # Delete depth maps
        del depth_maps
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        
    else:
        print(f"[INFO] Using provided isosurface value.")
        sdf_isosurface_value = args.isosurface_value
    
    # Adjust isosurface value
    print(f"[INFO] Adjusting SDF isosurface value to {sdf_isosurface_value}...")
    def sdf_function_wrapper(points, **kwargs):
        return sdf_function(points, **kwargs) - sdf_isosurface_value
    
    # Batchify the sdf function if necessary
    def batchified_sdf_function(points, **kwargs):
        all_sdf = []
        n_points = points.shape[0]
        
        if n_points > args.n_points_per_sdf_evaluation:
            n_batches = (n_points + args.n_points_per_sdf_evaluation - 1) // args.n_points_per_sdf_evaluation
        else:
            n_batches = 1
            
        for i_batch in range(n_batches):
            start_idx = i_batch * args.n_points_per_sdf_evaluation
            end_idx = min(start_idx + args.n_points_per_sdf_evaluation, n_points)
            batch_points = points[start_idx:end_idx]
            batch_sdf = sdf_function_wrapper(batch_points, **kwargs)
            all_sdf.append(batch_sdf)
        
        return torch.cat(all_sdf, dim=0)
    
    # Get pivots
    if args.use_tetra_points:
        print("[INFO] Using tetra points. Switching to 9 pivots.")
        args.n_pivots = 9
        pivots, pivot_scales = gaussians.get_tetra_points(let_gradients_flow=False)
        pivots = pivots.view(-1, 3)
        pivot_scales = pivot_scales.view(-1, 1)
        pivot_sdfs = batchified_sdf_function(pivots)
    else:
        if args.use_scores:
            pivots = get_pivots_by_scores(
                gaussians=gaussians,
                cameras=views,
                pipe=pipeline,
                background=background,
                score_ratio_threshold=args.score_threshold,
                kernel_size=0.0,
            ).view(-1, 3)
            pivot_results = pivots, torch.ones_like(pivots[:, 0])
            
        elif args.random_pivots:
            pivots, _ = sample_random_pivots(
                n_pivots_per_gaussian=args.n_pivots, 
                gaussians=gaussians, 
                sample_radius=args.random_radius,
                sdf_function=batchified_sdf_function,
            )
            pivot_results = pivots, torch.ones_like(pivots[:, 0])
            
        elif args.use_searched_pivots:
            pivots, _ = get_searched_pivots(
                gaussians, 
                search_iter=args.search_iter, 
                sdf_function=batchified_sdf_function, 
                std_factor=args.std_factor,
                step_size=args.search_step_size,
            )
            pivot_results = pivots, torch.ones_like(pivots[:, 0])
        
        else:
            pivot_results = get_intersecting_pivots_from_normals(
                n_pivots=args.n_pivots,
                gaussians=gaussians,
                normals=None,
                std_factor=args.std_factor,
                use_smallest_axis_as_normal = args.use_smallest_axis_as_normal,
                sdf_function=batchified_sdf_function if args.use_intersecting_pivots else None,
            )
        
        # Compute SDF values for pivots
        if args.use_intersecting_pivots:
            pivots, pivot_scales, pivot_sdfs = pivot_results
            pivots = pivots.view(-1, 3)
            pivot_scales = pivot_scales.view(-1, 1)
            pivot_sdfs = pivot_sdfs.view(-1)
        else:
            pivots, pivot_scales = pivot_results
            pivots = pivots.view(-1, 3)
            pivot_scales = pivot_scales.view(-1, 1)
            pivot_sdfs = batchified_sdf_function(pivots)
    
    # Compute valid mask
    if args.use_valid_mask:
        valid_mask = compute_valid_mask(points=pivots, views=views)
        pivot_sdfs[torch.logical_not(valid_mask)] = 0.5
        print(f"[INFO] Using valid mask for marching tetrahedra with shape {valid_mask.shape}")
        print(f"[INFO] Switching SDF values of invalid points to 0.5")
    else:
        valid_mask = None
        print("[INFO] Not using valid mask for marching tetrahedra")
    
    # Compute Delaunay triangulation
    t0 = time.time()
    tets = return_delaunay_tets(pivots, method="tetranerf").cpu()
    t1 = time.time()
    print(f"[INFO] Computed {tets.shape[0]} tets with Delaunay triangulation: {t1 - t0}s")
    
    # Extract mesh
    mesh, details = extract_mesh(
        delaunay_tets=tets.cuda(),
        pivots=pivots,
        pivots_sdf=pivot_sdfs,
        pivots_colors=None,
        pivots_scale=pivot_scales,
        filter_large_edges=args.filter_large_edges,
        collapse_large_edges=args.collapse_large_edges,
        return_details=True,
        mtet_on_cpu=args.mtet_on_cpu,
        valid=valid_mask,
    )
    torch.cuda.empty_cache()
    
    # Binary search
    if args.n_binary_steps > 0:        
        end_points = details['end_points']
        end_sdf = details['end_sdf']
        verts = refine_intersections_with_binary_search(
            end_points=end_points,
            end_sdf=end_sdf,
            sdf_function=batchified_sdf_function,
            n_binary_steps=args.n_binary_steps,
        )
        mesh.verts = verts

    # Compute vertex colors
    print("[INFO] Computing vertex colors...")
    verts_colors = evaluate_mesh_colors_all_vertices(
        views=views, 
        mesh=mesh,
        masks=None,
        use_scalable_renderer=True,
    ).view(-1, 3)
    
    # Create mesh
    mesh = trimesh.Trimesh(
        vertices=mesh.verts.cpu().numpy(), 
        faces=mesh.faces.cpu().numpy(), 
        vertex_colors=(verts_colors.cpu().numpy() * 255).astype(np.uint8), 
        process=False
    )

    # Export mesh
    mesh_save_path = os.path.join(model_path,f"mesh_{args.sdf_mode}_{args.n_pivots}pivots.ply")
    if args.use_scores:
        mesh_save_path = mesh_save_path.replace(".ply", "_scores.ply")
    if args.use_searched_pivots:
        mesh_save_path = mesh_save_path.replace(".ply", "_searched.ply")
    mesh.export(mesh_save_path)
    print(f"Mesh saved to:\n{mesh_save_path}")
    
    # Postprocess
    if args.postprocess:
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))

        print("[INFO] Start postprocessing to remove floaters and disconnected parts.")
        mesh = post_process_mesh(o3d_mesh, 1)
        print(f"[INFO] Postprocessing done.")
        
        mesh_save_path = mesh_save_path.replace(".ply", "_post.ply")
        o3d.io.write_triangle_mesh(mesh_save_path, mesh)
        print(f"[INFO] Postprocessed mesh saved to: \n{mesh_save_path}")


@torch.no_grad()
def main(
    dataset : ModelParams, 
    pipeline : PipelineParams, 
    args,
):
    # Load scene and Gaussian model
    gaussians = GaussianModel(dataset.sh_degree, use_unbounded_opacity=dataset.use_unbounded_opacity)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply"))
    if gaussians.learn_occupancy:
        gaussians.set_occupancy_mode("occupancy_shift")
    print(f"[INFO] Loaded Gaussian Model from {os.path.join(dataset.model_path, 'point_cloud', f'iteration_{args.iteration}', 'point_cloud.ply')}")
    print(f"[INFO]    > Number of Gaussians: {gaussians._xyz.shape[0]}")
    
    # Background color and kernel size
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    try:
        kernel_size = dataset.kernel_size
    except:
        print("No kernel size found in dataset, using 0.0")
        kernel_size = 0.0
    
    # Extract mesh
    marching_tetrahedra_with_binary_search(
        model_path=dataset.model_path, 
        views=scene.getTrainCameras(), 
        scene=scene, 
        gaussians=gaussians, 
        pipeline=pipeline, 
        background=background, 
        kernel_size=kernel_size, 
        args=args,
    )

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--rasterizer", default="ours", choices=["radegs", "ours"])
    
    # SDF function to use
    parser.add_argument(
        "--sdf_mode", 
        default="exact_computation", 
        choices=[
            "exact_computation", # Slower but exact
            "ours",
        ]
    )
    parser.add_argument("--n_points_per_sdf_evaluation", default=999_999_999, type=int)
    parser.add_argument("--dtype", default="int32", choices=["int32", "int64"])
    
    # Pivots
    parser.add_argument("--n_pivots", default=2, type=int)
    parser.add_argument("--use_smallest_axis_as_normal", action="store_true")
    parser.add_argument("--std_factor", default=3.0, type=float)
    parser.add_argument("--use_intersecting_pivots", action="store_true")
    parser.add_argument("--use_tetra_points", action="store_true")
    #   > Score-based pivots
    parser.add_argument("--use_scores", action="store_true")
    parser.add_argument("--score_threshold", default=0.75, type=float)
    #   > Random pivots spawned from Gaussians
    parser.add_argument("--random_pivots", action="store_true")
    parser.add_argument("--random_radius", default=2.0, type=float)
    #   > Refined pivots by searching in the direction of the normal
    parser.add_argument("--use_searched_pivots", action="store_true")
    parser.add_argument("--search_iter", default=10, type=int)
    parser.add_argument("--search_step_size", default=1.0, type=float)
    
    # Extraction
    parser.add_argument("--mtet_on_cpu", action="store_true")
    parser.add_argument("--use_valid_mask", action="store_true")
    parser.add_argument("--filter_large_edges", action="store_true")
    parser.add_argument("--collapse_large_edges", action="store_true")

    # Integration
    parser.add_argument("--n_binary_steps", default=8, type=int)
    parser.add_argument("--isosurface_value", default=-9999., type=float)  # 0.2 is a good value for GOF and occupancy
    
    # Postprocessing
    parser.add_argument("--postprocess", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    print(f"[ INFO ] Using rasterizer: {args.rasterizer}")
    if args.rasterizer == "radegs":
        from gaussian_renderer.radegs import render_radegs as render
        from gaussian_renderer.radegs import integrate_radegs as integrate
    elif args.rasterizer == "ours":
        from gaussian_renderer.ours import render_ours as render
    else:
        raise ValueError(f"Invalid rasterizer: {args.rasterizer}")
    
    # Change args if use scores
    if args.use_scores or args.random_pivots or args.use_searched_pivots:
        if args.random_pivots:
            print("[INFO] Using random pivots.")
        elif args.use_scores:
            print("[INFO] Using scores for pivots.")
        elif args.use_searched_pivots:
            print("[INFO] Using searched pivots.")
        args.use_tetra_points = False
        args.use_intersecting_pivots = False
        args.filter_large_edges = False
        args.collapse_large_edges = False
    
    # For integration mode
    args.compute_automatically_isosurface_value = (args.isosurface_value <= -9999.)
        
    main(
        model.extract(args), 
        pipeline.extract(args), 
        args,
    )
    