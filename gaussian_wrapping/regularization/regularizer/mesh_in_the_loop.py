from typing import Dict, Any, List
import torch
from arguments import PipelineParams
from scene import Scene
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.mesh import (
    Meshes, 
    MeshRasterizer,
    MeshRenderer,
    ScalableMeshRenderer,
    return_delaunay_tets,
)
from extraction.pivots import get_intersecting_pivots_from_normals
from extraction.mesh import extract_mesh
from regularization.mesh_in_the_loop import (
    compute_occupancy_sof,
    convert_occupancy_to_sdf,
    convert_sdf_to_occupancy,
)
from utils.geometry_utils import (
    depth_to_normal, 
    depth_to_normal_with_mask,
    is_in_view_frustum,
)


def initialize_mesh_in_the_loop_regularization(
    scene: Scene,
    gaussians: GaussianModel,
    milo_config: Dict[str, Any],
):
    # Set occupancy mode
    gaussians.set_occupancy_mode("occupancy_shift")
    
    # Define mesh renderer
    mesh_rasterizer = MeshRasterizer(
        cameras=scene.getTrainCameras().copy(),
        use_opengl=milo_config["rasterization_backend"] == "opengl",
    )
    if milo_config["use_scalable_renderer"]:
        mesh_renderer = ScalableMeshRenderer(mesh_rasterizer)
    else:
        mesh_renderer = MeshRenderer(mesh_rasterizer)
    
    # Build state
    milo_state = {
        "has_started": False,
        "delaunay_tets": None,
        "mesh_renderer": mesh_renderer,
        "reset_milo": False,
    }
    return milo_state


def reset_milo_state_at_next_iteration(milo_state):
    milo_state["reset_milo"] = True
    milo_state["delaunay_tets"] = None
    return milo_state


def compute_mesh_in_the_loop_regularization(
    iteration: int,
    train_cameras: List[Camera],
    viewpoint_cam: Camera,
    viewpoint_idx: int,
    render_pkg: Dict[str, torch.Tensor],
    gaussians: GaussianModel,
    pipe: PipelineParams,
    background: torch.Tensor,
    kernel_size: float,
    milo_config: Dict[str, Any],
    milo_state: Dict[str, Any],
    args,
) -> Dict[str, Any]:
    
    n_gaussians = gaussians._xyz.shape[0]
    n_pivots_per_gaussian = gaussians.n_pivots_per_gaussian
    
    # Check if first iteration
    first_iteration = not milo_state["has_started"]
    milo_state["has_started"] = True
    
    # Check if reset MILO
    reset_milo = milo_state["reset_milo"]
    if reset_milo:
        print(f"[INFO] Resetting MILO state at iteration {iteration}")
        milo_state["reset_milo"] = False
    
    # iteration since milo started
    iteration_since_milo = iteration - milo_config["start_iter"]
    
    # Reset Base occupancy logic
    reset_base_occupancy = False
    if (
        first_iteration
        or reset_milo
        or (
            iteration_since_milo % milo_config["reset_base_occupancy_every"] == 0
            and iteration >= milo_config["start_reset_base_occupancy_iter"]
            and iteration <= milo_config["end_reset_base_occupancy_iter"]
        )
    ):
        reset_base_occupancy = True
    
    # Reset Delaunay logic
    reset_delaunay = False
    if (
        first_iteration
        or reset_milo
        or (
            iteration_since_milo % milo_config["reset_delaunay_every"] == 0
            and iteration <= milo_config["end_reset_delaunay_iter"]
        )
    ):
        reset_delaunay = True
        milo_state["delaunay_tets"] = None
    
    # Compute pivots from normals
    pivots, pivot_scales = get_intersecting_pivots_from_normals(
        n_pivots=gaussians.n_pivots_per_gaussian,
        gaussians=gaussians,
        normals=None,
        std_factor=3.0,
        use_smallest_axis_as_normal=milo_config["use_smallest_axis_as_normal"],
        sdf_function=None,
    )  # (N_gaussians, n_pivots_per_gaussian, 3), (N_gaussians, n_pivots_per_gaussian, 1)
    
    # Get or update Delaunay triangulation
    if reset_delaunay:
        print(f"[INFO] Resetting Delaunay triangulation at iteration {iteration}")
        with torch.no_grad():
            delaunay_tets = return_delaunay_tets(pivots.detach().view(-1, 3), method=milo_config["delaunay_method"])  # (N_tets, 4)
        milo_state["delaunay_tets"] = delaunay_tets
    else:
        delaunay_tets = milo_state["delaunay_tets"]
        
    # Reset base occupancy
    if reset_base_occupancy:
        print(f"[INFO] Resetting base occupancy at iteration {iteration}")
        # Get current occupancy
        with torch.no_grad():
            current_occupancy = gaussians.get_occupancy  # (N_gaussians, n_pivots_per_gaussian)
        
        # Compute new base occupancy
        new_base_occupancy = compute_occupancy_sof(
            points=pivots.view(-1, 3),  # (N_pivots, 3)
            cameras=train_cameras, 
            gaussians=gaussians,
            pipeline=pipe,
            background=background,
            kernel_size=kernel_size,
        )  # (N_pivots,)
        new_base_occupancy = new_base_occupancy.view(*current_occupancy.shape)
        new_base_occupancy = new_base_occupancy.clamp(min=0.005, max=0.995)
        
        # Compute new occupancy
        if first_iteration:
            new_occupancy = new_base_occupancy
        else:
            ema_ratio = milo_config["reset_occupancy_ema_ratio"]
            new_occupancy = (
                ema_ratio * current_occupancy
                + (1. - ema_ratio) * new_base_occupancy
            ).clamp(min=0.005, max=0.995)

        # Reset occupancy
        gaussians.reset_occupancy(
            base_occupancy=new_base_occupancy, 
            occupancy=new_occupancy,
        )
        
    # Compute pivots SDF
    pivots_occupancy = gaussians.get_occupancy  # (N_gaussians, n_pivots_per_gaussian)
    if True:
        pivots_sdf = convert_occupancy_to_sdf(pivots_occupancy)  # (N_gaussians, n_pivots_per_gaussian)
    else:
        pivots_sdf = gaussians.get_sdf  # (N_gaussians, n_pivots_per_gaussian)
    
    # Compute pivots colors
    if milo_config["use_pivots_colors"]:
        # TODO: Implement pivots colors
        raise NotImplementedError("Pivots colors are not implemented")
    else:
        pivots_colors = None
    
    # Build mesh
    mesh = extract_mesh(
        delaunay_tets=delaunay_tets,
        pivots=pivots.view(-1, 3),
        pivots_sdf=pivots_sdf.view(-1),
        pivots_colors=pivots_colors.view(-1, 3) if pivots_colors is not None else None,
        pivots_scale=pivot_scales.view(-1),
        filter_large_edges=milo_config["filter_large_edges"],
        collapse_large_edges=milo_config["collapse_large_edges"],
        return_details=False,
        sdf_sh=None,
        mtet_on_cpu=False,
    )
    
    # Filter out faces not in view frustum
    with torch.no_grad():
        faces_mask = is_in_view_frustum(mesh.verts, viewpoint_cam)[mesh.faces].any(axis=1)
    mesh = Meshes(verts=mesh.verts, faces=mesh.faces[faces_mask])
    
    # Render mesh
    mesh_render_pkg = milo_state["mesh_renderer"](
        mesh=mesh, 
        # cameras=train_cameras, 
        # cam_idx=i_img,
        cameras=[viewpoint_cam],
        cam_idx=0,
        return_depth=True,
        return_normals=True,
        use_antialiasing=True,
        return_pix_to_face=False,
        check_errors=True,
    )
    mesh_depth = mesh_render_pkg['depth'][0].permute(2, 0, 1)  # (1, H, W)
    mesh_normal_view = mesh_render_pkg['normals'][0] @ viewpoint_cam.world_view_transform[:3,:3]  # (H, W, 3)
    mesh_normal_view = mesh_normal_view.permute(2, 0, 1)  # (3, H, W)
    rasterization_mask = mesh_depth[0] > 0.  # (H, W)
    if milo_config["use_pivots_colors"]:
        mesh_rgb = mesh_render_pkg['rgb'][0].permute(2, 0, 1)  # (3, H, W)
    else:
        mesh_rgb = torch.zeros_like(mesh_normal_view)  # (3, H, W)
        
    # RGB loss
    if milo_config["use_rgb_loss"]:
        assert milo_config["use_pivots_colors"]
        raise NotImplementedError("RGB loss is not implemented")
        
    # Depth Loss
    if milo_config["use_depth_loss"]:
        gaussians_depth = (
            (1. - milo_config["depth_ratio"]) * render_pkg["expected_depth"]  # (1, H, W)
            + milo_config["depth_ratio"] * render_pkg["median_depth"]  # (1, H, W)
        )  # (1, H, W)
        mesh_depth_loss = torch.log(1. + (mesh_depth - gaussians_depth).abs() / gaussians.spatial_lr_scale)  # (H, W)
        mesh_depth_loss = milo_config["depth_weight"] * (mesh_depth_loss * rasterization_mask).mean()
    else:
        mesh_depth_loss = torch.zeros(size=(), device=gaussians._xyz.device)

    # Normal Loss
    if milo_config["use_normal_loss"]:
        if milo_config["use_depth_normal"]:
            # Compute normals from Gaussian depth map
            if args.mask_depth_normal:
                depth_blend = (
                    (1. - milo_config["depth_ratio"]) * render_pkg["expected_depth"] 
                    + milo_config["depth_ratio"] * render_pkg["median_depth"]
                )  # (1, H, W)
                gaussians_normal_view, valid_gaussians_depth_points = depth_to_normal_with_mask(
                    viewpoint_cam, 
                    depth_blend,
                )  # (3, H, W), (H, W)
            else:
                depth_middepth_normal = depth_to_normal(
                    viewpoint_cam,
                    render_pkg["expected_depth"],  # (1, H, W)
                    render_pkg["median_depth"],  # (1, H, W)
                )  # (2, 3, H, W)
                gaussians_normal_view = (
                    (1. - milo_config["depth_ratio"]) * depth_middepth_normal[0]  # (3, H, W)
                    + milo_config["depth_ratio"] * depth_middepth_normal[1]  # (3, H, W)
                ) # (3, H, W)
        else:
            # Use rendered normals directly (already in view space)
            gaussians_normal_view = render_pkg["normal"]  # (3, H, W)
        normal_dot_product = (mesh_normal_view * gaussians_normal_view).sum(dim=0)  # (H, W)
        mesh_normal_loss = 1. - normal_dot_product.abs()  # (H, W)
        if args.mask_depth_normal:
            mesh_normal_loss = torch.where(
                valid_gaussians_depth_points.squeeze(),  # (H, W),
                mesh_normal_loss,  # (H, W),
                torch.zeros_like(mesh_normal_loss),  # (H, W),
            )  # (H, W)
        mesh_normal_loss = milo_config["normal_weight"] * (mesh_normal_loss * rasterization_mask).mean()
    else:
        mesh_normal_loss = torch.zeros(size=(), device=gaussians._xyz.device)
        
    # Loss to enforce Gaussian centers to be inside the mesh
    if milo_config["enforce_occupied_centers"]:
        assert pivots_occupancy.shape == (n_gaussians, n_pivots_per_gaussian)
        centers_occupancy = pivots_occupancy[:, -1]  # (N_gaussians,)
        occupied_centers_loss = milo_config["occupied_centers_weight"] * (
            milo_config["occupancy_isosurface"] - centers_occupancy
        ).clamp(min=0.).mean()
    else:
        occupied_centers_loss = torch.zeros(size=(), device=gaussians._xyz.device)

    # Total loss
    total_loss = (
        mesh_depth_loss
        + mesh_normal_loss
        + occupied_centers_loss
    )
    
    return {
        "milo_loss": total_loss,
        "mesh_depth_loss": mesh_depth_loss.detach(),
        "mesh_normal_loss": mesh_normal_loss.detach(),
        "occupied_centers_loss": occupied_centers_loss.detach(),
        "mesh_rgb": mesh_rgb.detach(),  # (3, H, W)
        "mesh_depth": mesh_depth.detach(),  # (1, H, W)
        "mesh_normal": mesh_normal_view.detach(),  # (3, H, W)
    }
