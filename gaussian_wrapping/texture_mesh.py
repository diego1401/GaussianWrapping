#adopted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/extract_mesh.py
import numpy as np
import torch
from scene import Scene
import os
import random
from argparse import ArgumentParser, BooleanOptionalAction
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import trimesh
from scene.mesh import Meshes, MeshRenderer, ScalableMeshRenderer, MeshRasterizer
from regularization.sdf.depth_fusion import frustum_cull_mesh
from tqdm import tqdm
import gc
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim
from random import randint


def main(
    dataset : ModelParams, 
    pipeline : PipelineParams, 
    args,
):
    # Get device
    device = torch.device(torch.cuda.current_device())
    
    # Load scene and Gaussian model
    gaussians = GaussianModel(dataset.sh_degree, use_unbounded_opacity=dataset.use_unbounded_opacity)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply"))
    if gaussians.learn_occupancy:
        gaussians.set_occupancy_mode("occupancy_shift")
    print(f"[INFO] Loaded Gaussian Model from {os.path.join(dataset.model_path, 'point_cloud', f'iteration_{args.iteration}', 'point_cloud.ply')}")
    print(f"[INFO]    > Number of Gaussians: {gaussians._xyz.shape[0]}")
    
    # Get cameras
    cameras = scene.getTrainCameras()
    
    # Background color and kernel size
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    try:
        kernel_size = dataset.kernel_size
    except:
        print("No kernel size found in dataset, using 0.0")
        kernel_size = 0.0
    
    # Render images with Gaussians
    images = []
    gaussians.active_sh_degree = args.sh_degree_for_texturing
    print(f"[INFO] Set SH degree to {gaussians.active_sh_degree}")
    for i_cam in range(len(cameras)):
        with torch.no_grad():
            render_pkg = render(
                viewpoint_camera=cameras[i_cam],
                pc=gaussians,
                pipe=pipeline,
                bg_color=background,
            )
            images.append(render_pkg['render'].cpu())
        
    # Load mesh
    mesh_name = os.path.basename(args.mesh).split(".")[0]
    mesh_extension = os.path.basename(args.mesh).split(".")[1]
    print(f"[INFO] Loading mesh from {args.mesh}")
    print(f"          > Mesh name: {mesh_name}")
    print(f"          > Mesh extension: {mesh_extension}")
    mesh = trimesh.load(args.mesh)
    
    # Get mesh args
    verts = torch.from_numpy(mesh.vertices).float().to(device="cuda")
    faces = torch.from_numpy(mesh.faces).to(device="cuda")
    _verts_colors = torch.from_numpy(mesh.visual.vertex_colors).float().to(device="cuda")[:, :3] / 255.0  # (N, 3)
    _verts_colors = torch.nn.Parameter(_verts_colors, requires_grad=True).to(device="cuda")
    print(f"[INFO] Vertex colors shape: {_verts_colors.shape}")
    print(f"[INFO] Vertex colors max: {_verts_colors.max()}, min: {_verts_colors.min()}")
    
    # Instantiates parameters and optimizer
    l = [{'params': [_verts_colors], 'lr': args.lr, "name": "verts_colors"}]
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    print(f"[INFO] Learnable parameters:")
    for param_group in optimizer.param_groups:
        print(param_group["name"], param_group["lr"], param_group["params"][0].shape)
    
    # Define mesh renderer
    if args.use_scalable_renderer:
        renderer = ScalableMeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras)
        )
    else:
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras)
        )
        
    # Texture refinement
    viewpoint_idx_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(args.n_iter), desc="Texture refinement progress")
    
    print(f"[INFO] Starting texture refinement with {args.n_iter} iterations")
    for i_iter in range(args.n_iter):
        # Get updated mesh
        mesh = Meshes(
            verts=verts,
            faces=faces,
            verts_colors=0. + _verts_colors,
        )
        
        # Get random viewpoint
        if not viewpoint_idx_stack:
            viewpoint_idx_stack = list(range(len(cameras)))
        _random_view_idx = randint(0, len(viewpoint_idx_stack)-1)
        viewpoint_idx = viewpoint_idx_stack.pop(_random_view_idx)
        
        # Render frustum-culled mesh
        rendered_image = renderer(
            mesh=frustum_cull_mesh(mesh, cameras[viewpoint_idx]),  # FIXME: Add znear
            cam_idx=viewpoint_idx,
            return_depth=True,
            return_normals=True,
            use_antialiasing=True,  
        )
        mesh_rgb = rendered_image['rgb'].squeeze(0).permute(2, 0, 1)
        
        gt_image = images[viewpoint_idx].to(device)
        Ll1 = l1_loss(mesh_rgb, gt_image)
        ssim_value = fused_ssim(mesh_rgb.unsqueeze(0), gt_image.unsqueeze(0))
        loss = ((1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim_value))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            
            if i_iter % 5 == 0:
                postfix_dict = {}
                postfix_dict["Loss"] = f"{ema_loss_for_log:.{7}f}"
                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(5)
        
        if i_iter % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
    print(f"[INFO] Texture refinement completed")
    
    # Create mesh
    with torch.no_grad():
        mesh = trimesh.Trimesh(
            vertices=mesh.verts.cpu().numpy(), 
            faces=mesh.faces.cpu().numpy(), 
            vertex_colors=(mesh.verts_colors.detach().clamp(0., 1.).cpu().numpy() * 255).astype(np.uint8), 
            process=False
        )
    output_path = os.path.join(dataset.model_path, mesh_name + f"_texture_refined_{i_iter}.{mesh_extension}")
    mesh.export(output_path)
    print(f"[INFO] Texture refined mesh saved to {output_path}")
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--rasterizer", default="ours", choices=["radegs", "ours"])
    
    # Mesh path
    parser.add_argument("--mesh", type=str)
    
    # texture refinement
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--sh_degree_for_texturing", type=int, default=0)
    parser.add_argument("--use_scalable_renderer", action=BooleanOptionalAction, default=True)

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
        
    main(
        model.extract(args), 
        pipeline.extract(args), 
        args,
    )
    