import torch
import gc
from tqdm import tqdm
# from utils.graphics_utils import depth_double_to_normal
from regularization.depth.depth_order import compute_depth_order_loss
# from scene import Scene, GaussianModel
# from scene.cameras import Camera


def initialize_depth_order_supervision(
    scene, 
    config:dict, 
    device='cuda'
):
    """
    Initializes depth priors using DepthAnythingV2.

    Args:
        scene: The scene object containing training cameras.
        config: Configuration dictionary for depth order supervision.
        device: The device to run the depth estimation model on.

    Returns:
        A list of depth priors (torch.Tensor) stored on CPU.
    """
    print("[INFO] Depth-order supervision enabled.")
    print("[INFO] Building depth priors from input RGB images...")
    viewpoint_stack = scene.getTrainCameras().copy()
    try:
        from regularization.depth.depthanythingv2 import load_depthanything, apply_depthanything
    except ImportError:
        print("[ERROR] DepthAnythingV2 not found. Please ensure it's installed.")
        raise

    dav2 = load_depthanything(
        checkpoint_dir=config["depthanythingv2_checkpoint_dir"],
        encoder=config["depthanythingv2_encoder"],
        device=device
    )
    dav2.eval()
    depth_priors = []
    with torch.no_grad():
        for i_image in tqdm(range(len(viewpoint_stack)), desc="Building depth priors"):
            gt_image = viewpoint_stack[i_image].original_image.permute(1, 2, 0) # H, W, C
            supervision_disparity = apply_depthanything(dav2, image=gt_image)
            supervision_disparity = (
                (supervision_disparity - supervision_disparity.min()) 
                / (supervision_disparity.max() - supervision_disparity.min())
            )
            supervision_depth = 1. / (0.1 + 0.9 * supervision_disparity)
            depth_priors.append(supervision_depth.squeeze().unsqueeze(0).to('cpu'))  # (1, H, W)

    del dav2
    gc.collect()
    torch.cuda.empty_cache()
    print("[INFO] Depth priors built.")
    return depth_priors


def compute_depth_order_regularization(
    iteration: int,
    rendered_depth: torch.Tensor,
    depth_priors: list,
    viewpoint_idx: int,
    gaussians,
    config: dict,
):
    """
    Computes the depth order regularization loss.

    Args:
        iteration: Current training iteration.
        rendered_depth: Tensor containing rendered depth. Has shape (1, H, W).
        depth_priors: List of precomputed depth priors.
        viewpoint_idx: Index of the current viewpoint camera.
        gaussians: The GaussianModel object.
        config: Configuration dictionary for depth order supervision.
        require_depth: Flag indicating if depth was rendered.

    Returns:
        torch.Tensor: The computed depth prior loss.
        torch.Tensor: The surface depth map used for the loss.
        torch.Tensor: The supervision depth map for the current view.
        float: The weight (lambda) applied to the depth order loss.
    """
    lambda_depth_order = config["weight_initial_value"]
    for i_depth_order_update_iter, depth_order_update_iter in enumerate(config["weight_update_iters"]):
        if iteration == depth_order_update_iter:
            print(f"[INFO] Updating depth order regularization weight to {config['weight_update_values'][i_depth_order_update_iter]} at iteration {iteration}.")
        if iteration >= depth_order_update_iter:
            lambda_depth_order = config["weight_update_values"][i_depth_order_update_iter]

    # If lambda_depth_order is 0, return 0 loss
    if lambda_depth_order <= 0:
        zero_loss = torch.tensor(0.0, device=gaussians.get_xyz.device)
        zero_depth = None
        zero_supervision = None
        return zero_loss, zero_depth, zero_supervision, lambda_depth_order
    
    # Resize supervision depth to match rendered depth size if necessary
    # Can be useful at beginning of training, when resolution warmup is active
    supervision_depth: torch.Tensor = depth_priors[viewpoint_idx].to(rendered_depth.device)
    _, sup_H, sup_W = supervision_depth.shape
    _, H, W = rendered_depth.shape
    if sup_H != H or sup_W != W:
        supervision_depth = torch.nn.functional.interpolate(
            supervision_depth.view(1, 1, sup_H, sup_W),  # (1, 1, sup_H, sup_W)
            (H, W), 
            mode="bilinear", 
            align_corners=True
        ).view(rendered_depth.shape)
        
    # Compute depth prior loss
    depth_prior_loss: torch.Tensor = lambda_depth_order * compute_depth_order_loss(
        depth=rendered_depth.squeeze(),
        prior_depth=supervision_depth.squeeze(),
        scene_extent=gaussians.spatial_lr_scale,
        max_pixel_shift_ratio=config["max_pixel_shift_ratio"],
        normalize_loss=config["normalize_loss"],
        log_space=config["log_space"],
        log_scale=config["log_scale"],
        reduction=config["reduction"],
        debug=False,
    )

    return depth_prior_loss, rendered_depth, supervision_depth, lambda_depth_order


# def get_depth_order_normals_for_logging(viewpoint_cam, supervision_depth):
#     """Computes normals from supervision depth for logging purposes."""
#     if supervision_depth.sum() == 0: # Handle case where lambda_depth_order is 0
#          return torch.zeros((3, viewpoint_cam.image_height, viewpoint_cam.image_width), device=supervision_depth.device)
#     # Ensure supervision_depth is on the correct device and has channel dim
#     supervision_depth_dev = supervision_depth.to(viewpoint_cam.R.device)
#     if supervision_depth_dev.dim() == 2:
#         supervision_depth_dev = supervision_depth_dev.unsqueeze(0)

#     # depth_double_to_normal expects (1, H, W)
#     supervision_normal = depth_double_to_normal(
#         viewpoint_cam,
#         supervision_depth_dev,
#         supervision_depth_dev # Use same depth for expected and median
#     )[0] # Get the first normal map (expected == median)
#     return supervision_normal