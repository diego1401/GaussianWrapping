import torch


def depth_order_weight_schedule(iteration:int, schedule:str="default"):
    if schedule == "default":
        lambda_depth_order = 0.
        if iteration > 3_000:
            lambda_depth_order = 1.
        if iteration > 7_000:
            lambda_depth_order = 0.1
        if iteration > 15_000:
            lambda_depth_order = 0.01
        if iteration > 20_000:
            lambda_depth_order = 0.001
        if iteration > 25_000:
            lambda_depth_order = 0.0001
    
    elif schedule == "strong":
        lambda_depth_order = 1.
        
    elif schedule == "weak":
        lambda_depth_order = 0.
        if iteration > 3_000:
            lambda_depth_order = 0.1
            
    elif schedule == "none":
        lambda_depth_order = 0.
        
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    
    return lambda_depth_order


def compute_depth_order_loss(
    depth:torch.Tensor, 
    prior_depth:torch.Tensor, 
    scene_extent:float=1., 
    max_pixel_shift_ratio:float=0.05,
    normalize_loss:bool=True,
    log_space:bool=False,
    log_scale:float=20.,
    reduction:str="mean",
    debug:bool=False,
):
    """Compute a loss encouraging pixels in 'depth' to have the same relative depth order as in 'prior_depth'.
    This loss does not require prior depth maps to be multi-view consistent nor to have accurate relative scale.

    Args:
        depth (torch.Tensor): A tensor of shape (H, W), (H, W, 1) or (1, H, W) containing the depth values.
        prior_depth (torch.Tensor): A tensor of shape (H, W), (H, W, 1) or (1, H, W) containing the prior depth values.
        scene_extent (float): The extent of the scene used to normalize the loss and make the loss invariant to the scene scale.
        max_pixel_shift_ratio (float, optional): The maximum pixel shift ratio. Defaults to 0.05, i.e. 5% of the image size.
        normalize_loss (bool, optional): Whether to normalize the loss. Defaults to True.
        reduction (str, optional): The reduction to apply to the loss. Can be "mean", "sum" or "none". Defaults to "mean".
    
    Returns:
        torch.Tensor: A scalar tensor.
            If reduction is "none", returns a tensor with same shape as depth containing the pixel-wise depth order loss.
    """
    height, width = depth.squeeze().shape
    pixel_coords = torch.stack(torch.meshgrid(
        torch.linspace(0, height - 1, height, dtype=torch.long, device=depth.device),
        torch.linspace(0, width - 1, width, dtype=torch.long, device=depth.device),
        indexing='ij'
    ), dim=-1).view(-1, 2)

    # Get random pixel shifts
    # TODO: Change the sampling so that shifts of (0, 0) are not possible
    max_pixel_shift = max(round(max_pixel_shift_ratio * max(height, width)), 1)
    pixel_shifts = torch.randint(-max_pixel_shift, max_pixel_shift + 1, pixel_coords.shape, device=depth.device)

    # Apply pixel shifts to pixel coordinates and clamp to image boundaries
    shifted_pixel_coords = (pixel_coords + pixel_shifts).clamp(
        min=torch.tensor([0, 0], device=depth.device), 
        max=torch.tensor([height - 1, width - 1], device=depth.device)
    )

    # Get depth values at shifted pixel coordinates
    shifted_depth = depth.squeeze()[
        shifted_pixel_coords[:, 0], 
        shifted_pixel_coords[:, 1]
    ].reshape(depth.shape)
    shifted_prior_depth = prior_depth.squeeze()[
        shifted_pixel_coords[:, 0], 
        shifted_pixel_coords[:, 1]
    ].reshape(depth.shape)

    # Compute pixel-wise depth order loss
    diff = (depth - shifted_depth) / scene_extent
    prior_diff = (prior_depth - shifted_prior_depth) / scene_extent
    if normalize_loss:
        prior_diff = prior_diff / prior_diff.detach().abs().clamp(min=1e-8)
    depth_order_loss = - (diff * prior_diff).clamp(max=0)
    if log_space:
        depth_order_loss = torch.log(1. + log_scale * depth_order_loss)
    
    # Reduce the loss
    if reduction == "mean":
        depth_order_loss = depth_order_loss.mean()
    elif reduction == "sum":
        depth_order_loss = depth_order_loss.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    
    if debug:
        return {
            "depth_order_loss": depth_order_loss,
            "diff": diff,
            "prior_diff": prior_diff,
            "shifted_depth": shifted_depth,
            "shifted_prior_depth": shifted_prior_depth,
        }
    return depth_order_loss