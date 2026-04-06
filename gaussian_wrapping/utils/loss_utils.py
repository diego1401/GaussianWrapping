#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from diff_gaussian_rasterization_ms._C import fusedssim, fusedssim_backward

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map, MU1, MU2, SIGMA1_sq, SIGMA2_sq, SIGMA12 = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2, MU1, MU2, SIGMA1_sq, SIGMA2_sq, SIGMA12)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, MU1, MU2, SIGMA1_sq, SIGMA2_sq, SIGMA12 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, MU1, MU2, SIGMA1_sq, SIGMA2_sq, SIGMA12, opt_grad)
        return None, None, grad, None


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)

    return ssim_map.mean()


# function L1_loss_appearance is fork from GOF https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/train.py
def L1_loss_appearance(image, gt_image, gaussians, view_idx, return_transformed_image=False):
    assert (gaussians.use_appearance_network or gaussians.use_exposure_compensation), "Appearance network or exposure compensation must be used"
    assert (
        (not gaussians.use_appearance_network)
        or (not gaussians.use_exposure_compensation)
    ), "Appearance network and exposure compensation cannot be used together"
    
    # Appearance network from GOF
    if gaussians.use_appearance_network:
        appearance_embedding = gaussians.get_appearance_embedding(view_idx)
        # center crop the image
        origH, origW = image.shape[1:]
        H = origH // 32 * 32
        W = origW // 32 * 32
        left = origW // 2 - W // 2
        top = origH // 2 - H // 2
        crop_image = image[:, top:top+H, left:left+W]
        crop_gt_image = gt_image[:, top:top+H, left:left+W]
        
        # down sample the image
        crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
        
        crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
        mapping_image = gaussians.appearance_network(crop_image_down)
        transformed_image = mapping_image * crop_image
        if return_transformed_image:
            transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
            return transformed_image
        else:
            return l1_loss(transformed_image, crop_gt_image)
        
    # Exposure compensation from PGSR
    elif gaussians.use_exposure_compensation:
        appearance_embedding = gaussians.get_appearance_embedding(view_idx)  # (2,)
        transformed_image = torch.addcmul(appearance_embedding[1], torch.exp(appearance_embedding[0]), image)  # (3, H, W)
        if return_transformed_image:
            return transformed_image
        else:
            return l1_loss(transformed_image, gt_image)
    
def get_img_grad_weight(img: torch.Tensor) -> torch.Tensor:
    """
    Function from PGSR:
    https://github.com/zju3dv/PGSR/blob/de24f1a38b350387e8d8fe381b2cd70c1ae946e7/utils/loss_utils.py#L92
    Computes a weight map based on the gradient of the RGB image.

    Args:
        img (torch.Tensor): The RGB image, with shape (3, H, W).

    Returns:
        torch.Tensor: The weight map, with shape (H, W).
    """
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)  # (1, H, W)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)  # (1, H, W)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)  # (2, H, W)
    grad_img, _ = torch.max(grad_img, dim=0)  # (H, W)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())  # (H, W)
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()  # (H, W)
    return grad_img  # (H, W)
