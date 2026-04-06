import os
import sys
import torch
import numpy as np

sys.path.append('./Depth-Anything-V2/')
from depth_anything_v2.dpt import DepthAnythingV2


def load_depthanything(
    checkpoint_dir='./Depth-Anything-V2/checkpoints/',
    encoder='vitl',  # or 'vits', 'vitb', 'vitg',
    device='cpu',
):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'depth_anything_v2_{encoder}.pth'), map_location='cpu'))
    model = model.to(device).eval()
    
    return model


@torch.no_grad()
def apply_depthanything(
    model:DepthAnythingV2, 
    image:torch.Tensor
):
    """_summary_

    Args:
        model (DepthAnythingV2): _description_
        image (torch.Tensor): Has shape (H, W, 3) and RGB format. Values are between 0 and 1.

    Returns:
        _type_: _description_
    """
    input_image = (image.flip(dims=(-1,)) * 255).int().cpu().numpy().astype(np.uint8)
    disp = model.infer_image(input_image)
    
    disp = torch.tensor(disp, device=image.device)
    return disp
