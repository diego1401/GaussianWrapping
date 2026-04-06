from typing import Union
import torch
from tetranerf.utils.extension import cpp as delaunay_cpp
from functional.pivots import extract_gaussian_pivots


@torch.no_grad()
def compute_delaunay_triangulation(
    means:Union[torch.Tensor, None]=None,
    scales:Union[torch.Tensor, None]=None,
    rotations:Union[torch.Tensor, None]=None,
    gaussian_idx:Union[torch.Tensor, None]=None,
    scale_pivots_with_downsample_ratio:bool=True,
    scale_pivots_factor:float=None,
    override_pivots:Union[torch.Tensor, None]=None,
) -> torch.Tensor:
    """Compute Delaunay tetrahedralization for a set of Gaussian pivots.
    Override pivots can be provided; otherwise, pivots will be extracted from Gaussians in gaussian_idx.
    Either (means, scales, rotations) or override_pivots must be provided.

    Args:
        means (Union[torch.Tensor, None], optional): Means of the Gaussians. Shape: (N, 3). Defaults to None.
        scales (Union[torch.Tensor, None], optional): Scales of the Gaussians. Shape: (N, 3). Defaults to None.
        rotations (Union[torch.Tensor, None], optional): Rotations of the Gaussians as quaternions. Shape: (N, 4). Defaults to None.
        gaussian_idx (Union[torch.Tensor, None], optional): Indices of the Gaussians to be used for generating pivots. 
            Shape: (N_selected,). Defaults to None.
        scale_pivots_with_downsample_ratio (bool, optional): If True, the scale of the pivots will be adjusted to match the downsample ratio. Defaults to True.
        scale_pivots_factor (float, optional): If provided, the scale of the pivots will be multiplied by this factor. Defaults to None.
        override_pivots (Union[torch.Tensor, None], optional): Override pivots. Shape: (N_pivots, 3). Defaults to None.

    Returns:
        torch.Tensor: Indices of the tetrahedra. Shape: (N_tetrahedra, 4).
    """
    assert (
        (
            (means is not None) and (scales is not None) and (rotations is not None)
        ) or (
            override_pivots is not None
        )
    )
    
    # Extract pivots from Gaussians.
    # If override_pivots is provided, use it instead of extracting pivots from Gaussians.
    if override_pivots is None:
        pivots, _ = extract_gaussian_pivots(
            means=means,
            scales=scales,
            rotations=rotations,
            gaussian_idx=gaussian_idx,
            scale_pivots_with_downsample_ratio=scale_pivots_with_downsample_ratio,
            scale_pivots_factor=scale_pivots_factor
        )
    else:
        pivots = override_pivots

    print(f"[INFO] Computing Delaunay tetrahedralization for {pivots.shape[0]} points...")
    delaunay_tets = delaunay_cpp.triangulate(pivots.detach()).cuda().long()
    torch.cuda.empty_cache()
    return delaunay_tets
