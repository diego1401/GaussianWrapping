from typing import Union
import torch
from torch_geometric.nn import knn


def get_knn_index(
    points:torch.Tensor, 
    k:int, 
    points2:Union[torch.Tensor, None]=None, 
    include_self:bool=False,
) -> torch.Tensor:
    """
    Return the k nearest neighbor indices of points in points2.
    If points2 is None, return the k nearest neighbor indices of points in points.
    If include_self is True, return the k nearest neighbor indices of points in points including the point itself.
    If include_self is False, return the k nearest neighbor indices of points in points excluding the point itself.

    Args:
        points (torch.Tensor): (n_pts, d)
        points2 (torch.Tensor, optional): (n_pts2, d). Defaults to None.
        k (int): number of nearest neighbors
        include_self (bool, optional): include the point itself in the nearest neighbors. Defaults to False.
            If points2 is not None, include_self is ignored.

    Returns:
        torch.Tensor: (n_pts, k)
    """
    if points2 is None:
        _k = k if include_self else k + 1
        knn_index = knn(points, points, k=_k)[1].view(len(points), _k)
        if not include_self:
            knn_index = knn_index[:, 1:]
    else:
        knn_index = knn(points2, points, k=k)[1].view(len(points), k)

    return knn_index


def find_affine_transform(
    X:torch.Tensor, 
    Y:torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finds affine transform (L, T) such that Y = X @ L + T

    Args:
        X (torch.Tensor): Has shape (..., N, d)
        Y (torch.Tensor): Has shape (..., N, d)

    Returns:
        M (torch.Tensor): Has shape (..., d+1, d+1)
        L (torch.Tensor): Has shape (..., d, d)
        T (torch.Tensor): Has shape (..., d)
    """
    d = X.shape[-1]
    
    _X = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    _Y = torch.cat([Y, torch.ones_like(Y[..., :1])], dim=-1)

    M = torch.linalg.lstsq(_X, _Y).solution
    L = M[..., :d, :d]
    T = M[..., d, :d]
    
    return M, L, T


def transform_points(
    X:torch.Tensor,
    L:torch.Tensor=None,
    T:torch.Tensor=None,
    M:torch.Tensor=None
) -> torch.Tensor:
    """Transforms points X by the affine transform (L, T).
    A matrix M with shape (..., d+1, d+1) can also be provided, 
    in which case L and T are extracted from M.
    
    Args:
        X (torch.Tensor): Has shape (..., N, d)
        L (torch.Tensor): Has shape (..., d, d)
        T (torch.Tensor): Has shape (..., d)
        M (torch.Tensor): Has shape (..., d+1, d+1)

    Returns:
        Y (torch.Tensor): Has shape (..., N, d)
    """
    assert ((L is not None) and (T is not None)) or (M is not None)
    if M is not None:
        d = X.shape[-1]
        L = M[..., :d, :d]
        T = M[..., d, :d]

    return X @ L + T[..., None, :]


def orthogonalize_basis(
    basis:torch.Tensor,
    use_biggest_axis_as_first_axis_for_gram_schmidt=True,
) -> torch.Tensor:
    """Orthogonalizes a basis of 3-dimensional vectors

    Args:
        basis (torch.Tensor): Has shape (..., 3, 3)

    Returns:
        basis (torch.Tensor): Has shape (..., 3, 3)
    """
    # TODO: Is torch.gather too slow?
    if use_biggest_axis_as_first_axis_for_gram_schmidt:
        # Sort basis by norm of each axis
        sorted_idx = torch.argsort(
            basis.norm(dim=-1, keepdim=True),  # (..., 3, 1)
            dim=1, 
            descending=True
        ).repeat(1, 1, basis.shape[-1]) # (..., 3, 3)
        ortho_basis = basis.gather(dim=1, index=sorted_idx).contiguous()  # (..., 3, 3)
        
        # Orthogonalize second axis
        second_ortho_axis = ortho_basis[:, 1] - (
            (ortho_basis[:, 1] * ortho_basis[:, 0]).sum(dim=-1, keepdim=True) 
            * ortho_basis[:, 0] / (ortho_basis[:, 0] ** 2).sum(dim=-1, keepdim=True)
        )
        
        # Orthogonalize third axis
        third_ortho_axis = ortho_basis[:, 2] - (
            (ortho_basis[:, 2] * ortho_basis[:, 0]).sum(dim=-1, keepdim=True) 
            * ortho_basis[:, 0] / (ortho_basis[:, 0] ** 2).sum(dim=-1, keepdim=True)
        ) - (
            (ortho_basis[:, 2] * second_ortho_axis).sum(dim=-1, keepdim=True)
            * second_ortho_axis / (second_ortho_axis ** 2).sum(dim=-1, keepdim=True)
        )

        # Back to original order
        invert_sorted_idx = torch.argsort(sorted_idx, dim=1)  # (..., 3, 3)
        ortho_basis = torch.cat(
            [ortho_basis[:, 0:1], second_ortho_axis[:, None], third_ortho_axis[:, None]], 
            dim=1
        ).gather(dim=1, index=invert_sorted_idx)
        
    else:
        ortho_basis = basis.clone()
        ortho_basis[:, 1] = ortho_basis[:, 1] - (
            (ortho_basis[:, 1] * ortho_basis[:, 0]).sum(dim=-1, keepdim=True) 
            * ortho_basis[:, 0] / (ortho_basis[:, 0] ** 2).sum(dim=-1, keepdim=True)
        )
        
    return ortho_basis


# =============================================================================
# FUNCTIONS BELOW ARE FROM pytorch3d
# =============================================================================

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)
