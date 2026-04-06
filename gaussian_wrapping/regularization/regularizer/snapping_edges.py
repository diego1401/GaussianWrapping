from typing import Tuple, Optional
from sklearn.neighbors import KDTree
import torch
from regularization.sdf.learnable import convert_occupancy_to_sdf, convert_sdf_to_occupancy
from scene.gaussian_model import GaussianModel
from scene.mesh import (
    Meshes, 
    return_delaunay_tets, 
    remove_duplicate_edges,
    remove_degenerate_edges,
    get_error_quadrics, 
    vstars_from_quadrics,
    vstars_from_quadrics_least_squares,
    quadrics_score, 
)
from utils.tetmesh import marching_tetrahedra
from utils.general_utils import torch_quantile
from utils.geometry_utils import flatten_voronoi_features
from tqdm import tqdm


### --- Utility functions ---


def write_snapping_mask(path: str, snapping_mask: torch.Tensor):
    """
    Saves the snapping mask tensor to a file at the specified path.

    Args:
        path (str): The target file path where the snapping mask will be saved.
        snapping_mask (torch.Tensor): The mask tensor indicating which points are retained after snapping.
    """
    torch.save(snapping_mask, path)


def read_snapping_mask(path: str):
    """
    Loads the snapping mask tensor from the specified file path.

    Args:
        path (str): The file path from which to load the mask.

    Returns:
        torch.BoolTensor: The loaded snapping mask converted to boolean type.
    """
    return torch.load(path)


def unflatten_distance_indices(index, n_neighbours):
    i = index // n_neighbours
    j = index % n_neighbours
    return i, j


def snapping_clustering(
    voronoi_points, 
    valid_edges_mask, 
    edges_u, 
    edges_v, 
    max_iters:int=100_000, 
    verbose: bool=False
) -> torch.Tensor:
    """
    This function will cluster the voronoi points based on the edges to collapse.
    TODO: Check if duplicate edges need to be removed or not.

    Args:
        voronoi_points (torch.Tensor): The voronoi points. Shape (N, 3)
        valid_edges_mask (torch.BoolTensor): Mask marking the edges to collapse. Shape (E,). If None, all edges are considered.
        edges_u (torch.Tensor): First vertex index of the edges. Shape (E,)
        edges_v (torch.Tensor): Second vertex index of the edges. Shape (E,)
        max_iters (int, optional): Maximum number of iterations. Defaults to 100_000.
        verbose (bool, optional): If True, prints progress. Defaults to False.

    Raises:
        ValueError: If the snapping clustering did not converge in the given number of iterations.

    Returns:
        torch.Tensor: The cluster indices. Shape (N,) with N_clusters unique values.
    """
    # Initialize cluster indices: each node is its own cluster
    cluster_idx = torch.arange(
        voronoi_points.shape[0], 
        device=voronoi_points.device
    )  # (N,)
    
    if valid_edges_mask is not None:
        if (valid_edges_mask.shape[0] == 0) or (valid_edges_mask.sum() == 0):
            # Nothing to do
            return cluster_idx

        # filter edges
        u = edges_u[valid_edges_mask]  # (E_valid,)
        v = edges_v[valid_edges_mask]  # (E_valid,)
    else:
        u = edges_u  # (E,)
        v = edges_v  # (E,)
        
    if u.shape[0] == 0:
        return cluster_idx

    for _ in tqdm(range(max_iters), desc="Snapping clustering", disable=not verbose):
        
        prev = cluster_idx.clone()  # (N,)
        lu = cluster_idx[u]  # (E_valid,)
        lv = cluster_idx[v]  # (E_valid,)
        
        # Minimum is representative of the cluster
        m = torch.minimum(lu, lv)  # (E_valid,)

        # Assign the min to both points
        cluster_idx[u] = m  # (E_valid,)
        cluster_idx[v] = m  # (E_valid,)

        # Check if we have converged
        if torch.equal(cluster_idx, prev):
            break
    else:
        raise ValueError(f"[WARNING] Snapping clustering did not converge in {max_iters} iterations")

    return cluster_idx  # (N,)


def update_snapping_clustering(
    snapping_tensor_0: torch.Tensor,
    snapping_tensor_1: torch.Tensor,
) -> torch.Tensor:
    """
    Used for iterative snapping clustering.
    
    Given:
    - an initial point cloud 0 with N_0 points that has been previously snapped to a point cloud 1 with N_1 clusters,
    - a point cloud 2 obtained by snapping the point cloud 1 into N_2 clusters,
    this function returns the snapping tensor mapping the points of the initial point cloud 0 to the N_2 final clusters.
    In other words, each point in the initial point cloud 0 is mapped to one of the N_2 final clusters based on the point cloud 2.

    Args:
        snapping_tensor_0 (torch.Tensor): The snapping tensor of the initial point cloud. Shape (N_0,) with N_1 unique values.
        snapping_tensor_1 (torch.Tensor): The snapping tensor of the point cloud 1. Shape (N_1,) with N_2 unique values.

    Returns:
        torch.Tensor: The snapping tensor mapping the points of the initial point cloud 0 to the N_2 final clusters. 
            Shape (N_0,) with N_2 unique values.
    """
    snapped_idx, snapped_idx_inv = torch.unique(snapping_tensor_0, return_inverse=True, sorted=True)
    
    # We want to map each point in PC0 to the PC0 index of the cluster representative from PC2.
    # To this end, we apply the three following mappings:
    #   - snapped_idx_inv   : Maps each point in PC0 to the PC1 index of its snapped point
    #   - snapping_tensor_1 : Maps each PC1 index to the PC1 index of the final snapped point (based on PC2)
    #   - snapped_idx       : Maps each PC1 index to the PC0 index of its snapped point
    
    return snapped_idx[snapping_tensor_1[snapped_idx_inv]]


# TODO: Merge get_mean_snapped_points into get_mean_snapped_points_weighted functionby:
#   - setting weights to 1 if weights are not provided
#   - setting weights_are_normalized to False if weights are not provided
def get_mean_snapped_points(
    points: torch.Tensor, 
    cluster_idx: torch.Tensor,
    return_sum: bool=False
) -> torch.Tensor:
    """
    Computes the mean position of points grouped by cluster indices.
    For each unique cluster index in `cluster_idx`, all points in `points` that belong to that cluster are averaged.
    Returns a tensor of mean points corresponding to valid clusters.

    Args:
        points (torch.Tensor): Tensor of shape (N, D) containing N points of D dimensions.
        cluster_idx (torch.Tensor): 1D integer tensor of length N indicating the cluster assignment for each point.

    Returns:
        torch.Tensor: Tensor of mean points for each valid cluster. The shape will be (num_valid_clusters, D) if D>1, or (num_valid_clusters,) if D==1. Only clusters with at least one point are included.
        If return_sum is True, returns a tuple of (cluster_points, count).
    """
    cluster_points = torch.zeros_like(points)  # (N, D)
    cluster_points.index_add_(0, cluster_idx, points)  # (N, D)
    count = torch.zeros(points.shape[0], device=points.device)  # (N,)
    count.index_add_(0, cluster_idx, torch.ones(points.shape[0], device=points.device))  # (N,)
    
    # If return_sum is True, return the sum of the points and the count
    if return_sum:
        return cluster_points, count  # (N_clusters, 3) and (N_clusters,)
    
    # Only keep clusters which have at least one point.
    valid_mask = count > 0  # (N,)
    if len(cluster_points.shape) > len(count.shape):
        return cluster_points[valid_mask] / count[valid_mask].unsqueeze(1)  # (N_clusters, D)
    else:
        return cluster_points[valid_mask] / count[valid_mask]  # (N_clusters, D)

def get_logical_or_snapped_points(
    mask: torch.Tensor, 
    cluster_idx: torch.Tensor,
    return_sum: bool=False
) -> torch.Tensor:
    """
    Computes the logical OR of points grouped by cluster indices.
    For each unique cluster index in `cluster_idx`, all points in `points` that belong to that cluster are logical ORed.
    Returns a tensor of logical OR points corresponding to valid clusters.

    Args:
        mask (torch.Tensor): Tensor of shape (N,) containing N boolean values.
        cluster_idx (torch.Tensor): 1D integer tensor of length N indicating the cluster assignment for each point.

    Returns:
        torch.Tensor: Tensor of mean points for each valid cluster. The shape will be (num_valid_clusters, D) if D>1, or (num_valid_clusters,) if D==1. Only clusters with at least one point are included.
        If return_sum is True, returns a tuple of (cluster_points, count).
    """
    assert mask.dtype == torch.bool, "Mask must be a boolean tensor"
    cluster_mask = torch.zeros_like(mask)  # (N,)
    cluster_mask.index_add_(0, cluster_idx, mask)  # (N,), add works like logical OR for bool types

    count = torch.zeros(mask.shape[0], device=mask.device)  # (N,)
    count.index_add_(0, cluster_idx, torch.ones(mask.shape[0], device=mask.device))  # (N,)
    valid_mask = count > 0  # (N,)

    return cluster_mask[valid_mask]


def get_mean_snapped_points_weighted(
    points: torch.Tensor,
    cluster_idx: torch.Tensor,
    weights: torch.Tensor,
    weights_are_normalized: bool=False
) -> torch.Tensor:
    """
    Computes the weighted mean position of points grouped by cluster indices.
    For each unique cluster index in `cluster_idx`, all points in `points` that belong to that cluster are averaged.
    Returns a tensor of mean points corresponding to valid clusters.
    
    Args:
        points (torch.Tensor): Points/features to be filtered. Shape (N, D)
        cluster_idx (torch.Tensor): Cluster assignment for each point. Shape (N,)
        weights (torch.Tensor): Weights for each point. Shape (N,) or (N, 1)
        weights_are_normalized (bool, optional): If True, the weights are already normalized. Defaults to False.

    Returns:
        torch.Tensor: The snapped points. Shape (N_clusters, D)
    """
    assert weights.shape[0] == points.shape[0], "Weights must have the same number of points as the points tensor"
    if len(weights.shape) != len(points.shape):
        weights = weights.unsqueeze(1)
    cluster_points = torch.zeros_like(points)  # (N, D)
    cluster_points.index_add_(0, cluster_idx, points * weights)  # (N, D)
    
    # Compute the normalizing factor
    normalizing_factor = torch.zeros_like(weights)  # (N, 1)
    normalizing_factor.index_add_(0, cluster_idx, weights)  # (N, 1)

    # Mask to filter empty clusters
    valid_mask = (normalizing_factor > 0).squeeze()  # (N,)

    # If weights are normalized, return the cluster points as is
    if weights_are_normalized:
        return cluster_points[valid_mask]  # (N_clusters, D)
    
    # Else, normalize the cluster points
    if len(cluster_points.shape) > len(normalizing_factor.shape):
        return cluster_points[valid_mask] / (normalizing_factor[valid_mask].unsqueeze(1) + 1e-10)  # (N_clusters, D)
    else:
        return cluster_points[valid_mask] / (normalizing_factor[valid_mask] + 1e-10)  # (N_clusters, D)


def get_softmax_snapped_points(
    points: torch.Tensor,
    cluster_idx: torch.Tensor,
    exp_vals: torch.Tensor,
    min_exp_val: float=1e-6
) -> torch.Tensor:
    """
    Computes the maximum position of points grouped by cluster indices.
    For each unique cluster index in `cluster_idx`, the maximum point in `points` that belongs to that cluster is returned.
    Returns a tensor of maximum points corresponding to valid clusters.
    
    Args:
        points (torch.Tensor): Points/features to be filtered. Shape (N, 3)
        cluster_idx (torch.Tensor): Cluster assignment for each point. Shape (N,) or (N, 1)
        exp_vals (torch.Tensor): Exponential values for each point. Shape (N,)
        min_exp_val (float, optional): Minimum value for the exponential values. Defaults to 1e-6.

    Returns:
        torch.Tensor: The snapped points. Shape (N_clusters, 3)
    """
    exp_vals_clamped = exp_vals.clamp(min=min_exp_val)  # (N,) or (N, 1)
    sum_per_group = torch.zeros_like(exp_vals_clamped)  # (N,) or (N, 1)
    sum_per_group = sum_per_group.index_add_(0, cluster_idx, exp_vals_clamped)  # (N,) or (N, 1)
    weights = exp_vals_clamped / sum_per_group[cluster_idx]  # (N,) or (N, 1)
    assert not torch.isinf(weights).any(), "Weights have inf values"  # (N,) or (N, 1)

    return get_mean_snapped_points_weighted(points, cluster_idx, weights, weights_are_normalized=False)  # (N_clusters, D)


def get_keep_one_snapped_points(
    points: torch.Tensor,
    snapping_mask: torch.BoolTensor
) -> torch.Tensor:
    """
    Selects and returns the points that correspond to the True entries in the provided snapping mask.

    Args:
        points (torch.Tensor): Tensor of shape (N, ...) representing points/features to be filtered.
        snapping_mask (torch.BoolTensor): Boolean mask of shape (N,) indicating which points to keep.

    Returns:
        torch.Tensor: A filtered tensor containing only the points where snapping_mask is True.
    """
    return points[snapping_mask]


# TODO: What should we do with the large edges?
# TODO: Add the option to change the SDF interpolation method in this function
# TODO: Add the option to include edge filtering/collapsing?
def get_mesh_and_points_and_sdf(
    tet_vertices: torch.Tensor, 
    tets: torch.Tensor, 
    sdf: torch.Tensor, 
    scales: torch.Tensor
) -> Tuple[Meshes, torch.Tensor, torch.Tensor]:
    
    print(f"[ERROR]: This function does not include the SDF interpolation method!")
    
    # Differentiable Marching Tetrahedra
    verts_list, scale_list, faces_list, interp_v = marching_tetrahedra(
        vertices=tet_vertices,
        tets=tets,
        sdf=sdf,
        scales=scales
    )
    end_points, end_sdf = verts_list[0]  # (N_verts, 2, 3) and (N_verts, 2, 1)
    end_scales = scale_list[0]  # (N_verts, 2, 1)

    norm_sdf = end_sdf.abs() / end_sdf.abs().sum(dim=1, keepdim=True)
    verts = end_points[:, 0, :] * norm_sdf[:, 1, :] + end_points[:, 1, :] * norm_sdf[:, 0, :]
    faces = faces_list[0]  # (N_faces, 3)

    # Filtering out large edges as in GOF
    # dmtet_distance = torch.norm(end_points[:, 0, :] - end_points[:, 1, :], dim=-1)
    # dmtet_scale = end_scales[:, 0, 0] + end_scales[:, 1, 0]
    # dmtet_vertex_mask = (dmtet_distance <= dmtet_scale)
    
    # dmtet_face_mask = dmtet_vertex_mask[faces].all(axis=1)
    # faces_mask = faces_mask & dmtet_face_mask

    # Build the Mesh object
    mesh = Meshes(verts=verts, 
                  faces=faces#[dmtet_face_mask]
    )
    return mesh, interp_v[0], end_sdf        


### --- Snapping mask functions ---


@torch.no_grad()
def get_snapping_tensor_keep_one(
    # gaussians: GaussianModel, 
    voronoi_points: torch.Tensor,
    voronoi_scale: torch.Tensor,
    voronoi_scale_min: torch.Tensor,
    K: int=10, 
    scale_factor: float=2.0, 
    verbose: bool=False,
):
    """
    Identifies and returns a mask indicating which Voronoi points (tetra points) should be retained,
    collapsing together spatially close points to reduce redundancy. This is based on the K-nearest
    neighbors of each point and a distance threshold defined by scale_factor.

    Args:
        voronoi_points (torch.Tensor): The voronoi points. Shape (N, 3)
        voronoi_scale (torch.Tensor): The voronoi scale. Shape (N, 1)  TODO: Check if shape is correct
        voronoi_scale_min (torch.Tensor): The voronoi scale min. Shape (N, 1)  TODO: Check if shape is correct
        K (int, optional): Number of nearest neighbors to consider for each candidate point. Default is 10.
        scale_factor (float, optional): Multiplier for the minimum scale to set the collapsing threshold. Default is 2.0.
        verbose (bool, optional): If True, prints progress and collapse statistics. Default is False.

    Returns:
        torch.BoolTensor: A boolean mask of length N, where True indicates the Voronoi point is retained (i.e., not collapsed).
    """
    # Create a knn tree for the voronoi points
    # TODO: Replace by get_knn_index in blender_utils.py
    knn_tree = KDTree(voronoi_points.detach().cpu().numpy())
    # Get the k-nearest neighbors for each voronoi point
    knn_indices = knn_tree.query(voronoi_points.detach().cpu().numpy(), k=K, return_distance=False)[:,1:]
    # Get the mean of the k-nearest neighbors
    voronoi_points_mean = torch.mean(voronoi_points[knn_indices], dim=1)

    point_to_neighbour_distances = torch.norm(voronoi_points[knn_indices] - voronoi_points.unsqueeze(1), dim=2)
    # mean_point_to_neighbour_distance = torch.mean(point_to_neighbour_distances, dim=1)

    # 2. Identify all valid snapping points
    close_points = point_to_neighbour_distances < voronoi_scale_min * scale_factor

    # Which voronoi points will be collapsed?
    has_valid_points = close_points.any(dim=1)

    # Identify a set of unique points to collapse and the points to collapse them with
    points_idx_to_collapse = torch.nonzero(has_valid_points).view(-1)
    points_idx_to_collapse_with = torch.from_numpy(knn_indices[points_idx_to_collapse.cpu()][close_points[points_idx_to_collapse].cpu()]).cuda()

    repeated_mask = torch.isin(points_idx_to_collapse,points_idx_to_collapse_with)

    ## Index of voronoi points to collapse
    unique_points_idx_to_collapse = points_idx_to_collapse[~repeated_mask]
    ## Number of points that we will collapse them with
    unique_points_idx_to_collapse_offset = close_points[unique_points_idx_to_collapse].sum(dim=1).cumsum(dim=0)
    ## Flatten list of points to collapse with
    unique_points_idx_to_collapse_with = torch.from_numpy(knn_indices[unique_points_idx_to_collapse.cpu()][close_points[unique_points_idx_to_collapse].cpu()]).view(-1).cuda()

    # 3. Collapse the voronoi points
    collapse_mask = torch.zeros(voronoi_points.shape[0], device=voronoi_points.device).bool()
    for idx in tqdm(points_idx_to_collapse, desc="Collapsing voronoi points", disable=not verbose):
        if collapse_mask[idx]:
            continue
        collapse_with = knn_indices[idx.cpu()][close_points[idx].cpu()]
        for collapse_with_idx in collapse_with:
            collapse_mask[collapse_with_idx] = True

    if verbose:
        non_collapsed_percentage = (1 - voronoi_points[collapse_mask].shape[0] / voronoi_points.shape[0]) * 100
        print(f"[INFO] Percentage of voronoi points that will be kept: {non_collapsed_percentage:.2f}%")
        print(f"[INFO] We now have {voronoi_points[~collapse_mask].shape[0] / 1e6:.2f} million voronoi points left")

    return ~collapse_mask


@torch.no_grad()
def get_snapping_tensor_mean(
    # gaussians: GaussianModel, 
    voronoi_points: torch.Tensor,
    voronoi_scale: torch.Tensor,
    voronoi_scale_min: torch.Tensor,
    K: int=30, 
    scale_factor: float=2.0, 
    verbose: bool=False
):
    """
    Identifies and returns a mask indicating which Voronoi points (tetra points) should be retained,
    collapsing together spatially close points to reduce redundancy. This is based on the K-nearest
    neighbors of each point and a distance threshold defined by scale_factor.

    Args:
        voronoi_points (torch.Tensor): The voronoi points. Shape (N, 3)
        voronoi_scale (torch.Tensor): The voronoi scale. Shape (N, 1)  TODO: Check if shape is correct
        voronoi_scale_min (torch.Tensor): The voronoi scale min. Shape (N, 1)  TODO: Check if shape is correct
        K (int, optional): Number of nearest neighbors to consider for each candidate point. Default is 10.
        scale_factor (float, optional): Multiplier for the minimum scale to set the collapsing threshold. Default is 2.0.
        verbose (bool, optional): If True, prints progress and collapse statistics. Default is False.

    Returns:
        torch.IntTensor: A tensor of indices indicating to which cluster each point belongs.
    """
    # Create a knn tree for the voronoi points
    knn_tree = KDTree(voronoi_points.detach().cpu().numpy())
    # Get the k-nearest neighbors for each voronoi point
    knn_indices = knn_tree.query(voronoi_points.detach().cpu().numpy(), k=K+1, return_distance=False)[:,1:] # (N_pivots, K)
    knn_indices = torch.from_numpy(knn_indices).cuda()
    point_to_neighbour_distances = torch.norm(
            voronoi_points[knn_indices]  # (N_pivots, K, 3)
            - voronoi_points.unsqueeze(1),  # (N_pivots, 1, 3)
            dim=2
    )  # (N_pivots, K)

    # Identify all valid snapping edges
    close_points = (
        point_to_neighbour_distances  # (N_pivots, K)
        < (
            voronoi_scale_min  # (N_pivots, 1)
            * scale_factor
        )
    ) # (N_pivots, K)
    
    edges_mask = close_points.view(-1)  # (N_pivots * K,)
    edges_u = torch.arange(voronoi_points.shape[0], device=voronoi_points.device).repeat_interleave(K)  # (N_pivots * K,)
    edges_v = knn_indices.view(-1)  # (N_pivots * K,)
    
    # Remove non-collapsed edges as well as duplicate edges
    edges = torch.stack([edges_u, edges_v], dim=1)  # (N_pivots * K, 2)
    edges = remove_duplicate_edges(edges[edges_mask])  # (E_unique, 2)
    edges_u, edges_v = edges[:, 0], edges[:, 1]  # (E_unique,) and (E_unique,)
    
    return snapping_clustering(
        voronoi_points=voronoi_points, 
        valid_edges_mask=None, 
        edges_u=edges_u, 
        edges_v=edges_v, 
        verbose=verbose
    )  # (N_pivots,)

    # Which voronoi points will be collapsed?
    has_valid_points = close_points.any(dim=1) # (N_pivots,)

    n_voronoi_points = point_to_neighbour_distances.shape[0]
    n_neighbours = point_to_neighbour_distances.shape[1]

    point_to_neighbour_distances[~close_points] = torch.inf
    n_distances_to_consider = close_points.view(-1).sum(dim=0).item()
    total = close_points.view(-1).shape[0]
    if verbose:
        print(f'total = {total/1e6:.2f}M')
        print(f'n_distances_to_consider = {n_distances_to_consider/1e6:.2f}M')
    all_distances_indices_sorted = torch.argsort(point_to_neighbour_distances.view(-1)) # (N_pivots * K,) # TODO: Check if long

    edges_u, neighbour_number_tensor = unflatten_distance_indices(all_distances_indices_sorted[:n_distances_to_consider], n_neighbours) # (N_pivots * K,) -> (N_pivots, K)
    edges_v = knn_indices[edges_u,neighbour_number_tensor]

    # edges_u, the points to collapse
    # edges_v, the points to collapse with
    
    raise ValueError("Edge collapsing is not implemented in this function yet")
    return snapping_clustering(voronoi_points, has_valid_points, edges_u, edges_v, verbose=verbose)


# TODO: Provide delaunay_tets as an input argument to avoid recomputing it?
# Maybe not, as we want to avoid self-intersecting tetrahedra for this computation. So recomputing the Delaunay makes sense.
@torch.no_grad()
def get_snapping_tensor_qem(
    # gaussians: GaussianModel,
    voronoi_points: torch.Tensor,
    voronoi_scale: torch.Tensor,
    voronoi_sdf: torch.Tensor,
    normalization_factor: float,
    threshold: Optional[float]=1e-3, 
    ratio: Optional[float]=0.25,
    average_w_face_area: bool=True,
    use_least_squares: bool=False,
    verbose: bool=False,
    use_faster_qem: bool=True,
):
    """
    This function takes as input the voronoi points and returns a snapping tensor based on the QEM of the edges.
    We start by listing the edges on the mesh that would be collapsed by the QEM algorithm.
    We then transform this list of mesh edges into a list of tetrahedra (pivots) edges to snap.
    To do so, we look at the tetrahedra the spawned the mesh vertices. Namely consider:
    - An edge to collapse (v1,v2)
    - We call t_i_p and t_i_n the pivots that created v_i
    - We then create the edges (t_1_p, t_2_p) and (t_1_n, t_2_n).
    - Both edges have as cost the cost of collapsing the edge (v1,v2) given by QEM.
    
    Args:
        voronoi_points (torch.Tensor): The voronoi points. Shape (N, 3)
        voronoi_scale (torch.Tensor): The voronoi scale. Shape (N, 1)  TODO: Check if shape is correct
        voronoi_sdf (torch.Tensor): The voronoi sdf. Shape (N, 1) or (N,) TODO: Check if shape is correct
        normalization_factor (float): The normalization factor for the QEM.
        threshold (float, optional): The threshold for the QEM score. Defaults to 1e-3.
        ratio (float, optional): The ratio of edges to collapse. Defaults to 0.75.
        average_w_face_area (bool, optional): If True, average the weights by the face area. Defaults to True.
        use_least_squares (bool, optional): If True, use least squares to compute the v_stars. Defaults to True.
        verbose (bool, optional): If True, prints progress and collapse statistics. Defaults to False.

    Returns:
        torch.IntTensor: A tensor of indices indicating to which cluster each point belongs. Shape (N,) with N_clusters unique values.
    """    
    assert (ratio is None) or (threshold is None), "ratio and threshold cannot be both provided"
    
    ## Get mesh
    print(f"[INFO] Computing Delaunay tetrahedra...")
    delaunay_tets = return_delaunay_tets(voronoi_points, method="tetranerf")
    print(f"[INFO] Computing mesh...")
    mesh, interp_v, end_sdf = get_mesh_and_points_and_sdf(
        tet_vertices=voronoi_points[None],
        tets=delaunay_tets,
        sdf=voronoi_sdf.reshape(1, -1),
        scales=voronoi_scale[None],
    )

    # v_i -> is the vertex given by interp_v[i, 0, :] and interp_v[i, 1, :]
    
    # Compute vertex quadrics
    Q = get_error_quadrics(mesh, average_w_face_area=average_w_face_area)  # (V, 4, 4)
    
    if use_faster_qem:
        scores_of_mesh_edges = (
            quadrics_score(Q[mesh.edges[:, 0]], mesh.verts[mesh.edges[:, 1]])
            + quadrics_score(Q[mesh.edges[:, 1]], mesh.verts[mesh.edges[:, 0]])
        )  # (E,)
    else:    
        # Compute edge quadrics
        Q = Q[mesh.edges[:, 0]] + Q[mesh.edges[:, 1]]  # (E, 4, 4)
        
        # Compute v_stars for all edges
        v1 = mesh.verts[mesh.edges[:, 0]]  # (E, 3)
        v2 = mesh.verts[mesh.edges[:, 1]]  # (E, 3)
        edge_midpoints = (v1 + v2) / 2.  # (E, 3)
        if use_least_squares:
            v_stars = vstars_from_quadrics_least_squares(Q)  # (E, 3)
        else:
            v_stars, _, _ = vstars_from_quadrics(Q, edge_midpoints)  # (E, 3)
        
        # Compute quadrics scores for all edges
        scores_of_mesh_edges = quadrics_score(Q, v_stars) # (E,)
    
    scores_of_mesh_edges = scores_of_mesh_edges / (normalization_factor ** 2)  # (E,)
    
    # FIXME: remove following print
    # ---------------------------------------------------
    print("-----Statistics of scores_of_mesh_edges-----")
    if threshold is not None:
        print(f"Threshold: {threshold:.8f}")
    else:
        print(f"Ratio: {ratio:.2f}")
    print(f"Shape: {scores_of_mesh_edges.shape}")
    print(f"Mean: {scores_of_mesh_edges.mean():.4f}")
    print(f"Std: {scores_of_mesh_edges.std():.4f}")
    print(f"Min: {scores_of_mesh_edges.min():.4f}")
    print(f"Max: {scores_of_mesh_edges.max():.4f}")
    n_quantiles = 10
    for i in range(n_quantiles):
        print(f"Quantile {i/n_quantiles:.2f}: {torch_quantile(scores_of_mesh_edges, q=i/n_quantiles):.8f}")
    print("--------------------------------------------")
    # ---------------------------------------------------

    # Sort edges by score
    sorted_scores, sorted_indices = torch.sort(scores_of_mesh_edges)  # (E,), (E,)
    edges_u_vertices = mesh.edges[sorted_indices, 0]  # Shape (E,) with values in [0, N_vertices-1]
    edges_v_vertices = mesh.edges[sorted_indices, 1]  # Shape (E,) with values in [0, N_vertices-1]

    # Create edges between voronoi points given by the mesh edges
    edges_u_positives = interp_v[edges_u_vertices, 0].unsqueeze(-1)  # (E, 1)
    edges_v_positives = interp_v[edges_v_vertices, 0].unsqueeze(-1)  # (E, 1)
    edges_u_negatives = interp_v[edges_u_vertices, 1].unsqueeze(-1)  # (E, 1)
    edges_v_negatives = interp_v[edges_v_vertices, 1].unsqueeze(-1)  # (E, 1)

    # We want the edges to keep the sorted order and be interleaved
    edges_u = torch.cat([edges_u_positives, edges_u_negatives], dim=1).view(-1)  # (E * 2,)
    edges_v = torch.cat([edges_v_positives, edges_v_negatives], dim=1).view(-1)  # (E * 2,)
    scores = torch.cat([sorted_scores.unsqueeze(-1), sorted_scores.unsqueeze(-1)], dim=1).view(-1)
    
    if threshold is not None:
        edge_mask = scores <= threshold
    else:
        edge_mask = scores <= torch_quantile(scores, q=1.-ratio).item()
    
    # FIXME: remove following print
    # ---------------------------------------------------
    print("------------Statistics of scores------------")
    if threshold is not None:
        print(f"Threshold: {threshold:.8f}")
    else:
        print(f"Ratio: {ratio:.2f}")
    print(f"Shape: {scores.shape}")
    print(f"edges below threshold: {edge_mask.sum().item()}")
    print(f"vertices involved: {torch.cat([edges_u, edges_v], dim=0).unique().shape[0]}")
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std: {scores.std():.4f}")
    print(f"Min: {scores.min():.4f}")
    print(f"Max: {scores.max():.4f}")
    n_quantiles = 10
    for i in range(n_quantiles):
        print(f"Quantile {i/n_quantiles:.2f}: {torch_quantile(scores, q=i/n_quantiles):.8f}")
    print("--------------------------------------------")
    # ---------------------------------------------------
    
    # Remove non-collapsed edges as well as duplicate edges
    edges = torch.stack([edges_u, edges_v], dim=1)  # (N_pivots * K, 2)
    edges = remove_duplicate_edges(edges[edge_mask])  # (E_unique, 2)
    edges_u, edges_v = edges[:, 0], edges[:, 1]  # (E_unique,) and (E_unique,)
    
    return snapping_clustering(
        voronoi_points=voronoi_points, 
        valid_edges_mask=None, 
        edges_u=edges_u, 
        edges_v=edges_v, 
        verbose=verbose,
    )
    
    # FIXME: scores_of_mesh_edges is not sorted, but edges_u and edges_v are?
    scores = torch.cat([scores_of_mesh_edges.unsqueeze(-1), scores_of_mesh_edges.unsqueeze(-1)], dim=1).view(-1)  

    ## Get score per voronoi point
    scores_per_voronoi_point = torch.zeros(voronoi_points.shape[0], device=voronoi_points.device)
    ### The score of each voronoi point is the sum of the scores of the edges that it is part of
    scores_per_voronoi_point.index_add_(0, edges_u, scores)
    scores_per_voronoi_point.index_add_(0, edges_v, scores)

    ## Get valid voronoi points
    has_valid_points = scores_per_voronoi_point < threshold
    
    # Return snapping tensor
    raise ValueError("Edge collapsing is not implemented in this function yet")
    return snapping_clustering(voronoi_points, has_valid_points, edges_u, edges_v, verbose=verbose), scores_per_voronoi_point


@torch.no_grad()
def get_snapping_tensor_along_mesh_edges(
    # gaussians: GaussianModel,
    voronoi_points: torch.Tensor,
    voronoi_scale: torch.Tensor,
    voronoi_scale_min: torch.Tensor,
    voronoi_sdf: torch.Tensor,
    scale_threshold: Optional[float]=None,
    ratio: Optional[float]=None,
    verbose: bool=False,
    tets: Optional[torch.Tensor]=None,
):
    """
    This function takes as input the voronoi points and returns a snapping tensor based on the proximity of the voronoi points along the mesh edges.
    We start by listing the edges on the mesh.
    We then lift this list of mesh edges to a list of edges between pivots to snap.
    To do so, we look at the tetrahedra spawning the mesh vertices. Namely consider:
    - An edge to collapse (v1,v2)
    - We call t_i_p and t_i_n the pivots that created v_i
    - We then create the edges (t_1_p, t_2_p) and (t_1_n, t_2_n).
    - Edges have as cost the distance between the two corresponding voronoi points.
    
    Args:
        voronoi_points (torch.Tensor): The voronoi points. Shape (N, 3)
        voronoi_scale (torch.Tensor): The voronoi scale. Shape (N, 1)  TODO: Check if shape is correct
        voronoi_sdf (torch.Tensor): The voronoi sdf. Shape (N, 1) or (N,) TODO: Check if shape is correct
        scale_factor (float): The scale factor for the distance.
        scale_threshold (float, optional): The threshold for the distance. Defaults to 1.1.
        ratio (float, optional): The ratio of edges to collapse. Defaults to 0.25.
        verbose (bool, optional): If True, prints progress and collapse statistics. Defaults to False.

    Returns:
        torch.IntTensor: A tensor of indices indicating to which cluster each point belongs. Shape (N,) with N_clusters unique values.
    """
    assert (ratio is not None) or (scale_threshold is not None), "either ratio or scale_threshold must be provided"
    assert (ratio is None) or (scale_threshold is None), "ratio and scale_threshold cannot be both provided"
    
    # Compute Delaunay tetrahedra
    if tets is None:
        print(f"[INFO] Computing Delaunay tetrahedra...")
        delaunay_tets = return_delaunay_tets(voronoi_points, method="tetranerf")
    else:
        print(f"[INFO] Using provided Delaunay tetrahedra for snapping...")
        delaunay_tets = tets
    
    # Compute mesh
    # v_i -> is the vertex given by interp_v[i, 0, :] and interp_v[i, 1, :]
    print(f"[INFO] Computing mesh...")
    mesh, interp_v, end_sdf = get_mesh_and_points_and_sdf(
        tet_vertices=voronoi_points[None],
        tets=delaunay_tets,
        sdf=voronoi_sdf.reshape(1, -1),
        scales=voronoi_scale[None],
    )
    
    # Get mesh edges
    edges_u_vertices = mesh.edges[:, 0]  # Shape (E,) with values in [0, N_vertices-1]
    edges_v_vertices = mesh.edges[:, 1]  # Shape (E,) with values in [0, N_vertices-1]

    # Lift mesh edges to edges between voronoi points
    edges_u_positives = interp_v[edges_u_vertices, 0].unsqueeze(-1)  # (E, 1)
    edges_v_positives = interp_v[edges_v_vertices, 0].unsqueeze(-1)  # (E, 1)
    edges_u_negatives = interp_v[edges_u_vertices, 1].unsqueeze(-1)  # (E, 1)
    edges_v_negatives = interp_v[edges_v_vertices, 1].unsqueeze(-1)  # (E, 1)

    edges_u = torch.cat([edges_u_positives, edges_u_negatives], dim=1).view(-1)  # (E * 2,)
    edges_v = torch.cat([edges_v_positives, edges_v_negatives], dim=1).view(-1)  # (E * 2,)
    
    non_degenerate_edges_mask = edges_u != edges_v
    edges_u, edges_v = edges_u[non_degenerate_edges_mask], edges_v[non_degenerate_edges_mask]
    
    # Compute scores.
    scores = torch.norm(
        voronoi_points[edges_u] - voronoi_points[edges_v], 
        dim=-1
    )  # (E * 2,)
    scores_normalization_factor = (voronoi_scale_min[edges_u] + voronoi_scale_min[edges_v]).view(-1) / 2.  # (E * 2,)
    scores = scores / scores_normalization_factor.clamp(min=1e-8)  # (E * 2,)
    
    # TODO: remove threshold and replace by scale_factor
    if scale_threshold is not None:
        edge_mask = scores <= scale_threshold
    else:
        edge_mask = scores <= torch_quantile(scores, q=1.-ratio).item()
    
    # FIXME: remove following print
    # ---------------------------------------------------
    print("------------Statistics of scores------------")
    if scale_threshold is not None:
        print(f"Threshold: {scale_threshold:.8f}")
    else:
        print(f"Ratio: {ratio:.2f}")
    print(f"Shape: {scores.shape}")
    print(f"edges below threshold: {edge_mask.sum().item()}")
    print(f"vertices involved: {torch.cat([edges_u, edges_v], dim=0).unique().shape[0]}")
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std: {scores.std():.4f}")
    print(f"Min: {scores.min():.4f}")
    print(f"Max: {scores.max():.4f}")
    n_quantiles = 10
    for i in range(n_quantiles):
        print(f"Quantile {i/n_quantiles:.2f}: {torch_quantile(scores, q=i/n_quantiles):.8f}")
    print("--------------------------------------------")
    # ---------------------------------------------------
    
    # Remove non-collapsed edges as well as duplicate edges
    edges = torch.stack([edges_u, edges_v], dim=1)  # (N_pivots * K, 2)
    edges = edges[edge_mask]  # (E_to_collapse, 2)
    # edges = remove_degenerate_edges(edges)  # (E_filtered, 2)
    edges = remove_duplicate_edges(edges)  # (E_unique, 2)
    edges_u, edges_v = edges[:, 0], edges[:, 1]  # (E_unique,) and (E_unique,)
    
    return snapping_clustering(
        voronoi_points=voronoi_points, 
        valid_edges_mask=None, 
        edges_u=edges_u, 
        edges_v=edges_v, 
        verbose=verbose,
    )


@torch.no_grad()
def get_snapping_tensor(
    # gaussians: GaussianModel, 
    voronoi_points: torch.Tensor,
    voronoi_scale: torch.Tensor,
    voronoi_scale_min: torch.Tensor,
    voronoi_sdf: torch.Tensor,
    K: int=10, 
    scale_factor: float=2.0, 
    qem_threshold: float=1e-3, 
    qem_ratio: float=0.25,
    qem_normalization_factor: float=100.,
    qem_average_w_face_area: bool=True,
    verbose: bool=False, 
    method: str="keep_one",
    tets: Optional[torch.Tensor]=None,
    attract_distance_scale_threshold: float=1.1,
    attract_distance_ratio: float=None,
):
    """
    This function will return the snapping tensor based on the provided method.
    The snapping tensor is an integer tensor of shape (N,) with N_clusters unique values.
    It indicates to which cluster each point should be snapped.

    Args:
        voronoi_points (torch.Tensor): The voronoi points. Shape (N, 3)
        voronoi_scale (torch.Tensor): The voronoi scale. Shape (N, 1)
        voronoi_scale_min (torch.Tensor): The voronoi scale min. Shape (N, 1)
        voronoi_sdf (torch.Tensor): The voronoi sdf. Shape (N, 1) or (N,) TODO: Check if shape is correct
        K (int, optional): Number of nearest neighbors to consider for each candidate point. Defaults to 10.
        scale_factor (float, optional): Multiplier for the minimum scale to set the collapsing threshold. Defaults to 2.0.
        qem_threshold (float, optional): Threshold for the QEM score. Defaults to 1e-3.
        qem_ratio (float, optional): Ratio of edges to collapse. Defaults to 0.75.
        qem_normalization_factor (float, optional): Normalization factor for the QEM. Defaults to 100.
        qem_average_w_face_area (bool, optional): If True, average the weights by the face area. Defaults to True.
        verbose (bool, optional): If True, prints progress and collapse statistics. Defaults to False.
        method (str, optional): The method to use. Defaults to "keep_one". Options: "mean", "mean_weighted", "keep_one", "max", "softmax", "mesh_edges".

    Raises:
        ValueError: If the method is invalid.

    Returns:
        torch.IntTensor: A tensor of indices indicating to which cluster each point belongs. Shape (N,) with N_clusters unique values.
    """
    if method in ["mean", "softmax"]: # Softmax is applied to the sdf values
        # return get_snapping_tensor_mean(gaussians, K, scale_factor, verbose)
        return get_snapping_tensor_mean(
            voronoi_points=voronoi_points,
            voronoi_scale=voronoi_scale,
            voronoi_scale_min=voronoi_scale_min,
            K=K,
            scale_factor=scale_factor,
            verbose=verbose,
        )
    elif method == "qem":
        return get_snapping_tensor_qem(
            voronoi_points=voronoi_points,
            voronoi_scale=voronoi_scale,
            voronoi_sdf=voronoi_sdf,
            threshold=qem_threshold,
            ratio=qem_ratio,
            normalization_factor=qem_normalization_factor,
            average_w_face_area=qem_average_w_face_area,
            verbose=verbose,
        )
    elif method == "mesh_edges":
        return get_snapping_tensor_along_mesh_edges(
            voronoi_points=voronoi_points,
            voronoi_scale=voronoi_scale,
            voronoi_scale_min=voronoi_scale_min,
            voronoi_sdf=voronoi_sdf,
            scale_threshold=attract_distance_scale_threshold, 
            ratio=attract_distance_ratio,
            verbose=verbose,
            tets=tets,
        )
    elif method == "keep_one":
        # return get_snapping_tensor_keep_one(gaussians, K, scale_factor, verbose)
        return get_snapping_tensor_keep_one(
            voronoi_points=voronoi_points,
            voronoi_scale=voronoi_scale,
            voronoi_scale_min=voronoi_scale_min,
            K=K,
            scale_factor=scale_factor,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Invalid method: {method}")


def get_snapped_points(
    points: torch.Tensor, 
    snapping_tensor: torch.BoolTensor, 
    method: str="keep_one", 
    **kwargs
) -> torch.Tensor:
    """
    This function will return the snapped points based on the snapping tensor and the method.
    Args:
        points (torch.Tensor): The points to snap. Shape (N, D)
        snapping_tensor (torch.BoolTensor): The snapping tensor. Shape (N,) with N_clusters unique values.
        method (str, optional): The method to use. Defaults to "keep_one". Options: "mean", "mean_weighted", "keep_one", "max", "softmax".
        kwargs (dict, optional): Additional keyword arguments.
        - return_sum (bool, optional): If True, return the sum of the points instead of the mean. Defaults to False.
        - exp_vals (torch.Tensor, optional): The exponential values to use for the softmax. Shape (N,)
    Returns:
        torch.Tensor: The snapped points. Shape (N_clusters, D)
    """
    if method == "mean":
        return get_mean_snapped_points(points, snapping_tensor, return_sum=kwargs.get("return_sum", False))
    elif method == "mean_weighted":
        return get_mean_snapped_points_weighted(points, snapping_tensor, kwargs["weights"])
    elif method == "keep_one":
        return get_keep_one_snapped_points(points, snapping_tensor)
    elif method == "max":
        raise ValueError("Max method not implemented yet")
        return get_max_snapped_points(points, snapping_tensor)
    elif method == "logical_or":
        return get_logical_or_snapped_points(points, snapping_tensor)
    elif method in ["softmax", "qem", "mesh_edges"]:
        return get_softmax_snapped_points(points, snapping_tensor, kwargs["exp_vals"])
    else:
        raise ValueError(f"Invalid method: {method}")


def get_snapped_parameters(
    parameters_pckg: dict, 
    snapping_tensor: torch.Tensor, 
    method: str="keep_one", 
    **kwargs
) -> dict:
    """
    This function will return the snapped parameters based on the snapping tensor and the method.
    Args:
        parameters_pckg (dict): The parameters to snap. Contains the following keys:
            - voronoi_points (torch.Tensor): The voronoi points. Shape (N, 3)
            - voronoi_scales (torch.Tensor): The voronoi scales. Shape (N, 1)
            - voronoi_sdf (torch.Tensor): The voronoi sdf. Shape (N, 1) TODO: Check if shape is correct
        snapping_tensor (torch.Tensor): The snapping tensor. Shape (N,) with N_clusters unique values.
        method (str, optional): The method to use. Defaults to "keep_one". Options: "mean", "mean_weighted", "keep_one", "max", "softmax".
        kwargs (dict, optional): Additional keyword arguments.
        - exp_vals (torch.Tensor, optional): The exponential values to use for the softmax. Shape (N,)
    Returns:
        dict: The snapped parameters. Contains the following keys:
            - voronoi_points (torch.Tensor): The snapped voronoi points. Shape (N_clusters, 3)
            - voronoi_scales (torch.Tensor): The snapped voronoi scales. Shape (N_clusters, 1)
            - voronoi_sdf (torch.Tensor): The snapped voronoi sdf. Shape (N_clusters, 1) TODO: Check if shape is correct
    """
    # Unpack main parameters
    voronoi_points = parameters_pckg["voronoi_points"]
    voronoi_scales = parameters_pckg["voronoi_scales"]
    voronoi_sdf = parameters_pckg["voronoi_sdf"]
        
    if method in ["softmax", "qem", "mesh_edges"]:
        assert "k" in kwargs, "k must be provided for softmax method"
        # Get unnormalized snapped points and count
        voronoi_points_snapped_sum, count = get_snapped_points(
            points=voronoi_points, 
            snapping_tensor=snapping_tensor,
            method="mean", 
            return_sum=True
        )
        
        # Mask to filter empty clusters
        mask = count > 0

        # Normalize the snapped points
        voronoi_points_snapped_mean_non_masked = torch.clone(voronoi_points_snapped_sum)  # (N, 3)
        voronoi_points_snapped_mean_non_masked[mask] = voronoi_points_snapped_mean_non_masked[mask] / count[mask].unsqueeze(1)  # (N_clusters, 3)

        # Compute the distance from initial points to snapped points
        distance_to_snapped_points = torch.norm(
            voronoi_points  # (N, 3)
            - voronoi_points_snapped_mean_non_masked[snapping_tensor],  # (N, 3)
            dim=1,
        )  # (N,)
        exp_vals = torch.exp(-kwargs["k"] * distance_to_snapped_points).clamp(min=1e-6)  # (N,)
        voronoi_points_snapped = voronoi_points_snapped_sum[mask] / count[mask].unsqueeze(1)  # (N_clusters, 3)
        
        # Store the exponential values for computing scales and sdf values of snapped points
        kwargs["exp_vals"] = exp_vals

    elif method in ["mean", "keep_one"]:
        voronoi_points_snapped = get_snapped_points(
            voronoi_points, 
            snapping_tensor, 
            method=method
        )

    else:
        raise ValueError(f"Invalid method: {method}")

    voronoi_scales_snapped = get_snapped_points(voronoi_scales, snapping_tensor, method=method, **kwargs)
    voronoi_sdf_snapped = get_snapped_points(voronoi_sdf, snapping_tensor, method=method, **kwargs)
    
    # TODO: Should we just automatically iterate over keys in parameters_pckg rather than manually checking for each key?
    if parameters_pckg.get("voronoi_scale_min") is not None:
        voronoi_scale_min_snapped = get_snapped_points(parameters_pckg["voronoi_scale_min"], snapping_tensor, method=method, **kwargs)
    if parameters_pckg.get("voronoi_occupancy_logit") is not None:
        occupancy_logit_snapped = get_snapped_points(parameters_pckg["voronoi_occupancy_logit"], snapping_tensor, method=method, **kwargs)
    if parameters_pckg.get("voronoi_features") is not None:
        voronoi_features_snapped = get_snapped_points(parameters_pckg["voronoi_features"], snapping_tensor, method=method, **kwargs)

    out_dict = {
        "voronoi_points": voronoi_points_snapped,
        "voronoi_scales": voronoi_scales_snapped,
        "voronoi_sdf": voronoi_sdf_snapped
    }
    if parameters_pckg.get("voronoi_scale_min") is not None:
        out_dict["voronoi_scale_min"] = voronoi_scale_min_snapped
    if parameters_pckg.get("voronoi_occupancy_logit") is not None:
        out_dict["voronoi_occupancy_logit"] = occupancy_logit_snapped
    if parameters_pckg.get("voronoi_features") is not None:
        out_dict["voronoi_features"] = voronoi_features_snapped
    return out_dict
