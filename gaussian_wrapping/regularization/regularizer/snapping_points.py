from sklearn.neighbors import KDTree
import torch
from utils.geometry_utils import flatten_voronoi_features
from regularization.sdf.learnable import convert_occupancy_to_sdf, convert_sdf_to_occupancy
from scene.gaussian_model import GaussianModel
from utils.tetmesh import marching_tetrahedra
from scene.mesh import Meshes, get_error_quadrics, quadrics_score, return_delaunay_tets
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

# FIXME: Isn't there a problem here?
# The edge mask valid_edges_mask is computed from the vertex mask, by checking if both vertices are valid.
# This can produce wrong behavior.
# Let's consider 4 vertices for instance: v_1, v_2, v_3 and v_4.
# Each v_i is linked to v_i+1 by an edge.
# Let's consider that v_1 and v_2 are very close, same for v_3 and v_4.
# Then we want to collapse two edges: (v_1, v_2) and (v_3, v_4).
# This should result in two clusters, linked by a single edge.
# However, since all points are "valid" because they end up being collapsed, then v2 and v3 are valid,
# so the valid_edges_mask will be True for the edge (v_2, v_3), even though it should be False.
# This ends up in a single cluster, containing all vertices (v_1, v_2, v_3, v_4).
# Since an edge mask is used anyway, shouldn't we just provide valid_edges_mask as an input argument?
def snapping_clustering(voronoi_points, has_valid_points, edges_u, edges_v, max_iters:int=100_000, verbose: bool=False):
    # Initialize cluster indices: each node is its own cluster
    cluster_idx = torch.arange(voronoi_points.shape[0], device=voronoi_points.device)

    # filter edges so at least both points are valid
    valid_edges_mask = has_valid_points[edges_u] & has_valid_points[edges_v]
    if valid_edges_mask.sum() == 0:
        # Nothing to do
        return cluster_idx

    # filter edges
    u = edges_u[valid_edges_mask]
    v = edges_v[valid_edges_mask]

    for _ in tqdm(range(max_iters), desc="Snapping clustering", disable=not verbose):
        
        prev = cluster_idx.clone()
        lu = cluster_idx[u]
        lv = cluster_idx[v]
        
        # Minimum is representative of the cluster
        m = torch.minimum(lu, lv)

        # Assign the min to both points
        cluster_idx[u] = m
        cluster_idx[v] = m

        # Check if we have converged
        if torch.equal(cluster_idx, prev):
            break
    else:
        raise ValueError(f"[WARNING] Snapping clustering did not converge in {max_iters} iterations")

    return cluster_idx

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
        torch.Tensor: Tensor of mean points for each valid cluster. The shape will be (num_valid_clusters, D) if D>1,
                      or (num_valid_clusters,) if D==1. Only clusters with at least one point are included.
    """
    cluster_points = torch.zeros_like(points)
    cluster_points.index_add_(0, cluster_idx, points)
    count = torch.zeros(points.shape[0], device=points.device)
    count.index_add_(0, cluster_idx, torch.ones(points.shape[0], device=points.device))
    # Only keep clusters which have at least one point.
    if return_sum:
        return cluster_points, count
    valid_mask = count > 0
    if len(cluster_points.shape) > len(count.shape):
        return cluster_points[valid_mask] / count[valid_mask].unsqueeze(1)
    else:
        return cluster_points[valid_mask] / count[valid_mask]

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
    """
    assert weights.shape[0] == points.shape[0], "Weights must have the same number of points as the points tensor"
    if len(weights.shape) != len(points.shape):
        weights = weights.unsqueeze(1)
    cluster_points = torch.zeros_like(points)
    cluster_points.index_add_(0, cluster_idx, points * weights)
    
    # Else, need to normalize the weights
    normalizing_factor = torch.zeros_like(weights)
    normalizing_factor.index_add_(0, cluster_idx, weights)
    valid_mask = (normalizing_factor > 0).squeeze()
    if weights_are_normalized:
        return cluster_points[valid_mask]
    # Only keep clusters which have at least one point.
    if len(cluster_points.shape) > len(normalizing_factor.shape):
        return cluster_points[valid_mask] / (normalizing_factor[valid_mask].unsqueeze(1) + 1e-10)
    else:
        return cluster_points[valid_mask] / (normalizing_factor[valid_mask] + 1e-10)

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
    """
    exp_vals_clamped = exp_vals.clamp(min=min_exp_val)
    sum_per_group = torch.zeros_like(exp_vals_clamped)
    sum_per_group = sum_per_group.index_add_(0, cluster_idx, exp_vals_clamped)
    weights = exp_vals_clamped / sum_per_group[cluster_idx]
    assert not torch.isinf(weights).any(), "Weights have inf values"

    return get_mean_snapped_points_weighted(points, cluster_idx, weights, weights_are_normalized=False)

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

def get_mesh_and_points_and_sdf(tet_vertices: torch.Tensor, tets: torch.Tensor, sdf: torch.Tensor, scales: torch.Tensor):
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
def get_snapping_tensor_keep_one(gaussians: GaussianModel, K: int=10, scale_factor: float=2.0, verbose: bool=False):
    """
    Identifies and returns a mask indicating which Voronoi points (tetra points) should be retained,
    collapsing together spatially close points to reduce redundancy. This is based on the K-nearest
    neighbors of each point and a distance threshold defined by scale_factor.

    Args:
        gaussians (GaussianModel): The Gaussian model providing access to Voronoi/tetra points and scales.
        K (int, optional): Number of nearest neighbors to consider for each candidate point. Default is 10.
        scale_factor (float, optional): Multiplier for the minimum scale to set the collapsing threshold. Default is 2.0.
        verbose (bool, optional): If True, prints progress and collapse statistics. Default is False.

    Returns:
        torch.BoolTensor: A boolean mask of length N, where True indicates the Voronoi point is retained (i.e., not collapsed).
    """
    # 1. Get voronoi points and knn-tree
    voronoi_points, voronoi_scale, voronoi_scale_min = gaussians.get_tetra_points(return_min_scales=True)
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
def get_snapping_tensor_mean(gaussians: GaussianModel, K: int=30, scale_factor: float=2.0, verbose: bool=False):
    """
    Identifies and returns a mask indicating which Voronoi points (tetra points) should be retained,
    collapsing together spatially close points to reduce redundancy. This is based on the K-nearest
    neighbors of each point and a distance threshold defined by scale_factor.

    Args:
        gaussians (GaussianModel): The Gaussian model providing access to Voronoi/tetra points and scales.
        K (int, optional): Number of nearest neighbors to consider for each candidate point. Default is 10.
        scale_factor (float, optional): Multiplier for the minimum scale to set the collapsing threshold. Default is 2.0.
        verbose (bool, optional): If True, prints progress and collapse statistics. Default is False.

    Returns:
        torch.IntTensor: A tensor of indices indicating to which cluster each point belongs.
    """
    voronoi_points, voronoi_scale, voronoi_scale_min = gaussians.get_tetra_points(return_min_scales=True)
    # Create a knn tree for the voronoi points
    knn_tree = KDTree(voronoi_points.detach().cpu().numpy())
    # Get the k-nearest neighbors for each voronoi point
    knn_indices = knn_tree.query(voronoi_points.detach().cpu().numpy(), k=K, return_distance=False)[:,1:] # (N_pivots, K)
    # Get the mean of the k-nearest neighbors
    voronoi_points_mean = torch.mean(voronoi_points[knn_indices], dim=1) # (N_pivots,)

    knn_indices = torch.from_numpy(knn_indices).cuda()
    # voronoi_points.unsqueeze(1) -> (N_pivots, 1, 3)
    # voronoi_points[knn_indices] -> (N_pivots, K, 3)
    point_to_neighbour_distances = torch.norm(voronoi_points[knn_indices] - voronoi_points.unsqueeze(1), dim=2) # (N_pivots, K)

    # 2. Identify all valid snapping points
    # voronoi_scale_min (N_pivots, 1)
    close_points = point_to_neighbour_distances < voronoi_scale_min * scale_factor # (N_pivots, K)

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

    return snapping_clustering(voronoi_points, has_valid_points, edges_u, edges_v, verbose=verbose)

@torch.no_grad()
def get_snapping_tensor_qem(gaussians: GaussianModel, threshold: float=1e-3, verbose: bool=False):
    """
    This function takes as input the gaussians and returns a snapping tensor based on the QEM of the edges.
    We start by listing the edges on the mesh that would be collapsed by the QEM algorithm.
    We then transform this list of mesh edges into a list of tetrahedra (pivots) edges to snap.
    To do so, we look at the tetrahedra the spawned the mesh vertices. Namely consider:
    - An edge to collapse (v1,v2)
    - We call t_i_p and t_i_n the pivots that created v_i
    - We then create the edges (t_1_p, t_2_p) and (t_1_n, t_2_n).
    - Both edges have as cost the cost of collapsing the edge (v1,v2) given by QEM.

    """

    # Get mesh
    ## Getting voronoi points
    voronoi_points, voronoi_scales = gaussians.get_tetra_points()
    current_occupancy = gaussians.get_occupancy  # (N_gaussians, 9)
    current_voronoi_sdf = convert_occupancy_to_sdf(
            flatten_voronoi_features(current_occupancy)
        )
    ## Get mesh
    print(f"[INFO] Computing Delaunay tetrahedra...")
    delaunay_tets = return_delaunay_tets(voronoi_points, method="tetranerf")
    print(f"[INFO] Computing mesh...")
    mesh, interp_v, end_sdf = get_mesh_and_points_and_sdf(
                    tet_vertices=voronoi_points[None],
                    tets=delaunay_tets,
                    sdf=current_voronoi_sdf.reshape(1, -1),
                    scales=voronoi_scales[None])

    # v_i -> is the vertex given by interp_v[i, 0, :] and interp_v[i, 1, :]
    
    # # Compute the cost for all edges
    Q = get_error_quadrics(mesh, normalization_factor=100., average_w_face_area=True)
    scores_of_mesh_edges = quadrics_score(Q[mesh.edges[:, 0]], mesh.verts[mesh.edges[:, 1]])+ \
             quadrics_score(Q[mesh.edges[:, 1]], mesh.verts[mesh.edges[:, 0]]) # (E,)

    ## Sort edges by score
    sorted_indices = torch.argsort(scores_of_mesh_edges)
    edges_u_vertices = mesh.edges[sorted_indices, 0] # Size (E,) values in [0, N_vertices-1]
    edges_v_vertices = mesh.edges[sorted_indices, 1] # Size (E,) values in [0, N_vertices-1]

    ### Create edges between voronoi points given by the mesh edges
    edges_u_positives = interp_v[edges_u_vertices, 0].unsqueeze(-1) 
    edges_v_positives = interp_v[edges_v_vertices, 0].unsqueeze(-1) 
    edges_u_negatives = interp_v[edges_u_vertices, 1].unsqueeze(-1) 
    edges_v_negatives = interp_v[edges_v_vertices, 1].unsqueeze(-1) 

    ## We want the edges to keep the sorted order and be interleaved
    edges_u = torch.cat([edges_u_positives, edges_u_negatives], dim=1).view(-1)
    edges_v = torch.cat([edges_v_positives, edges_v_negatives], dim=1).view(-1)
    scores = torch.cat([scores_of_mesh_edges.unsqueeze(-1), scores_of_mesh_edges.unsqueeze(-1)], dim=1).view(-1)

    ## Get score per voronoi point
    scores_per_voronoi_point = torch.zeros(voronoi_points.shape[0], device=voronoi_points.device)
    ### The score of each voronoi point is the sum of the scores of the edges that it is part of
    scores_per_voronoi_point.index_add_(0, edges_u, scores)
    scores_per_voronoi_point.index_add_(0, edges_v, scores)

    ## Get valid voronoi points
    has_valid_points = scores_per_voronoi_point < threshold
    
    # Return snapping tensor
    return snapping_clustering(voronoi_points, has_valid_points, edges_u, edges_v, verbose=verbose), scores_per_voronoi_point


@torch.no_grad()
def get_snapping_tensor(gaussians: GaussianModel, K: int=10, scale_factor: float=2.0, qem_threshold: float=1e-3, verbose: bool=False, method: str="keep_one"):
    if method in ["mean", "softmax"]: # Softmax is applied to the sdf values
        return get_snapping_tensor_mean(gaussians, K, scale_factor, verbose)
    elif method == "qem":
        return get_snapping_tensor_qem(gaussians, qem_threshold, verbose)
    elif method == "keep_one":
        return get_snapping_tensor_keep_one(gaussians, K, scale_factor, verbose)
    else:
        raise ValueError(f"Invalid method: {method}")

def get_snapped_points(points: torch.Tensor, snapping_tensor: torch.BoolTensor, method: str="keep_one", **kwargs) -> torch.Tensor:
    if method == "mean":
        return get_mean_snapped_points(points, snapping_tensor, return_sum=kwargs.get("return_sum", False))
    elif method == "mean_weighted":
        return get_mean_snapped_points_weighted(points, snapping_tensor, kwargs["weights"])
    elif method == "keep_one":
        return get_keep_one_snapped_points(points, snapping_tensor)
    elif method == "max":
        raise ValueError("Max method not implemented yet")
        return get_max_snapped_points(points, snapping_tensor)
    elif method == "softmax":
        return get_softmax_snapped_points(points, snapping_tensor, kwargs["exp_vals"])
    else:
        raise ValueError(f"Invalid method: {method}")


def get_snapped_parameters(parameters_pckg: dict, snapping_tensor: torch.BoolTensor, method: str="keep_one", **kwargs) -> dict:
    """
    This method will return the voronoi points, scales and sdf values.
    """
    # Unpack main parameters
    voronoi_points = parameters_pckg["voronoi_points"]
    voronoi_scales = parameters_pckg["voronoi_scales"]
    voronoi_sdf = parameters_pckg["voronoi_sdf"]
    if method == "softmax":
        assert "k" in kwargs, "k must be provided for softmax method"
        # Get voronoi points and some parameters
        voronoi_points_snapped_sum, count = get_snapped_points(voronoi_points, snapping_tensor,method="mean", return_sum=True)
        mask = count > 0
        voronoi_points_snapped_mean_non_masked = torch.clone(voronoi_points_snapped_sum)
        voronoi_points_snapped_mean_non_masked[mask] = voronoi_points_snapped_mean_non_masked[mask] / count[mask].unsqueeze(1)
        distance_to_snapped_points = torch.norm(voronoi_points - voronoi_points_snapped_mean_non_masked[snapping_tensor], dim=1)
        exp_vals = torch.exp(-kwargs["k"] * distance_to_snapped_points).clamp(min=1e-6)
        voronoi_points_snapped = voronoi_points_snapped_sum[mask] / count[mask].unsqueeze(1)
        kwargs["exp_vals"] = exp_vals
    elif method in ["mean", "keep_one"]:
        voronoi_points_snapped = get_snapped_points(voronoi_points, snapping_tensor, method=method)
    else:
        raise ValueError(f"Invalid method: {method}")

    voronoi_scales_snapped = get_snapped_points(voronoi_scales, snapping_tensor, method=method, **kwargs)
    voronoi_sdf_snapped = get_snapped_points(voronoi_sdf, snapping_tensor, method=method, **kwargs)

    return {
        "voronoi_points": voronoi_points_snapped,
        "voronoi_scales": voronoi_scales_snapped,
        "voronoi_sdf": voronoi_sdf_snapped
    }