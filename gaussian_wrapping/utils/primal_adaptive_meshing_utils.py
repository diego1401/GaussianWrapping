import os
from typing import Dict, Tuple
import torch
from scipy.spatial import KDTree
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from torch_geometric.nn import knn
from utils.geometry_utils import is_in_view_frustum
from utils.general_utils import (
    build_scaling_rotation, 
    robust_sigma_inv, 
    robust_gaussian_eval_shifted_points
)


'''
MeshFromDelaunay class
'''

def NDCnormalize(vertices, return_scale=False):
    """normalization in half unit ball"""
    vM = vertices.max(0)
    vm = vertices.min(0)
    scale = np.sqrt(((vM - vm) ** 2).sum(-1))
    mean = (vM + vm) / 2.0
    nverts = (vertices - mean) / scale
    if return_scale:
        return nverts, mean, scale
    return nverts

SIGNS = np.array(
    [
        [(-1) ** i, (-1) ** j, (-1) ** k]
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
)  # +1 or -1 for all coordinates

def tet_circumcenter(verts):
    # ba = b - a
    ba_x = verts[:, 1, 0] - verts[:, 0, 0]
    ba_y = verts[:, 1, 1] - verts[:, 0, 1]
    ba_z = verts[:, 1, 2] - verts[:, 0, 2]
    # ca = c - a
    ca_x = verts[:, 2, 0] - verts[:, 0, 0]
    ca_y = verts[:, 2, 1] - verts[:, 0, 1]
    ca_z = verts[:, 2, 2] - verts[:, 0, 2]
    # da = d - a
    da_x = verts[:, 3, 0] - verts[:, 0, 0]
    da_y = verts[:, 3, 1] - verts[:, 0, 1]
    da_z = verts[:, 3, 2] - verts[:, 0, 2]
    # Squares of lengths of the edges incident to `a'.
    len_ba = ba_x * ba_x + ba_y * ba_y + ba_z * ba_z
    len_ca = ca_x * ca_x + ca_y * ca_y + ca_z * ca_z
    len_da = da_x * da_x + da_y * da_y + da_z * da_z
    # Cross products of these edges.
    # c cross d
    cross_cd_x = ca_y * da_z - da_y * ca_z
    cross_cd_y = ca_z * da_x - da_z * ca_x
    cross_cd_z = ca_x * da_y - da_x * ca_y
    # d cross b
    cross_db_x = da_y * ba_z - ba_y * da_z
    cross_db_y = da_z * ba_x - ba_z * da_x
    cross_db_z = da_x * ba_y - ba_x * da_y
    # b cross c
    cross_bc_x = ba_y * ca_z - ca_y * ba_z
    cross_bc_y = ba_z * ca_x - ca_z * ba_x
    cross_bc_z = ba_x * ca_y - ca_x * ba_y
    # Calculate the denominator of the formula.
    div_den = (ba_x * cross_cd_x + ba_y *
               cross_cd_y + ba_z * cross_cd_z)
    # coplanar vertices
    mask_div_den = np.abs(div_den) == 0
    div_den[mask_div_den] = 1
    denominator = 0.5 / div_den
    # Calculate offset (from `a') of circumcenter.
    circ_x = (len_ba * cross_cd_x + len_ca * cross_db_x +
              len_da * cross_bc_x) * denominator
    circ_y = (len_ba * cross_cd_y + len_ca * cross_db_y +
              len_da * cross_bc_y) * denominator
    circ_z = (len_ba * cross_cd_z + len_ca * cross_db_z +
              len_da * cross_bc_z) * denominator

    out = np.column_stack((circ_x, circ_y, circ_z))+verts[:, 0]
    out[mask_div_den] = verts[mask_div_den].mean(1)
    return out

class MeshFromDelaunay(Delaunay):
    def __init__(self, points: np.array, add_corners=False, **kwargs) -> None:
        if add_corners:
            this_points = np.concatenate(
                (points, SIGNS))
        else:
            this_points = points
        super().__init__(this_points, **kwargs)
        self.add_corners = add_corners
        
        self.circum_centers = tet_circumcenter(self.points[self.simplices])
        self.barycenters = self.points[self.simplices].mean(1)
        self.triangle_faces, self.triangle_faces_neighbors = self.get_triangle_faces()
        self.in_mask = self.order_neighbors()
        self.triangle_areas = np.sqrt((np.cross(
            self.points[self.triangle_faces[:, 1]] - self.points[self.triangle_faces[:, 0]], self.points[self.triangle_faces[:, 2]] - self.points[self.triangle_faces[:, 0]])**2).sum(-1))
        self.triangle_max_length = self.get_triangle_max_length()
        
    def get_triangle_faces(self):
        opp_face = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        ii = np.arange(len(self.neighbors))
        triangle_faces = -np.ones((len(self.neighbors)*4, 3), dtype=int)
        for j in range(4):
            triangle_faces[4*ii + j] = self.simplices[:, opp_face[j]]
        triangle_faces_neighbors = np.column_stack((np.arange(
            len(self.neighbors)).repeat(4), self.neighbors.reshape(len(self.neighbors)*4)))
        return triangle_faces, triangle_faces_neighbors
    
    def get_triangle_max_length(self):
        l1 = ((self.points[self.triangle_faces[:, 1]] - self.points[self.triangle_faces[:, 0]])**2).sum(-1)
        l2 = ((self.points[self.triangle_faces[:, 2]] - self.points[self.triangle_faces[:, 0]])**2).sum(-1)
        l3 = ((self.points[self.triangle_faces[:, 2]] - self.points[self.triangle_faces[:, 1]])**2).sum(-1)
        return np.max(np.stack((l1, l2, l3)), 0)

    def face_orientation(self, p1, p2, p3, vp1):
        return (np.cross(p2 - p1, p3 - p1) * (vp1 - (p1+p2+p3)/3.)).sum(-1) > 0

    def order_neighbors(self):
        opp_vert = self.simplices.reshape(
            len(self.simplices[self.triangle_faces_neighbors]))
        in_mask = self.face_orientation(
            *np.transpose(self.points[self.triangle_faces], (1, 0, 2)), self.points[opp_vert])
        in_triangle = self.triangle_faces
        flipped_triangles = np.fliplr(in_triangle)
        self.triangle_faces = in_triangle * \
            in_mask[:, None] + (1-in_mask[:, None])*flipped_triangles
        return in_mask
    
    def sample_random_tet_points(self, n=1):
        weights = np.random.rand(n, *self.simplices.shape)
        weights /= weights.sum(-1, keepdims=True)
        x = self.points[self.simplices]
        bary_x = (weights[..., None]*x[None, ...]).sum(-2)
        return bary_x

    def get_surface(self, return_scores=False, color_func=None, treshold=0, len_threshold=None, return_indices=False):
        neigh_color = self.tet_colors[self.triangle_faces_neighbors]
        neigh_color[self.triangle_faces_neighbors == -1] = 0
        neigh_color = neigh_color > treshold
        # two triangles per face: select only one
        in_t = (neigh_color[:, 0] == 0)*(neigh_color[:, 1] == 1)
        in_triangle = self.triangle_faces[in_t]

        # select only the relevant vertices
        un = np.unique(in_triangle)
        inv = np.arange(in_triangle.max() + 1)
        inv[un] = np.arange(len(un))
        nvertices = self.points[un]
        in_triangle = inv[in_triangle]
        to_return = [nvertices]
        
        if not len_threshold is None:
            in_scores = self.triangle_max_length[in_t]
            to_return.append(in_triangle[in_scores<len_threshold])
        else:
            to_return.append(in_triangle)
        if return_scores:
            to_return.append(in_scores)
        elif not color_func is None:
            to_return.append(color_func[in_t])
        if return_indices:
            to_return.append(un)
        return to_return
    

'''
Vector Field Auxiliary functions
'''

def get_vector_field_quantities_aux(
    points: torch.Tensor,
    g_means: torch.Tensor,
    g_normals: torch.Tensor,
    g_scales: torch.Tensor,
    g_rotation: torch.Tensor,
    g_opacity: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Current version of the vector field quantities.
    """

    B, k_neighbors = g_means.shape[0], g_means.shape[1]
    p = (points.unsqueeze(1) - g_means) # (B, k, 1, 3)
    # Get G_i(x) | S^-1 @ R^T
    g_invscale_rot = build_scaling_rotation(
        s=1. / g_scales.view(-1, 3),
        r=g_rotation.view(-1, 4),
    ).transpose(-1, -2)  # (B * k, 3, 3)

    gi_x = robust_gaussian_eval_shifted_points(
        shifted_points=p.view(-1, 3),
        gaussian_invscale_rot=g_invscale_rot,
        gaussian_opacity=g_opacity.view(-1, 1),
    ).view(B, k_neighbors, 1) # (B, k, 1)

    # Get < n_i, x - mu_i >
    n_dot_x_minus_mu = torch.sum(g_normals * p, dim=-1, keepdim=True) # (B, k, 1)
    indicator_function = (n_dot_x_minus_mu >= 0).float()  # (B, k, 1)

    # Get transformed points \Sigma_i^{-1} @ (x - mu_i)
    sigma_inv = robust_sigma_inv(g_scales, g_rotation)
    transformed_points = torch.einsum('bkij, bkj -> bki', sigma_inv, p) # (B, k, 3)

    # Compute the normal field
    gaussian_quotient = gi_x / (1.0 - gi_x + 1e-8) # (B, k, 1)
    nabla_log = gaussian_quotient * transformed_points
    normal_field = torch.sum(indicator_function * nabla_log, dim=1)
    return_dict = {
        "normal_field": normal_field # (B, 3)
    }

    return return_dict

class GaussianVectorField:

    @torch.no_grad()
    def __init__(self, gaussians, nearest_neighbors_type:str = "kdtree", k_neighbors: int = 32):

        self.nearest_neighbors_type = nearest_neighbors_type
        if self.nearest_neighbors_type == "kdtree":
            means = gaussians.get_xyz.detach().cpu().numpy() # (N, 3)
            self.tree = KDTree(means)
        elif self.nearest_neighbors_type == "cache_neighbors":
            self.k_neighbors = k_neighbors
            self.cached_knn_index = knn(gaussians.get_xyz, gaussians.get_xyz, k=k_neighbors)[1].view(gaussians.get_xyz.shape[0], k_neighbors)
            print(f"[INFO] Cached KNN index with shape: {self.cached_knn_index.shape}")
        else:
            raise ValueError(f"Invalid nearest neighbors type: {nearest_neighbors_type}")

    def get_vector_field_quantities(self, 
            query_points: torch.Tensor, 
            gaussians,
            **kwargs
        ) -> Dict[str, torch.Tensor]:
        """
        Computes the curl of the vector field at the query points.

        Args:
            query_points: (N, 3)
            gaussians
            batch_size: int
            k_neighbors: int
        Returns:
            curls: (N, 3)
        """

        if self.nearest_neighbors_type == "kdtree":
            # Query the nearest neighbors
            assert "k_neighbors" in kwargs, "k_neighbors must be provided when using kdtree"
            k_neighbors = kwargs["k_neighbors"]
            device = query_points.device
            if isinstance(query_points, torch.Tensor):
                query_points = query_points.detach().cpu().numpy()
            _, nn_neigbors_indices = self.tree.query(query_points, k=k_neighbors)
            nn_neigbors_indices = torch.from_numpy(nn_neigbors_indices).to(device)
            query_points = torch.from_numpy(query_points).to(device)
        elif self.nearest_neighbors_type == "cache_neighbors":
            assert "sampled_from_gaussian_idx" in kwargs, "sampled_from_gaussian_idx must be provided when using cache_neighbors"
            sampled_from_gaussian_idx = kwargs["sampled_from_gaussian_idx"] # (N_sampled_points,)
            nn_neigbors_indices = self.cached_knn_index[sampled_from_gaussian_idx].to(query_points.device) # (N_sampled_points, k_neighbors)
            k_neighbors = self.k_neighbors
        else:
            raise ValueError(f"Invalid nearest neighbors type: {self.nearest_neighbors_type}")
        
        # Get relevant Gaussian quantities
        means = gaussians.get_xyz # (N, 3)
        normals = gaussians.convert_features_to_normals(normalize=True) # (N, 3)
        scales = gaussians.get_scaling_with_3D_filter  # (N, 3)
        rotations = gaussians._rotation  # (N, 4)
        opacity = gaussians.get_opacity_with_3D_filter # (N, 1)

        # Evaluate the vector field quantities
        N = query_points.shape[0]
        flat_indices = nn_neigbors_indices.view(-1)
        return get_vector_field_quantities_aux(
            query_points,
            means[flat_indices].view(N, k_neighbors, 3),
            normals[flat_indices].view(N, k_neighbors, 3),
            scales[flat_indices].view(N, k_neighbors, 3),
            rotations[flat_indices].view(N, k_neighbors, 4),
            opacity[flat_indices].view(N, k_neighbors, 1),
        )
    

'''
Sampling Auxiliary functions
'''
def compute_face_to_camera_minimum_distance(
    mesh, 
    cameras, 
    batch_size=100_000
) -> torch.Tensor:
    """Compute the minimum distance to any camera for each face of the mesh.

    Args:
        mesh (Meshes): The mesh.
        cameras (List[Camera]): The cameras to compute the minimum distance to.
        batch_size (int, optional): The batch size to use for the computation. Defaults to 100_000.

    Returns:
        torch.Tensor: The minimum distance to any camera for each face of the mesh.
            Has shape (N_faces,).
    """
    # Compute camera centers
    all_cam_centers = torch.cat([camera_i.camera_center[None] for camera_i in cameras], dim=0)

    # Initialize minimum face to camera distance
    min_face_to_cam_distance = 1_000_000. * torch.ones(mesh.faces.shape[0], device=mesh.faces.device)  # (N_faces,)

    # Do it per batch of faces to avoid OOM
    for i in range(0, mesh.faces.shape[0], batch_size):
        batch_faces = mesh.faces[i:i+batch_size]  # (batch_size, 3)
        batch_face_centers = mesh.verts[batch_faces].mean(dim=1)  # (batch_size, 3)
        
        # For each camera, check if the face is in the view frustum
        in_view_mask = torch.zeros(
            batch_faces.shape[0], len(cameras), 
            device=mesh.faces.device, 
            dtype=torch.bool
        )  # (batch_size, N_cams)
        for j, camera in enumerate(cameras):
            in_view_mask[:, j] = is_in_view_frustum(
                points=batch_face_centers,
                camera=camera,
            )
        
        # Compute distance to all cameras
        batch_face_dist = (batch_face_centers[:, None] - all_cam_centers[None]).norm(dim=2)  # (batch_size, N_cams)
        batch_face_dist[~in_view_mask] = 1_000_000.
        
        # For each face, get the minimum distance to any camera
        batch_face_min_dist = batch_face_dist.min(dim=1).values  # (batch_size,)
        min_face_to_cam_distance[i:i+batch_size] = batch_face_min_dist
        
    return min_face_to_cam_distance

def sample_mesh_proportional_to_camera(mesh, cameras, num_points, face_probs=None):
    """
    Sample points from a mesh proportional to the number of cameras.
    face_probs can be passed to skip recomputation when the mesh and cameras are unchanged.
    """
    # Get triangle verts — always needed for sampling
    face_verts = mesh.verts[mesh.faces]  # N_faces, 3, 3

    if face_probs is None:
        min_face_to_cam_distance = compute_face_to_camera_minimum_distance(
            mesh=mesh,
            cameras=cameras,
            batch_size=100_000,
        )  # (N_faces,)

        # Get triangle areas
        face_areas = torch.linalg.norm(
            torch.cross(face_verts[:, 1] - face_verts[:, 0],
                        face_verts[:, 2] - face_verts[:, 0],
                        dim=-1),  # (N_faces, 3)
            dim=1
        ) / 2.0  # (N_faces,)

        face_probs = face_areas / (min_face_to_cam_distance ** 2)
        face_probs = face_probs / torch.sum(face_probs)  # (N_faces,)

    # Sample triangles following the probabilities
    sampled_face_idx = torch.multinomial(input=face_probs, num_samples=num_points, replacement=True)  # (n_points,)
    sampled_face_verts = face_verts[sampled_face_idx]  # (num_points, 3, 3)

    # Sample barycentric coordinates (summing to 1) in each sampled triangle
    sampled_barycentric_coords_1 = torch.rand(num_points, 1, device=mesh.faces.device)  # (n_points, 1)
    sampled_barycentric_coords_2 = (
        torch.rand(num_points, 1, device=mesh.faces.device) * (1. - sampled_barycentric_coords_1)
    )  # (num_points, 1)
    sampled_barycentric_coords_3 = 1. - sampled_barycentric_coords_1 - sampled_barycentric_coords_2  # (num_points, 1)
    sampled_barycentric_coords = torch.cat([sampled_barycentric_coords_1, sampled_barycentric_coords_2, sampled_barycentric_coords_3], dim=1)  # (num_points, 3)
    
    # ---Compute means---
    means = (
        sampled_face_verts  # (num_points, 3, 3)
        * sampled_barycentric_coords[..., None]  # (n_points, 3, 1)
    ).sum(dim=1)  # (num_points, 3)

    normals = mesh.face_normals[sampled_face_idx]  # (num_points, 3)

    return means.cpu().detach().numpy(), normals.cpu().detach().numpy(), face_probs

'''
Plotting functions
'''

def plot_histogram(occupancy_values: torch.Tensor, filename: str, title: str, delta: float = 0.05):
    """
    Plots and saves a professional and beautiful histogram of occupancy values,
    zoomed into the region [0.5 - delta, 0.5 + delta], and sets the y-axis maximum to n_points.

    Args:
        occupancy_values (np.ndarray or Tensor): The occupancy values to plot.
        filename (str): Full path to save the image (including extension).
        title (str): Title of the plot.
        delta (float): Range around 0.5 to zoom in.
    """
    # Ensure values are numpy array and flattened
    if hasattr(occupancy_values, "detach"):
        occupancy_values = occupancy_values.detach().cpu().numpy()
    occupancy_values = occupancy_values.flatten()

    # Zoom in: select occupancy values in [0.5 - delta, 0.5 + delta]
    lower = 0.5 - delta
    upper = 0.5 + delta
    occupancy_values_zoomed = occupancy_values[(occupancy_values >= lower) & (occupancy_values <= upper)]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    n_points = occupancy_values.shape[0]

    plt.figure(figsize=(8, 6))
    n, bins, patches = plt.hist(occupancy_values_zoomed, bins=60, color="#1386e3", alpha=0.85, edgecolor="black", linewidth=1.2)
    plt.title(f"{title} (zoom [{lower:.3f}, {upper:.3f}])", fontsize=16, fontweight='bold', pad=16)
    plt.xlabel("Occupancy Value", fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.xlim([lower, upper])
    plt.ylim([0, n_points])
    plt.grid(axis="y", alpha=0.25, linestyle='--')
    plt.tight_layout()
    # You could use seaborn style for more beauty, but keeping it native matplotlib for safety.
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()