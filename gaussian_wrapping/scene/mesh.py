from typing import List, Union, Tuple, Optional
import numpy as np
import torch
import sys
sys.setrecursionlimit(10000) # Useful when running with python 3.9
import nvdiffrast.torch as dr
from scene.cameras import Camera
from utils.geometry_utils import transform_points_world_to_view

# Try importing cpp extension, handle potential ImportError
try:
    from tetranerf.utils.extension import cpp
except ImportError:
    cpp = None
    print("[WARNING] Could not import 'tetranerf.utils.extension.cpp'. Mesh regularization requires this.")

try:
    print("[INFO] Importing Delaunay from scipy...")
    from scipy.spatial import Delaunay
except ImportError:
    Delaunay = None
    print("[WARNING] Could not import 'scipy.spatial.Delaunay'. Mesh regularization requires this.")

def nvdiff_rasterization(
    camera,
    image_height:int, 
    image_width:int,
    verts:torch.Tensor, 
    faces:torch.Tensor,
    return_indices_only:bool=False,
    glctx=None,
    return_rast_out:bool=False,
    return_positions:bool=False,
):
    device = verts.device
        
    # Get full projection matrix
    camera_mtx = camera.full_proj_transform
    
    # Convert to homogeneous coordinates
    pos = torch.cat([verts, torch.ones([verts.shape[0], 1], device=device)], axis=1)
    
    # Transform points to NDC/clip space
    pos = torch.matmul(pos, camera_mtx)[None]
    
    # Rasterize with NVDiffRast
    # TODO: WARNING: pix_to_face is not in the correct range [-1, F-1] but in [0, F],
    # With 0 indicating that no triangle was hit.
    # So we need to subtract 1.
    rast_out, _ = dr.rasterize(glctx, pos=pos, tri=faces, resolution=[image_height, image_width])
    bary_coords, zbuf, pix_to_face = rast_out[..., :2], rast_out[..., 2], rast_out[..., 3].int() - 1
    
    if return_indices_only:
        return pix_to_face
    
    _output = (bary_coords, zbuf, pix_to_face)
    if return_rast_out:
        _output = _output + (rast_out,)
    if return_positions:
        _output = _output + (pos,)
    return _output

class Meshes(torch.nn.Module):
    """
    Meshes class for storing meshes parameters.
    """
    def __init__(
        self, 
        verts:torch.Tensor, 
        faces:torch.Tensor, 
        verts_colors:torch.Tensor=None
    ):
        super().__init__()
        assert verts_colors is None or verts_colors.shape[0] == verts.shape[0]
        self.verts = verts
        self.faces = faces.to(torch.int32)
        self.verts_colors = verts_colors
        self._edges = None
        self._faces_to_edges = None
        
    @property
    def device(self):
        return self.verts.device
        
    @property
    def face_normals(self):
        faces_verts = self.verts[self.faces]  # (F, 3, 3)
        faces_verts_normals = torch.cross(
            faces_verts[:,1] - faces_verts[:,0],  # (F, 3)
            faces_verts[:,2] - faces_verts[:,0],  # (F, 3)
            dim=-1
        )  # (F, 3)
        faces_verts_normals = torch.nn.functional.normalize(faces_verts_normals, dim=-1)  # (F, 3)
        return faces_verts_normals
    
    @property
    def vertex_normals(self):
        """Compute the vertex normals.
        Vertex normals are computed as the sum of the normals of all the faces it is part of,
        weighted by the face areas.
        """
        verts_normals = torch.zeros_like(self.verts)  # (V, 3)
        vertices_faces = self.verts[self.faces]  # (F, 3, 3)
        
        # Unnormalized faces normals.
        # Their magnitude is 2 x area of the triangle.
        faces_normals = torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],  # (F, 3)
            vertices_faces[:, 0] - vertices_faces[:, 1],  # (F, 3)
            dim=1,
        )  # (F, 3)
        
        # Add the faces normals to the verts normals
        verts_normals = verts_normals.index_add(
            0, self.faces[:, 0], faces_normals
        )
        verts_normals = verts_normals.index_add(
            0, self.faces[:, 1], faces_normals
        )
        verts_normals = verts_normals.index_add(
            0, self.faces[:, 2], faces_normals
        )
        
        # Normalize the verts normals
        return torch.nn.functional.normalize(
            verts_normals, eps=1e-6, dim=1
        )  # (V, 3)
    
    @property
    def edges(self):
        # Inspired from PyTorch3D: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/structures/meshes.html
        if self._edges is None:
            F = self.faces.shape[0]
            v0, v1, v2 = self.faces.chunk(3, dim=1)
            e01 = torch.cat([v0, v1], dim=1)  # (F, 2)
            e12 = torch.cat([v1, v2], dim=1)  # (F, 2)
            e20 = torch.cat([v2, v0], dim=1)  # (F, 2)
            
            # All edges including duplicates
            edges = torch.cat([e12, e20, e01], dim=0)  # (3 * F, 2)
            
            # Sort the edges in increasing vertex order to better identify duplicates
            edges, _ = edges.sort(dim=1)  # (3 * F, 2)
            
            # TODO: WARNING: Cast to long to avoid overflows
            edges = edges.to(torch.int64)
            
            # Remove duplicate edges: convert each edge (v0, v1) into an
            # integer hash = V * v0 + v1; this allows us to use the scalar version of
            # unique which is much faster than edges.unique(dim=1) which is very slow.
            # After finding the unique elements reconstruct the vertex indices as:
            # (v0, v1) = (hash / V, hash % V)
            # The inverse maps from unique_edges back to edges:
            # unique_edges[inverse_idxs] == edges
            # i.e. inverse_idxs[i] == j means that edges[i] == unique_edges[j]
            V = self.verts.shape[0]
            edges_hash = V * edges[:, 0] + edges[:, 1]  # (3 * F, )
            u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
            
            self._edges = torch.stack([u // V, u % V], dim=1)  # (E, 2)
            self._faces_to_edges = inverse_idxs.reshape(3, F).t()  # (F, 3)
            
            # TODO: WARNING: Cast back to int32
            self._edges = self._edges.to(torch.int32)
        
        return self._edges
    
    @property
    def faces_to_edges(self):
        if self._faces_to_edges is None:
            _ = self.edges  # Compute edges if not already computed
        return self._faces_to_edges
    
    @property
    def edges_to_faces(self):
        """
        Returns a tensor of shape (E, 2) giving, for each edge, the two faces it belongs to.
        If only one face, set the second index to be the same as the first one.
        """
        # Step 1: Getting two flat tensors establishing the correspondence between edge indices and face indices.
        # The edge at faces_to_edges[i] belongs to the face at faces_idx_per_edge[i].
        faces_idx_per_edge = torch.arange(0, self.faces.shape[0], device=self.faces.device)[:, None].repeat(1, 3)
        faces_idx_per_edge = faces_idx_per_edge.flatten()    
        faces_to_edges = self.faces_to_edges.flatten()
        
        # Step 2: Identify edges that belong to only one face
        _, _, edge_counts = faces_to_edges.unique(return_inverse=True, return_counts=True)
        edge_has_only_one_face = (edge_counts == 1)
        
        # Step 3: G
        # Each non unique edge idx in edge_in_faces belongs to the corresponding face in edge_to_faces_flattened
        edge_in_faces, edge_to_faces_flattened = faces_to_edges.sort()
        edge_to_faces_flattened = faces_idx_per_edge[edge_to_faces_flattened] 
        edge_in_faces_has_only_one_face = edge_has_only_one_face[edge_in_faces]

        # Step 4: Get a tensor with shape (E, 2) giving, for each edge, the two faces it belongs to.
        # If only one face, set the second index to be the same as the first one.
        edge_to_faces = - torch.ones(self.edges.shape[0], 2, device=self.edges.device, dtype=edge_to_faces_flattened.dtype)
        edge_to_faces[~edge_has_only_one_face] = edge_to_faces_flattened[~edge_in_faces_has_only_one_face].reshape(-1, 2)
        edge_to_faces[edge_has_only_one_face] = edge_to_faces_flattened[edge_in_faces_has_only_one_face].unsqueeze(-1).repeat(1, 2)
        
        return edge_to_faces
    
    @property
    def laplacian(self):
        """
        Inspired from PyTorch3D: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/laplacian_matrices.html
        
        Computes the laplacian matrix.
        The definition of the laplacian is
        L[i, j] =    -1       , if i == j
        L[i, j] = 1 / deg(i)  , if (i, j) is an edge
        L[i, j] =    0        , otherwise
        where deg(i) is the degree of the i-th vertex in the graph.

        Args:
            verts: tensor of shape (V, 3) containing the vertices of the graph
            edges: tensor of shape (E, 2) containing the vertex indices of each edge
        Returns:
            L: Sparse FloatTensor of shape (V, V)
        """
        V = self.verts.shape[0]

        e0, e1 = self.edges.unbind(1)

        idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
        idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
        idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

        # torch.sparse.check_sparse_tensor_invariants.enable()

        # First, we construct the adjacency matrix,
        # i.e. A[i, j] = 1 if (i,j) is an edge, or
        # A[e0, e1] = 1 &  A[e1, e0] = 1
        ones = torch.ones(idx.shape[1], dtype=torch.float32, device=self.device)
        A = torch.sparse_coo_tensor(idx, ones, (V, V), dtype=torch.float32)

        # the sum of i-th row of A gives the degree of the i-th vertex
        deg = torch.sparse.sum(A, dim=1).to_dense()

        # We construct the Laplacian matrix by adding the non diagonal values
        # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
        deg0 = deg[e0]
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
        deg1 = deg[e1]
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
        val = torch.cat([deg0, deg1])
        L = torch.sparse_coo_tensor(idx, val, (V, V), dtype=torch.float32)

        # Then we add the diagonal values L[i, i] = -1.
        idx = torch.arange(V, device=self.device)
        idx = torch.stack([idx, idx], dim=0)
        ones = torch.ones(idx.shape[1], dtype=torch.float32, device=self.device)
        L -= torch.sparse_coo_tensor(idx, ones, (V, V), dtype=torch.float32)

        return L 
    
    def cotangent_laplacian(self, eps:float=1e-12):
        """
        Inspired from PyTorch3D: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/laplacian_matrices.html
        
        Returns the Laplacian matrix with cotangent weights and the inverse of the
        face areas.

        Args:
            verts: tensor of shape (V, 3) containing the vertices of the graph
            faces: tensor of shape (F, 3) containing the vertex indices of each face
        Returns:
            2-element tuple containing
            - **L**: Sparse FloatTensor of shape (V,V) for the Laplacian matrix.
            Here, L[i, j] = cot a_ij + cot b_ij iff (i, j) is an edge in meshes.
            See the description above for more clarity.
            - **inv_areas**: FloatTensor of shape (V,) containing the inverse of sum of
            face areas containing each vertex
        """
        verts = self.verts
        faces = self.faces
        V, F = verts.shape[0], faces.shape[0]

        face_verts = verts[faces]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

        # Side lengths of each triangle, of shape (F,)
        # A is the side opposite v1, B is opposite v2, and C is opposite v3
        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1)
        C = (v0 - v1).norm(dim=1)

        # Area of each triangle (with Heron's formula); shape is (F,)
        s = 0.5 * (A + B + C)
        # note that the area can be negative (close to 0) causing nans after sqrt()
        # we clip it to a small positive value
        # pyre-fixme[16]: `float` has no attribute `clamp_`.
        area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=eps).sqrt()

        # Compute cotangents of angles, of shape (sum(F_n), 3)
        A2, B2, C2 = A * A, B * B, C * C
        cota = (B2 + C2 - A2) / area
        cotb = (A2 + C2 - B2) / area
        cotc = (A2 + B2 - C2) / area
        cot = torch.stack([cota, cotb, cotc], dim=1)
        cot /= 4.0

        # Construct a sparse matrix by basically doing:
        # L[v1, v2] = cota
        # L[v2, v0] = cotb
        # L[v0, v1] = cotc
        ii = faces[:, [1, 2, 0]]
        jj = faces[:, [2, 0, 1]]
        idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
        L = torch.sparse_coo_tensor(idx, cot.view(-1), (V, V), dtype=torch.float32)

        # Make it symmetric; this means we are also setting
        # L[v2, v1] = cota
        # L[v0, v2] = cotb
        # L[v1, v0] = cotc
        L += L.t()

        # For each vertex, compute the sum of areas for triangles containing it.
        idx = faces.view(-1).to(torch.int64)
        inv_areas = torch.zeros(V, dtype=torch.float32, device=verts.device)
        val = torch.stack([area] * 3, dim=1).view(-1)
        inv_areas.scatter_add_(0, idx, val)
        idx = inv_areas > 0
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        inv_areas[idx] = 1.0 / inv_areas[idx]
        inv_areas = inv_areas.view(-1, 1)

        return L, inv_areas
    
    def get_edge_to_edge_idx(self, edges:torch.Tensor) -> torch.Tensor:
        """Given a tensor of edges with shape (E, 2), 
        return a tensor of shape (E) giving the index of the edge in mesh.edges.

        Args:
            mesh (Meshes): The mesh to get the edge to edge index for.
            edges (torch.Tensor): A tensor of shape (E, 2) giving the edges to get the index for.

        Returns:
            torch.Tensor: A tensor of shape (E) giving the index of the edge in mesh.edges.
        """
        
        # Sort the edges in increasing vertex order
        sort_edges, _ = edges.sort(dim=1)
        
        # TODO: WARNING: Cast to long to avoid overflows
        sort_edges = sort_edges.to(torch.int64)
        all_edges = self.edges.to(torch.int64)
        
        # Compute the hash of the edges
        V = self.verts.shape[0]
        edges_hash = V * sort_edges[:, 0] + sort_edges[:, 1]
        
        # Compute the hash of all edges
        all_edges_hash = V * all_edges[:, 0] + all_edges[:, 1]

        # Create a sparse tensor to map hashes to edge indices
        max_hash = all_edges_hash.max()
        hash_to_idx = torch.sparse_coo_tensor(
            indices=all_edges_hash.unsqueeze(0),  # Hashes
            values=torch.arange(0, self.edges.shape[0], device=self.edges.device),  # Edge indices
            size=(max_hash+1,)
        )
        
        # Use the sparse tensor to map the edge hashes to edge indices
        return hash_to_idx.index_select(0, edges_hash).to_dense()
    
    def submesh(
        self, 
        vert_idx:Optional[torch.Tensor]=None, 
        face_idx:Optional[torch.Tensor]=None,
        vert_mask:Optional[torch.Tensor]=None,
        face_mask:Optional[torch.Tensor]=None,
    ):
        assert (
            (vert_idx is not None) or (vert_mask is not None) 
            or
            (face_idx is not None) or (face_mask is not None)
        ), "Either vert_idx, vert_mask, face_idx, or face_mask must be provided"

        if (vert_idx is not None) or (vert_mask is not None):
            if vert_mask is None:
                vert_mask = torch.zeros(self.verts.shape[0], dtype=torch.bool, device=self.verts.device)
                vert_mask[vert_idx] = True
            face_mask = vert_mask[self.faces].all(dim=1)

        elif (face_idx is not None) or (face_mask is not None):
            if face_mask is None:
                face_mask = torch.zeros(self.faces.shape[0], dtype=torch.bool, device=self.verts.device)
                face_mask[face_idx] = True
            vert_mask = torch.zeros(self.verts.shape[0], dtype=torch.bool, device=self.verts.device)
            vert_mask[self.faces[face_mask]] = True
        
        old_vert_idx_to_new_vert_idx = torch.zeros(self.verts.shape[0], dtype=self.faces.dtype, device=self.verts.device)
        old_vert_idx_to_new_vert_idx[vert_mask] = torch.arange(vert_mask.sum(), dtype=self.faces.dtype, device=self.verts.device)
        
        new_verts = self.verts[vert_mask]
        new_verts_colors = None if self.verts_colors is None else self.verts_colors[vert_mask]
        new_faces = old_vert_idx_to_new_vert_idx[self.faces][face_mask]
        
        return Meshes(verts=new_verts, faces=new_faces, verts_colors=new_verts_colors)

    def angle_defect(self) -> float:
        """
        Computes how far the angles of the mesh are from 60 degrees.

        Returns:
            angle_defect (float): The mean angle defect of the mesh.
        """
        faces_verts = self.verts[self.faces]
        v0 = faces_verts[:,0]
        v1 = faces_verts[:,1]
        v2 = faces_verts[:,2]
        
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        a = torch.linalg.norm(e0, dim=-1, keepdim=True)
        b = torch.linalg.norm(e1, dim=-1, keepdim=True)
        c = torch.linalg.norm(e2, dim=-1, keepdim=True)

        cos_A = torch.sum(-e2 * e0, dim=-1, keepdim=True) / (c * a + 1e-6)
        cos_B = torch.sum(-e0 * e1, dim=-1, keepdim=True) / (a * b + 1e-6)
        cos_C = torch.sum(-e1 * e2, dim=-1, keepdim=True) / (b * c + 1e-6)
        all_cos = torch.cat([cos_A, cos_B, cos_C], dim=0)

        # How far angles are from cos(pi/3)=0.5
        return torch.mean(torch.abs(all_cos - 0.5))

    def return_triangle_areas(self) -> torch.Tensor:
        """
        Computes the areas of the triangles in the mesh.
        
        Returns:
            face_areas (torch.Tensor): The areas of the triangles in the mesh. (N_faces,)
        """
        
        return compute_triangle_areas(verts=self.verts, faces=self.faces)


def remove_duplicate_edges(
    edges:torch.Tensor,
    already_sorted:bool=False,
) -> torch.Tensor:
    """Given a tensor of edges with shape (E, 2), return a tensor of shape (E_unique, 2) by removing duplicate edges.

    Args:
        edges (torch.Tensor): A tensor of shape (E, 2) giving the edges to remove duplicates from.

    Returns:
        torch.Tensor: A tensor of shape (E_unique, 2) giving the unique edges.
    """
    if edges.shape[0] == 0:
        return edges
    
    V = edges.max() + 1
    
    if not already_sorted:
        edges = edges.sort(dim=1).values  # (E, 2)
    
    # Cast to long to avoid overflows
    edges = edges.to(torch.int64)  # (E, 2)
    
    # Compute the hash of the edges
    edges_hash = V * edges[:, 0] + edges[:, 1]  # (E,)
    
    # Call scalar version of unique, much faster than vectorized version
    u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
    
    # Convert hash to vertex indices
    unique_edges = torch.stack([u // V, u % V], dim=1)  # (E_unique, 2)
    
    return unique_edges


def remove_degenerate_edges(
    edges:torch.Tensor,
):
    """
    Given a tensor of edges with shape (E, 2), return a tensor of shape (E_filtered, 2) by removing degenerate edges.

    Args:
        edges (torch.Tensor): A tensor of shape (E, 2) giving the edges to remove degenerate edges from.

    Returns:
        torch.Tensor: A tensor of shape (E_filtered, 2) giving the filtered edges.
    
    """
    edge_mask = edges[:, 0] != edges[:, 1]
    return edges[edge_mask]


def compute_triangle_areas(
    verts:torch.Tensor, 
    faces:torch.Tensor
) -> torch.Tensor:
    """
    Computes the areas of the triangles given by the faces and vertices.
    
    Args:
        verts (torch.Tensor): The vertices. Shape (N_verts, 3)
        faces (torch.Tensor): The faces. Shape (N_faces, 3)
    
    Returns:
        face_areas (torch.Tensor): The areas of the triangles. (N_faces,)
    """
    # Get triangle verts
    faces_verts = verts[faces]
    v0 = faces_verts[:,0]  # (N_faces, 3)
    v1 = faces_verts[:,1]  # (N_faces, 3)
    v2 = faces_verts[:,2]  # (N_faces, 3)
    
    # Get triangle areas
    face_areas = torch.linalg.norm(
        torch.cross(v1 - v0, v2 - v0,dim=-1),  # (N_faces, 3)
        dim=1
    ) / 2.0  # (N_faces,)

    return face_areas


def combine_meshes(
    meshes:List[Meshes],
) -> Meshes:
    """Combines multiple meshes into a single mesh.

    Args:
        meshes (List[Meshes]): List of meshes to combine.

    Returns:
        Meshes: Combined mesh.
    """
    all_verts = torch.zeros(0, 3, dtype=meshes[0].verts.dtype, device=meshes[0].verts.device)
    all_faces = torch.zeros(0, 3, dtype=meshes[0].faces.dtype, device=meshes[0].faces.device)
    all_verts_colors = None if meshes[0].verts_colors is None else torch.zeros(0, 3, dtype=meshes[0].verts_colors.dtype, device=meshes[0].verts_colors.device)
    
    n_total_verts = 0
    
    for _, mesh in enumerate(meshes):
        all_verts = torch.cat([all_verts, mesh.verts], dim=0)
        all_faces = torch.cat([all_faces, mesh.faces + n_total_verts], dim=0)
        if all_verts_colors is not None:
            all_verts_colors = torch.cat([all_verts_colors, mesh.verts_colors], dim=0)
        n_total_verts += mesh.verts.shape[0]
    
    return Meshes(verts=all_verts, faces=all_faces, verts_colors=all_verts_colors)

def laplacian_smoothing_loss(
    mesh:Meshes, 
    method:str="uniform",
    reduce:bool=True,
) -> torch.Tensor:
    """Computes the Laplacian smoothing loss for a mesh.

    Args:
        mesh (Meshes): The mesh to compute the Laplacian smoothing loss for.
        method (str, optional): The method to use for the Laplacian smoothing loss.
            Defaults to "uniform", can also be "cot" or "cotcurv".
        reduce (bool, optional): Whether to reduce the loss to a scalar.
            Defaults to True.

    Raises:
        ValueError: If the method is not one of "uniform", "cot", or "cotcurv".

    Returns:
        torch.Tensor: The Laplacian smoothing loss.
    """
    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = mesh.laplacian
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = mesh.cotangent_laplacian()
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                # pyre-fixme[58]: `/` is not supported for operand types `float` and
                #  `Tensor`.
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    if method == "uniform":
        loss = L.mm(mesh.verts)
    elif method == "cot":
        # pyre-fixme[61]: `norm_w` is undefined, or not always defined.
        loss = L.mm(mesh.verts) * norm_w - mesh.verts
    elif method == "cotcurv":
        # pyre-fixme[61]: `norm_w` may not be initialized here.
        loss = (L.mm(mesh.verts) - L_sum * mesh.verts) * norm_w
    
    loss = loss.norm(dim=1)
    return loss.mean() if reduce else loss

def normal_consistency_loss(
    mesh:Meshes,
    reduce:bool=True,
) -> torch.Tensor:
    """Computes the normal consistency loss for a mesh.
    
    Args:
        mesh (Meshes): The mesh to compute the normal consistency loss for.
        
    Returns:
        torch.Tensor: The normal consistency loss.
    """
    if False:
        edge_verts_normals = mesh.vertex_normals[mesh.edges]  # Shape (E, 2, 3)
        assert (
            edge_verts_normals.shape[0] == mesh.edges.shape[0]
            and edge_verts_normals.shape[1] == 2
            and edge_verts_normals.shape[2] == 3
        ), "edge_verts_normals should be of shape (E, 2, 3)"
        
        edge_verts_normals_dot_product = (edge_verts_normals[..., 0] * edge_verts_normals[..., 1]).sum(dim=-1)
        result = (1. - edge_verts_normals_dot_product)
    
    if True:
        face_normals = mesh.face_normals[:, None, :]  # Shape (F, 1, 3)
        face_verts_normals = mesh.vertex_normals[mesh.faces]  # Shape (F, 3, 3)
        face_verts_normals_dot_product = (face_normals * face_verts_normals).sum(dim=-1)  # Shape (F, 3)
        result = (1. - face_verts_normals_dot_product)
    
    if reduce:
        return result.mean()
    else:
        return result
    
def get_error_quadrics(mesh: Meshes, average_w_face_area: bool = True, verbose: bool = False):
    """
    Compute the error quadrics for vertices of a mesh.
    
    Args:
        mesh (Meshes): The mesh to compute the error quadrics for.
        average_w_face_area (bool, optional): Whether to average the quadrics by the face area. Defaults to True.

    Returns:
        torch.Tensor: The error quadrics. Shape (V, 4, 4)
    """
    
    verts = mesh.verts  # (V, 3)
    
    if verbose:
        if torch.isnan(verts).any() or torch.isinf(verts).any():
            print(f"\n[DEBUG] verts has {torch.isnan(verts).sum().item()} / {verts.flatten().shape[0]} nan values")
            print(f"[DEBUG] verts has {torch.isinf(verts).sum().item()} / {verts.flatten().shape[0]} inf values")
    
    # Compute the fundamental quadrics for each face
    face_centers = verts[mesh.faces].mean(dim=-2)  # (F, 3)
    if average_w_face_area: 
        face_areas = compute_triangle_areas(verts=verts, faces=mesh.faces) # (F,)
    else:
        face_areas = torch.ones(mesh.faces.shape[0], device=mesh.device)
    
    if verbose:
        if torch.isnan(face_areas).any() or torch.isinf(face_areas).any():
            print(f"[DEBUG] face_areas has {torch.isnan(face_areas).sum().item()} / {face_areas.flatten().shape[0]} nan values")
            print(f"[DEBUG] face_areas has {torch.isinf(face_areas).sum().item()} / {face_areas.flatten().shape[0]} inf values")
        
    face_normals = mesh.face_normals  # (F, 3)
    
    if verbose:
        if torch.isnan(face_normals).any() or torch.isinf(face_normals).any():
            print(f"[DEBUG] face_normals has {torch.isnan(face_normals).sum().item()} / {face_normals.flatten().shape[0]} nan values")
    
    face_centers_proj = (face_normals * face_centers).sum(dim=-1, keepdim=True)  # (F, 1)
    
    if verbose:
        if torch.isnan(face_centers_proj).any() or torch.isinf(face_centers_proj).any():
            print(f"[DEBUG] face_centers_proj has {torch.isnan(face_centers_proj).sum().item()} / {face_centers_proj.flatten().shape[0]} nan values")
            print(f"[DEBUG] face_centers_proj has {torch.isinf(face_centers_proj).sum().item()} / {face_centers_proj.flatten().shape[0]} inf values")
    
    face_planes = torch.cat(
        [
            face_normals, 
            -face_centers_proj,
        ], 
        dim=-1
    )  # (F, 4)
    fundamental_quadrics = face_planes[:, :, None] @ face_planes[:, None, :]  # (F, 4, 4)
    if average_w_face_area:
        # From section 3.4.1:https://www.cs.cmu.edu/~garland/thesis/thesis-onscreen.pdf
        fundamental_quadrics = fundamental_quadrics * face_areas.view(-1, 1, 1)
    # Add the fundamental face quadrics to the verts error quadrics
    error_quadrics = torch.zeros(verts.shape[0], 4, 4, device=mesh.device)  # (V, 4, 4)
    error_quadrics = error_quadrics.index_add(
        0, mesh.faces[:, 0], fundamental_quadrics
    )
    error_quadrics = error_quadrics.index_add(
        0, mesh.faces[:, 1], fundamental_quadrics
    )
    error_quadrics = error_quadrics.index_add(
        0, mesh.faces[:, 2], fundamental_quadrics
    )
    
    return error_quadrics


def vstars_from_quadrics(Q, P, eps=.05):
    """Compute optimal vertices positions

    Parameters
    ----------
    Q: torch.tensor 
        Nx4x4 of quadric matrices
    P: torch.tensor 
        Nx3 tensor of positions

    Returns
    -------
    vstars: torch.tensor 
        Nx3 optimal vertices 
    eigs: torch.tensor 
        Nx3 eigen values of quadric matrices
    """
    A = Q[:, :3, :3]
    b = -Q[:, 3, :3]
    u, eigs, vh = torch.linalg.svd(A) # TODO: Fix nan in backprop

    eigs2 = torch.zeros_like(eigs)
    mask_s = (eigs[:, 0, None] > 0.) & ((eigs / eigs[:, 0, None]) > eps)  # TODO: Could the nan come from here? Like, if all eigenvalues are 0 for some reason?
    eigs2[mask_s] = 1.0 / eigs[mask_s]

    base_pos = P
    vstars = base_pos + (
        vh.transpose(1, 2)
        @ torch.diag_embed(eigs2)
        @ u.transpose(1, 2)
        @ (b[..., None] - A @ base_pos[..., None])
    ).squeeze(-1)
    return vstars, vh, eigs


def vstars_from_quadrics_least_squares(Q, **kwargs):
    """
    Compute the v_stars from the quadrics using least squares.

    Args:
        Q (torch.Tensor): The quadrics. Shape (N, 4, 4)

    Returns:
        torch.Tensor: The v_stars. Shape (N, 3)
    """
    A = Q[:, :3, :3]  # (N, 3, 3)
    B = -Q[:, 3, :3].unsqueeze(-1)  # (N, 3, 1)
    
    # TODO: If A is not full rank, use a provided base_point tensor as v_star
    
    v_star = torch.linalg.lstsq(A, B).solution  # (N, 3, 1)

    return v_star.squeeze(-1)  # (N, 3)


def get_contraction_points(
    verts: torch.Tensor, 
    edges: torch.Tensor, 
    error_quadrics: torch.Tensor,
    return_edge_quadrics: bool = False,
    return_qem_cost_quantities: bool = False,
):
    # Compute the edge quadrics
    edge_quadrics = error_quadrics[edges[:, 0]] + error_quadrics[edges[:, 1]]  # (E, 4, 4)
    
    M = torch.zeros_like(edge_quadrics)
    M[:, :3, :4] = edge_quadrics[:, :3, :4]
    M[:, 3, 3] = 1.
    
    b = torch.zeros(edges.shape[0], 4, device=error_quadrics.device)
    b[:, 3] = 1.
    
    # compute solvable mask
    solvable_mask = torch.linalg.det(M).abs() > 0.
    
    # Solve the linear system Mv = b
    v = torch.zeros(edges.shape[0], 4, device=error_quadrics.device) # (E, 4)
    v[solvable_mask] = torch.linalg.solve(M[solvable_mask], b[solvable_mask])
    
    if return_qem_cost_quantities:
        return (v.unsqueeze(-1), edge_quadrics, solvable_mask) # (E, 4, 1), (E, 4, 4), (E,)
    return (v[..., :3], edge_quadrics, solvable_mask) if return_edge_quadrics else v[..., :3]

def quadrics_score(quadrics, points):
    """
    Computes the QEM score for a given set of quadrics and points.

    Args:
        quadrics (torch.Tensor): The quadrics. Shape (N, 4, 4)
        points (torch.Tensor): The points. Shape (N, 3)

    Returns:
        torch.Tensor: The QEM score. Shape (N, )
    """
    new_points = torch.cat((points, torch.ones_like(points[..., :1])), -1)  # (N, 4)
    return (
        new_points[..., None, :]  # (N, 1, 4)
        @ quadrics  # (N, 4, 4)
        @ new_points[..., None]  # (N, 4, 1)
    ).squeeze(-1).squeeze(-1)  # (N, )

def compute_qem_loss(
    mesh: Meshes, 
    threshold: Optional[float] = None, 
    reduce: bool = True, 
    use_least_squares: bool = False,
    normalization_factor: float = 1.,
    average_w_face_area: bool = True,
    use_faster_qem: bool = True,
):
    
    # Compute v_bar and quadrics 
    # Using robust code from: https://github.com/nissmar/PoNQ/blob/main/src/utils/PoNQ_to_mesh.py

    # Compute the error quadrics per vertex
    Q = get_error_quadrics(
        mesh,
        average_w_face_area=average_w_face_area,
    ).nan_to_num(0.)  # (V, 4, 4)
    
    if torch.isnan(Q).any() or torch.isinf(Q).any():
        is_nan_mask = torch.isnan(Q)
        is_inf_mask = torch.isinf(Q)
        print(f"\n[DEBUG] Q has {is_nan_mask.sum().item()} / {Q.flatten().shape[0]} nan values")
        print(f"[DEBUG] Q has {is_inf_mask.sum().item()} / {Q.flatten().shape[0]} inf values")
    
    # Compute the error quadrics per edge
    if use_faster_qem:
        scores = (
            quadrics_score(Q[mesh.edges[:, 0]], mesh.verts[mesh.edges[:, 1]])
            + quadrics_score(Q[mesh.edges[:, 1]], mesh.verts[mesh.edges[:, 0]])
        )  # (E,)
    else:
        Q = Q[mesh.edges[:, 0]] + Q[mesh.edges[:, 1]]  # (E, 4, 4)
        
        # Compute the v_stars
        v1 = mesh.verts[mesh.edges[:, 0]]  # (E, 3)
        v2 = mesh.verts[mesh.edges[:, 1]]  # (E, 3)
        edge_midpoints = (v1 + v2) / 2.  # (E, 3)
        edge_midpoints = edge_midpoints.nan_to_num(0.)  # (E, 3)
        if use_least_squares:
            v_stars = vstars_from_quadrics_least_squares(Q)  # (E, 3)
        else:        
            v_stars, _, _ = vstars_from_quadrics(Q, edge_midpoints)  # (E, 3)
            
        if torch.isnan(v_stars).any() or torch.isinf(v_stars).any():
            is_nan_mask = torch.isnan(v_stars)
            is_inf_mask = torch.isinf(v_stars)
            print(f"[DEBUG] v_stars has {is_nan_mask.sum().item()} / {v_stars.flatten().shape[0]} nan values")
            print(f"[DEBUG] v_stars has {is_inf_mask.sum().item()} / {v_stars.flatten().shape[0]} inf values")

        # The quantity vT(Q1 +Q2 )v is the cost of contracting the corresponding edge.
        # Compute the cost for all edges
        # (E,)
        scores = quadrics_score(Q, v_stars)  # (E,)
    
    scores = scores / (normalization_factor ** 2)  # (E,)
    # edge_costs = edge_costs[solvable_mask] # We only consider solvable edges
    
    if torch.isnan(scores).any() or torch.isinf(scores).any():
        is_nan_mask = torch.isnan(scores)
        is_inf_mask = torch.isinf(scores)
        print(f"[DEBUG] scores has {is_nan_mask.sum().item()} / {scores.flatten().shape[0]} nan values")
        print(f"[DEBUG] scores has {is_inf_mask.sum().item()} / {scores.flatten().shape[0]} inf values")
    
    if threshold is not None:
        # We could threshold the cost so that only low cost edges are promoted to contract.
        raise NotImplementedError("Threshold not implemented")

    # Compute the loss
    loss = scores.mean() if reduce else scores.squeeze()
    return loss

def return_delaunay_tets(points: torch.Tensor, method: str) -> torch.Tensor:
    if method == "tetranerf":
        with torch.no_grad():
            return cpp.triangulate(points.detach()).cuda().long()
    elif method == "scipy":
        # Slower but easier to install
        return torch.from_numpy(
            Delaunay(points.detach().cpu().numpy()).simplices.astype(np.int32)
        ).to(points.device).long()
    else:
        raise ValueError(f"Invalid method: {method}")

def v_star_attraction_loss(
    mesh: Meshes, 
    points: torch.Tensor, 
    interpolation_vertices: torch.Tensor, 
    reduce: bool = True, 
    random_batch_size: int = None,
    use_least_squares: bool = False,
    normalization_factor: float = 1.,
    average_w_face_area: bool = True,
) -> torch.Tensor:
    """
    Compute the v_star attraction loss for a mesh.
    """
    # Compute the per-vertex error quadrics
    error_quadrics = get_error_quadrics(
        mesh,
        average_w_face_area=average_w_face_area,
    )  # (V, 4, 4)
    
    # Compute the per-edge error quadrics
    v1_index = mesh.edges[:,0]
    v2_index = mesh.edges[:,1]
    Q = error_quadrics[v1_index] + error_quadrics[v2_index]  # (E, 4, 4)
    
    # Compute the v_stars
    v1 = mesh.verts[v1_index]  # (E, 3)
    v2 = mesh.verts[v2_index]  # (E, 3)
    edge_midpoints = (v1 + v2) / 2.  # (E, 3)
    if use_least_squares:
        v_stars = vstars_from_quadrics_least_squares(Q)  # (E, 3)
    else:
        v_stars, _, _ = vstars_from_quadrics(Q, edge_midpoints)  # (E, 3)
    
    
    if torch.isnan(v_stars).any() or torch.isinf(v_stars).any():
        is_nan_mask = torch.isnan(v_stars) | torch.isinf(v_stars)
        print(f"[DEBUG] nan mask: {is_nan_mask.shape}")
        raise ValueError("v_stars is nan")
    
    # Compute the v_star hat
    t_star_p = points[interpolation_vertices[v1_index, 0]] + points[interpolation_vertices[v2_index, 0]]
    t_star_n = points[interpolation_vertices[v1_index, 1]] + points[interpolation_vertices[v2_index, 1]]
    v_star_hat = (t_star_p + t_star_n) / 2.
    
    # Compute the loss
    loss = (v_stars - v_star_hat).norm(dim=-1) / normalization_factor
    if random_batch_size is not None:
        loss = loss[torch.randperm(loss.shape[0])[:random_batch_size]]

    return loss.mean() if reduce else loss


def attract_pivots_along_mesh_edges_loss_fn(
    mesh: Meshes,
    pivots: torch.Tensor,
    interpolation_vertices: torch.Tensor,
    distance_normalization_factor: float,
    reduce: bool = True,
    weight_by_qem: bool = True,
    use_least_squares: bool = False,
    qem_normalization_factor: float = 1.,
    average_w_face_area: bool = True,
    use_faster_qem: bool = True,
) -> torch.Tensor:
    """
    Loss to attract the pivots/voronoi points along the mesh edges.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # Get mesh edges
    edges_u_vertices = mesh.edges[:, 0]  # Shape (E,) with values in [0, N_vertices-1]
    edges_v_vertices = mesh.edges[:, 1]  # Shape (E,) with values in [0, N_vertices-1]

    # Lift mesh edges to edges between voronoi points
    edges_u_positives = interpolation_vertices[edges_u_vertices, 0].unsqueeze(-1)  # (E, 1)
    edges_v_positives = interpolation_vertices[edges_v_vertices, 0].unsqueeze(-1)  # (E, 1)
    edges_u_negatives = interpolation_vertices[edges_u_vertices, 1].unsqueeze(-1)  # (E, 1)
    edges_v_negatives = interpolation_vertices[edges_v_vertices, 1].unsqueeze(-1)  # (E, 1)

    edges_u = torch.cat([edges_u_positives, edges_u_negatives], dim=1).view(-1)  # (E * 2,)
    edges_v = torch.cat([edges_v_positives, edges_v_negatives], dim=1).view(-1)  # (E * 2,)
    
    # Compute scores.
    # Should we normalize by using the minimum scales of the corresponding Gaussians?
    scores = torch.norm(
        pivots[edges_u] - pivots[edges_v], 
        dim=-1
    ) / distance_normalization_factor  # (E * 2,)
    
    if weight_by_qem:
        with torch.no_grad():
            # Compute the per-vertex error quadrics
            error_quadrics = get_error_quadrics(
                mesh,
                average_w_face_area=average_w_face_area,
            )  # (V, 4, 4)
            
            if use_faster_qem:
                qem_scores = (
                    quadrics_score(error_quadrics[mesh.edges[:, 0]], mesh.verts[mesh.edges[:, 1]])
                    + quadrics_score(error_quadrics[mesh.edges[:, 1]], mesh.verts[mesh.edges[:, 0]])
                )  # (E,)
            else:
                # Compute the per-edge error quadrics
                v1_index = mesh.edges[:,0]
                v2_index = mesh.edges[:,1]
                
                Q = error_quadrics[v1_index] + error_quadrics[v2_index]  # (E, 4, 4)
                
                # Compute v_stars for all edges
                v1 = mesh.verts[mesh.edges[:, 0]]  # (E, 3)
                v2 = mesh.verts[mesh.edges[:, 1]]  # (E, 3)
                edge_midpoints = (v1 + v2) / 2.  # (E, 3)
                if use_least_squares:
                    v_stars = vstars_from_quadrics_least_squares(Q)  # (E, 3)
                else:
                    v_stars, _, _ = vstars_from_quadrics(Q, edge_midpoints)  # (E, 3)
                    
                # Compute quadrics scores for all edges
                qem_scores = quadrics_score(Q, v_stars)  # (E,)
                
            qem_scores = qem_scores / (qem_normalization_factor ** 2)  # (E,)
            
            # Lift scores to edges between voronoi points
            qem_scores = torch.cat([qem_scores, qem_scores], dim=0)  # (E * 2,)
        
        # Weight scores by QEM scores
        # TODO: We should work on this weighting function
        scores = scores * torch.exp(-qem_scores)
        
    return scores.mean() if reduce else scores


class RasterizationSettings():
    """
    Rasterization settings for meshes.
    """
    def __init__(
        self, 
        image_size=(1080, 1920),
        blur_radius=0.0,
        faces_per_pixel=1,
        ):
        self.image_size = image_size
        self.blur_radius = blur_radius
        self.faces_per_pixel = faces_per_pixel


class Fragments():
    def __init__(self, bary_coords, zbuf, pix_to_face):
        self.bary_coords = bary_coords  # Shape (1, height, width, 1, 3)
        self.zbuf = zbuf  # Shape (1, height, width, 1)
        self.pix_to_face = pix_to_face  # Shape (1, height, width, 1)


class MeshRasterizer(torch.nn.Module):
    """
    Class for rasterizing meshes with NVDiffRast.
    """
    def __init__(
        self, 
        cameras:Union[List[Camera], Camera]=None,
        raster_settings:RasterizationSettings=None,
        use_opengl=True,
    ):
        super().__init__()
        
        if cameras is None:
            if raster_settings is None:
                raster_settings = RasterizationSettings()
            self.raster_settings = raster_settings
            self.height, self.width = raster_settings.image_size
            self.cameras = None
        else:
            if isinstance(cameras, Camera):
                cameras = [cameras]
            # Get height and width if provided in cameras
            self.height = cameras[0].image_height
            self.width = cameras[0].image_width
            self.raster_settings = RasterizationSettings(
                image_size=(self.height, self.width),
            )
            self.cameras = cameras
        
        if use_opengl:
            self.gl_context = dr.RasterizeGLContext()
        else:
            self.gl_context = dr.RasterizeCudaContext()
            
    def forward(
        self, 
        mesh:Meshes, 
        cameras:List[Camera]=None,
        cam_idx:int=0,
        return_only_pix_to_face:bool=False,
        return_rast_out:bool=False,
        return_positions:bool=False,
    ):
        if cameras is None:
            if self.cameras is None:
                raise ValueError("cameras must be provided either in the constructor or in the forward method")
            cameras = self.cameras
        
        if isinstance(cameras, Camera):
            render_camera = cameras
        else:
            render_camera = cameras[cam_idx]

        height, width = render_camera.image_height, render_camera.image_width
        nvdiff_rast_out = nvdiff_rasterization(
            camera=render_camera,
            image_height=height, 
            image_width=width,
            verts=mesh.verts,
            faces=mesh.faces,
            return_indices_only=False,
            glctx=self.gl_context,
            return_rast_out=return_rast_out,
            return_positions=return_positions,
        )
        bary_coords, zbuf, pix_to_face = nvdiff_rast_out[:3]
        if return_rast_out:
            rast_out = nvdiff_rast_out[3]
        if return_positions:
            pos = nvdiff_rast_out[4]
        
        if return_only_pix_to_face:
            return pix_to_face.view(1, height, width, 1)
        bary_coords = torch.cat([bary_coords, 1. - bary_coords.sum(dim=-1, keepdim=True)], dim=-1)
        
        # TODO: Zbuf is still in NDC space, should convert to camera space
        fragments = Fragments(
            bary_coords.view(1, height, width, 1, 3),
            zbuf.view(1, height, width, 1),
            pix_to_face.view(1, height, width, 1),
        )
        _output = (fragments,)
        if return_rast_out:
            _output = _output + (rast_out,)
        if return_positions:
            _output = _output + (pos,)
        return _output
        
    
class MeshRenderer(torch.nn.Module):
    """
    Class for rendering meshes with NVDiffRast and a shader.
    """
    def __init__(self, rasterizer:MeshRasterizer):
        super().__init__()
        self.rasterizer = rasterizer
        # TODO: Add shader
        
    def forward(
        self, 
        mesh:Meshes, 
        cameras:Union[List[Camera], Camera]=None, 
        cam_idx=0,
        return_depth=False,
        return_normals=False,
        use_antialiasing=True,
        return_pix_to_face=False,
        check_errors=True,
    ):
        fragments, rast_out, pos = self.rasterizer(mesh, cameras, cam_idx, return_rast_out=True, return_positions=True)
        if cameras is None:
            cameras = self.rasterizer.cameras
        if isinstance(cameras, Camera):
            cameras = [cameras]
        
        return_colors = mesh.verts_colors is not None

        output_pkg = {}
        
        # Compute per-vertex features to render
        features = torch.zeros(mesh.verts.shape[0], 0, device=mesh.verts.device)        
        
        if return_depth:
            depth_idx = features.shape[-1]
            verts_depth = transform_points_world_to_view(mesh.verts, [cameras[cam_idx]])[..., 2].squeeze()  # Shape (N, )
            features = torch.cat([features, verts_depth.view(mesh.verts.shape[0], 1)], dim=-1)
            
        if return_colors:
            color_idx = features.shape[-1]
            features = torch.cat([features, mesh.verts_colors], dim=-1)  # Shape (N, n_features)
        
        # Compute image
        feature_img, _ = dr.interpolate(features[None], rast_out, mesh.faces)  # Shape (1, H, W, n_features)
        
        # Antialiasing for propagating gradients
        if use_antialiasing:
            feature_img = dr.antialias(feature_img, rast_out, pos, mesh.faces)  # Shape (1, H, W, n_features)
        
        if return_depth:
            output_pkg["depth"] = feature_img[..., depth_idx:depth_idx+1]  # Shape (1, H, W)
        if return_colors:
            output_pkg["rgb"] = feature_img[..., color_idx:color_idx+3]  # Shape (1, H, W, 3)
            
        # Compute per-face normals
        if return_normals:
            valid_mask = fragments.pix_to_face >= 0  # Shape (1, H, W, 1)
            if check_errors:
                error_mask = fragments.pix_to_face >= mesh.faces.shape[0]
                error_encountered = torch.sum(error_mask)
                if error_encountered > 0:
                    print(f"[WARNING] Rasterized {error_encountered} pixels with invalid triangle index.")
                    fragments.pix_to_face = torch.clamp(fragments.pix_to_face, min=0, max=mesh.faces.shape[0] - 1)
                    valid_mask = valid_mask & ~error_mask
            output_pkg["normals"] = mesh.face_normals[fragments.pix_to_face].squeeze()[None] * valid_mask  # Shape (1, H, W, 3)
            # if use_antialiasing:
            #     output_pkg["normals"] = dr.antialias(output_pkg["normals"], rast_out, pos, mesh.faces)  # Shape (1, H, W, 3)
            
        if return_pix_to_face:
            output_pkg["pix_to_face"] = fragments.pix_to_face

        return output_pkg


def fuse_fragments(fragments1:Fragments, fragments2:Fragments):
    raster_mask1 = fragments1.pix_to_face > -1
    raster_mask2 = fragments2.pix_to_face > -1
    
    # raster_mask1 = fragments1.zbuf > 0.
    # raster_mask2 = fragments2.zbuf > 0.
    
    no_raster_mask = (~raster_mask1) & (~raster_mask2)
    
    zbuf1 = torch.where(raster_mask1, fragments1.zbuf, 1000.)
    zbuf2 = torch.where(raster_mask2, fragments2.zbuf, 1000.)
    
    all_zbufs = torch.cat([zbuf1[..., None], zbuf2[..., None]], dim=-1)  # Shape (1, H, W, 1, 2)
    zbuf, argzbuf = torch.min(all_zbufs, dim=-1)  # argzbuf is of shape (1, H, W, 1)
    zbuf[no_raster_mask] = 0.
    
    all_pix_to_face = torch.cat(
        [
            fragments1.pix_to_face[..., None], 
            fragments2.pix_to_face[..., None]
        ], 
        dim=-1
    )  # Shape (1, H, W, 1, 2)
    pix_to_face = torch.gather(
        all_pix_to_face, 
        dim=-1, 
        index=argzbuf[..., None]
    )[..., 0]  # Shape (1, H, W, 1)
    pix_to_face[no_raster_mask] = -1
    
    all_bary_coords = torch.cat([fragments1.bary_coords[..., None], fragments2.bary_coords[..., None]], dim=-1)
    bary_coords = torch.gather(
        all_bary_coords, 
        dim=-1, 
        index=argzbuf[..., None, None].expand(-1, -1, -1, -1, 3, -1)
    )[..., 0]  # Shape (1, H, W, 1, 3)

    return Fragments(
        bary_coords,
        zbuf,
        pix_to_face,
    )

    
class ScalableMeshRenderer(torch.nn.Module):
    """
    Class for rendering big meshes with NVDiffRast.
    """
    def __init__(self, rasterizer:MeshRasterizer):
        super().__init__()
        self.rasterizer = rasterizer
        # TODO: Add shader
        
    def forward(
        self, 
        mesh:Meshes, 
        cameras:Union[List[Camera], Camera]=None, 
        cam_idx:int=0,
        return_depth:bool=False,
        return_normals:bool=False,
        use_antialiasing:bool=True,
        return_pix_to_face:bool=False,
        check_errors:bool=True,
        max_triangles_in_batch:int=2**22  # Reduced from 2**24 to avoid subtriangle overflow
    ):
        n_passes = (mesh.faces.shape[0] + max_triangles_in_batch - 1) // max_triangles_in_batch
        
        fragments = None
        idx_shift = 0
        for i_pass in range(n_passes):
            start_idx = i_pass * max_triangles_in_batch
            end_idx = min(start_idx + max_triangles_in_batch, mesh.faces.shape[0])
            
            # Compute submesh
            sub_faces = mesh.faces[start_idx:end_idx]
            sub_mesh = Meshes(verts=mesh.verts, faces=sub_faces)
            
            # Rasterize submesh
            _fragments, _, pos = self.rasterizer(sub_mesh, cameras, cam_idx, return_rast_out=True, return_positions=True)
            
            # Combine fragments
            if fragments is None:
                fragments = _fragments
            else:
                # _fragments.pix_to_face = _fragments.pix_to_face + idx_shift
                _fragments.pix_to_face = torch.where(
                    _fragments.pix_to_face > -1, 
                    _fragments.pix_to_face + idx_shift, 
                    _fragments.pix_to_face
                )
                fragments = fuse_fragments(fragments, _fragments)
                # fragments = _fragments
            
            # Update idx shift
            idx_shift = idx_shift + len(sub_faces)
        
        # Filter mesh and fragments to keep only rasterized faces. This will decrease the number of faces to at most H * W.
        # Reducing the number of faces is necessary to avoid errors when running dr.interpolate and dr.antialias
        if True:
            filtered_face_idx, filtered_pix_to_face = fragments.pix_to_face.unique(return_inverse=True)
            filtered_face_idx = filtered_face_idx[1:]
            filtered_faces = mesh.faces[filtered_face_idx]
            filtered_pix_to_face = filtered_pix_to_face - 1
            mesh = Meshes(verts=mesh.verts, faces=filtered_faces, verts_colors=mesh.verts_colors)
            fragments.pix_to_face = filtered_pix_to_face
        
        # Rebuild rast_out
        rast_out = torch.zeros(*fragments.zbuf.shape[:-1], 4, device=fragments.zbuf.device)
        rast_out[..., :2] = fragments.bary_coords[..., 0, :2]
        rast_out[..., 2:3] = fragments.zbuf
        rast_out[..., 3:4] = fragments.pix_to_face.float() + 1
        
        if cameras is None:
            cameras = self.rasterizer.cameras
        if isinstance(cameras, Camera):
            cameras = [cameras]
        
        return_colors = mesh.verts_colors is not None

        output_pkg = {}
        
        # Compute per-vertex features to render
        features = torch.zeros(mesh.verts.shape[0], 0, device=mesh.verts.device)
        
        if return_depth:
            depth_idx = features.shape[-1]
            verts_depth = transform_points_world_to_view(mesh.verts, [cameras[cam_idx]])[..., 2].squeeze()  # Shape (N, )
            features = torch.cat([features, verts_depth.view(mesh.verts.shape[0], 1)], dim=-1)
            
        if return_colors:
            color_idx = features.shape[-1]
            features = torch.cat([features, mesh.verts_colors], dim=-1)  # Shape (N, n_features)
        
        # Compute image
        if True:
            feature_img, _ = dr.interpolate(features[None], rast_out, mesh.faces)  # Shape (1, H, W, n_features)
        else:
            pix_to_verts = mesh.faces[fragments.pix_to_face]  # Shape (1, H, W, 1, 3)
            pix_to_features = features[pix_to_verts]  # Shape (1, H, W, 1, 3, n_features)
            feature_img = (pix_to_features * fragments.bary_coords[..., None]).sum(dim=-2)  # Shape (1, H, W, 1, n_features)
            feature_img = feature_img.squeeze(-2)  # Shape (1, H, W, n_features)
        
        # Antialiasing for propagating gradients
        if use_antialiasing:
            feature_img = dr.antialias(feature_img, rast_out, pos, mesh.faces)  # Shape (1, H, W, n_features)
        
        if return_depth:
            output_pkg["depth"] = feature_img[..., depth_idx:depth_idx+1]  # Shape (1, H, W)

        if return_colors:
            output_pkg["rgb"] = feature_img[..., color_idx:color_idx+3]  # Shape (1, H, W, 3)
            
        # Compute per-face normals
        if return_normals:
            valid_mask = fragments.pix_to_face >= 0  # Shape (1, H, W, 1)
            if check_errors:
                error_mask = fragments.pix_to_face >= mesh.faces.shape[0]
                error_encountered = torch.sum(error_mask)
                if error_encountered > 0:
                    print(f"[WARNING] Rasterized {error_encountered} pixels with invalid triangle index.")
                    fragments.pix_to_face = torch.clamp(fragments.pix_to_face, min=0, max=mesh.faces.shape[0] - 1)
                    valid_mask = valid_mask & ~error_mask
            output_pkg["normals"] = mesh.face_normals[fragments.pix_to_face].squeeze()[None] * valid_mask  # Shape (1, H, W, 3)
            # if use_antialiasing:
            #     output_pkg["normals"] = dr.antialias(output_pkg["normals"], rast_out, pos, mesh.faces)  # Shape (1, H, W, 3)
            
        if return_pix_to_face:
            output_pkg["pix_to_face"] = fragments.pix_to_face
            
        #### TO REMOVE
        output_pkg["fragments"] = fragments
        output_pkg["rast_out"] = rast_out
        #### TO REMOVE

        return output_pkg