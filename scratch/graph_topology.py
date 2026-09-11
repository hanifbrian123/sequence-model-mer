"""
graph_topology.py — Topologi graf wajah berdasarkan MediaPipe Face Mesh.

Menggunakan koneksi FACEMESH_TESSELATION yang sesungguhnya (~2.900 edges)
untuk membangun adjacency matrix yang merepresentasikan struktur spasial wajah
secara akurat. Mendukung multi-hop adjacency untuk GCN orde tinggi.
"""

import numpy as np
import torch
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import mediapipe.solutions.face_mesh as mp_face_mesh
# MediaPipe FACEMESH_TESSELATION edges (478 landmarks, ~2900 triangular edges)
# Sumber: mediapipe.solutions.face_mesh.FACEMESH_TESSELATION
# Import secara dinamis untuk menghindari dependency issue saat import modul
def _get_mediapipe_tesselation_edges():
    """Mengambil edge list dari MediaPipe FACEMESH_TESSELATION."""
    try:
        edges = set()
        for connection in mp_face_mesh.FACEMESH_TESSELATION:
            i, j = connection
            edges.add((i, j))
            edges.add((j, i))
        return list(edges)
    except Exception:
        try:
            edges = set()
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                i, j = connection
                edges.add((i, j))
                edges.add((j, i))
            return list(edges)
        except Exception:
            # Fallback jika mediapipe gagal di-load karena TF DLL
            edges = set()
            for i in range(468 - 1):
                edges.add((i, i + 1))
                edges.add((i + 1, i))
            return list(edges)


def _get_mediapipe_contour_edges():
    """Mengambil edge list dari MediaPipe FACEMESH_CONTOURS (mata, bibir, alis)."""
    try:
        edges = set()
        for connection in mp_face_mesh.FACEMESH_CONTOURS:
            i, j = connection
            edges.add((i, j))
            edges.add((j, i))
        return list(edges)
    except Exception:
        try:
            edges = set()
            for connection in mp_face_mesh.FACEMESH_CONTOURS:
                i, j = connection
                edges.add((i, j))
                edges.add((j, i))
            return list(edges)
        except Exception:
            return []


class FaceMeshTopology:
    """
    Membangun adjacency matrix dari topologi wajah MediaPipe Face Mesh.
    
    Fitur:
    - Edges dari FACEMESH_TESSELATION + FACEMESH_CONTOURS
    - Self-loops untuk stabilitas GCN
    - Symmetric normalized adjacency: D^(-1/2) * A * D^(-1/2)
    - Multi-hop adjacency (A^k) untuk menangkap relasi jarak jauh
    
    Args:
        num_nodes: Jumlah node (468 untuk MediaPipe tanpa refined, 478 dengan refined)
        use_contours: Jika True, tambahkan edges dari FACEMESH_CONTOURS
        add_self_loops: Jika True, tambahkan self-loop (diagonal = 1)
    """
    
    def __init__(
        self,
        num_nodes: int = 468,
        use_contours: bool = True,
        add_self_loops: bool = True,
    ):
        self.num_nodes = num_nodes
        
        # 1. Kumpulkan semua edges
        all_edges = _get_mediapipe_tesselation_edges()
        if use_contours:
            all_edges.extend(_get_mediapipe_contour_edges())
        
        # 2. Filter edges yang dalam range num_nodes
        filtered_edges = [(i, j) for i, j in all_edges if i < num_nodes and j < num_nodes]
        self.num_edges = len(set(filtered_edges))
        
        # 3. Bangun adjacency matrix
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i, j in filtered_edges:
            A[i, j] = 1.0
            A[j, i] = 1.0  # Symmetric
        
        # 4. Tambah self-loops
        if add_self_loops:
            A = A + np.eye(num_nodes, dtype=np.float32)
        
        # 5. Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.zeros_like(D)
        D_inv_sqrt[D > 0] = np.power(D[D > 0], -0.5)
        D_mat = np.diag(D_inv_sqrt)
        A_norm = D_mat @ A @ D_mat
        
        # 6. Convert ke PyTorch tensor
        self.A_raw = torch.tensor(A, dtype=torch.float32)
        self.A_tensor = torch.tensor(A_norm, dtype=torch.float32)
    
    def get_multihop_adjacency(self, k: int = 2) -> torch.Tensor:
        """
        Menghasilkan adjacency matrix multi-hop (A^k normalized).
        
        Berguna untuk menangkap relasi antara node yang berjarak >1 hop.
        Misalnya k=2 menghubungkan node yang berjarak 2 langkah.
        
        Args:
            k: Jumlah hop
            
        Returns:
            Normalized A^k tensor
        """
        A = self.A_raw.numpy()
        
        # Hitung A^k
        A_k = np.linalg.matrix_power(A, k)
        
        # Binarize (>0 → 1)
        A_k = (A_k > 0).astype(np.float32)
        
        # Normalize
        D = np.sum(A_k, axis=1)
        D_inv_sqrt = np.zeros_like(D)
        D_inv_sqrt[D > 0] = np.power(D[D > 0], -0.5)
        D_mat = np.diag(D_inv_sqrt)
        A_k_norm = D_mat @ A_k @ D_mat
        
        return torch.tensor(A_k_norm, dtype=torch.float32)
    
    def get_stacked_adjacency(self, max_hop: int = 2) -> torch.Tensor:
        """
        Menghasilkan stacked adjacency matrices [max_hop, V, V].
        Setiap slice adalah A^k normalized untuk k = 1..max_hop.
        
        Berguna untuk multi-scale graph convolution.
        """
        adjacencies = []
        for k in range(1, max_hop + 1):
            if k == 1:
                adjacencies.append(self.A_tensor)
            else:
                adjacencies.append(self.get_multihop_adjacency(k))
        
        return torch.stack(adjacencies, dim=0)


def build_face_topology(
    num_nodes: int = 468,
    device: str | torch.device = "cpu",
) -> FaceMeshTopology:
    """
    Factory function untuk membangun topologi graf wajah.
    
    Args:
        num_nodes: Jumlah node landmark (468 atau 478)
        device: Device target untuk tensor
    
    Returns:
        FaceMeshTopology instance dengan tensor di device yang ditentukan
    """
    topo = FaceMeshTopology(num_nodes=num_nodes)
    topo.A_tensor = topo.A_tensor.to(device)
    topo.A_raw = topo.A_raw.to(device)
    
    print(f"[TOPOLOGY] Graf wajah MediaPipe siap!")
    print(f"  Nodes: {topo.num_nodes}")
    print(f"  Edges: {topo.num_edges}")
    print(f"  Adjacency shape: {topo.A_tensor.shape}")
    
    return topo
