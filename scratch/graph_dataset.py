"""
graph_dataset.py — Dataset wrapper untuk konversi pixel image ke graph representation.

Mengekstrak fitur node yang kaya dari MediaPipe Face Mesh landmarks:
  - Channel 0-1: Koordinat (x, y) normalized
  - Channel 2-3: Velocity (dx, dy) — displacement antar frame
  - Channel 4: Jarak ke centroid wajah
  - Channel 5-6: Fitur tekstur lokal (mean & std patch piksel)
  - Channel 7: Sudut relatif terhadap centroid
  - Channel 8: Displacement magnitude dari frame-0 (onset)

Mendukung graph augmentation untuk regularisasi.
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Jumlah fitur channel per node
NUM_GRAPH_CHANNELS = 9


class GraphDatasetWrapper(Dataset):
    """
    Wrapper dataset yang mengkonversi pixel-based ImageToVideoDataset
    menjadi graph-based representation menggunakan MediaPipe Face Mesh.
    
    Args:
        pixel_dataset: Dataset yang mengembalikan (video_tensor, label)
        num_nodes: Jumlah landmark wajah (468 untuk MediaPipe)
        patch_radius: Radius patch untuk fitur tekstur lokal
        augment: Jika True, terapkan graph augmentation saat training
        jitter_std: Standar deviasi untuk coordinate jittering augmentation
        node_drop_rate: Probabilitas drop node saat augmentation
    """
    
    def __init__(
        self,
        pixel_dataset: Dataset,
        num_nodes: int = 468,
        patch_radius: int = 3,
        augment: bool = False,
        jitter_std: float = 0.005,
        node_drop_rate: float = 0.05,
    ):
        self.pixel_dataset = pixel_dataset
        self.num_nodes = num_nodes
        self.patch_radius = patch_radius
        self.augment = augment
        self.jitter_std = jitter_std
        self.node_drop_rate = node_drop_rate
        
        # Inisialisasi MediaPipe Face Mesh
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.3,
            )
        except Exception as e:
            print(f"⚠️ MediaPipe tidak tersedia: {e}")
            self.mp_face_mesh = None
        
        # Propagate labels
        if hasattr(pixel_dataset, 'labels'):
            self.labels = pixel_dataset.labels
        elif hasattr(pixel_dataset, 'datasets'):
            self.labels = []
            for ds in pixel_dataset.datasets:
                if hasattr(ds, 'labels'):
                    self.labels.extend(ds.labels)
    
    def __len__(self) -> int:
        return len(self.pixel_dataset)
    
    def _extract_landmarks(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Mengekstrak koordinat landmark dari satu frame.
        
        Returns:
            Array [num_nodes, 2] berisi koordinat (x, y) normalized [0, 1]
        """
        landmarks = np.zeros((self.num_nodes, 2), dtype=np.float32)
        
        if self.mp_face_mesh is None:
            # Fallback: sinusoidal pattern
            base_x = np.linspace(0.1, 0.9, self.num_nodes)
            landmarks[:, 0] = base_x
            landmarks[:, 1] = np.sin(base_x * np.pi) * 0.4 + 0.5
            return landmarks
        
        results = self.mp_face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for i, lm in enumerate(results.multi_face_landmarks[0].landmark):
                if i >= self.num_nodes:
                    break
                landmarks[i, 0] = lm.x
                landmarks[i, 1] = lm.y
        else:
            # Fallback jika wajah tidak terdeteksi
            base_x = np.linspace(0.1, 0.9, self.num_nodes)
            landmarks[:, 0] = base_x
            landmarks[:, 1] = np.sin(base_x * np.pi) * 0.4 + 0.5
        
        return landmarks
    
    def _extract_texture_features(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Mengekstrak fitur tekstur lokal di sekitar setiap landmark.
        
        Returns:
            (mean_features, std_features): Masing-masing array [num_nodes]
        """
        r = self.patch_radius
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        means = np.zeros(self.num_nodes, dtype=np.float32)
        stds = np.zeros(self.num_nodes, dtype=np.float32)
        
        for v in range(self.num_nodes):
            x_pixel = int(np.clip(landmarks[v, 0] * (width - 1), 0, width - 1))
            y_pixel = int(np.clip(landmarks[v, 1] * (height - 1), 0, height - 1))
            
            # Extract patch
            y_min = max(0, y_pixel - r)
            y_max = min(height, y_pixel + r + 1)
            x_min = max(0, x_pixel - r)
            x_max = min(width, x_pixel + r + 1)
            
            patch = gray[y_min:y_max, x_min:x_max]
            if patch.size > 0:
                means[v] = np.mean(patch)
                stds[v] = np.std(patch)
        
        return means, stds
    
    def _build_graph_features(
        self,
        video_tensor: torch.Tensor,
    ) -> np.ndarray:
        """
        Mengkonversi video tensor ke graph features [C, T, V].
        
        C = NUM_GRAPH_CHANNELS (9 fitur)
        T = jumlah frame
        V = jumlah node
        """
        channels, num_frames, height, width = video_tensor.shape
        
        # Konversi ke numpy uint8 RGB
        np_video = (video_tensor.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)
        
        # Step 1: Ekstrak landmarks untuk semua frame
        all_landmarks = np.zeros((num_frames, self.num_nodes, 2), dtype=np.float32)
        for f in range(num_frames):
            frame_rgb = np_video[f]
            if channels == 1 or frame_rgb.shape[-1] == 1:
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
            elif frame_rgb.shape[-1] == 4:
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2RGB)
            all_landmarks[f] = self._extract_landmarks(frame_rgb)
        
        # Step 2: Hitung centroid per frame
        centroids = np.mean(all_landmarks, axis=1)  # [T, 2]
        
        # Step 3: Bangun fitur 9-channel
        graph_data = np.zeros((NUM_GRAPH_CHANNELS, num_frames, self.num_nodes), dtype=np.float32)
        
        for f in range(num_frames):
            frame_rgb = np_video[f]
            if frame_rgb.shape[-1] != 3:
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB) if frame_rgb.shape[-1] == 1 \
                    else cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2RGB)
            
            lm = all_landmarks[f]  # [V, 2]
            cx, cy = centroids[f]
            
            # Channel 0-1: Koordinat normalized
            graph_data[0, f, :] = lm[:, 0]
            graph_data[1, f, :] = lm[:, 1]
            
            # Channel 2-3: Velocity (dx, dy) — displacement dari frame sebelumnya
            if f > 0:
                prev_lm = all_landmarks[f - 1]
                graph_data[2, f, :] = lm[:, 0] - prev_lm[:, 0]
                graph_data[3, f, :] = lm[:, 1] - prev_lm[:, 1]
            # else: 0 untuk frame pertama
            
            # Channel 4: Jarak ke centroid (euclidean, normalized)
            dx = lm[:, 0] - cx
            dy = lm[:, 1] - cy
            dist = np.sqrt(dx ** 2 + dy ** 2)
            max_dist = dist.max() if dist.max() > 0 else 1.0
            graph_data[4, f, :] = dist / max_dist
            
            # Channel 5-6: Fitur tekstur lokal
            means, stds = self._extract_texture_features(frame_rgb, lm, height, width)
            graph_data[5, f, :] = means
            graph_data[6, f, :] = stds
            
            # Channel 7: Sudut relatif terhadap centroid
            angles = np.arctan2(dy, dx) / np.pi  # Normalize ke [-1, 1]
            graph_data[7, f, :] = angles
            
            # Channel 8: Displacement magnitude dari frame-0 (onset)
            if f > 0:
                onset_lm = all_landmarks[0]
                disp_x = lm[:, 0] - onset_lm[:, 0]
                disp_y = lm[:, 1] - onset_lm[:, 1]
                graph_data[8, f, :] = np.sqrt(disp_x ** 2 + disp_y ** 2)
        
        return graph_data
    
    def _apply_augmentation(self, graph_data: np.ndarray) -> np.ndarray:
        """
        Terapkan graph augmentation untuk regularisasi.
        
        - Coordinate jittering: Tambah noise Gaussian ke koordinat
        - Node drop: Set fitur beberapa node ke 0
        """
        if not self.augment:
            return graph_data
        
        data = graph_data.copy()
        C, T, V = data.shape
        
        # 1. Coordinate jittering pada channel 0-1 (x, y)
        noise = np.random.normal(0, self.jitter_std, (2, T, V)).astype(np.float32)
        data[0:2] += noise
        
        # 2. Node drop: random set fitur node ke 0
        if self.node_drop_rate > 0:
            drop_mask = np.random.random(V) < self.node_drop_rate
            data[:, :, drop_mask] = 0.0
        
        return data
    
    def __getitem__(self, idx: int):
        video_tensor, label = self.pixel_dataset[idx]
        
        # Bangun fitur graf
        graph_data = self._build_graph_features(video_tensor)
        
        # Augmentation
        graph_data = self._apply_augmentation(graph_data)
        
        return torch.tensor(graph_data, dtype=torch.float32), label


class PrecomputedGraphDataset(Dataset):
    """
    Dataset yang membaca graf yang sudah di-precompute dari disk.
    
    Setiap file .npy berisi dict {"graph": [C, T, V], "label": int}.
    
    Args:
        precomputed_dir: Direktori berisi file sample_*.npy
        total_samples: Jumlah total sampel
        augment: Jika True, terapkan augmentation
        jitter_std: Noise std untuk coordinate jitter
        node_drop_rate: Rate drop node
    """
    
    def __init__(
        self,
        precomputed_dir: str,
        total_samples: int,
        augment: bool = False,
        jitter_std: float = 0.005,
        node_drop_rate: float = 0.05,
    ):
        self.precomputed_dir = precomputed_dir
        self.total_samples = total_samples
        self.augment = augment
        self.jitter_std = jitter_std
        self.node_drop_rate = node_drop_rate
        
        # Load semua label ke RAM
        self.labels = []
        for idx in range(total_samples):
            filepath = os.path.join(self.precomputed_dir, f"sample_{idx}.npy")
            data = np.load(filepath, allow_pickle=True).item()
            self.labels.append(data["label"])
    
    def __len__(self) -> int:
        return self.total_samples
    
    def _apply_augmentation(self, graph_data: np.ndarray) -> np.ndarray:
        """Augmentation pada data graf."""
        if not self.augment:
            return graph_data
        
        data = graph_data.copy()
        C, T, V = data.shape
        
        # Coordinate jittering
        if C >= 2:
            noise = np.random.normal(0, self.jitter_std, (2, T, V)).astype(np.float32)
            data[0:2] += noise
        
        # Node drop
        if self.node_drop_rate > 0:
            drop_mask = np.random.random(V) < self.node_drop_rate
            data[:, :, drop_mask] = 0.0
        
        return data
    
    def __getitem__(self, idx: int):
        filepath = os.path.join(self.precomputed_dir, f"sample_{idx}.npy")
        data = np.load(filepath, allow_pickle=True).item()
        
        graph_data = data["graph"]
        label = data["label"]
        
        # Augmentation
        graph_data = self._apply_augmentation(graph_data)
        
        return torch.tensor(graph_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def precompute_graphs(
    dataset_wrapper: GraphDatasetWrapper,
    save_dir: str,
    desc: str = "Precomputing Graphs",
) -> int:
    """
    Utility untuk precompute dan menyimpan graf ke disk.
    
    Args:
        dataset_wrapper: GraphDatasetWrapper instance
        save_dir: Direktori penyimpanan file .npy
        desc: Deskripsi progress bar
    
    Returns:
        Jumlah sampel yang berhasil disimpan
    """
    os.makedirs(save_dir, exist_ok=True)
    
    success = 0
    for idx in tqdm(range(len(dataset_wrapper)), desc=desc):
        try:
            graph_tensor, label = dataset_wrapper[idx]
            
            sample_data = {
                "graph": graph_tensor.numpy(),
                "label": int(label) if isinstance(label, (int, np.integer)) else label.item(),
            }
            
            np.save(os.path.join(save_dir, f"sample_{idx}.npy"), sample_data, allow_pickle=True)
            success += 1
        except Exception as e:
            print(f"⚠️ Gagal precompute sampel {idx}: {e}")
    
    print(f"\n✅ {success}/{len(dataset_wrapper)} sampel berhasil disimpan ke {save_dir}")
    return success
