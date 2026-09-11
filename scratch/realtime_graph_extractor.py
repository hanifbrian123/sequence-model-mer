"""
realtime_graph_extractor.py — Ekstraksi fitur graph 7-channel secara real-time.

Mengkonversi stream frame webcam menjadi graph representation yang kompatibel
dengan model GCN-GRU. Menggunakan sliding window 16-frame buffer.

7-Channel Features per node:
    CH0: Koordinat X normalized [0, 1]
    CH1: Koordinat Y normalized [0, 1]
    CH2: Global displacement X (dari anchor/frame pertama)
    CH3: Global displacement Y (dari anchor/frame pertama)
    CH4: Temporal velocity X (displacement dari frame sebelumnya)
    CH5: Temporal velocity Y (displacement dari frame sebelumnya)
    CH6: Magnitude geometris (sqrt(dx² + dy²) global displacement)

Output shape: [1, 7, 16, 468] — siap untuk GCNGRUClassifier.forward()
"""

import os
# Fix protobuf 4.x / TensorFlow 2.10 conflict (mediapipe triggers TF import)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2
import numpy as np
import torch
import mediapipe as mp


class RealtimeGraphExtractor:
    """
    Ekstraksi fitur graph 7-channel secara real-time dari webcam frames.
    
    Mengelola sliding window buffer 16 frame terakhir dan menghasilkan
    tensor graph yang siap diinferensi oleh GCN-GRU model.
    
    Args:
        num_nodes: Jumlah landmark node (468 untuk MediaPipe)
        num_frames: Jumlah frame dalam window (16)
        num_channels: Jumlah channel fitur (7)
    """

    def __init__(
        self,
        num_nodes: int = 468,
        num_frames: int = 16,
        num_channels: int = 7,
    ):
        self.num_nodes = num_nodes
        self.num_frames = num_frames
        self.num_channels = num_channels

        # Inisialisasi MediaPipe Face Mesh
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                mp_face_mesh = mp.solutions.face_mesh
            else:
                import mediapipe as mp
                mp_face_mesh = mp.solutions.face_mesh
        except Exception:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,       # Tracking mode untuk video
            max_num_faces=1,
            refine_landmarks=True,         # Detail ekstra untuk mata & bibir
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )

        # Sliding window buffer: menyimpan landmarks [num_frames, num_nodes, 2]
        self.landmark_buffer: list[np.ndarray] = []

        # Anchor frame (frame pertama dalam buffer) untuk global displacement
        self.anchor_landmarks: np.ndarray | None = None

        # Status tracking
        self.face_detected = False
        self.frames_collected = 0

    def reset(self):
        """Reset buffer dan anchor. Panggil saat kehilangan tracking."""
        self.landmark_buffer.clear()
        self.anchor_landmarks = None
        self.frames_collected = 0
        self.face_detected = False

    def is_ready(self) -> bool:
        """Apakah buffer sudah penuh (≥16 frame) dan siap untuk inferensi."""
        return len(self.landmark_buffer) >= self.num_frames

    def get_buffer_count(self) -> int:
        """Jumlah frame dalam buffer saat ini."""
        return len(self.landmark_buffer)

    def extract_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """
        Ekstrak 468 landmarks dari satu frame BGR.
        
        Args:
            frame_bgr: Frame BGR dari OpenCV (uint8)
        
        Returns:
            Array [num_nodes, 2] koordinat (x, y) normalized [0, 1],
            atau None jika wajah tidak terdeteksi.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            self.face_detected = False
            return None

        self.face_detected = True
        landmarks = np.zeros((self.num_nodes, 2), dtype=np.float32)

        face_lm = results.multi_face_landmarks[0]
        for i, lm in enumerate(face_lm.landmark):
            if i >= self.num_nodes:
                break
            landmarks[i, 0] = lm.x
            landmarks[i, 1] = lm.y

        return landmarks

    def process_frame(self, frame_bgr: np.ndarray) -> bool:
        """
        Proses satu frame: ekstrak landmarks dan tambahkan ke buffer.
        
        Args:
            frame_bgr: Frame BGR dari OpenCV
        
        Returns:
            True jika wajah terdeteksi dan landmarks berhasil ditambahkan
        """
        landmarks = self.extract_landmarks(frame_bgr)

        if landmarks is None:
            # Wajah hilang — reset buffer untuk konsistensi temporal
            if len(self.landmark_buffer) > 0:
                self.reset()
            return False

        # Set anchor pada frame pertama
        if self.anchor_landmarks is None:
            self.anchor_landmarks = landmarks.copy()

        # Tambah ke buffer (sliding window)
        self.landmark_buffer.append(landmarks)

        # Jaga ukuran buffer = num_frames (buang frame terlama)
        if len(self.landmark_buffer) > self.num_frames:
            self.landmark_buffer.pop(0)
            # Update anchor ke frame terlama dalam buffer saat ini
            self.anchor_landmarks = self.landmark_buffer[0].copy()

        self.frames_collected += 1
        return True

    def build_graph_tensor(self) -> torch.Tensor | None:
        """
        Bangun tensor graph 7-channel dari buffer landmarks saat ini.
        
        Returns:
            Tensor [1, 7, 16, 468] siap untuk model inference,
            atau None jika buffer belum penuh.
        """
        if not self.is_ready():
            return None

        # Konversi buffer ke numpy array [T, V, 2]
        landmarks_array = np.array(self.landmark_buffer[-self.num_frames:])
        anchor = self.anchor_landmarks

        # Hitung displacement features
        global_displacement = landmarks_array - anchor[np.newaxis, :, :]  # [T, V, 2]

        temporal_displacement = np.zeros_like(landmarks_array)  # [T, V, 2]
        temporal_displacement[1:] = landmarks_array[1:] - landmarks_array[:-1]

        # Build 7-channel graph data [C=7, T=16, V=468]
        graph_data = np.zeros(
            (self.num_channels, self.num_frames, self.num_nodes),
            dtype=np.float32,
        )

        for f in range(self.num_frames):
            lm = landmarks_array[f]  # [V, 2]

            # CH0-1: Koordinat normalized
            graph_data[0, f, :] = lm[:, 0]
            graph_data[1, f, :] = lm[:, 1]

            # CH2-3: Global displacement (dari anchor)
            graph_data[2, f, :] = global_displacement[f, :, 0]
            graph_data[3, f, :] = global_displacement[f, :, 1]

            # CH4-5: Temporal velocity
            graph_data[4, f, :] = temporal_displacement[f, :, 0]
            graph_data[5, f, :] = temporal_displacement[f, :, 1]

            # CH6: Magnitude geometris
            dx_g = global_displacement[f, :, 0]
            dy_g = global_displacement[f, :, 1]
            graph_data[6, f, :] = np.sqrt(dx_g ** 2 + dy_g ** 2)

        # Konversi ke tensor [1, 7, 16, 468]
        tensor = torch.tensor(graph_data, dtype=torch.float32).unsqueeze(0)
        return tensor

    def get_current_landmarks(self) -> np.ndarray | None:
        """
        Ambil landmarks dari frame terakhir (untuk visualisasi).
        
        Returns:
            Array [num_nodes, 2] atau None
        """
        if len(self.landmark_buffer) > 0:
            return self.landmark_buffer[-1]
        return None

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()
