"""
gcn_gru_model.py — Arsitektur GCN-GRU untuk Micro-Expression Recognition.

Diekstrak dari notebook GCN_Training_Testing_CASME2 Final.ipynb.

Arsitektur:
    GCNLayer(in_channels → 64)  — Spatial feature extraction layer 1
    GCNLayer(64 → 64)           — Spatial feature extraction layer 2
    BiGRU(64 → 32, 2-layer)     — Temporal sequence modeling
    FC(64 → 32 → 5)             — Classification head

Input:  [Batch, 7, 16, 468] — (batch, channels, frames, nodes)
Output: [Batch, 5]           — logits untuk 5 kelas ekspresi

Label Mapping:
    0: Anger
    1: Disgust/Surprise
    2: Fear
    3: Happiness
    4: Neutral
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Satu layer Graph Convolutional Network standar.
    
    Operasi: Mish(BN(Linear(A @ X)))
    
    Args:
        in_channels: Jumlah channel input per node
        out_channels: Jumlah channel output per node
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [Batch, Nodes, In_Channels]
            A: Normalized adjacency matrix [Nodes, Nodes]
        
        Returns:
            Output [Batch, Nodes, Out_Channels]
        """
        # 1. Agregasi Spasial: A @ X
        out = torch.matmul(A, x)

        # 2. Transformasi Linear
        out = self.linear(out)

        # 3. Batch Normalization
        batch_size, num_nodes, num_channels = out.shape
        out = out.view(-1, num_channels)
        out = self.bn(out)
        out = out.view(batch_size, num_nodes, num_channels)

        return F.mish(out)


class GCNGRUClassifier(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network dengan GRU untuk
    Micro-Expression Recognition.
    
    Pipeline:
        1. Per-frame spatial extraction via 2 GCN layers
        2. Mean pooling per frame (468 nodes → 1 vector)
        3. Temporal modeling via 2-layer bidirectional GRU
        4. Classification head dari output frame terakhir
    
    Args:
        num_nodes: Jumlah node landmark (468 untuk MediaPipe)
        in_channels: Jumlah channel fitur per node (7)
        gcn_hidden: Hidden dimension GCN layers
        gru_hidden: Hidden dimension GRU
        fc_hidden: Hidden dimension FC head
        num_classes: Jumlah kelas output (5)
        A_base: Base adjacency matrix [num_nodes, num_nodes].
                Jika None, menggunakan identity matrix (fallback).
    """

    def __init__(
        self,
        num_nodes: int = 468,
        in_channels: int = 7,
        gcn_hidden: int = 64,
        gru_hidden: int = 64,
        fc_hidden: int = 32,
        num_classes: int = 5,
        A_base: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.gru_hidden = gru_hidden
        self.gcn_hidden = gcn_hidden
        self.fc_hidden = fc_hidden
        self.num_frames = 16

        # Adjacency matrix — registered as buffer (moves with model to device)
        if A_base is not None:
            self.register_buffer('A', A_base.clone())
        else:
            print("[WARNING] Tidak ada adjacency matrix diberikan. Menggunakan Identity.")
            self.register_buffer('A', torch.eye(num_nodes))

        # --- SPATIAL EXTRACTOR (GCN) ---
        self.gcn1 = GCNLayer(in_channels, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden)

        # --- TEMPORAL EXTRACTOR (Bidirectional GRU) ---
        self.gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=fc_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # --- CLASSIFIER HEAD ---
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden, fc_hidden),  # 64 → 32
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(fc_hidden, num_classes),  # 32 → 5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor. Mendukung dua format:
               - [Batch, Channels=7, Frames=16, Nodes=468] → auto-permute
               - [Batch, Frames, Nodes, Channels]
        
        Returns:
            Logits tensor [Batch, num_classes]
        """
        # Auto-permute: [B, C=7, T, V] → [B, T, V, C]
        if x.shape[1] == 7:
            x = x.permute(0, 2, 3, 1)

        batch_size, num_frames, num_nodes, channels = x.shape

        # Ekstraksi Fitur Spasial per Frame via GCN
        frame_features = []
        for t in range(num_frames):
            x_t = x[:, t, :, :]  # [Batch, Nodes, Channels]

            h_spatial = self.gcn1(x_t, self.A)
            h_spatial = self.gcn2(h_spatial, self.A)  # [Batch, Nodes, GCN_Hidden]

            # Mean Pooling: 468 nodes → 1 vector
            h_pooled = torch.mean(h_spatial, dim=1)  # [Batch, GCN_Hidden]
            frame_features.append(h_pooled)

        # Susun temporal sequence
        temporal_seq = torch.stack(frame_features, dim=1)  # [Batch, Frames, GCN_Hidden]

        # GRU Temporal
        gru_out, _ = self.gru(temporal_seq)  # [Batch, Frames, GRU_Hidden]

        # Ambil output frame terakhir
        out_last_frame = gru_out[:, -1, :]  # [Batch, GRU_Hidden]

        # Klasifikasi
        logits = self.fc(out_last_frame)  # [Batch, Num_Classes]
        return logits

    def get_num_params(self) -> int:
        """Total parameter model."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self) -> int:
        """Parameter yang bisa di-train."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
