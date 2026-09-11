"""
gcn_gru.py — Faithful implementation of GCNGRUClassifier from external repository.

Architecture:
    Spatial extraction : 2-layer GCN (in_channels=7 -> 64 -> 64) with Mish & BatchNorm
    Spatial pooling    : Mean pooling across 468 facial landmark nodes
    Temporal modeling  : 2-layer Bidirectional GRU (64 -> 64) across 16 frames
    Classification head: Linear(64 -> 32) -> Mish -> Dropout(0.3) -> Linear(32 -> num_classes)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_normalized_adjacency(num_nodes: int = 468) -> torch.Tensor:
    """Build symmetrically normalized adjacency matrix D^(-1/2) * A * D^(-1/2)."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    A = A + np.eye(num_nodes, dtype=np.float32)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5, where=D > 0)
    D_inv_sqrt[D == 0] = 0.0
    D_mat = np.diag(D_inv_sqrt)
    A_norm = D_mat @ A @ D_mat
    return torch.tensor(A_norm, dtype=torch.float32)


class GCNLayer(nn.Module):
    """Single Graph Convolutional Network layer: Mish(BN(Linear(A @ X)))."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Nodes, In_Channels]
        # A: [Nodes, Nodes]
        out = torch.matmul(A, x)
        out = self.linear(out)
        batch_size, num_nodes, num_channels = out.shape
        out = out.view(-1, num_channels)
        out = self.bn(out)
        out = out.view(batch_size, num_nodes, num_channels)
        return F.mish(out)


class GCNGRUClassifier(nn.Module):
    """Spatio-Temporal GCN with Bidirectional GRU for Micro-Expression Recognition."""

    def __init__(
        self,
        num_nodes: int = 468,
        in_channels: int = 7,
        gcn_hidden: int = 64,
        gru_hidden: int = 64,
        fc_hidden: int = 32,
        num_classes: int = 5,
        num_frames: int = 16,
        dropout: float = 0.3,
        A_base: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.gru_hidden = gru_hidden
        self.gcn_hidden = gcn_hidden
        self.fc_hidden = fc_hidden
        self.num_frames = num_frames

        if A_base is not None:
            self.register_buffer("A", A_base.clone())
        else:
            self.register_buffer("A", build_normalized_adjacency(num_nodes))

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
            dropout=dropout,
        )

        # --- CLASSIFIER HEAD ---
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden, fc_hidden),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C=7, T=16, V=468] or [B, T=16, V=468, C=7]
        Returns:
            Logits of shape [B, num_classes]
        """
        # Auto-permute if channel is first [B, C=7, T, V] -> [B, T, V, C]
        if x.shape[1] == 7:
            x = x.permute(0, 2, 3, 1)

        batch_size, num_frames, num_nodes, channels = x.shape
        frame_features = []
        for t in range(num_frames):
            x_t = x[:, t, :, :]  # [B, V, C]
            h_spatial = self.gcn1(x_t, self.A)
            h_spatial = self.gcn2(h_spatial, self.A)
            h_pooled = torch.mean(h_spatial, dim=1)  # [B, gcn_hidden]
            frame_features.append(h_pooled)

        temporal_seq = torch.stack(frame_features, dim=1)  # [B, T, gcn_hidden]
        gru_out, _ = self.gru(temporal_seq)  # [B, T, gru_hidden]
        out_last_frame = gru_out[:, -1, :]  # [B, gru_hidden]
        return self.fc(out_last_frame)
