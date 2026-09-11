"""
temporal_transformer.py — Faithful implementation of CNNTemporalViViT from external repository.

Processes temporal sequence of precomputed CNN feature vectors (e.g. ResNet-50 2048-D)
using a Multi-Head Self-Attention Temporal Transformer Encoder with [CLS] token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTemporalViViT(nn.Module):
    """Temporal Transformer Encoder on top of precomputed CNN features."""

    def __init__(
        self,
        num_classes: int = 5,
        num_frames: int = 16,
        feature_dim: int = 2048,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.feature_dim = feature_dim
        self.d_model = d_model

        # 1. Linear Projection & Feature Adapter
        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # 2. Learnable [CLS] Token & Temporal Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_frames + 1, d_model) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. Temporal Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, num_frames, feature_dim] or [B, feature_dim]
        Returns:
            Logits of shape [B, num_classes]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, self.num_frames, 1)
        elif x.dim() == 4:
            x = x.flatten(2)

        B, T, D = x.shape
        x = self.feat_proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if x.size(1) == self.pos_embed.size(1):
            x = x + self.pos_embed
        else:
            pos_emb = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=x.size(1),
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
            x = x + pos_emb

        x = self.pos_drop(x)
        feats = self.transformer_encoder(x)
        cls_out = feats[:, 0]
        return self.classifier(cls_out)
