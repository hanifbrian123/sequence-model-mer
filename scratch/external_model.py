"""
model.py — Model ViViT (Video Vision Transformer) untuk Micro-Expression Recognition.

Mendukung dua mode input:
  - in_channels=3: RGB image classification (untuk dataset static image)
  - in_channels=5: Hybrid 5-channel (RGB + Muscle Map + Geometric Map) untuk full pipeline

Menggunakan HuggingFace Transformers VivitModel sebagai backbone, 
dengan classification head yang dilengkapi dropout dan layer norm.

Fitur baru:
  - Transfer learning dari pretrained Kinetics-400
  - Channel adapter untuk input non-RGB (5 channel)
  - Freeze/unfreeze backbone untuk fine-tuning bertahap
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VivitConfig, VivitModel


class ViViTMLP(nn.Module):
    def __init__(
        self, 
        num_classes: int = 7, 
        num_frames: int = 16, 
        patch_size: int = 16, 
        in_channels: int = 3,
        image_size: int = 224,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 0,
    ):
        """
        Args:
            num_classes: Jumlah kelas emosi target.
            num_frames: Jumlah frame temporal (16 untuk pseudo-video dari static image).
            patch_size: Ukuran patch untuk ViViT tokenization.
            in_channels: Jumlah channel input (3=RGB, 5=RGB+Muscle+Geometric).
            image_size: Resolusi input (default 224x224).
            dropout: Dropout rate untuk classification head.
            pretrained: Jika True, load pretrained weights dari Kinetics-400.
            freeze_backbone: Jika True, freeze backbone ViViT (hanya train head).
            unfreeze_last_n: Jumlah layer terakhir yang di-unfreeze (0 = freeze semua).
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self._pretrained = pretrained
        
        # Channel adapter untuk input non-3-channel (misalnya 5ch hybrid)
        self.channel_adapter = None
        if in_channels != 3:
            self.channel_adapter = nn.Sequential(
                nn.Conv3d(in_channels, 3, kernel_size=1, bias=False),
                nn.BatchNorm3d(3),
            )
        
        # ================================================================
        # Backbone ViViT Transformer
        # ================================================================
        if pretrained:
            # Load pretrained dari Kinetics-400 (Google ViViT-B/16x2)
            print("  [MODEL] Loading pretrained ViViT from google/vivit-b-16x2-kinetics400...")
            self.vivit = VivitModel.from_pretrained(
                "google/vivit-b-16x2-kinetics400",
                ignore_mismatched_sizes=True,
                use_safetensors=True,  # Bypass torch.load CVE issue with PyTorch < 2.6
            )
            # Adaptasi positional embeddings jika num_frames berbeda dari pretrained (32)
            if self.vivit.config.num_frames != num_frames:
                print(f"  [MODEL] Interpolating temporal positional embeddings from "
                      f"{self.vivit.config.num_frames} frames to {num_frames} frames...")
                self._interpolate_temporal_pos_embed(num_frames)
        else:
            # Training dari nol (tidak disarankan untuk dataset kecil)
            print("  [MODEL] Initializing ViViT from scratch (no pretrained weights)")
            configuration = VivitConfig(
                num_frames=num_frames,
                video_size=[num_frames, image_size, image_size],
                patch_size=patch_size,
                num_channels=3,  # Selalu 3 karena channel adapter
                image_size=image_size,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
            )
            self.vivit = VivitModel(configuration)
        
        # ================================================================
        # Classification Head dengan LayerNorm dan Dropout
        # ================================================================
        hidden_size = self.vivit.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )
        
        # ================================================================
        # Freeze/Unfreeze Strategy
        # ================================================================
        if freeze_backbone:
            self.freeze_backbone(unfreeze_last_n=unfreeze_last_n)
            
    def _interpolate_temporal_pos_embed(self, new_num_frames: int):
        """
        Interpolates pretrained temporal positional embeddings to match new_num_frames.
        """
        old_pos = self.vivit.embeddings.position_embeddings  # [1, 1 + old_t * spatial, dim]
        cls_pos = old_pos[:, :1, :]
        patch_pos = old_pos[:, 1:, :]

        tubelet_t = self.vivit.config.tubelet_size[0]
        tubelet_h = self.vivit.config.tubelet_size[1]
        tubelet_w = self.vivit.config.tubelet_size[2]
        
        old_num_frames = self.vivit.config.num_frames
        old_t = old_num_frames // tubelet_t
        new_t = new_num_frames // tubelet_t
        
        h = self.vivit.config.image_size // tubelet_h
        w = self.vivit.config.image_size // tubelet_w
        spatial = h * w
        dim = self.vivit.config.hidden_size

        # Reshape to [1, old_t, spatial, dim] -> permute to [spatial, dim, old_t]
        patch_pos = patch_pos.view(1, old_t, spatial, dim).permute(0, 2, 3, 1).contiguous().view(spatial, dim, old_t)
        # Linear interpolation across temporal dimension
        new_patch_pos = F.interpolate(patch_pos, size=new_t, mode='linear', align_corners=False)
        # Reshape back to [1, new_t * spatial, dim]
        new_patch_pos = new_patch_pos.permute(2, 0, 1).contiguous().view(1, new_t * spatial, dim)

        new_pos = torch.cat([cls_pos, new_patch_pos], dim=1)
        self.vivit.embeddings.position_embeddings = nn.Parameter(new_pos)
        self.vivit.config.num_frames = new_num_frames
    
    def freeze_backbone(self, unfreeze_last_n: int = 0):
        """
        Freeze seluruh backbone ViViT, kecuali N layer terakhir.
        
        Args:
            unfreeze_last_n: Jumlah encoder layer terakhir yang tetap trainable.
                             0 = freeze semua backbone.
        """
        # Freeze semua parameter backbone
        for param in self.vivit.parameters():
            param.requires_grad = False
        
        # Unfreeze last N encoder layers
        if unfreeze_last_n > 0 and hasattr(self.vivit, 'encoder'):
            encoder_layers = self.vivit.encoder.layer
            total_layers = len(encoder_layers)
            start_unfreeze = max(0, total_layers - unfreeze_last_n)
            
            for i in range(start_unfreeze, total_layers):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True
            
            print(f"  [MODEL] Unfroze last {unfreeze_last_n} encoder layers "
                  f"(layers {start_unfreeze}-{total_layers-1})")
        
        # Selalu unfreeze layernorm terakhir jika ada
        if hasattr(self.vivit, 'layernorm'):
            for param in self.vivit.layernorm.parameters():
                param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  [MODEL] Backbone frozen. Trainable: {trainable:,}/{total:,} "
              f"({trainable/total*100:.1f}%)")
    
    def unfreeze_backbone(self, unfreeze_last_n: int | None = None):
        """
        Unfreeze backbone ViViT (seluruhnya atau sebagian).
        
        Args:
            unfreeze_last_n: Jika None, unfreeze seluruh backbone.
                             Jika int, unfreeze N layer terakhir saja.
        """
        if unfreeze_last_n is None:
            # Unfreeze seluruh backbone
            for param in self.vivit.parameters():
                param.requires_grad = True
            print("  [MODEL] Backbone fully unfrozen.")
        else:
            # Pertama freeze semua, lalu unfreeze N terakhir
            self.freeze_backbone(unfreeze_last_n=unfreeze_last_n)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  [MODEL] Trainable: {trainable:,}/{total:,} "
              f"({trainable/total*100:.1f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
               - Jika 5D [B, C, F, H, W]: video/pseudo-video format → permute ke [B, F, C, H, W]
               - Format yang diharapkan oleh VivitModel: [B, F, C, H, W]
        
        Returns:
            Logits tensor [B, num_classes]
        """
        # Input dari FeatureFuser: [B, C, F, H, W]
        # VivitModel expects: [B, F, C, H, W]
        if x.dim() == 5:
            # Channel adapter: [B, C_in, F, H, W] → [B, 3, F, H, W]
            if self.channel_adapter is not None:
                x = self.channel_adapter(x)
            
            # Permute: [B, C, F, H, W] → [B, F, C, H, W]
            x = x.permute(0, 2, 1, 3, 4)
        
        outputs = self.vivit(x)
        
        # Gunakan [CLS] token representation (indeks 0)
        cls_output = outputs.last_hidden_state[:, 0]
        
        return self.classifier(cls_output)
    
    def get_num_params(self) -> int:
        """Menghitung total parameter model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Menghitung parameter yang bisa di-train."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# FAST PRECOMPUTED CNN + TEMPORAL VIVIT ARCHITECTURE
# ============================================================================

class CNNTemporalViViT(nn.Module):
    """
    Temporal Vision Transformer ringan & berkecepatan tinggi yang menerima
    input representasi fitur spasial CNN yang sudah diekstrak (precomputed).
    
    Input : [B, num_frames, feature_dim]
    Output: [B, num_classes] (Logits)
    
    Akselerasi: Menghilangkan kebutuhan render frame spasial pixel 224x224
    dan tubelet tokenization setiap epoch, mempercepat training hingga >300x.
    """
    def __init__(
        self,
        num_classes: int = 7,
        num_frames: int = 16,
        feature_dim: int = 2048,   # Default ResNet50 (2048), ResNet18 (512), EffNet (1280)
        d_model: int = 512,        # Dimensi transformer internal
        nhead: int = 8,            # Jumlah attention heads
        num_layers: int = 4,       # Jumlah temporal transformer encoder layers
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
        
        # Init weights
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
            x: Tensor fitur spasial [B, num_frames, feature_dim]
               atau [B, feature_dim] (akan diexpand ke [B, num_frames, feature_dim])
        Returns:
            Logits [B, num_classes]
        """
        if x.dim() == 2:
            # Expand static image feature ke pseudo temporal sequence
            x = x.unsqueeze(1).repeat(1, self.num_frames, 1)
        elif x.dim() == 4:
            # Format [B, C, F, H] fallback
            x = x.flatten(2)
            
        B, T, D = x.shape
        
        # Proyeksikan ke d_model
        x = self.feat_proj(x)  # [B, T, d_model]
        
        # Tambahkan CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, T+1, d_model]
        
        # Tambahkan Positional Embeddings
        if x.size(1) == self.pos_embed.size(1):
            x = x + self.pos_embed
        else:
            # Interpolasi positional embedding jika panjang frame dinamis
            pos_emb = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=x.size(1),
                mode="linear",
                align_corners=False
            ).permute(0, 2, 1)
            x = x + pos_emb
            
        x = self.pos_drop(x)
        
        # Multi-Head Temporal Self-Attention
        feats = self.transformer_encoder(x)  # [B, T+1, d_model]
        
        # Ambil [CLS] token output
        cls_out = feats[:, 0]  # [B, d_model]
        
        return self.classifier(cls_out)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
        
    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EndToEndCNNViViT(nn.Module):
    """
    Model wrapper lengkap untuk fase realtime / inferensi live.
    Menggabungkan CNN Backbone Spasial + Temporal ViViT menjadi satu modul terpadu.
    """
    def __init__(self, cnn_backbone: nn.Module, temporal_vivit: CNNTemporalViViT):
        super().__init__()
        self.cnn_backbone = cnn_backbone
        self.temporal_vivit = temporal_vivit
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, C, F, H, W] (Video Tensor)
        Output: [B, num_classes] (Logits)
        """
        B, C, F_len, H, W = x.shape
        # Permute & reshape ke format batch gambar [B * F, C, H, W]
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * F_len, C, H, W)
        
        with torch.no_grad():
            cnn_feats = self.cnn_backbone(x_reshaped)  # [B * F, feature_dim]
            if cnn_feats.dim() > 2:
                cnn_feats = F.adaptive_avg_pool2d(cnn_feats, (1, 1)).flatten(1)
                
        # Reshape kembali ke urutan temporal [B, F, feature_dim]
        temporal_feats = cnn_feats.view(B, F_len, -1)
        return self.temporal_vivit(temporal_feats)