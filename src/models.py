"""Sequence models. Input tensor shape (B, C, T, H, W)."""
import torch
import torch.nn as nn
from torchvision.models import video as V
import torchvision


def _replace_fc(fc_in, num_classes, dropout):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(fc_in, num_classes))


class ResNetGRU(nn.Module):
    """Per-frame ResNet18 (ImageNet) features + BiGRU. Input (B,C,T,H,W)."""
    def __init__(self, num_classes, dropout=0.5, hidden=256, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=w)
        self.feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.gru = nn.GRU(self.feat_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, num_classes))

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        f = self.backbone(x).reshape(B, T, self.feat_dim)
        o, _ = self.gru(f)
        o = o.mean(dim=1)
        return self.head(o)


class TwoStream(nn.Module):
    """Two 3D-CNN backbones (motion + appearance) with concat-feature fusion.
    Input: tuple (x_a, x_b), each (B,3,T,H,W)."""
    def __init__(self, num_classes, backbone="r2plus1d_18", dropout=0.5, pretrained=True):
        super().__init__()
        weights_map = {
            "r2plus1d_18": V.R2Plus1D_18_Weights.KINETICS400_V1,
            "r3d_18": V.R3D_18_Weights.KINETICS400_V1,
            "mc3_18": V.MC3_18_Weights.KINETICS400_V1,
        }
        ctor = getattr(V, backbone)

        def make():
            net = ctor(weights=weights_map[backbone] if pretrained else None)
            fc_in = net.fc.in_features
            net.fc = nn.Identity()
            return net, fc_in
        self.stream_a, fa = make()   # motion (flow)
        self.stream_b, fb = make()   # appearance (onset_ref/rgb)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fa + fb, num_classes))

    def forward(self, x):
        xa, xb = x
        fa = self.stream_a(xa)
        fb = self.stream_b(xb)
        return self.head(torch.cat([fa, fb], dim=1))


def _adapt_in_channels(model, in_ch):
    """Adapt r2plus1d/r3d/mc3 stem first conv to `in_ch` input channels, inflating
    pretrained 3-channel weights (extra channels = mean of the RGB kernels)."""
    if in_ch == 3:
        return model
    old = model.stem[0]  # Conv3d(3, C, ...)
    new = nn.Conv3d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=(old.bias is not None))
    with torch.no_grad():
        w = old.weight  # (C,3,kt,kh,kw)
        if in_ch < 3:
            new.weight.copy_(w[:, :in_ch])
        else:
            new.weight[:, :3].copy_(w)
            mean_k = w.mean(dim=1, keepdim=True)
            for i in range(3, in_ch):
                new.weight[:, i:i + 1].copy_(mean_k)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.stem[0] = new
    return model


def build_model(cfg, num_classes):
    in_ch = cfg.get("in_channels", 3)
    if cfg.get("modality") == "two_stream":
        return TwoStream(num_classes, backbone=cfg.get("backbone", "r2plus1d_18"),
                         dropout=cfg.get("dropout", 0.5), pretrained=cfg.get("pretrained", True))
    name = cfg["backbone"]
    dropout = cfg.get("dropout", 0.5)
    pretrained = cfg.get("pretrained", True)
    if name in ("r2plus1d_18", "r3d_18", "mc3_18"):
        weights_map = {
            "r2plus1d_18": V.R2Plus1D_18_Weights.KINETICS400_V1,
            "r3d_18": V.R3D_18_Weights.KINETICS400_V1,
            "mc3_18": V.MC3_18_Weights.KINETICS400_V1,
        }
        ctor = getattr(V, name)
        model = ctor(weights=weights_map[name] if pretrained else None)
        model.fc = _replace_fc(model.fc.in_features, num_classes, dropout)
        model = _adapt_in_channels(model, in_ch)
        return model
    if name == "resnet_gru":
        return ResNetGRU(num_classes, dropout=dropout,
                         hidden=cfg.get("gru_hidden", 256), pretrained=pretrained)
    raise ValueError(f"unknown backbone {name}")
