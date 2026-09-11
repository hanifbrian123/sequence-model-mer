"""Sequence models. Input tensor shape (B, C, T, H, W)."""
import torch
import torch.nn as nn
from torchvision.models import video as V
import torchvision


def _replace_fc(fc_in, num_classes, dropout):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(fc_in, num_classes))


class ResNetGRU(nn.Module):
    """Per-frame ResNet18 (ImageNet) features + BiGRU. Input (B,C,T,H,W)."""
    def __init__(self, num_classes, dropout=0.5, hidden=256, pretrained=True,
                 in_ch=3):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=w)
        if in_ch != 3:
            # Inflate conv1 to in_ch channels so the focus/segmentation channel
            # can be fed to the RNN backbone too. Extra channels = mean of the
            # pretrained RGB kernels, matching the 3D-CNN stem adaptation.
            old = net.conv1
            new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                if in_ch < 3:
                    new.weight.copy_(old.weight[:, :in_ch])
                else:
                    new.weight[:, :3].copy_(old.weight)
                    mean_k = old.weight.mean(dim=1, keepdim=True)
                    for i in range(3, in_ch):
                        new.weight[:, i:i + 1].copy_(mean_k)
            net.conv1 = new
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


class ViViTWrapper(nn.Module):
    """ViViT-B (Kinetics-pretrained video transformer, HuggingFace) as a sequence
    backbone. Input (B,C,T,H,W) -> permute to ViViT's (B,T,C,H,W) pixel_values.
    Exposes gradual-unfreeze + layer-wise-LR-decay (LLRD) helpers for adaptive
    finetuning on tiny data (246 samples): keep pretrained weights nearly intact
    (early layers tiny LR / optional freeze), only strongly move the new head."""
    CKPT = "google/vivit-b-16x2-kinetics400"

    def __init__(self, num_classes, dropout=0.3, pretrained=True, grad_checkpoint=True):
        super().__init__()
        from transformers import VivitForVideoClassification, VivitConfig
        if pretrained:
            self.net = VivitForVideoClassification.from_pretrained(
                self.CKPT, num_labels=num_classes, ignore_mismatched_sizes=True,
                hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout)
        else:
            cfg = VivitConfig(num_labels=num_classes, hidden_dropout_prob=dropout)
            self.net = VivitForVideoClassification(cfg)
        if grad_checkpoint:
            self.net.gradient_checkpointing_enable()
        self.num_layers = self.net.config.num_hidden_layers

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()   # (B,C,T,H,W) -> (B,T,C,H,W)
        return self.net(pixel_values=x).logits

    def set_backbone_requires_grad(self, flag):
        """Freeze/unfreeze everything except the classification head."""
        for n, p in self.net.named_parameters():
            if not n.startswith("classifier"):
                p.requires_grad = flag

    def _depth_of(self, name):
        """0 = embeddings (input side), i+1 = encoder block i, L+1 = head."""
        L = self.num_layers
        if name.startswith("classifier"):
            return L + 1
        if "encoder.layer." in name:
            return int(name.split("encoder.layer.")[1].split(".")[0]) + 1
        return 0

    def llrd_param_groups(self, base_lr, head_lr, decay, weight_decay):
        """Discriminative (layer-wise decayed) LR param groups. Deeper layers
        (closer to head) get larger LR; embeddings get base_lr*decay^L; the head
        gets head_lr. No weight decay on biases / LayerNorm."""
        L = self.num_layers
        no_decay = ("bias", "layernorm", "layer_norm")
        groups = {}
        for n, p in self.net.named_parameters():
            d = self._depth_of(n)
            lr = head_lr if d == L + 1 else base_lr * (decay ** (L - d))
            wd = 0.0 if any(k in n.lower() for k in no_decay) else weight_decay
            key = (round(float(lr), 10), wd)
            groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": wd})
            groups[key]["params"].append(p)
        return list(groups.values())


class MultiTaskModel(nn.Module):
    """Shared backbone, emotion head, plus an auxiliary Action Unit head.

    `others` is 45% of the data and is defined negatively — everything that is
    not one of the four named emotions — so it has no emotional signature to
    learn, and 76% of the champion's errors involve it. It does have an AU
    signature. Predicting AUs alongside the emotion gives the backbone a dense,
    physically grounded target, which is what a 192-sample problem is short of.

    The AU head is training-only scaffolding: ``forward`` returns emotion logits
    first, and every evaluation path reads only that, so an uploaded clip never
    needs AU annotation.
    """
    def __init__(self, backbone, feat_dim, num_classes, num_au, dropout=0.5):
        super().__init__()
        self.backbone = backbone
        self.emotion = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(feat_dim, num_classes))
        self.au = nn.Sequential(nn.Dropout(dropout),
                                nn.Linear(feat_dim, num_au))

    def forward(self, x):
        features = self.backbone(x)
        return self.emotion(features), self.au(features)


class RegionAttention(nn.Module):
    """Segmentation attention: learn how much each facial region matters.

    Input is ``(clip, masks)`` where the K region masks sum to 1 at every pixel.
    The gate is ``sum_k w_k * mask_k``, so at initialisation every ``w_k`` is 1,
    the gate is exactly 1 everywhere, and the clip passes through untouched. The
    model can only depart from that if the data pays for it.

    This is deliberately the opposite of every spatial experiment that has failed
    here: a soft ellipse, translation compensation, ECC stabilisation and face
    parsing all DELETED signal and all lost. Nothing is deleted here — regions
    are re-weighted, and a region can even be amplified rather than suppressed.

    ``static`` learns one weight vector for the whole dataset (K parameters, the
    safest thing to add to 192 samples). ``dynamic`` predicts the weights from
    the clip itself, which is more expressive and far easier to overfit.
    """
    def __init__(self, backbone, num_regions, mode="static", hidden=16):
        super().__init__()
        self.backbone = backbone
        self.mode = mode
        self.num_regions = num_regions
        if mode == "static":
            self.log_weight = nn.Parameter(torch.zeros(num_regions))
        elif mode == "dynamic":
            self.log_weight = nn.Parameter(torch.zeros(num_regions))
            self.predictor = nn.Sequential(
                nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                nn.Linear(3, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, num_regions))
            nn.init.zeros_(self.predictor[-1].weight)
            nn.init.zeros_(self.predictor[-1].bias)
        else:
            raise ValueError(f"unknown region attention mode: {mode}")

    def region_weights(self, clip):
        logits = self.log_weight.unsqueeze(0)
        if self.mode == "dynamic":
            logits = logits + self.predictor(clip)
        return torch.exp(logits)

    def forward(self, x):
        clip, masks = x                                  # (B,C,T,H,W), (B,K,H,W)
        weights = self.region_weights(clip)              # (B,K)
        gate = (masks * weights[:, :, None, None]).sum(dim=1, keepdim=True)
        return self.backbone(clip * gate.unsqueeze(2))   # broadcast over time


def _video_resnet_depth(name):
    """0 = stem (input side) ... 4 = layer4, 5 = classifier head.

    Matches the depth convention ViViTWrapper already uses so both backbone
    families can share one layer-wise-LR-decay implementation.
    """
    if name.startswith("fc."):
        return 5
    for index in (1, 2, 3, 4):
        if name.startswith(f"layer{index}."):
            return index
    return 0


def _resnet_gru_depth(name):
    """Same convention for ResNetGRU; the GRU counts as part of the head."""
    if name.startswith("head.") or name.startswith("gru."):
        return 5
    for index in (1, 2, 3, 4):
        if name.startswith(f"backbone.layer{index}."):
            return index
    return 0


def _multitask_depth(inner_depth_fn):
    """Reuse a backbone's depth map through the multi-task wrapper's prefixes."""
    def depth_of(name):
        if name.startswith("emotion.") or name.startswith("au."):
            return 5
        if name.startswith("backbone."):
            return inner_depth_fn(name[len("backbone."):])
        return 0
    return depth_of


def _region_depth(inner_depth_fn):
    """Depth map through the RegionAttention wrapper; the gate counts as head."""
    def depth_of(name):
        if name.startswith("log_weight") or name.startswith("predictor."):
            return 5
        if name.startswith("backbone."):
            return inner_depth_fn(name[len("backbone."):])
        return 0
    return depth_of


def _depth_fn_for(model):
    if isinstance(model, RegionAttention):
        return _region_depth(_depth_fn_for(model.backbone))
    if isinstance(model, MultiTaskModel):
        inner = (_resnet_gru_depth if isinstance(model.backbone, ResNetGRU)
                 else _video_resnet_depth)
        return _multitask_depth(inner)
    if isinstance(model, ResNetGRU):
        return _resnet_gru_depth
    return _video_resnet_depth


def cnn_llrd_param_groups(model, base_lr, head_lr, decay, weight_decay,
                          head_depth=5):
    """Layer-wise-LR-decay groups for the 3D-CNN / ResNet+GRU backbones.

    246 samples cannot support moving a Kinetics-pretrained stem as fast as a
    randomly initialised head. Early layers already encode edges and motion;
    training them at the head's learning rate destroys that. Layer i gets
    ``base_lr * decay**(head_depth-1-i)`` so layer4 keeps ``base_lr`` and the
    stem is slowed by ``decay**4``. Biases and normalization get no weight decay.
    """
    depth_of = _depth_fn_for(model)
    no_decay = ("bias", "norm")
    groups = {}
    for name, parameter in model.named_parameters():
        depth = depth_of(name)
        lr = (head_lr if depth >= head_depth
              else base_lr * (decay ** (head_depth - 1 - depth)))
        wd = 0.0 if any(token in name.lower() for token in no_decay) else weight_decay
        key = (round(float(lr), 12), float(wd))
        groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": wd})
        groups[key]["params"].append(parameter)
    return list(groups.values())


def set_backbone_requires_grad(model, flag):
    """Freeze/unfreeze everything except the classification head.

    Dispatches to ViViTWrapper's own implementation when present so callers do
    not need to know which backbone they hold.
    """
    if hasattr(model, "set_backbone_requires_grad"):
        model.set_backbone_requires_grad(flag)
        return
    depth_of = _depth_fn_for(model)
    for name, parameter in model.named_parameters():
        if depth_of(name) < 5:
            parameter.requires_grad = flag


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
        feat_dim = model.fc.in_features
        num_au = int(cfg.get("num_au", 0)) if cfg.get("au_multitask", False) else 0
        if num_au > 0:
            model.fc = nn.Identity()
            model = _adapt_in_channels(model, in_ch)
            return MultiTaskModel(model, feat_dim, num_classes, num_au, dropout)
        model.fc = _replace_fc(feat_dim, num_classes, dropout)
        model = _adapt_in_channels(model, in_ch)
        if cfg.get("region_attention", False):
            return RegionAttention(
                model, int(cfg.get("num_regions", 6)),
                mode=cfg.get("region_attention_mode", "static"),
                hidden=int(cfg.get("region_attention_hidden", 16)))
        return model
    if name == "resnet_gru":
        return ResNetGRU(num_classes, dropout=dropout,
                         hidden=cfg.get("gru_hidden", 256), pretrained=pretrained,
                         in_ch=in_ch)
    if name == "vivit":
        return ViViTWrapper(num_classes, dropout=dropout, pretrained=pretrained,
                            grad_checkpoint=cfg.get("grad_checkpoint", True))
    if name == "stgcn":
        from casme.models.gcn import STGCN
        return STGCN(in_channels=in_ch, num_classes=num_classes, dropout=dropout)
    if name == "cnn_temporal_vivit":
        from casme.models.temporal_transformer import CNNTemporalViViT
        return CNNTemporalViViT(
            num_classes=num_classes,
            num_frames=int(cfg.get("T", 16)),
            feature_dim=int(cfg.get("feature_dim", 2048)),
            d_model=int(cfg.get("d_model", 512)),
            nhead=int(cfg.get("nhead", 8)),
            num_layers=int(cfg.get("num_layers", 4)),
            dim_feedforward=int(cfg.get("dim_feedforward", 1024)),
            dropout=float(cfg.get("dropout", 0.3)),
        )
    raise ValueError(f"unknown backbone {name}")

