"""Shallow 2D CNNs for apex optical-flow images. Tiny param count = strong
regularization on 246 samples (the recipe that reaches 0.8+ on CASME II)."""
import torch
import torch.nn as nn


class STSTNet(nn.Module):
    """Reproduction of STSTNet: 3 parallel single-conv streams (3/5/8 filters)
    on the 3-channel flow image, concatenated, then FC. Very few parameters."""
    def __init__(self, num_classes, in_ch=3, dropout=0.5):
        super().__init__()
        def stream(f):
            return nn.Sequential(nn.Conv2d(in_ch, f, 3, padding=1), nn.BatchNorm2d(f),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.s1, self.s2, self.s3 = stream(3), stream(5), stream(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(3 + 5 + 8, num_classes)

    def forward(self, x):
        h = torch.cat([self.s1(x), self.s2(x), self.s3(x)], dim=1)
        h = self.pool(h).flatten(1)
        return self.fc(self.drop(h))


class ShallowCNN(nn.Module):
    """Slightly larger shallow CNN (3 conv blocks + GAP). Still tiny vs 3D-CNN."""
    def __init__(self, num_classes, in_ch=3, width=32, dropout=0.5):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.net = nn.Sequential(block(in_ch, width), block(width, width * 2),
                                 block(width * 2, width * 4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(width * 4, num_classes)

    def forward(self, x):
        h = self.pool(self.net(x)).flatten(1)
        return self.fc(self.drop(h))


def build_shallow(cfg, num_classes):
    name = cfg.get("shallow", "ststnet")
    in_ch = cfg.get("in_channels", 3)
    dp = cfg.get("dropout", 0.5)
    if name == "ststnet":
        return STSTNet(num_classes, in_ch=in_ch, dropout=dp)
    if name == "shallowcnn":
        return ShallowCNN(num_classes, in_ch=in_ch, width=cfg.get("width", 32), dropout=dp)
    raise ValueError(name)
