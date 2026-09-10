"""Apex optical-flow dataset (STSTNet-style): ONE compact flow image per sample.

Onset->apex flow = flow128[key][apex_pos] (flow cache is onset-referenced, so the
apex-position frame IS the onset->apex field). Channels: [u, v, strain].
Returns (3, H, W) — a 2D image, NOT a sequence. Small net on this generalizes far
better than a 33M-param 3D-CNN on the full sequence (246 samples).
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def strain_field(u, v):
    duy, dux = np.gradient(u)
    dvy, dvx = np.gradient(v)
    exy = 0.5 * (duy + dvx)
    return np.sqrt(dux * dux + dvy * dvy + 2.0 * exy * exy)


class ApexFlowDataset(Dataset):
    def __init__(self, samples, arrays, cfg, train):
        self.samples = samples          # dicts: key, subject, label, apex_pos
        self.arrays = arrays            # key -> (L,H,W,2) flow
        self.cfg = cfg
        self.train = train
        self.s = cfg["img_size"]
        self.flow_clip = cfg.get("flow_clip", 3.0)
        self.strain_clip = cfg.get("strain_clip", 0.3)

    def __len__(self):
        return len(self.samples)

    def _apex_field(self, s):
        flow = self.arrays[s["key"]]          # (L,H,W,2)
        L = flow.shape[0]
        ap = s.get("apex_pos", -1)
        idx = ap if (ap is not None and 1 <= ap < L) else L // 2
        return flow[idx].astype(np.float32)   # (H,W,2)

    def __getitem__(self, i):
        s = self.samples[i]
        f = self._apex_field(s)               # (H,W,2)
        H, W, _ = f.shape
        # spatial crop
        cs = self.cfg.get("base_size", H)
        # (flow cached at H=W=128); random/center crop to keep a margin then resize
        crop = self.cfg.get("crop", H)
        if self.train:
            top = np.random.randint(0, H - crop + 1) if H > crop else 0
            left = np.random.randint(0, W - crop + 1) if W > crop else 0
        else:
            top, left = (H - crop) // 2, (W - crop) // 2
        f = f[top:top + crop, left:left + crop, :]
        flip = self.train and self.cfg.get("hflip", True) and np.random.rand() < 0.5
        if flip:
            f = f[:, ::-1, :].copy()
            f[..., 0] = -f[..., 0]
        u, v = f[..., 0], f[..., 1]
        st = strain_field(u, v)
        c, sc = self.flow_clip, self.strain_clip
        u = np.clip(u, -c, c) / c
        v = np.clip(v, -c, c) / c
        st = np.clip(st, 0, sc) / sc * 2.0 - 1.0
        img = np.stack([u, v, st], axis=0)    # (3,crop,crop)
        # resize to img_size
        if img.shape[1] != self.s:
            t = torch.from_numpy(img).unsqueeze(0).float()
            t = torch.nn.functional.interpolate(t, size=(self.s, self.s),
                                                mode="bilinear", align_corners=False)
            img = t.squeeze(0).numpy()
        return torch.from_numpy(img).float(), int(s["label"])
