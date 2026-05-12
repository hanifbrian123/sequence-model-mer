"""Datasets turning cached clips into (C,T,H,W) tensors.

Single-stream (SeqDataset): rgb / onset_ref / diff / flow.
Two-stream (TwoStreamDataset): shares temporal indices + spatial crop/flip across
a motion (flow) stream and an appearance stream so both are frame-aligned.
Arrays are preloaded into RAM (dict key->np array) for fast LOSO reuse.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def sample_indices(L, T, train, jitter=True, hi_idx=None):
    """T frame indices spanning [0, hi_idx]; per-segment temporal jitter in train.
    hi_idx defaults to L-1 (full clip). Set to apex position for onset->apex span."""
    top = L - 1 if hi_idx is None else int(np.clip(hi_idx, 1, L - 1))
    span = top + 1
    if span <= T:
        return np.round(np.linspace(0, top, T)).astype(int)
    bounds = np.linspace(0, span, T + 1)
    idx = []
    for i in range(T):
        lo, hi = bounds[i], bounds[i + 1]
        c = np.random.uniform(lo, hi) if (train and jitter) else (lo + hi) / 2.0
        idx.append(min(int(c), top))
    return np.array(idx, dtype=int)


def pick_crop(H, W, s, train):
    if train:
        top = np.random.randint(0, H - s + 1) if H > s else 0
        left = np.random.randint(0, W - s + 1) if W > s else 0
    else:
        top, left = (H - s) // 2, (W - s) // 2
    return top, left


def build_rgb(clip, top, left, s, flip, color_factor, input_mode):
    """clip: (T,H,W,3) uint8 -> (T,s,s,3) float32, kinetics-normalized."""
    clip = clip[:, top:top + s, left:left + s, :].astype(np.float32) / 255.0
    if flip:
        clip = clip[:, :, ::-1, :].copy()
    if color_factor is not None:
        clip = np.clip(clip * color_factor, 0, 1)
    if input_mode == "onset_ref":
        clip = np.clip((clip - clip[0:1]) * 0.5 + 0.5, 0, 1)
    elif input_mode == "diff":
        d = np.zeros_like(clip); d[1:] = clip[1:] - clip[:-1]
        clip = np.clip(d * 0.5 + 0.5, 0, 1)
    return (clip - KINETICS_MEAN) / KINETICS_STD


def _optical_strain(u, v):
    """Optical strain magnitude from a flow field (per frame).
    e = sqrt(exx^2 + eyy^2 + 2*exy^2), exy = 0.5(du/dy + dv/dx).
    Captures facial tissue deformation — a classic micro-expression feature."""
    out = np.empty(u.shape, dtype=np.float32)
    for t in range(u.shape[0]):
        duy, dux = np.gradient(u[t])   # d/drow(y), d/dcol(x)
        dvy, dvx = np.gradient(v[t])
        exx, eyy = dux, dvy
        exy = 0.5 * (duy + dvx)
        out[t] = np.sqrt(exx * exx + eyy * eyy + 2.0 * exy * exy)
    return out


def build_flow(flow, top, left, s, flip, flow_clip, third="mag", strain_clip=1.0):
    """flow: (T,H,W,2) float -> (T,s,s,C) float32 in ~[-1,1].
    third: "mag" -> [u,v,mag]; "strain" -> [u,v,strain]; "both" -> [u,v,mag,strain]."""
    flow = flow[:, top:top + s, left:left + s, :].astype(np.float32)
    if flip:
        flow = flow[:, :, ::-1, :].copy()
        flow[..., 0] = -flow[..., 0]
    u, v = flow[..., 0], flow[..., 1]
    c = flow_clip
    mag = np.clip(np.sqrt(u * u + v * v), 0, c) / c * 2.0 - 1.0
    strain = np.clip(_optical_strain(u, v), 0, strain_clip) / strain_clip * 2.0 - 1.0
    un = np.clip(u, -c, c) / c
    vn = np.clip(v, -c, c) / c
    if third == "both":
        chans = [un, vn, mag, strain]
    elif third == "strain":
        chans = [un, vn, strain]
    else:
        chans = [un, vn, mag]
    return np.stack(chans, axis=-1)


def random_erase(clip, prob, area_range=(0.02, 0.15), aspect=(0.5, 2.0)):
    """Cutout/random-erasing on a (T,H,W,C) clip. Same spatial box across ALL
    frames (temporal consistency) -> simulates occluding a face region, forcing
    the model to use the rest of the face. Fill = 0 (per-channel neutral-ish).
    Cheap regularizer that directly attacks the large train/test overfit gap."""
    if np.random.rand() > prob:
        return clip
    T, H, W, C = clip.shape
    for _ in range(10):
        area = H * W * np.random.uniform(*area_range)
        ar = np.random.uniform(*aspect)
        h = int(round(np.sqrt(area * ar))); w = int(round(np.sqrt(area / ar)))
        if h < H and w < W:
            top = np.random.randint(0, H - h); left = np.random.randint(0, W - w)
            clip[:, top:top + h, left:left + w, :] = 0.0
            return clip
    return clip


def _to_cthw(clip):
    return torch.from_numpy(np.transpose(clip, (3, 0, 1, 2)).copy()).float()


class SeqDataset(Dataset):
    def __init__(self, samples, arrays, cfg, train):
        self.samples = samples
        self.arrays = arrays
        self.cfg = cfg
        self.train = train
        self.T = cfg["T"]
        self.s = cfg["img_size"]
        self.input_mode = cfg.get("input_mode", "rgb")
        self.modality = cfg.get("modality", "rgb")
        self.flow_clip = cfg.get("flow_clip", 3.0)
        self.span = cfg.get("temporal_span", "onset_offset")  # or "onset_apex"

    def __len__(self):
        return len(self.samples)

    def _hi(self, s, L):
        if self.span == "onset_apex" and s.get("apex_pos", -1) >= 1:
            return int(s["apex_pos"])
        return None

    def __getitem__(self, i):
        s = self.samples[i]
        arr = self.arrays[s["key"]]
        L = arr.shape[0]
        idx = sample_indices(L, self.T, self.train, jitter=self.cfg.get("temporal_jitter", True),
                             hi_idx=self._hi(s, L))
        clip = arr[idx]
        H, W = clip.shape[1], clip.shape[2]
        top, left = pick_crop(H, W, self.s, self.train)
        flip = self.train and self.cfg.get("hflip", True) and np.random.rand() < 0.5
        if self.modality == "flow":
            out = build_flow(clip, top, left, self.s, flip, self.flow_clip,
                             third=self.cfg.get("flow_third", "mag"),
                             strain_clip=self.cfg.get("strain_clip", 1.0))
        else:
            cj = self.cfg.get("color_jitter", 0)
            cf = (1.0 + np.random.uniform(-cj, cj)) if (self.train and cj > 0) else None
            out = build_rgb(clip, top, left, self.s, flip, cf, self.input_mode)
        re = self.cfg.get("random_erase", 0.0)
        if self.train and re > 0:
            out = random_erase(out, re)
        return _to_cthw(out), int(s["label"])


class TwoStreamDataset(Dataset):
    """Returns ((x_flow, x_appearance), label) with shared temporal+spatial params."""
    def __init__(self, samples, arrays_a, arrays_b, cfg, train):
        self.samples = samples
        self.arrays_a = arrays_a   # flow: key -> (L,H,W,2)
        self.arrays_b = arrays_b   # appearance: key -> (L,H,W,3) uint8
        self.cfg = cfg
        self.train = train
        self.T = cfg["T"]
        self.s = cfg["img_size"]
        self.flow_clip = cfg.get("flow_clip", 3.0)
        self.b_input_mode = cfg.get("stream_b_input_mode", "onset_ref")
        self.span = cfg.get("temporal_span", "onset_offset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        fa = self.arrays_a[s["key"]]   # (La,H,W,2)
        fb = self.arrays_b[s["key"]]   # (Lb,H,W,3)  (same L: same onset..offset)
        L = min(fa.shape[0], fb.shape[0])
        hi = int(s["apex_pos"]) if (self.span == "onset_apex" and s.get("apex_pos", -1) >= 1) else None
        idx = sample_indices(L, self.T, self.train, jitter=self.cfg.get("temporal_jitter", True),
                             hi_idx=hi)
        ca, cb = fa[idx], fb[idx]
        H, W = ca.shape[1], ca.shape[2]
        top, left = pick_crop(H, W, self.s, self.train)
        flip = self.train and self.cfg.get("hflip", True) and np.random.rand() < 0.5
        xa = build_flow(ca, top, left, self.s, flip, self.flow_clip,
                        third=self.cfg.get("flow_third", "mag"),
                        strain_clip=self.cfg.get("strain_clip", 1.0))
        cj = self.cfg.get("color_jitter", 0)
        cf = (1.0 + np.random.uniform(-cj, cj)) if (self.train and cj > 0) else None
        xb = build_rgb(cb, top, left, self.s, flip, cf, self.b_input_mode)
        return (_to_cthw(xa), _to_cthw(xb)), int(s["label"])
