"""Datasets turning cached clips into (C,T,H,W) tensors.

Single-stream (SeqDataset): rgb / onset_ref / diff / flow.
Two-stream (TwoStreamDataset): shares temporal indices + spatial crop/flip across
a motion (flow) stream and an appearance stream so both are frame-aligned.
Arrays are preloaded into RAM (dict key->np array) for fast LOSO reuse.
"""
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def estimate_apex_from_flow(flow, spatial_quantile=0.9, border_fraction=0.08,
                            smooth_radius=0, search_max_fraction=0.55):
    """Estimate an expression apex from onset-referenced optical flow.

    This is deliberately label-free so the same temporal endpoint can be used
    for uploaded videos.  Per-frame motion energy is the mean of the strongest
    spatial responses inside a lightly cropped face region; a short temporal
    smoother makes the estimate less sensitive to a single noisy flow field.
    """
    flow = np.asarray(flow)
    if flow.ndim != 4 or flow.shape[-1] < 2:
        raise ValueError("flow must have shape (T,H,W,2+)")
    length, height, width = flow.shape[:3]
    if length < 2:
        return 0
    margin_y = min(height // 3, max(0, int(round(height * border_fraction))))
    margin_x = min(width // 3, max(0, int(round(width * border_fraction))))
    y1 = height - margin_y if margin_y else height
    x1 = width - margin_x if margin_x else width
    roi = flow[:, margin_y:y1, margin_x:x1, :2].astype(np.float32)
    magnitude = np.sqrt(np.square(roi[..., 0]) + np.square(roi[..., 1]))
    flat = magnitude.reshape(length, -1)
    quantile = float(np.clip(spatial_quantile, 0.0, 1.0))
    thresholds = np.quantile(flat, quantile, axis=1)
    energy = np.asarray([
        values[values >= threshold].mean() if np.any(values >= threshold)
        else values.mean()
        for values, threshold in zip(flat, thresholds)
    ], dtype=np.float64)
    radius = max(0, int(smooth_radius))
    if radius:
        kernel = np.ones(2 * radius + 1, dtype=np.float64)
        numerator = np.convolve(energy, kernel, mode="same")
        denominator = np.convolve(np.ones_like(energy), kernel, mode="same")
        energy = numerator / denominator
    # Frame zero is the reference and cannot be a useful endpoint.
    energy[0] = -np.inf
    max_index = int(np.clip(
        round((length - 1) * float(search_max_fraction)), 1, length - 1))
    return int(np.argmax(energy[:max_index + 1]))


def resolve_apex_position(arr, sample, cfg, train):
    """Resolve annotated or label-free apex according to the experiment role."""
    default_source = cfg.get("apex_source", "annotation")
    source_key = "train_apex_source" if train else "eval_apex_source"
    source = cfg.get(source_key, default_source)
    if source == "annotation":
        return int(sample.get("apex_pos", -1))
    if source == "flow_energy":
        if cfg.get("modality", "rgb") != "flow":
            raise ValueError("flow_energy apex estimation requires flow modality")
        return estimate_apex_from_flow(
            arr,
            spatial_quantile=cfg.get("apex_energy_quantile", 0.95),
            border_fraction=cfg.get("apex_border_fraction", 0.08),
            smooth_radius=cfg.get("apex_smooth_radius", 0),
            search_max_fraction=cfg.get("apex_search_max_fraction", 0.55))
    if source == "fraction":
        fraction = float(cfg.get("apex_fraction", 0.55))
        return int(np.clip(round((len(arr) - 1) * fraction), 1, len(arr) - 1))
    raise ValueError(f"unknown apex source: {source}")


def sample_indices(L, T, train, jitter=True, hi_idx=None, lo_idx=0, phase=0.5):
    """T frame indices spanning [lo_idx, hi_idx]; per-segment temporal jitter in
    train. hi_idx defaults to L-1 (full clip). lo_idx>0 = shifted windowing.
    For deterministic evaluation/TTA, ``phase`` chooses a fixed point inside
    each temporal segment (0=start, 0.5=center, 1=end)."""
    top = L - 1 if hi_idx is None else int(np.clip(hi_idx, 1, L - 1))
    lo = int(np.clip(lo_idx, 0, max(0, top - 1)))
    span = top - lo + 1
    if span <= T:
        return np.round(np.linspace(lo, top, T)).astype(int)
    bounds = np.linspace(lo, top + 1, T + 1)
    idx = []
    for i in range(T):
        a, b = bounds[i], bounds[i + 1]
        if train and jitter:
            c = np.random.uniform(a, b)
        else:
            p = float(np.clip(phase, 0.0, 1.0))
            c = a + p * (b - a)
        idx.append(min(int(c), top))
    return np.array(idx, dtype=int)


def resize_clip(clip, size):
    """Resize a (T,H,W,C) clip spatially to (T,size,size,C) with bilinear interp."""
    if size is None or (clip.shape[1] == size and clip.shape[2] == size):
        return clip
    out = np.empty((clip.shape[0], size, size, clip.shape[3]), dtype=clip.dtype)
    for t in range(clip.shape[0]):
        out[t] = cv2.resize(clip[t], (size, size), interpolation=cv2.INTER_LINEAR)
    return out


def pick_crop(H, W, s, train, mode="center"):
    if train:
        top = np.random.randint(0, H - s + 1) if H > s else 0
        left = np.random.randint(0, W - s + 1) if W > s else 0
    else:
        max_top, max_left = max(0, H - s), max(0, W - s)
        positions = {
            "center": (max_top // 2, max_left // 2),
            "top_left": (0, 0),
            "top_right": (0, max_left),
            "bottom_left": (max_top, 0),
            "bottom_right": (max_top, max_left),
        }
        if mode not in positions:
            raise ValueError(f"unknown deterministic crop mode: {mode}")
        top, left = positions[mode]
    return top, left


def resolve_window_start(L, hi_idx, train, cfg, view=None):
    """Resolve a window start without leaking evaluation randomness into training.

    Training ``rand_start`` remains stochastic. Evaluation shifting is fixed and
    reproducible; a TTA view may explicitly override it with ``start_fraction``.
    Fractions are relative to the available [0, hi] span.
    """
    view = view or {}
    top = L - 1 if hi_idx is None else int(np.clip(hi_idx, 1, L - 1))
    if train and cfg.get("rand_start", False):
        return int(np.random.randint(0, max(1, top // 2 + 1)))
    if train:
        return 0
    if "start_fraction" in view:
        frac = float(view["start_fraction"])
    elif cfg.get("eval_rand_start", False):
        # Legacy flag now means a deterministic shifted evaluation. The former
        # random implementation consumed NumPy RNG every validation epoch and
        # changed the subsequent training trajectory.
        frac = float(cfg.get("eval_start_fraction", 0.25))
    else:
        frac = 0.0
    return int(np.clip(round(top * frac), 0, max(0, top - 1)))


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


def build_flow(flow, top, left, s, flip, flow_clip, third="mag", strain_clip=1.0,
               compensation="none", roi="none"):
    """flow: (T,H,W,2) float -> (T,s,s,C) float32 in ~[-1,1].
    third: "mag" -> [u,v,mag]; "strain" -> [u,v,strain]; "both" -> [u,v,mag,strain]."""
    flow = flow.astype(np.float32)
    if compensation == "translation":
        # A spatial median is robust to the small moving facial regions and
        # estimates residual global head/camera translation without labels.
        translation = np.median(flow, axis=(1, 2), keepdims=True)
        flow = flow - translation
    elif compensation != "none":
        raise ValueError(f"unknown flow compensation: {compensation}")
    if roi != "none":
        height, width = flow.shape[1:3]
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        xx = (xx + 0.5) / max(1, width)
        yy = (yy + 0.5) / max(1, height)
        # Soft ellipse suppresses corners/hair/background while retaining the
        # central registered face. The feather avoids an artificial hard edge.
        distance = np.sqrt(
            np.square((xx - 0.5) / 0.46) + np.square((yy - 0.52) / 0.50))
        mask = np.clip((1.08 - distance) / 0.16, 0.0, 1.0)
        if roi == "upper_face":
            mask *= np.clip((0.68 - yy) / 0.10, 0.0, 1.0)
        elif roi == "lower_face":
            mask *= np.clip((yy - 0.34) / 0.10, 0.0, 1.0)
        elif roi != "face_ellipse":
            raise ValueError(f"unknown flow ROI: {roi}")
        flow = flow * mask[None, :, :, None]
    flow = flow[:, top:top + s, left:left + s, :]
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


def prepare_seq_array(arr, sample, cfg, train=False, view=None):
    """Apply the canonical single-stream preprocessing recipe to one array.

    This function is intentionally shared by ``SeqDataset`` and deployment
    inference. Keeping temporal sampling, crop/flip handling, normalization,
    flow channel construction, and resize in one place prevents an artifact
    from being evaluated with preprocessing different from LOSO.
    """
    view = view or {}
    length = arr.shape[0]
    temporal_span = cfg.get("temporal_span", "onset_offset")
    if "apex_fraction" in view:
        apex_position = int(np.clip(
            round((length - 1) * float(view["apex_fraction"])),
            1, length - 1))
    else:
        apex_position = resolve_apex_position(arr, sample, cfg, train)
    hi = int(apex_position) \
        if temporal_span == "onset_apex" and apex_position >= 1 else None
    lo = resolve_window_start(length, hi, train, cfg, view)
    indices = sample_indices(
        length, cfg["T"], train,
        jitter=cfg.get("temporal_jitter", True), hi_idx=hi, lo_idx=lo,
        phase=view.get("temporal_phase", 0.5))
    clip = arr[indices]
    height, width = clip.shape[1], clip.shape[2]
    size = cfg["img_size"]
    top, left = pick_crop(height, width, size, train,
                          mode=view.get("crop", "center"))
    flip = (train and cfg.get("hflip", True) and np.random.rand() < 0.5) \
        if train else bool(view.get("flip", False))
    if cfg.get("modality", "rgb") == "flow":
        output = build_flow(
            clip, top, left, size, flip, cfg.get("flow_clip", 3.0),
            third=cfg.get("flow_third", "mag"),
            strain_clip=cfg.get("strain_clip", 1.0),
            compensation=cfg.get("flow_compensation", "none"),
            roi=cfg.get("flow_roi", "none"))
    else:
        jitter = cfg.get("color_jitter", 0)
        color_factor = (1.0 + np.random.uniform(-jitter, jitter)) \
            if train and jitter > 0 else None
        output = build_rgb(
            clip, top, left, size, flip, color_factor,
            cfg.get("input_mode", "rgb"))
    erase_probability = cfg.get("random_erase", 0.0)
    if train and erase_probability > 0:
        output = random_erase(output, erase_probability)
    output = resize_clip(output, cfg.get("resize_to", None))
    return _to_cthw(output)


class SeqDataset(Dataset):
    def __init__(self, samples, arrays, cfg, train, view=None):
        self.samples = samples
        self.arrays = arrays
        self.cfg = cfg
        self.train = train
        self.view = view or {}
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
        return prepare_seq_array(
            arr, s, self.cfg, train=self.train, view=self.view), int(s["label"])


class TwoStreamDataset(Dataset):
    """Returns ((x_flow, x_appearance), label) with shared temporal+spatial params."""
    def __init__(self, samples, arrays_a, arrays_b, cfg, train, view=None):
        self.samples = samples
        self.arrays_a = arrays_a   # flow: key -> (L,H,W,2)
        self.arrays_b = arrays_b   # appearance: key -> (L,H,W,3) uint8
        self.cfg = cfg
        self.train = train
        self.view = view or {}
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
        lo = resolve_window_start(L, hi, self.train, self.cfg, self.view)
        idx = sample_indices(L, self.T, self.train, jitter=self.cfg.get("temporal_jitter", True),
                             hi_idx=hi, lo_idx=lo,
                             phase=self.view.get("temporal_phase", 0.5))
        ca, cb = fa[idx], fb[idx]
        H, W = ca.shape[1], ca.shape[2]
        top, left = pick_crop(H, W, self.s, self.train,
                              mode=self.view.get("crop", "center"))
        flip = (self.train and self.cfg.get("hflip", True) and np.random.rand() < 0.5) \
            if self.train else bool(self.view.get("flip", False))
        xa = build_flow(ca, top, left, self.s, flip, self.flow_clip,
                        third=self.cfg.get("flow_third", "mag"),
                        strain_clip=self.cfg.get("strain_clip", 1.0),
                        compensation=self.cfg.get("flow_compensation", "none"))
        cj = self.cfg.get("color_jitter", 0)
        cf = (1.0 + np.random.uniform(-cj, cj)) if (self.train and cj > 0) else None
        xb = build_rgb(cb, top, left, self.s, flip, cf, self.b_input_mode)
        return (_to_cthw(xa), _to_cthw(xb)), int(s["label"])
