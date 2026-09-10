"""ONSET-FREE app inference for CASME II micro-expression (realtime-friendly).

Unlike infer.py (which needs the onset frame to compute onset-referenced flow),
this runs on ANY sliding window of T frames from a camera — NO onset detection /
spotting required. It uses the frame-difference ("diff") motion representation
(cheap: just subtraction) with the r3d_18 model trained on it.

Deployable model: train with train_final.py on an onset-free diff config, e.g.
  python src/train_final.py --config configs/iter_36_r3d_diff_res128.json \
      --seeds 42 123 2024 --out models/onsetfree_diff

Usage:
  python src/infer_onsetfree.py --frames <folder> --models models/onsetfree_diff
"""
import os
import re
import sys
import json
import argparse
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from casme.models.models import build_model
from casme.serving.inference_utils import predict_array

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_RE = re.compile(r"(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def list_frames(folder):
    files = []
    for fn in os.listdir(folder):
        m = IMG_RE.search(fn)
        if m:
            files.append((int(m.group(1)), os.path.join(folder, fn)))
    files.sort()
    return [p for _, p in files]


def load_clip(frame_paths, base):
    """Read frames -> (L, base, base, 3) uint8 RGB. No onset reference."""
    out = []
    for p in frame_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)               # BGR
        img = cv2.resize(img, (base, base), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(out, axis=0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="folder with a window of frames")
    ap.add_argument("--models", default="models/onsetfree_diff")
    args = ap.parse_args()

    mdir = os.path.join(REPO, args.models) if not os.path.isabs(args.models) else args.models
    deploy = json.load(open(os.path.join(mdir, "deploy.json")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(deploy)
    cfg["pretrained"] = False

    models = []
    for ck in deploy["checkpoints"]:
        m = build_model(cfg, deploy["num_classes"]).to(device)
        m.load_state_dict(torch.load(os.path.join(mdir, ck), map_location=device))
        m.eval()
        models.append(m)

    frames = list_frames(args.frames)
    if len(frames) < 2:
        raise SystemExit(f"need >=2 frames in {args.frames}, found {len(frames)}")
    clip_full = load_clip(frames, deploy["base_size"])      # (L,base,base,3)

    input_mode = deploy.get("input_mode", "diff")
    probs = predict_array(models, clip_full, deploy, device)

    names = deploy["class_names"]
    order = np.argsort(-probs)
    print(f"\nframes: {len(frames)} | models: {len(models)} | input: {input_mode} (ONSET-FREE, no spotting)")
    print(f"PREDICTION: {names[order[0]]}  (p={probs[order[0]]:.3f})\n")
    print("all class probabilities:")
    for i in order:
        print(f"  {names[i]:>12}: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
