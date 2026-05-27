"""App-facing inference for CASME II micro-expression sequence models.

Given a folder of sequence frames (e.g. reg_img46.jpg .. reg_img86.jpg, the
onset..offset range), computes onset-referenced TV-L1 optical flow, runs the
trained ensemble with light TTA (center + horizontal flip), and prints the
predicted emotion with class probabilities.

Usage:
  python src/infer.py --frames <folder> --models models
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
from dataset import build_flow, sample_indices, _to_cthw
from models import build_model

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


def compute_onset_flow(frame_paths, base):
    tv = cv2.optflow.DualTVL1OpticalFlow_create()
    g0 = cv2.resize(cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE), (base, base),
                    interpolation=cv2.INTER_AREA)
    flows = []
    for p in frame_paths:
        g = cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (base, base),
                       interpolation=cv2.INTER_AREA)
        flows.append(tv.calc(g0, g, None).astype(np.float32))
    return np.stack(flows, axis=0)  # (L,base,base,2)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="folder with sequence frames")
    ap.add_argument("--models", default="models")
    args = ap.parse_args()

    mdir = os.path.join(REPO, args.models) if not os.path.isabs(args.models) else args.models
    deploy = json.load(open(os.path.join(mdir, "deploy.json")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {"backbone": deploy["backbone"], "dropout": deploy["dropout"],
           "pretrained": False, "modality": deploy["modality"],
           "in_channels": deploy.get("in_channels", 3)}

    models = []
    for ck in deploy["checkpoints"]:
        m = build_model(cfg, deploy["num_classes"]).to(device)
        m.load_state_dict(torch.load(os.path.join(mdir, ck), map_location=device))
        m.eval()
        models.append(m)

    frames = list_frames(args.frames)
    if len(frames) < 2:
        raise SystemExit(f"need >=2 frames in {args.frames}, found {len(frames)}")
    flow = compute_onset_flow(frames, deploy["base_size"])  # (L,base,base,2)

    T, s = deploy["T"], deploy["img_size"]
    idx = sample_indices(flow.shape[0], T, train=False)
    clip = flow[idx]
    H, W = clip.shape[1], clip.shape[2]
    top, left = (H - s) // 2, (W - s) // 2

    probs = None
    for flip in [False, True]:  # light TTA
        x = build_flow(clip, top, left, s, flip, deploy["flow_clip"],
                       third=deploy.get("flow_third", "mag"),
                       strain_clip=deploy.get("strain_clip", 1.0))
        xt = _to_cthw(x).unsqueeze(0).to(device)
        for m in models:
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                out = m(xt)
            p = torch.softmax(out.float(), dim=1).cpu().numpy()[0]
            probs = p if probs is None else probs + p
    probs /= (2 * len(models))

    names = deploy["class_names"]
    order = np.argsort(-probs)
    print(f"\nframes: {len(frames)} | models: {len(models)} | flow (onset-ref TV-L1)")
    print(f"PREDICTION: {names[order[0]]}  (p={probs[order[0]]:.3f})\n")
    print("all class probabilities:")
    for i in order:
        print(f"  {names[i]:>12}: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
