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
from casme.models.models import build_model
from casme.serving.inference_utils import predict_array
from casme.data.flow_pipeline import onset_flow_from_grays, resize_gray

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


def compute_onset_flow(frame_paths, base, deploy=None):
    deploy = deploy or {}
    grays = []
    for p in frame_paths:
        image = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"could not read frame: {p}")
        grays.append(resize_gray(image, base))
    flow, _ = onset_flow_from_grays(
        grays, preset=deploy.get("flow_preset", "default"),
        stabilize=deploy.get("flow_stabilize", "none"),
        clahe=deploy.get("flow_clahe", False))
    return flow


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="folder with sequence frames")
    ap.add_argument("--models", default="models")
    ap.add_argument("--apex_index", type=int, default=None,
                    help="optional zero-based apex frame; default: flow-energy estimate")
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
    flow = compute_onset_flow(frames, deploy["base_size"], deploy)

    sample = None if args.apex_index is None else {"apex_pos": args.apex_index}
    probs = predict_array(models, flow, deploy, device, sample=sample)

    names = deploy["class_names"]
    order = np.argsort(-probs)
    print(f"\nframes: {len(frames)} | models: {len(models)} | "
          f"views: {len(deploy.get('tta_views', [])) or deploy.get('tta', 1)} | "
          "flow (onset-ref TV-L1)")
    print(f"PREDICTION: {names[order[0]]}  (p={probs[order[0]]:.3f})\n")
    print("all class probabilities:")
    for i in order:
        print(f"  {names[i]:>12}: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
