"""Measure how far accuracy falls when the camera records fewer frames per second.

CASME II is recorded at 200 fps. A phone records ~30 fps. A micro-expression
lasting ~0.2 s therefore spans ~41 frames in the dataset but only ~6 frames on a
phone, while the model consumes T=16. This asks what that costs, before any app
code is written on top of an assumption that may not hold.

The simulation is faithful because the cached flow is onset-referenced:
``arr[t]`` is TV-L1(onset, frame_t), so dropping intermediate frames is exactly
equivalent to a camera never having captured them. No recomputation is implied.

Evaluation-side only: weights are reused from saved fold checkpoints, so nothing
is retrained and the sealed audit split is never touched. Label-free apex
estimation runs on the decimated array, matching deployment.

    python src/eval_fps_robustness.py \
        --source_runs experiments/protocol_v2/iter_47_r3d_auto_apex_s42_v2_dev_p5 \
        --fps 200 120 60 30 \
        --out experiments/protocol_v2/robustness/fps_auto_apex_s42.json
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from casme.training.engine import predict
from casme.evaluation.metrics import compute_metrics
from casme.models.models import build_model
from casme.evaluation.protocol_v2 import load_or_create_protocol, protocol_subject_sets
from casme.training.run_protocol_v2 import _load_problem

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_FPS = 200.0


def decimate(array, source_fps, target_fps):
    """Keep only the frames a slower camera would have captured."""
    if target_fps >= source_fps:
        return array
    length = array.shape[0]
    step = source_fps / float(target_fps)
    indices = np.unique(np.round(np.arange(0.0, length, step)).astype(int))
    indices = indices[indices < length]
    if indices.size < 2:
        indices = np.array([0, length - 1], dtype=int)
    return array[indices]


def load_source(directory):
    with open(os.path.join(directory, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("role") != "dev" or not summary.get("complete", False):
        raise ValueError(f"source must be a complete development run: {directory}")
    if not summary.get("fold_checkpoints_saved", False):
        raise ValueError(
            f"source run saved no fold checkpoints, cannot re-evaluate: {directory}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_runs", nargs="+", required=True,
                        help="complete dev runs that saved fold checkpoints")
    parser.add_argument("--weights", nargs="*", type=float, default=None,
                        help="fusion weights; defaults to equal weighting")
    parser.add_argument("--fps", nargs="+", type=float,
                        default=[200.0, 120.0, 60.0, 30.0])
    parser.add_argument("--protocol", default="protocols/accuracy_v5.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs = [os.path.abspath(d) for d in args.source_runs]
    summaries = [load_source(d) for d in runs]
    if args.weights:
        if len(args.weights) != len(runs):
            raise ValueError("one weight is required per source run")
        weights = np.asarray(args.weights, dtype=np.float64)
    else:
        weights = np.ones(len(runs), dtype=np.float64)
    weights = weights / weights.sum()

    fingerprints = {s["protocol_samples_sha256"] for s in summaries}
    if len(fingerprints) != 1:
        raise ValueError("source runs were evaluated on different protocols")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg0 = summaries[0]["config"]
    num_classes = len(cfg0["class_names"])
    samples, arrays = _load_problem(cfg0, print)
    protocol_path = (args.protocol if os.path.isabs(args.protocol)
                     else os.path.join(REPO, args.protocol))
    protocol = load_or_create_protocol(samples, num_classes, protocol_path)
    if protocol["samples_sha256"] != summaries[0]["protocol_samples_sha256"]:
        raise ValueError("cached protocol does not match the source runs")
    _, dev_subjects, dev_folds = protocol_subject_sets(protocol)

    # Frame statistics must be quoted over the same samples the metrics are
    # computed on, otherwise the counts read as nonsense (e.g. "235 of 192").
    dev_samples = [row for row in samples if row["subject"] in dev_subjects]
    lengths = np.asarray([arrays[row["key"]].shape[0] for row in dev_samples])
    print(f"clip length at {SOURCE_FPS:.0f} fps over {len(dev_samples)} dev samples: "
          f"median {np.median(lengths):.0f} frames, min {lengths.min()}, "
          f"max {lengths.max()}")

    results = []
    for target_fps in args.fps:
        started = time.time()
        decimated = {key: decimate(value, SOURCE_FPS, target_fps)
                     for key, value in arrays.items()}
        kept = np.asarray([decimated[row["key"]].shape[0] for row in dev_samples])
        pooled_keys = []
        pooled_labels, pooled_probs, pooled_subjects = [], [], []
        per_fold = []
        for fold_id, validation_subjects in dev_folds:
            validation = [row for row in samples
                          if row["subject"] in validation_subjects]
            fused = None
            for directory, summary, weight in zip(runs, summaries, weights):
                cfg = summary["config"]
                paths = sorted(glob.glob(os.path.join(
                    directory, "fold_checkpoints", f"fold{fold_id}_ep*.pt")))
                if not paths:
                    raise ValueError(
                        f"no checkpoints for fold {fold_id} in {directory}")
                model_cfg = dict(cfg)
                model_cfg["pretrained"] = False
                accumulator = None
                for path in paths:
                    model = build_model(model_cfg, num_classes).to(device)
                    model.load_state_dict(torch.load(path, map_location=device))
                    model.eval()
                    probabilities = predict(model, validation, decimated, cfg,
                                            device, tta=cfg.get("tta", 1))
                    accumulator = (probabilities if accumulator is None
                                   else accumulator + probabilities)
                    del model
                member = accumulator / len(paths)
                fused = (weight * member if fused is None
                         else fused + weight * member)
            labels = np.asarray([row["label"] for row in validation])
            metrics = compute_metrics(labels, fused.argmax(1), num_classes)
            per_fold.append({"fold": fold_id, "UF1": metrics["UF1"],
                             "UAR": metrics["UAR"], "ACC": metrics["ACC"]})
            pooled_labels.extend(labels.tolist())
            pooled_probs.extend(fused)
            pooled_subjects.extend(row["subject"] for row in validation)
            pooled_keys.extend(row["key"] for row in validation)
            torch.cuda.empty_cache()

        pooled_probs = np.asarray(pooled_probs, dtype=np.float32)
        metrics = compute_metrics(pooled_labels, pooled_probs.argmax(1), num_classes)

        # Emit a protocol-shaped run directory per fps so that compare_protocol.py
        # can gate these against each other instead of eyeballing point scores.
        fps_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                               f"{os.path.splitext(os.path.basename(args.out))[0]}"
                               f"_fps{int(target_fps)}")
        os.makedirs(fps_dir, exist_ok=True)
        np.savez(os.path.join(fps_dir, "probs.npz"),
                 keys=np.asarray(pooled_keys), subject=np.asarray(pooled_subjects),
                 label=np.asarray(pooled_labels), probs=pooled_probs)
        with open(os.path.join(fps_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump({"name": os.path.basename(fps_dir), "role": "dev",
                       "complete": True, "num_classes": num_classes,
                       "fps": target_fps, "source_runs": runs,
                       "weights": weights.tolist(),
                       "config_sha256": summaries[0]["config_sha256"],
                       "protocol_samples_sha256":
                           summaries[0]["protocol_samples_sha256"],
                       "fold_checkpoints_saved": False, **metrics}, handle, indent=2)
        row = {
            "fps": target_fps,
            "median_frames": float(np.median(kept)),
            "min_frames": int(kept.min()),
            "frames_below_T": int((kept < cfg0["T"]).sum()),
            "n_samples": int(len(pooled_labels)),
            "UF1": metrics["UF1"], "UAR": metrics["UAR"], "ACC": metrics["ACC"],
            "per_fold": per_fold, "elapsed_sec": time.time() - started,
        }
        results.append(row)
        print(f"{target_fps:6.0f} fps | median {row['median_frames']:5.1f} frames "
              f"| {row['frames_below_T']:3d}/{row['n_samples']} clips shorter than T "
              f"| UF1={row['UF1']:.4f} UAR={row['UAR']:.4f} ACC={row['ACC']:.4f}")

    baseline = next((r for r in results if r["fps"] == SOURCE_FPS), results[0])
    for row in results:
        row["delta_UF1_vs_source"] = row["UF1"] - baseline["UF1"]

    report = {
        "source_runs": runs, "weights": weights.tolist(),
        "source_fps": SOURCE_FPS, "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print()
    print(pd.DataFrame([{k: v for k, v in r.items() if k != "per_fold"}
                        for r in results]).to_string(index=False))


if __name__ == "__main__":
    main()
