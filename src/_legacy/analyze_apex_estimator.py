"""Aggregate-only validation of label-free flow-energy apex estimation.

The script intentionally emits no sample keys, subjects, frames, or flow
visualizations.  It uses development subjects from the fixed protocol and
reports only error aggregates, keeping the licensed data sealed from review.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from dataset import estimate_apex_from_flow
from protocol_v2 import load_or_create_protocol, protocol_subject_sets
from run_protocol_v2 import _apex_position


def _metric(errors, lengths):
    errors = np.asarray(errors, dtype=np.float64)
    lengths = np.maximum(np.asarray(lengths, dtype=np.float64), 1.0)
    return {
        "n": int(len(errors)),
        "mae_frames": float(np.mean(np.abs(errors))),
        "median_ae_frames": float(np.median(np.abs(errors))),
        "mean_normalized_ae": float(np.mean(np.abs(errors) / lengths)),
        "within_2_frames": float(np.mean(np.abs(errors) <= 2)),
        "within_4_frames": float(np.mean(np.abs(errors) <= 4)),
        "bias_frames": float(np.mean(errors)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="cache/flow144")
    parser.add_argument("--protocol", default="protocols/accuracy_v5.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quantiles", type=float, nargs="+",
                        default=[0.8, 0.9, 0.95])
    parser.add_argument("--smooth_radii", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--search_max_fractions", type=float, nargs="+",
                        default=[1.0])
    parser.add_argument("--border_fraction", type=float, default=0.08)
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    manifest = pd.read_csv(os.path.join(cache_dir, "manifest.csv"))
    samples = [
        {"key": row["key"], "subject": int(row["subject"]),
         "label": int(row["label"]), "apex_pos": _apex_position(row)}
        for _, row in manifest.iterrows()
    ]
    protocol = load_or_create_protocol(
        samples, int(manifest["label"].max()) + 1, os.path.abspath(args.protocol))
    _, dev_subjects, _ = protocol_subject_sets(protocol)
    dev_rows = manifest[manifest["subject"].astype(int).isin(dev_subjects)]

    settings = [(float(q), int(r), float(f)) for q in args.quantiles
                for r in args.smooth_radii
                for f in args.search_max_fractions]
    errors = {setting: [] for setting in settings}
    lengths = {setting: [] for setting in settings}
    for _, row in dev_rows.iterrows():
        annotated = _apex_position(row)
        if annotated < 1:
            continue
        flow = np.load(os.path.join(cache_dir, row["npy"]), mmap_mode="r")
        for setting in settings:
            q, radius, fraction = setting
            estimate = estimate_apex_from_flow(
                flow, spatial_quantile=q,
                border_fraction=args.border_fraction,
                smooth_radius=radius,
                search_max_fraction=fraction)
            errors[setting].append(estimate - annotated)
            lengths[setting].append(len(flow))

    results = []
    for q, radius, fraction in settings:
        result = {"spatial_quantile": q, "smooth_radius": radius,
                  "search_max_fraction": fraction,
                  **_metric(errors[(q, radius, fraction)],
                            lengths[(q, radius, fraction)])}
        results.append(result)
    results.sort(key=lambda item: (item["mean_normalized_ae"],
                                   item["mae_frames"],
                                   -item["within_4_frames"]))
    report = {
        "scope": "development subjects only",
        "protocol_samples_sha256": protocol["samples_sha256"],
        "border_fraction": args.border_fraction,
        "best": results[0],
        "candidates": results,
    }
    output = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
