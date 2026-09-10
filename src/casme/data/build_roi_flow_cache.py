"""Restrict the flow cache to chosen facial regions ("focus on the AU areas").

This is the destructive counterpart of `RegionAttention`: instead of learning a
weight per region, it hard-selects a subset and zeroes the rest. Worth measuring
rather than assuming, but the cost is visible in the annotation: AU6 (cheek
raiser, 13 samples) and AU14 (dimpler, 27 samples) live in the cheeks, so
dropping cheeks discards the primary cue for 40 of 246 samples.

Cheeks are also the single largest region at 33% of face area, so this is a much
bigger cut than face parsing, which removed only 4.7% of motion energy.
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd

from build_region_masks import REGION_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_dir", default="cache/flow144")
    parser.add_argument("--region_dir", default="cache/regionmask144")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--keep", nargs="+", required=True,
                        help=f"regions to keep, from {REGION_NAMES}")
    parser.add_argument("--feather", type=float, default=0.0,
                        help="unused placeholder; masks are already soft")
    args = parser.parse_args()

    unknown = [name for name in args.keep if name not in REGION_NAMES]
    if unknown:
        raise ValueError(f"unknown regions {unknown}; valid: {REGION_NAMES}")
    keep_index = [REGION_NAMES.index(name) for name in args.keep]

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    flow_dir = os.path.join(repo, args.flow_dir)
    region_dir = os.path.join(repo, args.region_dir)
    out_dir = os.path.join(repo, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    manifest = pd.read_csv(os.path.join(flow_dir, "manifest.csv"))
    kept_area = []
    kept_energy = []
    for position, row in manifest.iterrows():
        masks = np.load(os.path.join(region_dir, row["key"] + ".npy")).astype(np.float32)
        gate = masks[keep_index].sum(axis=0)          # (H,W) in [0,1]
        flow = np.load(os.path.join(flow_dir, row["npy"]))
        dtype = flow.dtype
        magnitude = np.sqrt(flow[..., 0].astype(np.float32) ** 2
                            + flow[..., 1].astype(np.float32) ** 2)
        energy = magnitude[len(magnitude) // 2:].mean(axis=0)
        kept_energy.append(float((energy * gate).sum() / max(1e-9, energy.sum())))
        kept_area.append(float(gate.mean()))
        masked = flow.astype(np.float32) * gate[None, :, :, None]
        np.save(os.path.join(out_dir, row["npy"]), masked.astype(dtype))
        if (position + 1) % 50 == 0:
            print(f"  {position + 1}/{len(manifest)}", flush=True)

    shutil.copy(os.path.join(flow_dir, "manifest.csv"),
                os.path.join(out_dir, "manifest.csv"))
    kept_area = np.asarray(kept_area)
    kept_energy = np.asarray(kept_energy)
    print(f"DONE -> {out_dir}")
    print(f"region dipertahankan : {args.keep}")
    print(f"region dibuang       : {[n for n in REGION_NAMES if n not in args.keep]}")
    print(f"luas dipertahankan   : {kept_area.mean() * 100:.1f}%")
    print(f"energi gerakan dipertahankan: {kept_energy.mean() * 100:.1f}% "
          f"(dibuang {100 - kept_energy.mean() * 100:.1f}%)")


if __name__ == "__main__":
    main()
