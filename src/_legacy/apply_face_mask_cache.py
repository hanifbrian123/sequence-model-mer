"""Write a face-parsed copy of a flow cache: flow * per-sample FaceMesh mask.

Baking the mask into a separate cache keeps the training path untouched — a
config only swaps ``cache_dir`` — so a parsed run differs from its baseline in
exactly one variable and stays reproducible from the config hash alone.

Measured on cache/flow144, the parse removes 4.7% of motion energy on average
(the soft ellipse of iter_49 removed 28.5%), and 83% of what it removes sits in
the bottom band: chin and neck, not hair. The forehead — the densest band in the
frame — is kept, which is the opposite of what the ellipse did to it.
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_dir", default="cache/flow144")
    parser.add_argument("--mask_dir", default="cache/facemask144")
    parser.add_argument("--out_dir", default="cache/flow144_parse")
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    flow_dir = os.path.join(repo, args.flow_dir)
    mask_dir = os.path.join(repo, args.mask_dir)
    out_dir = os.path.join(repo, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    manifest = pd.read_csv(os.path.join(flow_dir, "manifest.csv"))
    written = missing = 0
    removed_energy = []
    for position, row in manifest.iterrows():
        key = row["key"]
        mask_path = os.path.join(mask_dir, key + ".npy")
        flow = np.load(os.path.join(flow_dir, row["npy"]))
        dtype = flow.dtype
        if not os.path.exists(mask_path):
            missing += 1
            np.save(os.path.join(out_dir, row["npy"]), flow)
            continue
        mask = np.load(mask_path).astype(np.float32) / 255.0
        if mask.shape != flow.shape[1:3]:
            raise ValueError(
                f"mask {mask.shape} does not match flow {flow.shape[1:3]} for {key}")
        before = np.abs(flow.astype(np.float32)).sum()
        masked = flow.astype(np.float32) * mask[None, :, :, None]
        removed_energy.append(1.0 - float(np.abs(masked).sum()) / max(1e-9, float(before)))
        np.save(os.path.join(out_dir, row["npy"]), masked.astype(dtype))
        written += 1
        if (position + 1) % 50 == 0:
            print(f"  {position + 1}/{len(manifest)}", flush=True)

    shutil.copy(os.path.join(flow_dir, "manifest.csv"),
                os.path.join(out_dir, "manifest.csv"))
    removed_energy = np.asarray(removed_energy)
    print(f"DONE masked={written} mask_missing={missing} -> {out_dir}")
    if len(removed_energy):
        print(f"energy removed: mean={removed_energy.mean()*100:.2f}% "
              f"max={removed_energy.max()*100:.2f}%")


if __name__ == "__main__":
    main()
