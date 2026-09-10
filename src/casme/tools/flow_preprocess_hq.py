"""Accuracy-first stabilized high-quality TV-L1 cache builder.

Licensed frames are processed programmatically and are never displayed.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from casme.data.flow_pipeline import make_tvl1, onset_flow_from_grays, resize_gray


CROPPED = r"D:/AI-Projects/casmeII-facesleuth-r/dataset/Cropped"


def _frame_path(row, number):
    return os.path.join(CROPPED, row["sub_folder"], row["filename"],
                        f"reg_img{number}.jpg")


def cache_sample(row, args, optical_flow):
    key = f'{row["sub_folder"]}__{row["filename"]}'
    output_path = os.path.join(args.out_dir, key + ".npy")
    if os.path.exists(output_path):
        return output_path, None, 0
    grays = []
    for number in range(int(row["onset"]), int(row["offset"]) + 1):
        path = _frame_path(row, number)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            grays.append(resize_gray(image, args.base_size))
    if len(grays) < 2:
        return None, f"too few frames for {key}", 0
    flows, failures = onset_flow_from_grays(
        grays, preset=args.preset, stabilize=args.stabilize,
        clahe=args.clahe, optical_flow=optical_flow)
    temporary_path = output_path + f".tmp.{os.getpid()}.npy"
    np.save(temporary_path, flows.astype(np.float16))
    os.replace(temporary_path, output_path)
    return output_path, None, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_size", type=int, default=176)
    parser.add_argument("--preset", choices=("default", "hq"), default="hq")
    parser.add_argument("--stabilize", choices=("none", "translation", "euclidean", "affine"),
                        default="euclidean")
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--finalize_only", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    cv2.setNumThreads(int(os.environ.get("CV2_THREADS", "6")))
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    errors = []
    total_failures = 0
    start = time.time()
    index = pd.read_csv(args.index)
    optical_flow = make_tvl1(args.preset)
    if args.finalize_only:
        missing = []
        for _, row in index.iterrows():
            key = f'{row["sub_folder"]}__{row["filename"]}'
            path = os.path.join(args.out_dir, key + ".npy")
            if not os.path.exists(path):
                missing.append(key)
                continue
            manifest.append({
                "key": key, "npy": os.path.basename(path),
                "subject": int(row["subject"]), "label": int(row["label"]),
                "emotion": row.get("emotion", ""), "apex": int(row["apex"]),
                "onset": int(row["onset"]), "offset": int(row["offset"]),
            })
        if missing:
            raise RuntimeError(f"cannot finalize: {len(missing)} cached arrays missing")
        pd.DataFrame(manifest).to_csv(
            os.path.join(args.out_dir, "manifest.csv"), index=False)
        metadata = {
            "base_size": args.base_size, "flow_preset": args.preset,
            "flow_stabilize": args.stabilize, "flow_clahe": args.clahe,
            "cached_samples": len(manifest), "errors": 0,
            "finalized": True,
        }
        with open(os.path.join(args.out_dir, "preprocess.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print(json.dumps(metadata, indent=2), flush=True)
        return
    for position, (_, row) in enumerate(index.iterrows(), 1):
        if (position - 1) % args.num_shards != args.shard_index:
            continue
        path, error, failures = cache_sample(row, args, optical_flow)
        total_failures += failures
        if error:
            errors.append(error)
            continue
        manifest.append({
            "key": f'{row["sub_folder"]}__{row["filename"]}',
            "npy": os.path.basename(path), "subject": int(row["subject"]),
            "label": int(row["label"]), "emotion": row.get("emotion", ""),
            "apex": int(row["apex"]), "onset": int(row["onset"]),
            "offset": int(row["offset"]),
        })
        if position % 10 == 0:
            print(f"HQ flow cached {position}/{len(index)} elapsed={time.time()-start:.0f}s",
                  flush=True)
    manifest_name = ("manifest.csv" if args.num_shards == 1
                     else f"manifest.shard{args.shard_index:02d}.csv")
    pd.DataFrame(manifest).to_csv(os.path.join(args.out_dir, manifest_name), index=False)
    metadata = {
        "base_size": args.base_size, "flow_preset": args.preset,
        "flow_stabilize": args.stabilize, "flow_clahe": args.clahe,
        "cached_samples": len(manifest), "errors": len(errors),
        "stabilization_failures": total_failures,
        "elapsed_sec": time.time() - start,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
    }
    metadata_name = ("preprocess.json" if args.num_shards == 1
                     else f"preprocess.shard{args.shard_index:02d}.json")
    with open(os.path.join(args.out_dir, metadata_name), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)
    for error in errors:
        print("ERR:", error)


if __name__ == "__main__":
    main()
