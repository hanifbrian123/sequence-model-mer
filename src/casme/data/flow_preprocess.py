"""Precompute onset-referenced TV-L1 optical flow for each CASME II sequence.

For each frame i in [onset, offset], compute flow(onset -> i) at base_size.
Stored as (L, H, W, 2) float16 .npy. Motion-only representation removes static
appearance -> far less identity overfitting than raw RGB.
Pixels are only processed programmatically (never displayed).
"""
import os
import argparse
import numpy as np
import pandas as pd
import cv2

CROPPED = r"D:/AI-Projects/casmeII-facesleuth-r/dataset/Cropped"


def make_tvl1():
    tv = cv2.optflow.DualTVL1OpticalFlow_create()
    return tv


def frame_path(sub_folder, filename, n):
    return os.path.join(CROPPED, sub_folder, filename, f"reg_img{n}.jpg")


def load_gray(sub_folder, filename, n, base):
    p = frame_path(sub_folder, filename, n)
    if not os.path.exists(p):
        return None
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (base, base), interpolation=cv2.INTER_AREA)


def cache_flow(row, base, out_dir, tv):
    sub_folder, filename = row["sub_folder"], row["filename"]
    onset, offset = int(row["onset"]), int(row["offset"])
    key = f"{sub_folder}__{filename}"
    out_path = os.path.join(out_dir, key + ".npy")
    if os.path.exists(out_path):
        return out_path, None

    g0 = load_gray(sub_folder, filename, onset, base)
    if g0 is None:
        return None, f"no onset frame {key}"
    flows = []
    for n in range(onset, offset + 1):
        g = load_gray(sub_folder, filename, n, base)
        if g is None:
            continue
        f = tv.calc(g0, g, None)  # (H,W,2) float32, flow onset->n
        flows.append(f.astype(np.float16))
    if len(flows) < 2:
        return None, f"too few frames {key}"
    arr = np.stack(flows, axis=0)  # (L,H,W,2) float16
    np.save(out_path, arr)
    return out_path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_size", type=int, default=128)
    args = ap.parse_args()

    cv2.setNumThreads(int(os.environ.get("CV2_THREADS", "6")))
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.index)
    tv = make_tvl1()
    manifest, errors = [], []
    import time
    t0 = time.time()
    for i, row in df.iterrows():
        path, err = cache_flow(row, args.base_size, args.out_dir, tv)
        if err:
            errors.append(err); continue
        manifest.append({
            "key": f"{row['sub_folder']}__{row['filename']}",
            "npy": os.path.basename(path),
            "subject": int(row["subject"]), "label": int(row["label"]),
            "emotion": row.get("emotion", ""), "apex": int(row["apex"]),
            "onset": int(row["onset"]), "offset": int(row["offset"]),
        })
        if (i + 1) % 25 == 0:
            print(f"flow cached {i+1}/{len(df)}  elapsed={time.time()-t0:.0f}s", flush=True)
    mdf = pd.DataFrame(manifest)
    mdf.to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)
    print(f"DONE flow cached={len(manifest)} errors={len(errors)} "
          f"base={args.base_size} elapsed={time.time()-t0:.0f}s")
    for e in errors:
        print("ERR:", e)


if __name__ == "__main__":
    main()
