"""Motion-magnified onset-referenced TV-L1 flow cache (linear Eulerian-style).

For each frame i, amplify its deviation from the onset (neutral) frame:
    g_amp_i = g0 + alpha * blur(g_i - g0)
then compute TV-L1 flow(g0 -> g_amp_i). Spatial blur limits noise amplification.
Amplifying the subtle intensity change improves optical-flow SNR for the tiny
motions that define micro-expressions.

Stored as (L,H,W,2) float16 .npy, same format as flow_preprocess.
Pixels processed programmatically only (never displayed).
"""
import os
import argparse
import numpy as np
import pandas as pd
import cv2

CROPPED = r"D:/AI-Projects/casmeII-facesleuth-r/dataset/Cropped"


def frame_path(sub_folder, filename, n):
    return os.path.join(CROPPED, sub_folder, filename, f"reg_img{n}.jpg")


def load_gray(sub_folder, filename, n, base):
    p = frame_path(sub_folder, filename, n)
    if not os.path.exists(p):
        return None
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (base, base), interpolation=cv2.INTER_AREA).astype(np.float32)


def cache_one(row, base, alpha, blur, out_dir, tv):
    sub_folder, filename = row["sub_folder"], row["filename"]
    onset, offset = int(row["onset"]), int(row["offset"])
    key = f"{sub_folder}__{filename}"
    out_path = os.path.join(out_dir, key + ".npy")
    if os.path.exists(out_path):
        return out_path, None
    g0 = load_gray(sub_folder, filename, onset, base)
    if g0 is None:
        return None, f"no onset {key}"
    g0b = cv2.GaussianBlur(g0, (0, 0), blur) if blur > 0 else g0
    flows = []
    for n in range(onset, offset + 1):
        g = load_gray(sub_folder, filename, n, base)
        if g is None:
            continue
        gb = cv2.GaussianBlur(g, (0, 0), blur) if blur > 0 else g
        g_amp = np.clip(g0 + alpha * (gb - g0b), 0, 255).astype(np.uint8)
        f = tv.calc(g0.astype(np.uint8), g_amp, None)  # flow onset -> amplified
        flows.append(f.astype(np.float16))
    if len(flows) < 2:
        return None, f"too few {key}"
    np.save(out_path, np.stack(flows, axis=0))
    return out_path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_size", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=5.0, help="magnification factor")
    ap.add_argument("--blur", type=float, default=1.0, help="gaussian sigma (0=off)")
    args = ap.parse_args()

    cv2.setNumThreads(int(os.environ.get("CV2_THREADS", "6")))
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.index)
    tv = cv2.optflow.DualTVL1OpticalFlow_create()
    manifest, errors = [], []
    import time
    t0 = time.time()
    for i, row in df.iterrows():
        path, err = cache_one(row, args.base_size, args.alpha, args.blur, args.out_dir, tv)
        if err:
            errors.append(err); continue
        manifest.append({
            "key": f"{row['sub_folder']}__{row['filename']}",
            "npy": os.path.basename(path), "subject": int(row["subject"]),
            "label": int(row["label"]), "emotion": row.get("emotion", ""),
            "apex": int(row["apex"]), "onset": int(row["onset"]), "offset": int(row["offset"]),
        })
        if (i + 1) % 25 == 0:
            print(f"mag cached {i+1}/{len(df)} elapsed={time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(manifest).to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)
    print(f"DONE mag cached={len(manifest)} errors={len(errors)} "
          f"alpha={args.alpha} blur={args.blur} base={args.base_size} "
          f"elapsed={time.time()-t0:.0f}s")
    for e in errors:
        print("ERR:", e)


if __name__ == "__main__":
    main()
