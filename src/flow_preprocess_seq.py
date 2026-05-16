"""Sequential (consecutive-frame) TV-L1 optical flow for CASME II sequences.

Unlike flow_preprocess.py (onset-referenced: flow(onset -> i)), this computes
flow(i -> i+1) between consecutive frames -> instantaneous velocity field. A
genuinely different motion representation; useful as a DECORRELATED fusion member
alongside the onset-referenced flow. Stored (L-1, H, W, 2) float16.
Pixels processed programmatically only (never displayed).
"""
import os, argparse, time
import numpy as np
import pandas as pd
import cv2

CROPPED = r"D:/AI-Projects/casmeII-facesleuth-r/dataset/Cropped"


def load_gray(sub_folder, filename, n, base):
    p = os.path.join(CROPPED, sub_folder, filename, f"reg_img{n}.jpg")
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
    prev = load_gray(sub_folder, filename, onset, base)
    if prev is None:
        return None, f"no onset frame {key}"
    flows = []
    for n in range(onset + 1, offset + 1):
        g = load_gray(sub_folder, filename, n, base)
        if g is None:
            continue
        f = tv.calc(prev, g, None)          # flow prev -> current (consecutive)
        flows.append(f.astype(np.float16))
        prev = g
    if len(flows) < 2:
        return None, f"too few frames {key}"
    np.save(out_path, np.stack(flows, axis=0))
    return out_path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_size", type=int, default=144)
    args = ap.parse_args()
    cv2.setNumThreads(int(os.environ.get("CV2_THREADS", "6")))
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.index)
    tv = cv2.optflow.DualTVL1OpticalFlow_create()
    manifest, errors, t0 = [], [], time.time()
    for i, row in df.iterrows():
        path, err = cache_flow(row, args.base_size, args.out_dir, tv)
        if err:
            errors.append(err); continue
        manifest.append({"key": f"{row['sub_folder']}__{row['filename']}",
                         "npy": os.path.basename(path), "subject": int(row["subject"]),
                         "label": int(row["label"]), "emotion": row.get("emotion", ""),
                         "apex": int(row["apex"]), "onset": int(row["onset"]),
                         "offset": int(row["offset"])})
        if (i + 1) % 25 == 0:
            print(f"seqflow cached {i+1}/{len(df)} elapsed={time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(manifest).to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)
    print(f"DONE seqflow cached={len(manifest)} errors={len(errors)} "
          f"base={args.base_size} elapsed={time.time()-t0:.0f}s")
    for e in errors:
        print("ERR:", e)


if __name__ == "__main__":
    main()
