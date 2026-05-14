"""Cache CASME II sequences to disk as uint8 arrays (L, H, W, 3) RGB.

Reads frames in [onset, offset] per sample, resizes to a base size, and saves
one .npy per sample. Temporal sampling to fixed T is done later at train time.
Pixels are only processed programmatically (never displayed).
"""
import os
import argparse
import numpy as np
import pandas as pd
import cv2

CROPPED = r"D:/AI-Projects/casmeII-facesleuth-r/dataset/Cropped"


def frame_path(sub_folder, filename, n):
    return os.path.join(CROPPED, sub_folder, filename, f"reg_img{n}.jpg")


def cache_sample(row, base_size, out_dir):
    sub_folder, filename = row["sub_folder"], row["filename"]
    onset, offset = int(row["onset"]), int(row["offset"])
    key = f"{sub_folder}__{filename}"
    out_path = os.path.join(out_dir, key + ".npy")
    if os.path.exists(out_path):
        return out_path, None  # already cached

    frames = []
    for n in range(onset, offset + 1):
        p = frame_path(sub_folder, filename, n)
        if not os.path.exists(p):
            continue  # skip any gap in numbering
        img = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            continue
        img = cv2.resize(img, (base_size, base_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    if len(frames) < 2:
        return None, f"too few frames for {key}"
    arr = np.stack(frames, axis=0).astype(np.uint8)  # (L,H,W,3)
    np.save(out_path, arr)
    return out_path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.index)
    manifest = []
    errors = []
    for i, row in df.iterrows():
        path, err = cache_sample(row, args.base_size, args.out_dir)
        if err:
            errors.append(err)
            continue
        manifest.append({
            "key": f"{row['sub_folder']}__{row['filename']}",
            "npy": os.path.basename(path),
            "subject": int(row["subject"]),
            "label": int(row["label"]),
            "emotion": row.get("emotion", ""),
            "apex": int(row["apex"]),
            "onset": int(row["onset"]),
            "offset": int(row["offset"]),
        })
        if (i + 1) % 50 == 0:
            print(f"cached {i+1}/{len(df)}")
    mdf = pd.DataFrame(manifest)
    man_path = os.path.join(args.out_dir, "manifest.csv")
    mdf.to_csv(man_path, index=False)
    print(f"DONE cached={len(manifest)} errors={len(errors)} base_size={args.base_size}")
    for e in errors:
        print("ERR:", e)
    print("manifest:", man_path)


if __name__ == "__main__":
    main()
