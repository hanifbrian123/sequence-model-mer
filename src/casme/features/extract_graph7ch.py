"""
extract_graph7ch.py — Precompute 7-channel facial graph features for CASME II.

Faithfully replicates the 7-channel graph representation from the external repository:
- Channels:
    CH0, CH1: Normalized (X, Y) coordinates in [0, 1]
    CH2, CH3: Global displacement (X_t - X_0, Y_t - Y_0) from onset/anchor frame
    CH4, CH5: Temporal velocity (X_t - X_{t-1}, Y_t - Y_{t-1})
    CH6:      Geometric magnitude sqrt(dx_global^2 + dy_global^2)
- Nodes: First 468 landmarks of MediaPipe Face Mesh
- Frames: 16 frames uniformly sampled per clip
- Output shape: (7, 16, 468) float32 numpy arrays in cache/graph7ch_468/
- Writes cache/graph7ch_468/manifest.csv
"""
import os
import time
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SOURCE_CACHE = os.path.join(REPO, "cache", "landmarks478")
TARGET_CACHE = os.path.join(REPO, "cache", "graph7ch_468")


def compute_7ch_features(landmarks: np.ndarray, num_nodes: int = 468, num_frames: int = 16) -> np.ndarray:
    """
    Compute 7-channel features from raw landmark sequence.
    
    Args:
        landmarks: np.ndarray of shape (L, >=468, >=2)
        num_nodes: Number of facial nodes to use (default 468)
        num_frames: Number of temporal frames (default 16)
        
    Returns:
        np.ndarray of shape (7, num_frames, num_nodes) float32
    """
    length = len(landmarks)
    indices = np.round(np.linspace(0, length - 1, num_frames)).astype(int)
    lm = landmarks[indices, :num_nodes, :2].astype(np.float32)  # (T=16, V=468, 2)

    anchor = lm[0:1]  # (1, V=468, 2)
    glob_disp = lm - anchor  # (T, V, 2)

    temp_disp = np.zeros_like(lm)  # (T, V, 2)
    temp_disp[1:] = lm[1:] - lm[:-1]

    mag = np.sqrt(glob_disp[..., 0] ** 2 + glob_disp[..., 1] ** 2)[..., None]  # (T, V, 1)

    # Concat along channel axis: [CH0-1, CH2-3, CH4-5, CH6] -> (T, V, 7)
    feat = np.concatenate([lm, glob_disp, temp_disp, mag], axis=-1)

    # Transpose to (C=7, T=16, V=468)
    return np.transpose(feat, (2, 0, 1)).astype(np.float32)


def main():
    print("=== Extracting 7-Channel Facial Graph Features (468 nodes, 16 frames) ===")
    os.makedirs(TARGET_CACHE, exist_ok=True)

    manifest_path = os.path.join(SOURCE_CACHE, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Source manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    print(f"Total samples to extract: {len(manifest)}")

    out_rows = []
    start_time = time.time()

    for idx, row in manifest.iterrows():
        src_npy = os.path.join(SOURCE_CACHE, row["npy"])
        arr = np.load(src_npy)  # [L, 478, 3]

        graph_tensor = compute_7ch_features(arr, num_nodes=468, num_frames=16)

        out_fn = f"{row['key']}.npy"
        out_path = os.path.join(TARGET_CACHE, out_fn)
        np.save(out_path, graph_tensor)

        new_row = dict(row)
        new_row["npy"] = out_fn
        out_rows.append(new_row)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(manifest):
            print(f"  [{idx + 1:3d}/{len(manifest)}] Extracted {row['key']} -> {graph_tensor.shape}")

    out_manifest = pd.DataFrame(out_rows)
    out_manifest.to_csv(os.path.join(TARGET_CACHE, "manifest.csv"), index=False)
    elapsed = time.time() - start_time
    print(f"Extraction complete in {elapsed:.2f}s. Manifest saved to {TARGET_CACHE}/manifest.csv")


if __name__ == "__main__":
    main()
