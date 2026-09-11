"""
extract_resnet50.py — Precompute 16-frame ResNet-50 features for all CASME II clips.

Faithfully replicates the CNN feature extraction stage of the external repository:
- Pretrained ResNet-50 (ImageNet DEFAULT weights)
- Output: 2048-D feature vector per frame
- 16 frames uniformly sampled per clip
- Saves (16, 2048) float32 numpy arrays in cache/features_resnet50/
- Writes cache/features_resnet50/manifest.csv
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SOURCE_CACHE = os.path.join(REPO, "cache", "frames128")
TARGET_CACHE = os.path.join(REPO, "cache", "features_resnet50")


def get_feature_extractor(device):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def sample_indices(length, T=16):
    if length <= T:
        return np.round(np.linspace(0, length - 1, T)).astype(int)
    return np.round(np.linspace(0, length - 1, T)).astype(int)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Extracting ResNet-50 Features ({device}) ===")
    os.makedirs(TARGET_CACHE, exist_ok=True)

    manifest_path = os.path.join(SOURCE_CACHE, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Source manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    print(f"Total samples to extract: {len(manifest)}")

    extractor = get_feature_extractor(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    out_rows = []
    start_time = time.time()

    with torch.no_grad():
        for idx, row in manifest.iterrows():
            src_npy = os.path.join(SOURCE_CACHE, row["npy"])
            arr = np.load(src_npy)  # [L, 128, 128, 3] uint8

            # Sample 16 frames uniformly
            indices = sample_indices(len(arr), T=16)
            sampled_frames = arr[indices]

            # Transform each frame
            tensors = []
            for frame in sampled_frames:
                pil_img = Image.fromarray(frame)
                tensors.append(transform(pil_img))

            batch = torch.stack(tensors).to(device)  # [16, 3, 224, 224]
            features = extractor(batch).cpu().numpy().astype(np.float32)  # [16, 2048]

            out_fn = f"{row['key']}.npy"
            out_path = os.path.join(TARGET_CACHE, out_fn)
            np.save(out_path, features)

            new_row = dict(row)
            new_row["npy"] = out_fn
            out_rows.append(new_row)

            if (idx + 1) % 25 == 0 or (idx + 1) == len(manifest):
                print(f"  [{idx + 1:3d}/{len(manifest)}] Extracted {row['key']} -> {features.shape}")

    out_manifest = pd.DataFrame(out_rows)
    out_manifest.to_csv(os.path.join(TARGET_CACHE, "manifest.csv"), index=False)
    elapsed = time.time() - start_time
    print(f"Extraction complete in {elapsed:.1f}s. Manifest saved to {TARGET_CACHE}/manifest.csv")


if __name__ == "__main__":
    main()
