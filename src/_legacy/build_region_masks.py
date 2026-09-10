"""Per-sample facial REGION masks (a partition, not a cutout).

This is the input side of segmentation attention. Unlike `build_face_mask.py`,
which produces one keep/drop mask, this produces K soft masks that sum to 1 at
every pixel. Nothing is discarded: the model receives the same total signal and
only learns how much to weight each anatomical region.

That distinction matters here. Every destructive spatial experiment in this
project has lost — soft ellipse 0.6370, translation compensation 0.5987, ECC
stabilisation 0.5906, face parsing neutral at best. A partition cannot lose
information, so it tests the "attention" idea without repeating the "removal"
mistake.

Regions follow Action Unit territories rather than a grid, because that is what
the labels are actually made of: brows carry AU1/AU2/AU4, eyes AU5/AU6/AU7,
mouth AU12/AU15/AU17/AU25.
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

# MediaPipe FaceMesh landmark groups.
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 129, 358, 49, 279]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
         0, 37, 39, 40, 185, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LEFT_CHEEK = [116, 117, 118, 119, 120, 100, 142, 205, 206, 207, 187, 123]
RIGHT_CHEEK = [345, 346, 347, 348, 349, 329, 371, 425, 426, 427, 411, 352]
FOREHEAD = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109, 151, 108, 337]

REGIONS = {
    "brow": LEFT_BROW + RIGHT_BROW,
    "eye": LEFT_EYE + RIGHT_EYE,
    "nose": NOSE,
    "mouth": MOUTH,
    "cheek": LEFT_CHEEK + RIGHT_CHEEK,
    "forehead": FOREHEAD,
}
REGION_NAMES = list(REGIONS)


def _mesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.2)


def _landmarks(face_mesh, frame_rgb):
    result = face_mesh.process(frame_rgb)
    if not result.multi_face_landmarks:
        return None
    height, width = frame_rgb.shape[:2]
    points = result.multi_face_landmarks[0].landmark
    return np.asarray([[p.x * width, p.y * height] for p in points],
                      dtype=np.float32)


def region_masks(points, size, blur_fraction=0.06):
    """Soft, normalised masks: (K,size,size) float32 summing to 1 per pixel.

    Each region starts as its landmark hull, is blurred so boundaries are soft
    (a hard edge would inject gradients the face does not have), then the stack
    is normalised. Normalising is what makes this a partition rather than K
    independent crops, and is why the sum of all regions reconstructs the input
    exactly when every weight is 1.
    """
    stack = np.zeros((len(REGION_NAMES), size, size), dtype=np.float32)
    blur = max(1, int(round(size * blur_fraction)))
    ksize = 2 * blur + 1
    for index, name in enumerate(REGION_NAMES):
        mask = np.zeros((size, size), dtype=np.uint8)
        hull = cv2.convexHull(points[REGIONS[name]].astype(np.float32))
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        stack[index] = cv2.GaussianBlur(mask, (ksize, ksize), 0).astype(np.float32)
    total = stack.sum(axis=0, keepdims=True)
    # Pixels no region claims (jaw edge, background) are shared out evenly so
    # the partition still sums to 1 and no signal silently disappears.
    empty = total[0] < 1e-3
    if empty.any():
        stack[:, empty] = 1.0 / len(REGION_NAMES)
        total = stack.sum(axis=0, keepdims=True)
    return stack / np.maximum(total, 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", default="cache/frames128")
    parser.add_argument("--out_dir", default="cache/regionmask144")
    parser.add_argument("--size", type=int, default=144)
    parser.add_argument("--probe_frames", type=int, default=5)
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frames_dir = os.path.join(repo, args.frames_dir)
    out_dir = os.path.join(repo, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    manifest = pd.read_csv(os.path.join(frames_dir, "manifest.csv"))
    face_mesh = _mesh()

    written = failed = 0
    coverage = np.zeros(len(REGION_NAMES))
    for position, row in manifest.iterrows():
        clip = np.load(os.path.join(frames_dir, row["npy"]))
        length = clip.shape[0]
        indices = np.unique(np.linspace(
            0, length - 1, min(args.probe_frames, length)).round().astype(int))
        found = []
        for index in indices:
            frame = clip[index]
            if frame.shape[0] != args.size:
                frame = cv2.resize(frame, (args.size, args.size),
                                   interpolation=cv2.INTER_LINEAR)
            points = _landmarks(face_mesh, np.ascontiguousarray(frame))
            if points is not None:
                found.append(points)
        if found:
            stack = region_masks(np.mean(found, axis=0), args.size)
        else:
            stack = np.full((len(REGION_NAMES), args.size, args.size),
                            1.0 / len(REGION_NAMES), dtype=np.float32)
            failed += 1
        np.save(os.path.join(out_dir, row["key"] + ".npy"),
                stack.astype(np.float16))
        coverage += stack.sum(axis=(1, 2)) / (args.size * args.size)
        written += 1
        if (position + 1) % 50 == 0:
            print(f"  {position + 1}/{len(manifest)}", flush=True)
    face_mesh.close()
    manifest.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    coverage /= max(1, written)
    print(f"DONE written={written} detection_failed={failed} "
          f"regions={len(REGION_NAMES)}")
    print("porsi luas rata-rata per region:")
    for name, value in zip(REGION_NAMES, coverage):
        print(f"   {name:10s} {value * 100:5.1f}%")
    print(f"jumlah semua region = {coverage.sum() * 100:.1f}% "
          f"(harus 100,0% -- bukti tidak ada yang hilang)")


if __name__ == "__main__":
    main()
