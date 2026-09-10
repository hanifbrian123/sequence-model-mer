"""Build per-sample face-parsing masks with MediaPipe FaceMesh.

Motivation, measured on this cache rather than assumed: 27.5% of the optical
flow energy sits outside the coarse face ellipse, and the top band (hairline /
forehead) is the densest region of the whole frame at 1.35x average density.
A soft ellipse (iter_49) cannot separate that: it clips corners and jaw
indiscriminately and scored 0.6370 against a 0.6816 baseline.

This produces a real parse instead: the convex hull of the 468 FaceMesh
landmarks, which follows the actual face boundary of each subject, so hair,
neck, ears and background are excluded while every pixel of facial skin, brows,
eyes, nose and mouth is kept.

Masks are computed once and cached as uint8 arrays. No frame is displayed or
exported; only the derived binary mask is written.
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

# Landmarks that trace the face silhouette in MediaPipe FaceMesh.
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]


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


def _mask_from_landmarks(points, size, dilate_fraction, feather_fraction):
    """Convex hull of the mesh, slightly grown, with a feathered edge.

    A hard edge would inject a synthetic high-contrast boundary into the flow
    field; the feather keeps the transition smooth so the network does not see
    an artificial gradient where none exists on the face.
    """
    hull = cv2.convexHull(points.astype(np.float32))
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    grow = max(0, int(round(size * dilate_fraction)))
    if grow:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
        mask = cv2.dilate(mask, kernel)
    blur = max(0, int(round(size * feather_fraction)))
    if blur:
        ksize = 2 * blur + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return mask


def build(frames_dir, out_dir, size, probe_frames, dilate_fraction,
          feather_fraction):
    os.makedirs(out_dir, exist_ok=True)
    manifest = pd.read_csv(os.path.join(frames_dir, "manifest.csv"))
    face_mesh = _mesh()
    written = failed = 0
    coverage = []
    fallback_keys = []
    for position, row in manifest.iterrows():
        key = row["key"]
        out_path = os.path.join(out_dir, key + ".npy")
        clip = np.load(os.path.join(frames_dir, row["npy"]))   # (L,H,W,3) RGB
        length = clip.shape[0]
        # Probe a few frames across the clip: a single frame can fail detection
        # on a blink or a partly occluded face, and the crops are registered so
        # the hulls agree closely when detection does succeed.
        indices = np.unique(np.linspace(
            0, length - 1, min(probe_frames, length)).round().astype(int))
        hulls = []
        for index in indices:
            frame = clip[index]
            if frame.shape[0] != size:
                frame = cv2.resize(frame, (size, size),
                                   interpolation=cv2.INTER_LINEAR)
            points = _landmarks(face_mesh, np.ascontiguousarray(frame))
            if points is not None:
                hulls.append(points[FACE_OVAL])
        if hulls:
            merged = np.concatenate(hulls, axis=0)
            mask = _mask_from_landmarks(merged, size, dilate_fraction,
                                        feather_fraction)
        else:
            # Never silently drop a sample: fall back to keeping everything so
            # the run stays comparable, and report which samples fell back.
            mask = np.full((size, size), 255, dtype=np.uint8)
            failed += 1
            fallback_keys.append(key)
        np.save(out_path, mask)
        coverage.append(float(mask.mean()) / 255.0)
        written += 1
        if (position + 1) % 50 == 0:
            print(f"  {position + 1}/{len(manifest)}", flush=True)
    face_mesh.close()
    coverage = np.asarray(coverage)
    manifest.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    print(f"DONE written={written} detection_failed={failed} size={size}")
    print(f"mask coverage: mean={coverage.mean()*100:.1f}% "
          f"min={coverage.min()*100:.1f}% max={coverage.max()*100:.1f}%")
    if fallback_keys:
        print(f"fallback (all-ones) keys: {fallback_keys}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", default="cache/frames128")
    parser.add_argument("--out_dir", default="cache/facemask144")
    parser.add_argument("--size", type=int, default=144)
    parser.add_argument("--probe_frames", type=int, default=5)
    parser.add_argument("--dilate_fraction", type=float, default=0.02)
    parser.add_argument("--feather_fraction", type=float, default=0.03)
    args = parser.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build(os.path.join(repo, args.frames_dir), os.path.join(repo, args.out_dir),
          args.size, args.probe_frames, args.dilate_fraction,
          args.feather_fraction)


if __name__ == "__main__":
    main()
