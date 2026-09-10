"""Reusable TV-L1 preprocessing for cached training data and uploaded video."""
import cv2
import numpy as np


def make_tvl1(preset="default"):
    """Create TV-L1 with an explicit reproducible quality preset."""
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    if preset == "default":
        return optical_flow
    if preset != "hq":
        raise ValueError(f"unknown TV-L1 preset: {preset}")
    # Accuracy-first settings. Compared with OpenCV defaults, use more pyramid
    # levels and warps and a tighter convergence criterion. This is intentionally
    # expensive and targets uploaded-video inference rather than realtime use.
    optical_flow.setScalesNumber(7)
    optical_flow.setWarpingsNumber(10)
    optical_flow.setOuterIterations(15)
    optical_flow.setInnerIterations(40)
    optical_flow.setEpsilon(0.005)
    optical_flow.setMedianFiltering(5)
    return optical_flow


def resize_gray(image, base_size):
    interpolation = (cv2.INTER_AREA if max(image.shape[:2]) >= base_size
                     else cv2.INTER_CUBIC)
    return cv2.resize(image, (base_size, base_size), interpolation=interpolation)


def normalize_gray(image, clahe=False):
    image = np.asarray(image, dtype=np.uint8)
    if not clahe:
        return image
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)


def stabilize_to_reference(reference, frame, motion="euclidean",
                           iterations=100, epsilon=1e-6):
    """Remove global camera/head motion with label-free ECC registration."""
    if motion == "none":
        return frame, True
    modes = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }
    if motion not in modes:
        raise ValueError(f"unknown stabilization motion: {motion}")
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(iterations), float(epsilon))
    height, width = reference.shape
    mask = np.zeros_like(reference, dtype=np.uint8)
    margin_y = max(1, int(round(height * 0.05)))
    margin_x = max(1, int(round(width * 0.05)))
    mask[margin_y:height - margin_y, margin_x:width - margin_x] = 255
    try:
        cv2.findTransformECC(
            reference, frame, warp, modes[motion], criteria, mask, 5)
        aligned = cv2.warpAffine(
            frame, warp, (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101)
        return aligned, True
    except cv2.error:
        return frame, False


def onset_flow_from_grays(grays, preset="default", stabilize="none",
                          clahe=False, optical_flow=None):
    """Compute onset-referenced flow from same-sized grayscale frames."""
    if len(grays) < 2:
        raise ValueError("at least two frames are required")
    processed = [normalize_gray(frame, clahe=clahe) for frame in grays]
    reference = processed[0]
    tvl1 = optical_flow or make_tvl1(preset)
    flows = []
    stabilization_failures = 0
    for frame in processed:
        if stabilize != "none":
            frame, success = stabilize_to_reference(
                reference, frame, motion=stabilize)
            stabilization_failures += int(not success)
        flows.append(tvl1.calc(reference, frame, None).astype(np.float32))
    return np.stack(flows, axis=0), stabilization_failures
