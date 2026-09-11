import os
import glob
import numpy as np
import mediapipe as mp
import cv2
from tqdm import tqdm

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_landmarks(frames):
    """
    frames: numpy array (T, H, W, 3) in RGB
    Returns: numpy array (T, 478, 3) where last dim is (x, y, z)
    """
    T, H, W, _ = frames.shape
    landmarks_seq = np.zeros((T, 478, 3), dtype=np.float32)
    
    for i in range(T):
        frame = frames[i]
        if frame.dtype != np.uint8:
            # If frames are float [0, 1] or similar
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
                
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            for j, lm in enumerate(landmarks):
                landmarks_seq[i, j, 0] = lm.x
                landmarks_seq[i, j, 1] = lm.y
                landmarks_seq[i, j, 2] = lm.z
        else:
            # If face not detected, use previous frame's landmarks if i > 0
            if i > 0:
                landmarks_seq[i] = landmarks_seq[i-1]
            # else it stays zeros
            
    return landmarks_seq

def process_dataset(frames_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    frame_files = glob.glob(os.path.join(frames_dir, "*.npy"))
    
    for f in tqdm(frame_files, desc="Extracting MediaPipe Landmarks"):
        basename = os.path.basename(f)
        out_path = os.path.join(out_dir, basename)
        
        if os.path.exists(out_path):
            continue
            
        frames = np.load(f)
        landmarks = extract_landmarks(frames)
        np.save(out_path, landmarks)

if __name__ == "__main__":
    frames_dir = "cache/frames128"
    out_dir = "cache/landmarks478"
    print(f"Extracting landmarks from {frames_dir} to {out_dir}")
    process_dataset(frames_dir, out_dir)
    print("Done!")
