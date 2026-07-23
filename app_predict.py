"""Reusable inference API for apps (see private/INFERENCE_GUIDE.md).

    from app_predict import MicroExpressionModel
    from infer import list_frames  # or roll your own chronological sort
    m = MicroExpressionModel("models/emotion_single")
    print(m.predict(list_frames("path/to/clip_folder")))
"""
import os
import sys
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from models import build_model                              # noqa: E402
from infer import compute_onset_flow                        # noqa: E402
from inference_utils import predict_array                   # noqa: E402


class MicroExpressionModel:
    def __init__(self, models_dir, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        subdirs = [os.path.join(models_dir, d) for d in sorted(os.listdir(models_dir))
                   if os.path.isdir(os.path.join(models_dir, d))
                   and os.path.exists(os.path.join(models_dir, d, "deploy.json"))]
        self.dirs = subdirs if subdirs else [models_dir]   # ensemble vs single
        self.groups = []
        for d in self.dirs:
            dep = json.load(open(os.path.join(d, "deploy.json")))
            cfg = dict(dep)
            cfg["pretrained"] = False
            models = []
            for ck in dep["checkpoints"]:
                m = build_model(cfg, dep["num_classes"]).to(self.device).eval()
                m.load_state_dict(torch.load(os.path.join(d, ck), map_location=self.device))
                models.append(m)
            self.groups.append((models, dep))
        self.class_names = self.groups[0][1]["class_names"]
        if any(dep["class_names"] != self.class_names for _, dep in self.groups):
            raise ValueError("all deployment artifacts must use the same class order")

    @torch.no_grad()
    def predict(self, frame_paths):
        """frame_paths: chronological list of cropped-face frame paths (frame[0]=onset)."""
        flow_cache = {}
        probs = None
        for models, dep in self.groups:
            base_size = dep["base_size"]
            flow_key = (base_size, dep.get("flow_preset", "default"),
                        dep.get("flow_stabilize", "none"), dep.get("flow_clahe", False))
            if flow_key not in flow_cache:
                flow_cache[flow_key] = compute_onset_flow(
                    frame_paths, base_size, dep)
            current = predict_array(models, flow_cache[flow_key], dep, self.device)
            probs = current if probs is None else probs + current
        probs /= len(self.groups)
        i = int(probs.argmax())
        return {"label": self.class_names[i], "confidence": float(probs[i]),
                "probs": {c: float(p) for c, p in zip(self.class_names, probs)}}
