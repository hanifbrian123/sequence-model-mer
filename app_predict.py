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
from dataset import build_flow, sample_indices, _to_cthw   # noqa: E402
from models import build_model                              # noqa: E402
from infer import compute_onset_flow                        # noqa: E402


class MicroExpressionModel:
    def __init__(self, models_dir, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        subdirs = [os.path.join(models_dir, d) for d in sorted(os.listdir(models_dir))
                   if os.path.isdir(os.path.join(models_dir, d))
                   and os.path.exists(os.path.join(models_dir, d, "deploy.json"))]
        self.dirs = subdirs if subdirs else [models_dir]   # ensemble vs single
        self.members = []
        for d in self.dirs:
            dep = json.load(open(os.path.join(d, "deploy.json")))
            cfg = {"backbone": dep["backbone"], "dropout": dep["dropout"],
                   "pretrained": False, "modality": dep["modality"],
                   "in_channels": dep.get("in_channels", 3)}
            for ck in dep["checkpoints"]:
                m = build_model(cfg, dep["num_classes"]).to(self.device).eval()
                m.load_state_dict(torch.load(os.path.join(d, ck), map_location=self.device))
                self.members.append((m, dep))
        self.class_names = self.members[0][1]["class_names"]
        self.base_size = self.members[0][1]["base_size"]

    @torch.no_grad()
    def predict(self, frame_paths):
        """frame_paths: chronological list of cropped-face frame paths (frame[0]=onset)."""
        flow = compute_onset_flow(frame_paths, self.base_size)
        probs = None
        for m, dep in self.members:
            T, s = dep["T"], dep["img_size"]
            idx = sample_indices(flow.shape[0], T, train=False)
            clip = flow[idx]
            H, W = clip.shape[1], clip.shape[2]
            top, left = (H - s) // 2, (W - s) // 2
            for flip in (False, True):
                x = build_flow(clip, top, left, s, flip, dep["flow_clip"],
                               third=dep.get("flow_third", "mag"),
                               strain_clip=dep.get("strain_clip", 1.0))
                xt = _to_cthw(x).unsqueeze(0).to(self.device)
                p = torch.softmax(m(xt).float(), 1).cpu().numpy()[0]
                probs = p if probs is None else probs + p
        probs /= (2 * len(self.members))
        i = int(probs.argmax())
        return {"label": self.class_names[i], "confidence": float(probs[i]),
                "probs": {c: float(p) for c, p in zip(self.class_names, probs)}}
