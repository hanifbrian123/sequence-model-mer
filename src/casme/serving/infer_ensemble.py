"""Ensemble inference: average several trained models over one flow computation.

Each model dir under --models_root has deploy.json + checkpoint(s). Flow is
computed once (onset-referenced TV-L1) at base_size; each model then builds its
own input (3ch [u,v,mag] or 4ch [u,v,mag,strain]) and predicts. Probabilities
are averaged. Reproduces the reported ensemble (~UF1 0.72) offline.

Usage:
  python src/infer_ensemble.py --frames <folder> --models_root models/emotion_ensemble
"""
import os, sys, json, argparse
import numpy as np, torch, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from casme.models.models import build_model
from infer import list_frames, compute_onset_flow
from casme.serving.inference_utils import predict_array

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLS = ['happiness', 'disgust', 'repression', 'surprise', 'others']


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True)
    ap.add_argument('--models_root', default='models/emotion_ensemble')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = os.path.join(REPO, args.models_root) if not os.path.isabs(args.models_root) else args.models_root

    dirs = [os.path.join(root, d) for d in sorted(os.listdir(root))
            if os.path.exists(os.path.join(root, d, 'deploy.json'))]
    if not dirs:
        raise SystemExit(f'no model dirs with deploy.json under {root}')

    # load all models + their deploy meta
    groups = []
    names = CLS
    for d in dirs:
        dep = json.load(open(os.path.join(d, 'deploy.json')))
        if groups and dep['class_names'] != names:
            raise ValueError(f'class order mismatch in {d}')
        names = dep['class_names']
        cfg = dict(dep)
        cfg['pretrained'] = False
        models = []
        for ck in dep['checkpoints']:
            m = build_model(cfg, dep['num_classes']).to(device)
            m.load_state_dict(torch.load(os.path.join(d, ck), map_location=device))
            m.eval()
            models.append(m)
        groups.append((models, dep))

    frames = list_frames(args.frames)
    if len(frames) < 2:
        raise SystemExit(f'need >=2 frames, found {len(frames)}')
    flow_cache = {}
    probs = None
    total_checkpoints = 0
    for models, dep in groups:
        base_size = dep['base_size']
        flow_key = (base_size, dep.get('flow_preset', 'default'),
                    dep.get('flow_stabilize', 'none'), dep.get('flow_clahe', False))
        if flow_key not in flow_cache:
            flow_cache[flow_key] = compute_onset_flow(frames, base_size, dep)
        member_probs = predict_array(models, flow_cache[flow_key], dep, device)
        probs = member_probs if probs is None else probs + member_probs
        total_checkpoints += len(models)
    # Equal weight per independently configured artifact; checkpoints and TTA
    # are averaged within an artifact so one recipe cannot win by file count.
    probs /= len(groups)

    order = np.argsort(-probs)
    print(f"\nframes: {len(frames)} | ensemble of {len(groups)} artifacts "
          f"({total_checkpoints} checkpoints)")
    print(f"PREDICTION: {names[order[0]]}  (p={probs[order[0]]:.3f})\n")
    for i in order:
        print(f"  {names[i]:>12}: {probs[i]:.4f}")


if __name__ == '__main__':
    main()
