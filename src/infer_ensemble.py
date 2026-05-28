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
from dataset import build_flow, sample_indices, _to_cthw
from models import build_model
from infer import list_frames, compute_onset_flow

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
    members = []
    base_size = None
    names = CLS
    for d in dirs:
        dep = json.load(open(os.path.join(d, 'deploy.json')))
        base_size = dep['base_size']  # all members share base_size (144)
        names = dep['class_names']    # use deploy class names (supports objective 6-class)
        cfg = {'backbone': dep['backbone'], 'dropout': dep['dropout'], 'pretrained': False,
               'modality': dep['modality'], 'in_channels': dep.get('in_channels', 3)}
        for ck in dep['checkpoints']:
            m = build_model(cfg, dep['num_classes']).to(device)
            m.load_state_dict(torch.load(os.path.join(d, ck), map_location=device))
            m.eval()
            members.append((m, dep))

    frames = list_frames(args.frames)
    if len(frames) < 2:
        raise SystemExit(f'need >=2 frames, found {len(frames)}')
    flow = compute_onset_flow(frames, base_size)   # (L,base,base,2)

    probs = None
    for m, dep in members:
        T, s = dep['T'], dep['img_size']
        idx = sample_indices(flow.shape[0], T, train=False)
        clip = flow[idx]
        H, W = clip.shape[1], clip.shape[2]
        top, left = (H - s) // 2, (W - s) // 2
        for fl in [False, True]:  # light TTA
            x = build_flow(clip, top, left, s, fl, dep['flow_clip'],
                           third=dep.get('flow_third', 'mag'), strain_clip=dep.get('strain_clip', 1.0))
            xt = _to_cthw(x).unsqueeze(0).to(device)
            with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                out = m(xt)
            p = torch.softmax(out.float(), 1).cpu().numpy()[0]
            probs = p if probs is None else probs + p
    probs /= (2 * len(members))

    order = np.argsort(-probs)
    print(f"\nframes: {len(frames)} | ensemble of {len(members)} models")
    print(f"PREDICTION: {names[order[0]]}  (p={probs[order[0]]:.3f})\n")
    for i in order:
        print(f"  {names[i]:>12}: {probs[i]:.4f}")


if __name__ == '__main__':
    main()
