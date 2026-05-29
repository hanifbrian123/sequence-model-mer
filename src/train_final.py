"""Train the FINAL deployable model(s) on ALL data with the best config.

Trains one model per seed on the full dataset (no held-out fold) and saves
checkpoints + a deploy metadata json. Inference (infer.py) averages the seeds
with TTA, mirroring the best LOSO ensemble config.

Usage:
  python src/train_final.py --config configs/iter_10.json --seeds 42 123 2024 --out models
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import SeqDataset
from models import build_model
from engine import make_loader, class_weights
from run_experiment import DEFAULTS, load_config, preload_arrays

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_all(seed, samples, arrays, cfg, num_classes, device, log):
    cfg = dict(cfg); cfg["seed"] = seed
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(cfg, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    w = class_weights(samples, num_classes, device) if cfg.get("class_weighting", True) else None
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.get("label_smoothing", 0.0))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("amp", True))
    loader = make_loader(samples, arrays, cfg, train=True)
    for ep in range(cfg["epochs"]):
        model.train()
        tot, correct, ls = 0, 0, 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True) if not isinstance(x, (list, tuple)) \
                else [t.to(device) for t in x]
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=cfg.get("amp", True)):
                out = model(x); loss = criterion(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            bs = y.size(0); ls += loss.item() * bs; tot += bs
            correct += (out.argmax(1) == y).sum().item()
        sched.step()
        log(f"  [seed {seed}] ep{ep:02d} loss={ls/tot:.4f} acc={correct/tot:.3f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = os.path.join(REPO, args.out)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def log(m):
        print(m, flush=True)

    class_names = cfg["class_names"]; num_classes = len(class_names)
    manifest = pd.read_csv(os.path.join(REPO, cfg["cache_dir"],
                                        cfg.get("manifest_name", "manifest.csv")))
    arrays = preload_arrays(os.path.join(REPO, cfg["cache_dir"]), manifest)

    def apex_pos(r):
        try:
            return max(int(r["apex"]) - int(r["onset"]), -1)
        except (KeyError, ValueError, TypeError):
            return -1
    samples = [dict(key=r["key"], subject=int(r["subject"]), label=int(r["label"]),
                    apex_pos=apex_pos(r)) for _, r in manifest.iterrows()]
    log(f"training final on ALL {len(samples)} samples, seeds={args.seeds}")

    ckpts = []
    for seed in args.seeds:
        model = train_all(seed, samples, arrays, cfg, num_classes, device, log)
        p = os.path.join(out_dir, f"final_seed{seed}.pt")
        torch.save(model.state_dict(), p)
        ckpts.append(os.path.basename(p))
        log(f"saved {p}")
        del model; torch.cuda.empty_cache()

    deploy = {
        "checkpoints": ckpts, "class_names": class_names, "num_classes": num_classes,
        "backbone": cfg["backbone"], "modality": cfg.get("modality", "flow"),
        "T": cfg["T"], "img_size": cfg["img_size"], "base_size": cfg["base_size"],
        "flow_clip": cfg.get("flow_clip", 3.0), "dropout": cfg.get("dropout", 0.5),
        "flow_third": cfg.get("flow_third", "mag"), "strain_clip": cfg.get("strain_clip", 1.0),
        "in_channels": cfg.get("in_channels", 3),
        "pretrained": False, "tta": cfg.get("tta", 5),
        "kinetics_mean": [0.43216, 0.394666, 0.37645],
        "kinetics_std": [0.22803, 0.22145, 0.216989],
        "source_config": os.path.basename(args.config),
    }
    json.dump(deploy, open(os.path.join(out_dir, "deploy.json"), "w"), indent=2)
    log(f"wrote {os.path.join(out_dir, 'deploy.json')}")


if __name__ == "__main__":
    main()
