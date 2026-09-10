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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from casme.models.models import build_model
from casme.training.engine import (EMA, build_criterion, configure_training,
                    deterministic_tta_views, make_loader, train_one_epoch)
from run_experiment import DEFAULTS, load_config, preload_arrays

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _snapshot_state(model, ema=None):
    """Return a CPU checkpoint, optionally using EMA weights, without mutation."""
    backup = None
    if ema is not None:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if backup is not None:
        model.load_state_dict(backup)
    return state


def train_all(seed, samples, arrays, cfg, num_classes, device, log):
    """Train on all samples and return the same tail/snapshot recipe as LOSO."""
    cfg = dict(cfg); cfg["seed"] = seed
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(cfg, num_classes).to(device)
    training = configure_training(model, cfg)
    criterion = build_criterion(samples, cfg, num_classes, device)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("amp", True))
    loader = make_loader(samples, arrays, cfg, train=True)
    epochs = int(cfg["epochs"])
    ema_decay = float(cfg.get("ema", 0.0))
    ema_start = int(cfg.get("ema_start_epoch", max(1, epochs // 5)))
    ema = EMA(model, ema_decay) if ema_decay > 0 else None
    eval_last_k = max(1, int(cfg.get("eval_last_k", 1)))
    snapshots = []

    for ep in range(epochs):
        loss, accuracy = train_one_epoch(
            model, loader, criterion, scaler, cfg, device, ep, training,
            ema=ema, ema_start=ema_start, log_fn=log)
        training["scheduler"].step()
        if ema is not None and ep == ema_start - 1:
            ema.reset(model)
        log(f"  [seed {seed}] ep{ep:02d} loss={loss:.4f} acc={accuracy:.3f}")

        if training["snapshot_cycles"] > 1:
            collect = (ep + 1) % training["cycle_length"] == 0
        else:
            collect = ep >= epochs - eval_last_k
        if collect:
            snapshots.append({
                "epoch": ep,
                "state_dict": _snapshot_state(model, ema=ema),
                "weights": "ema" if ema is not None else "raw",
            })
    if not snapshots:
        snapshots.append({"epoch": epochs - 1,
                          "state_dict": _snapshot_state(model, ema=ema),
                          "weights": "ema" if ema is not None else "raw"})
    del model
    torch.cuda.empty_cache()
    return snapshots


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
        snapshots = train_all(seed, samples, arrays, cfg, num_classes, device, log)
        for snap in snapshots:
            p = os.path.join(out_dir, f"final_seed{seed}_ep{snap['epoch']:02d}.pt")
            torch.save(snap["state_dict"], p)
            ckpts.append(os.path.basename(p))
            log(f"saved {p} ({snap['weights']})")

    deploy = {
        "checkpoints": ckpts, "class_names": class_names, "num_classes": num_classes,
        "backbone": cfg["backbone"], "modality": cfg.get("modality", "flow"),
        "input_mode": cfg.get("input_mode", "rgb"), "resize_to": cfg.get("resize_to", None),
        "T": cfg["T"], "img_size": cfg["img_size"], "base_size": cfg["base_size"],
        "flow_clip": cfg.get("flow_clip", 3.0), "dropout": cfg.get("dropout", 0.5),
        "flow_third": cfg.get("flow_third", "mag"), "strain_clip": cfg.get("strain_clip", 1.0),
        "flow_preset": cfg.get("flow_preset", "default"),
        "flow_stabilize": cfg.get("flow_stabilize", "none"),
        "flow_clahe": cfg.get("flow_clahe", False),
        "flow_compensation": cfg.get("flow_compensation", "none"),
        "flow_roi": cfg.get("flow_roi", "none"),
        "in_channels": cfg.get("in_channels", 3),
        "temporal_span": cfg.get("temporal_span", "onset_offset"),
        "eval_apex_source": cfg.get("eval_apex_source", "annotation"),
        "apex_fraction": cfg.get("apex_fraction", 0.55),
        "apex_energy_quantile": cfg.get("apex_energy_quantile", 0.95),
        "apex_border_fraction": cfg.get("apex_border_fraction", 0.08),
        "apex_smooth_radius": cfg.get("apex_smooth_radius", 0),
        "apex_search_max_fraction": cfg.get(
            "apex_search_max_fraction", 0.55),
        "temporal_jitter": cfg.get("temporal_jitter", True),
        "eval_rand_start": cfg.get("eval_rand_start", False),
        "eval_start_fraction": cfg.get("eval_start_fraction", 0.25),
        "gru_hidden": cfg.get("gru_hidden", 256),
        "grad_checkpoint": cfg.get("grad_checkpoint", True),
        "pretrained": False, "tta": cfg.get("tta", 5),
        "tta_views": deterministic_tta_views(cfg, cfg.get("tta", 5)),
        "checkpoint_aggregation": "probability_mean",
        "checkpoint_recipe": ("snapshot_cycles" if cfg.get("snapshot_cycles", 0) > 1
                              else "last_k_epochs"),
        "loss": cfg.get("loss", "ce"),
        "label_smoothing": cfg.get("label_smoothing", 0.0),
        "ema": cfg.get("ema", 0.0),
        "kinetics_mean": [0.43216, 0.394666, 0.37645],
        "kinetics_std": [0.22803, 0.22145, 0.216989],
        "source_config": os.path.basename(args.config),
    }
    json.dump(deploy, open(os.path.join(out_dir, "deploy.json"), "w"), indent=2)
    log(f"wrote {os.path.join(out_dir, 'deploy.json')}")


if __name__ == "__main__":
    main()
