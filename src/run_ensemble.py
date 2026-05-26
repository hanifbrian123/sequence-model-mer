"""Multi-seed ensemble LOSO: train the same config with several seeds and average
per-sample softmax probs (pooled across folds). Reduces LOSO variance.

Usage:
  python src/run_ensemble.py --config configs/iter_07.json --seeds 42 123 2024
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import train_fold
from metrics import compute_metrics
from run_experiment import DEFAULTS, load_config, preload_arrays, plot_confusion

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one_seed(seed, samples, subjects, arrays, cfg, num_classes, device, log):
    cfg = dict(cfg); cfg["seed"] = seed
    torch.manual_seed(seed); np.random.seed(seed)
    key_probs = {}
    for fi, subj in enumerate(subjects):
        val_s = [s for s in samples if s["subject"] == subj]
        train_s = [s for s in samples if s["subject"] != subj]
        ft = time.time()
        val_probs, _ = train_fold(train_s, val_s, arrays, cfg, num_classes, device, lambda m: None)
        for s, pr in zip(val_s, val_probs):
            key_probs[s["key"]] = pr
        log(f"  [seed {seed}] fold {fi+1}/{len(subjects)} sub{subj:02d} "
            f"acc={np.mean(val_probs.argmax(1)==[x['label'] for x in val_s]):.3f} "
            f"({time.time()-ft:.0f}s)")
    return key_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = os.path.splitext(os.path.basename(args.config))[0]
    if args.tag:
        name += f"_{args.tag}"
    exp_dir = os.path.join(REPO, "experiments", name)
    os.makedirs(exp_dir, exist_ok=True)
    logf = open(os.path.join(exp_dir, "run.log"), "w", encoding="utf-8")

    def log(msg):
        print(msg); logf.write(msg + "\n"); logf.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    log(f"=== ENSEMBLE {name} seeds={args.seeds} ===")
    log(f"time {datetime.now().isoformat()} device {device}")
    log("config:\n" + json.dumps(cfg, indent=2))
    json.dump(cfg, open(os.path.join(exp_dir, "config.json"), "w"), indent=2)

    class_names = cfg["class_names"]; num_classes = len(class_names)
    if cfg.get("modality") == "two_stream":
        man_a = pd.read_csv(os.path.join(REPO, cfg["cache_dir_a"], "manifest.csv"))
        man_b = pd.read_csv(os.path.join(REPO, cfg["cache_dir_b"], "manifest.csv"))
        common = set(man_a["key"]) & set(man_b["key"])
        manifest = man_a[man_a["key"].isin(common)].reset_index(drop=True)
        arrays = (preload_arrays(os.path.join(REPO, cfg["cache_dir_a"]), manifest),
                  preload_arrays(os.path.join(REPO, cfg["cache_dir_b"]), manifest))
    else:
        manifest = pd.read_csv(os.path.join(REPO, cfg["cache_dir"], "manifest.csv"))
        arrays = preload_arrays(os.path.join(REPO, cfg["cache_dir"]), manifest)
    log(f"loaded arrays; samples={len(manifest)}")

    def apex_pos(r):
        try:
            return max(int(r["apex"]) - int(r["onset"]), -1)
        except (KeyError, ValueError, TypeError):
            return -1
    samples = [dict(key=r["key"], subject=int(r["subject"]), label=int(r["label"]),
                    apex_pos=apex_pos(r)) for _, r in manifest.iterrows()]
    subjects = sorted(set(s["subject"] for s in samples))

    seed_probs = []
    for seed in args.seeds:
        log(f"\n--- SEED {seed} ---")
        kp = run_one_seed(seed, samples, subjects, arrays, cfg, num_classes, device, log)
        seed_probs.append(kp)
        # per-seed metric
        yt = [s["label"] for s in samples]
        yp = [int(np.argmax(kp[s["key"]])) for s in samples]
        m1 = compute_metrics(yt, yp, num_classes)
        log(f"  seed {seed}: UF1={m1['UF1']:.4f} UAR={m1['UAR']:.4f} ACC={m1['ACC']:.4f}")

    # average probs across seeds
    yt = [s["label"] for s in samples]
    avg = [np.mean([sp[s["key"]] for sp in seed_probs], axis=0) for s in samples]
    yp = [int(np.argmax(a)) for a in avg]
    m = compute_metrics(yt, yp, num_classes)
    log("\n=== ENSEMBLE FINAL (pooled LOSO, avg over seeds) ===")
    log(f"UF1 = {m['UF1']:.4f}\nUAR = {m['UAR']:.4f}\nACC = {m['ACC']:.4f}")
    log("per-class F1: " + ", ".join(f"{c}={v:.3f}" for c, v in zip(class_names, m['per_class_f1'])))
    log("confusion matrix (rows=true):")
    log("        " + " ".join(f"{c[:6]:>6}" for c in class_names))
    for i, row in enumerate(m["confusion_matrix"]):
        log(f"{class_names[i][:7]:>7} " + " ".join(f"{v:6d}" for v in row))

    np.savez(os.path.join(exp_dir, "probs.npz"),
             keys=np.array([s["key"] for s in samples]),
             label=np.array(yt), probs=np.array(avg, dtype=np.float32))
    pd.DataFrame({"key": [s["key"] for s in samples], "subject": [s["subject"] for s in samples],
                  "label": yt, "pred": yp}).to_csv(os.path.join(exp_dir, "predictions.csv"), index=False)
    pd.DataFrame(m["confusion_matrix"], index=class_names, columns=class_names
                 ).to_csv(os.path.join(exp_dir, "confusion_matrix.csv"))
    plot_confusion(m["confusion_matrix"], class_names, os.path.join(exp_dir, "confusion_matrix.png"))

    elapsed = round(time.time() - t0, 1)
    summary = {"name": name, "seeds": args.seeds, "elapsed_sec": elapsed,
               **{k: m[k] for k in ("UF1", "UAR", "ACC", "n")},
               "per_class_f1": m["per_class_f1"], "class_names": class_names, "config": cfg}
    json.dump(summary, open(os.path.join(exp_dir, "summary.json"), "w"), indent=2)

    ledger = os.path.join(REPO, "results_ledger.csv")
    row = {"name": name, "time": datetime.now().isoformat(),
           "UF1": round(m["UF1"], 4), "UAR": round(m["UAR"], 4), "ACC": round(m["ACC"], 4),
           "n_folds": len(subjects), "backbone": cfg["backbone"], "T": cfg["T"],
           "img_size": cfg["img_size"], "input_mode": f"ens_{cfg.get('modality','rgb')}",
           "epochs": cfg["epochs"], "lr": cfg["lr"], "elapsed_sec": elapsed,
           "index": cfg["index"], "num_classes": num_classes}
    pd.DataFrame([row]).to_csv(ledger, mode="a", header=not os.path.exists(ledger), index=False)
    log(f"\nsaved to {exp_dir}\nelapsed {elapsed}s")
    logf.close()


if __name__ == "__main__":
    main()
