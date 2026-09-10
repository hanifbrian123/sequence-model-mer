"""LOSO orchestrator for CASME II sequence models, with full logging.

Usage:
  python src/run_experiment.py --config configs/iter_01.json [--max_folds N] [--tag smoke]
"""
import os
import sys
import json
import time
import argparse
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import train_fold
from metrics import compute_metrics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "seed": 42,
    "T": 16,
    "img_size": 112,
    "base_size": 128,
    "input_mode": "rgb",
    "backbone": "r2plus1d_18",
    "pretrained": True,
    "dropout": 0.5,
    "epochs": 25,
    "lr": 1e-4,
    "weight_decay": 1e-3,
    "batch_size": 8,
    "class_weighting": True,
    "label_smoothing": 0.1,
    "hflip": True,
    "color_jitter": 0.1,
    "temporal_jitter": True,
    "amp": True,
    "eval_last_k": 3,
    "tta": 1,
    "cache_dir": "cache/frames128",
    "index": "cache/index_emotion.csv",
    "class_names": ["happiness", "disgust", "repression", "surprise", "others"],
}


def load_config(path):
    with open(path) as f:
        user = json.load(f)
    cfg = dict(DEFAULTS)
    cfg.update(user)
    return cfg


def preload_arrays(cache_dir, manifest):
    arrays = {}
    for _, r in manifest.iterrows():
        arrays[r["key"]] = np.load(os.path.join(cache_dir, r["npy"]))
    return arrays


def plot_confusion(cm, class_names, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = np.array(cm)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print("confusion plot failed:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_folds", type=int, default=0, help="0 = all subjects")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = os.path.splitext(os.path.basename(args.config))[0]
    if args.tag:
        name = f"{name}_{args.tag}"
    exp_dir = os.path.join(REPO, "experiments", name)
    os.makedirs(exp_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, "run.log")
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    log(f"=== EXPERIMENT {name} ===")
    log(f"time: {datetime.now().isoformat()}")
    log(f"device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'})")
    log(f"platform: {platform.platform()} | torch {torch.__version__}")
    log("config:\n" + json.dumps(cfg, indent=2))
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    class_names = cfg["class_names"]
    num_classes = len(class_names)
    if cfg.get("modality") == "two_stream":
        man_a = pd.read_csv(os.path.join(REPO, cfg["cache_dir_a"], "manifest.csv"))
        man_b = pd.read_csv(os.path.join(REPO, cfg["cache_dir_b"], "manifest.csv"))
        # align on common keys
        common = set(man_a["key"]) & set(man_b["key"])
        manifest = man_a[man_a["key"].isin(common)].reset_index(drop=True)
        log(f"two-stream: {len(manifest)} common samples "
            f"(a={cfg['cache_dir_a']}, b={cfg['cache_dir_b']})")
        log("loading arrays into RAM...")
        arrays_a = preload_arrays(os.path.join(REPO, cfg["cache_dir_a"]), manifest)
        arrays_b = preload_arrays(os.path.join(REPO, cfg["cache_dir_b"]), manifest)
        arrays = (arrays_a, arrays_b)
        log(f"loaded a={len(arrays_a)} b={len(arrays_b)} arrays")
    else:
        manifest = pd.read_csv(os.path.join(REPO, cfg["cache_dir"],
                                            cfg.get("manifest_name", "manifest.csv")))
        log(f"samples: {len(manifest)} | classes: {num_classes} {class_names}")
        log("loading arrays into RAM...")
        arrays = preload_arrays(os.path.join(REPO, cfg["cache_dir"]), manifest)
        log(f"loaded {len(arrays)} arrays")

    def apex_pos(r):
        # position of apex within the cached onset..offset clip (index 0 = onset)
        try:
            return max(int(r["apex"]) - int(r["onset"]), -1)
        except (KeyError, ValueError, TypeError):
            return -1
    samples = [dict(key=r["key"], subject=int(r["subject"]), label=int(r["label"]),
                    apex_pos=apex_pos(r))
               for _, r in manifest.iterrows()]
    subjects = sorted(set(s["subject"] for s in samples))
    if args.max_folds > 0:
        subjects = subjects[:args.max_folds]
        log(f"[SMOKE] limiting to {args.max_folds} folds: {subjects}")

    hist_f = open(os.path.join(exp_dir, "epoch_history.jsonl"), "w", encoding="utf-8")
    pooled_true, pooled_pred, pooled_key, pooled_subj = [], [], [], []
    pooled_probs = []
    per_fold = []

    for fi, subj in enumerate(subjects):
        val_s = [s for s in samples if s["subject"] == subj]
        train_s = [s for s in samples if s["subject"] != subj]
        log(f"\n--- Fold {fi+1}/{len(subjects)} | test sub{subj:02d} "
            f"({len(val_s)} test, {len(train_s)} train) ---")
        ft = time.time()
        val_probs, history = train_fold(train_s, val_s, arrays, cfg, num_classes, device, log)
        preds = val_probs.argmax(1)
        for s, p, pr in zip(val_s, preds, val_probs):
            pooled_true.append(s["label"]); pooled_pred.append(int(p))
            pooled_key.append(s["key"]); pooled_subj.append(subj)
            pooled_probs.append(pr.tolist())
        for h in history:
            h2 = dict(h); h2["fold"] = fi; h2["subject"] = subj
            hist_f.write(json.dumps(h2) + "\n")
        hist_f.flush()
        acc = float(np.mean([int(p) == s["label"] for s, p in zip(val_s, preds)]))
        per_fold.append({"fold": fi, "subject": subj, "n_test": len(val_s),
                         "acc": acc, "sec": round(time.time() - ft, 1)})
        log(f"    fold acc={acc:.3f} time={time.time()-ft:.1f}s "
            f"| running pooled acc={np.mean(np.array(pooled_true)==np.array(pooled_pred)):.3f}")

    hist_f.close()
    m = compute_metrics(pooled_true, pooled_pred, num_classes)
    log("\n=== FINAL (pooled LOSO) ===")
    log(f"UF1 = {m['UF1']:.4f}")
    log(f"UAR = {m['UAR']:.4f}")
    log(f"ACC = {m['ACC']:.4f}")
    log(f"per-class F1: " + ", ".join(f"{c}={v:.3f}" for c, v in zip(class_names, m['per_class_f1'])))
    log(f"per-class recall: " + ", ".join(f"{c}={v:.3f}" for c, v in zip(class_names, m['per_class_recall'])))
    log("confusion matrix (rows=true):")
    log("        " + " ".join(f"{c[:6]:>6}" for c in class_names))
    for i, row in enumerate(m["confusion_matrix"]):
        log(f"{class_names[i][:7]:>7} " + " ".join(f"{v:6d}" for v in row))

    # save artifacts
    pd.DataFrame(per_fold).to_csv(os.path.join(exp_dir, "per_fold.csv"), index=False)
    pd.DataFrame({"key": pooled_key, "subject": pooled_subj,
                  "label": pooled_true, "pred": pooled_pred,
                  "correct": [int(a == b) for a, b in zip(pooled_true, pooled_pred)]}
                 ).to_csv(os.path.join(exp_dir, "predictions.csv"), index=False)
    # per-sample softmax probs (aligned with predictions.csv order) for late-fusion ensembles
    np.savez(os.path.join(exp_dir, "probs.npz"),
             keys=np.array(pooled_key), subject=np.array(pooled_subj),
             label=np.array(pooled_true), probs=np.array(pooled_probs, dtype=np.float32))
    pd.DataFrame(m["confusion_matrix"], index=class_names, columns=class_names
                 ).to_csv(os.path.join(exp_dir, "confusion_matrix.csv"))
    plot_confusion(m["confusion_matrix"], class_names, os.path.join(exp_dir, "confusion_matrix.png"))

    elapsed = round(time.time() - t0, 1)
    summary = {"name": name, "time": datetime.now().isoformat(), "elapsed_sec": elapsed,
               "n_folds": len(subjects), "num_classes": num_classes,
               "class_names": class_names, **{k: m[k] for k in ("UF1", "UAR", "ACC", "n")},
               "per_class_f1": m["per_class_f1"], "per_class_recall": m["per_class_recall"],
               "config": cfg}
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # append to ledger
    ledger = os.path.join(REPO, "results_ledger.csv")
    row = {"name": name, "time": datetime.now().isoformat(),
           "UF1": round(m["UF1"], 4), "UAR": round(m["UAR"], 4), "ACC": round(m["ACC"], 4),
           "n_folds": len(subjects), "backbone": cfg["backbone"], "T": cfg["T"],
           "img_size": cfg["img_size"], "input_mode": cfg["input_mode"],
           "epochs": cfg["epochs"], "lr": cfg["lr"], "elapsed_sec": elapsed,
           "index": cfg["index"], "num_classes": num_classes}
    hdr = not os.path.exists(ledger)
    pd.DataFrame([row]).to_csv(ledger, mode="a", header=hdr, index=False)

    log(f"\nsaved to {exp_dir}")
    log(f"elapsed {elapsed}s")
    logf.close()


if __name__ == "__main__":
    main()
