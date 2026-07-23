"""Re-evaluate saved development-fold weights with deploy-time preprocessing.

Only evaluation-side fields may differ from the source run. This supports fair
comparisons of label-free temporal hypotheses without repeating identical
training or touching the sealed audit subjects.
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import predict
from metrics import compute_metrics
from models import build_model
from protocol_v2 import (config_fingerprint, load_or_create_protocol,
                         protocol_subject_sets)
from run_experiment import load_config, plot_confusion
from run_protocol_v2 import _load_problem


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_ONLY_KEYS = {
    "name", "tta", "tta_views", "eval_apex_source", "apex_fraction",
    "apex_energy_quantile", "apex_border_fraction", "apex_smooth_radius",
    "apex_search_max_fraction", "eval_rand_start", "eval_start_fraction",
}


def _training_projection(cfg):
    return {key: value for key, value in cfg.items() if key not in EVAL_ONLY_KEYS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_run", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", default="protocols/accuracy_v5.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source_run = os.path.abspath(args.source_run)
    with open(os.path.join(source_run, "summary.json"), encoding="utf-8") as handle:
        source = json.load(handle)
    if source.get("role") != "dev" or not source.get("complete", False):
        raise ValueError("source must be a complete development run")
    if not source.get("fold_checkpoints_saved", False):
        raise ValueError("source run did not save fold checkpoints")
    cfg = load_config(args.config)
    if _training_projection(cfg) != _training_projection(source["config"]):
        raise ValueError("evaluation config changes training/model fields")

    output = os.path.abspath(args.out)
    os.makedirs(output, exist_ok=True)

    def log(message):
        print(message, flush=True)

    start = time.time()
    samples, arrays = _load_problem(cfg, log)
    protocol_path = (args.protocol if os.path.isabs(args.protocol)
                     else os.path.join(REPO, args.protocol))
    protocol = load_or_create_protocol(
        samples, len(cfg["class_names"]), protocol_path)
    if protocol["samples_sha256"] != source["protocol_samples_sha256"]:
        raise ValueError("source and requested protocols differ")
    _, dev_subjects, dev_folds = protocol_subject_sets(protocol)
    folds = [(fold_id, subjects, dev_subjects - subjects)
             for fold_id, subjects in dev_folds]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(cfg["class_names"])
    pooled_keys, pooled_subjects, pooled_labels, pooled_probs = [], [], [], []
    per_fold = []
    checkpoint_dir = os.path.join(source_run, "fold_checkpoints")
    for fold_id, validation_subjects, _ in folds:
        validation = [row for row in samples
                      if row["subject"] in validation_subjects]
        paths = sorted(glob.glob(os.path.join(
            checkpoint_dir, f"fold{fold_id}_ep*.pt")))
        if not paths:
            raise ValueError(f"no saved checkpoints for fold {fold_id}")
        accumulator = None
        model_cfg = dict(cfg)
        model_cfg["pretrained"] = False
        for path in paths:
            model = build_model(model_cfg, num_classes).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            probabilities = predict(
                model, validation, arrays, cfg, device, tta=cfg.get("tta", 1))
            accumulator = (probabilities if accumulator is None
                           else accumulator + probabilities)
            del model
        probabilities = accumulator / len(paths)
        labels = np.asarray([row["label"] for row in validation])
        metrics = compute_metrics(labels, probabilities.argmax(1), num_classes)
        per_fold.append({"fold": fold_id, "n_subjects": len(validation_subjects),
                         "n_samples": len(validation), "UF1": metrics["UF1"],
                         "UAR": metrics["UAR"], "ACC": metrics["ACC"]})
        for row, probability in zip(validation, probabilities):
            pooled_keys.append(row["key"])
            pooled_subjects.append(row["subject"])
            pooled_labels.append(row["label"])
            pooled_probs.append(probability)
        torch.cuda.empty_cache()

    pooled_probs = np.asarray(pooled_probs, dtype=np.float32)
    predictions = pooled_probs.argmax(1)
    metrics = compute_metrics(pooled_labels, predictions, num_classes)
    np.savez(os.path.join(output, "probs.npz"), keys=np.asarray(pooled_keys),
             subject=np.asarray(pooled_subjects), label=np.asarray(pooled_labels),
             probs=pooled_probs)
    pd.DataFrame({"key": pooled_keys, "subject": pooled_subjects,
                  "label": pooled_labels, "pred": predictions,
                  "correct": np.asarray(pooled_labels) == predictions}).to_csv(
                      os.path.join(output, "predictions.csv"), index=False)
    pd.DataFrame(per_fold).to_csv(os.path.join(output, "per_fold.csv"), index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=cfg["class_names"],
                 columns=cfg["class_names"]).to_csv(
                     os.path.join(output, "confusion_matrix.csv"))
    plot_confusion(metrics["confusion_matrix"], cfg["class_names"],
                   os.path.join(output, "confusion_matrix.png"))
    summary = {
        "name": os.path.basename(output), "role": "dev", "complete": True,
        "time": datetime.now().isoformat(), "elapsed_sec": time.time() - start,
        "config_sha256": config_fingerprint(cfg),
        "protocol_samples_sha256": protocol["samples_sha256"],
        "n_folds": len(folds), "n_subjects": len(set(pooled_subjects)),
        "num_classes": num_classes, "reused_training_from": source_run,
        "fold_checkpoints_saved": False, **metrics, "config": cfg,
    }
    with open(os.path.join(output, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"output": output, "UF1": metrics["UF1"],
                      "UAR": metrics["UAR"], "ACC": metrics["ACC"]}, indent=2))


if __name__ == "__main__":
    main()
