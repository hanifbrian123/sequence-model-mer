"""Run fixed subject-group development CV or a locked audit evaluation.

Development runs never train on or evaluate the audit subjects. Audit mode
requires a gate-generated lock file whose config and protocol hashes match.
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from casme.training.engine import train_fold
from casme.evaluation.metrics import compute_metrics
from casme.evaluation.protocol_v2 import (config_fingerprint, load_or_create_protocol,
                         protocol_subject_sets)
from run_experiment import load_config, plot_confusion, preload_arrays


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _apex_position(row):
    try:
        return max(int(row["apex"]) - int(row["onset"]), -1)
    except (KeyError, ValueError, TypeError):
        return -1


def _load_problem(cfg, log):
    if cfg.get("modality") == "two_stream":
        manifest_a = pd.read_csv(os.path.join(
            REPO, cfg["cache_dir_a"], "manifest.csv"))
        manifest_b = pd.read_csv(os.path.join(
            REPO, cfg["cache_dir_b"], "manifest.csv"))
        common = set(manifest_a["key"]) & set(manifest_b["key"])
        manifest = manifest_a[manifest_a["key"].isin(common)].reset_index(drop=True)
        log(f"loading {len(manifest)} aligned two-stream cached arrays")
        arrays = (
            preload_arrays(os.path.join(REPO, cfg["cache_dir_a"]), manifest),
            preload_arrays(os.path.join(REPO, cfg["cache_dir_b"]), manifest),
        )
    else:
        manifest = pd.read_csv(os.path.join(
            REPO, cfg["cache_dir"], cfg.get("manifest_name", "manifest.csv")))
        log(f"loading {len(manifest)} cached arrays")
        arrays = preload_arrays(os.path.join(REPO, cfg["cache_dir"]), manifest)
    samples = [
        {"key": row["key"], "subject": int(row["subject"]),
         "label": int(row["label"]), "apex_pos": _apex_position(row)}
        for _, row in manifest.iterrows()
    ]
    _attach_au_labels(samples, cfg, log)
    _attach_region_masks(samples, cfg, log)
    return samples, arrays


def _attach_region_masks(samples, cfg, log):
    """Attach K soft facial-region masks when segmentation attention is on.

    Same side-file pattern as the AU labels: nothing about the cached flow or
    the protocol fingerprint changes, so a region-attention run stays directly
    comparable to every run that came before it.
    """
    needs_masks = (cfg.get("region_attention", False)
                   or cfg.get("focus_channel", "none") == "region"
                   or cfg.get("load_region_masks", False))
    if not needs_masks:
        return
    directory = os.path.join(REPO, cfg.get("region_mask_dir",
                                           "cache/regionmask144"))
    expected = int(cfg.get("num_regions", 6))
    missing = []
    for sample in samples:
        path = os.path.join(directory, sample["key"] + ".npy")
        if not os.path.exists(path):
            missing.append(sample["key"])
            continue
        masks = np.load(path).astype(np.float32)
        if masks.shape[0] != expected:
            raise ValueError(
                f"{path} has {masks.shape[0]} regions, config says {expected}")
        sample["region"] = masks
    if missing:
        raise ValueError(
            f"{len(missing)} samples have no region mask, e.g. {missing[:3]}")
    log(f"segmentation attention: {expected} region per sampel dari "
        f"{os.path.basename(directory)} "
        f"(mode={cfg.get('region_attention_mode', 'static')})")


def _attach_au_labels(samples, cfg, log):
    """Attach Action Unit multi-labels when the config asks for the aux task.

    Loaded from a side file keyed by cache key so the protocol fingerprint
    (key/subject/label) is untouched and every previous run stays comparable.
    """
    if not cfg.get("au_multitask", False):
        return
    path = os.path.join(REPO, cfg.get("au_labels", "cache/au_labels.csv"))
    table = pd.read_csv(path)
    columns = [c for c in table.columns if c.startswith("au") and c != "au"]
    lookup = {row["key"]: [float(row[c]) for c in columns]
              for _, row in table.iterrows()}
    missing = [s["key"] for s in samples if s["key"] not in lookup]
    if missing:
        raise ValueError(
            f"{len(missing)} samples have no AU labels, e.g. {missing[:3]}")
    for sample in samples:
        sample["au"] = lookup[sample["key"]]
    if int(cfg.get("num_au", 0)) != len(columns):
        raise ValueError(
            f"config num_au={cfg.get('num_au')} but {path} has {len(columns)} "
            f"AU columns: {columns}")
    log(f"AU multi-task: {len(columns)} units from {os.path.basename(path)} "
        f"({', '.join(columns)})")


def _verify_audit_lock(lock_path, cfg_hash, protocol):
    if not lock_path:
        raise ValueError("audit mode requires --lock from compare_protocol.py")
    with open(lock_path, encoding="utf-8") as handle:
        lock = json.load(handle)
    if not lock.get("gate_passed", False):
        raise ValueError("candidate lock does not contain a passed gate")
    allowed = lock.get("allowed_config_sha256", [lock.get("config_sha256")])
    if cfg_hash not in allowed:
        raise ValueError("locked config hash does not match requested config")
    if lock.get("protocol_samples_sha256") != protocol["samples_sha256"]:
        raise ValueError("locked protocol does not match current protocol")
    return lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", choices=("dev", "audit"), default="dev")
    parser.add_argument("--protocol", default="protocols/accuracy_v5.json")
    parser.add_argument("--lock", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--split", choices=("grouped", "loso_dev", "loso_all"), default="grouped",
        help="grouped = 4 balanced subject folds (cheap screening); "
             "loso_dev = leave-one-subject-out over development subjects only, "
             "audit stays sealed; loso_all = 26-fold LOSO as reported in the "
             "literature, which USES THE AUDIT SUBJECTS and breaks the seal")
    parser.add_argument(
        "--break_audit_seal", action="store_true",
        help="required acknowledgement for --split loso_all")
    parser.add_argument("--max_folds", type=int, default=0,
                        help="development smoke only; 0 runs all dev folds")
    parser.add_argument("--dry_run", action="store_true",
                        help="validate split counts without training")
    parser.add_argument("--save_fold_checkpoints", action="store_true",
                        help="save fixed tail/snapshot weights for deterministic re-evaluation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_hash = config_fingerprint(cfg)
    stem = os.path.splitext(os.path.basename(args.config))[0]
    split_tag = "" if args.split == "grouped" else f"_{args.split}"
    name = (f"{stem}_v2_{args.role}{split_tag}"
            + (f"_{args.tag}" if args.tag else ""))
    output_dir = os.path.join(REPO, "experiments", "protocol_v2", name)
    os.makedirs(output_dir, exist_ok=True)
    log_file = open(os.path.join(output_dir, "run.log"), "w", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        log_file.write(str(message) + "\n")
        log_file.flush()

    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"=== PROTOCOL V2 {name} ===")
    log(f"time: {datetime.now().isoformat()} | role: {args.role}")
    log(f"device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")
    log(f"platform: {platform.platform()} | torch {torch.__version__}")
    samples, arrays = _load_problem(cfg, log)
    protocol_path = (args.protocol if os.path.isabs(args.protocol)
                     else os.path.join(REPO, args.protocol))
    protocol = load_or_create_protocol(
        samples, len(cfg["class_names"]), protocol_path)
    audit_subjects, dev_subjects, dev_folds = protocol_subject_sets(protocol)
    log(f"fixed split loaded: {len(dev_subjects)} development subjects, "
        f"{len(audit_subjects)} sealed audit subjects")

    if args.role == "audit":
        _verify_audit_lock(args.lock, cfg_hash, protocol)
        cfg = dict(cfg)
        cfg["monitor_val"] = False
        folds = [(0, audit_subjects, dev_subjects)]
        log("audit lock verified; validation metrics are hidden during epochs")
    elif args.split == "loso_dev":
        # Leave-one-subject-out over development subjects only. Same protocol
        # family the literature reports, and each fold trains on ~183 samples
        # instead of ~144, so an idea that only pays off with more data is not
        # rejected by a data-starved screen. The audit subjects stay sealed.
        folds = [(index, {subject}, dev_subjects - {subject})
                 for index, subject in enumerate(sorted(dev_subjects))]
        log(f"LOSO over {len(folds)} development subjects "
            f"(audit subjects {sorted(audit_subjects)} remain sealed)")
    elif args.split == "loso_all":
        if not args.break_audit_seal:
            raise ValueError(
                "--split loso_all evaluates the sealed audit subjects. Pass "
                "--break_audit_seal to acknowledge that this consumes the "
                "one-time clean check.")
        every_subject = sorted(dev_subjects | audit_subjects)
        folds = [(index, {subject}, set(every_subject) - {subject})
                 for index, subject in enumerate(every_subject)]
        log(f"!! LOSO over ALL {len(folds)} subjects -- AUDIT SEAL BROKEN. "
            f"This number is comparable to published LOSO results, and the "
            f"audit split can no longer serve as an untouched final check.")
    else:
        folds = [(fold_id, subjects, dev_subjects - subjects)
                 for fold_id, subjects in dev_folds]
        if args.max_folds > 0:
            folds = folds[:args.max_folds]
            log(f"SMOKE mode: {len(folds)} development folds")

    if args.dry_run:
        for position, (_, validation_subjects, training_subjects) in enumerate(folds, 1):
            n_train = sum(row["subject"] in training_subjects for row in samples)
            n_validation = sum(row["subject"] in validation_subjects for row in samples)
            log(f"dry fold {position}/{len(folds)}: {len(training_subjects)} train subjects / "
                f"{len(validation_subjects)} validation subjects; "
                f"{n_train} train / {n_validation} validation samples")
        log("dry-run split validation complete; no model was trained")
        log_file.close()
        return

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    num_classes = len(cfg["class_names"])
    histories = []
    pooled_keys, pooled_subjects, pooled_labels, pooled_probs = [], [], [], []
    per_fold = []
    for position, (fold_id, validation_subjects, training_subjects) in enumerate(folds, 1):
        train_samples = [row for row in samples if row["subject"] in training_subjects]
        validation_samples = [row for row in samples
                              if row["subject"] in validation_subjects]
        log(f"fold {position}/{len(folds)}: {len(training_subjects)} train subjects / "
            f"{len(validation_subjects)} validation subjects; "
            f"{len(train_samples)} train / {len(validation_samples)} validation samples")
        fold_start = time.time()
        snapshot_fn = None
        if args.save_fold_checkpoints:
            checkpoint_dir = os.path.join(output_dir, "fold_checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

            def snapshot_fn(epoch, state_dict, current_fold=fold_id):
                path = os.path.join(
                    checkpoint_dir, f"fold{current_fold}_ep{epoch:02d}.pt")
                torch.save(state_dict, path)
                log(f"      saved re-evaluation checkpoint {os.path.basename(path)}")

        probabilities, history = train_fold(
            train_samples, validation_samples, arrays, cfg, num_classes, device, log,
            snapshot_fn=snapshot_fn)
        predictions = probabilities.argmax(axis=1)
        labels = np.array([row["label"] for row in validation_samples])
        fold_metrics = compute_metrics(labels, predictions, num_classes)
        per_fold.append({
            "fold": fold_id, "n_subjects": len(validation_subjects),
            "n_samples": len(validation_samples),
            "UF1": fold_metrics["UF1"], "UAR": fold_metrics["UAR"],
            "ACC": fold_metrics["ACC"], "elapsed_sec": time.time() - fold_start,
        })
        log(f"fold result: UF1={fold_metrics['UF1']:.4f} "
            f"UAR={fold_metrics['UAR']:.4f} ACC={fold_metrics['ACC']:.4f}")
        for row, probability in zip(validation_samples, probabilities):
            pooled_keys.append(row["key"])
            pooled_subjects.append(row["subject"])
            pooled_labels.append(row["label"])
            pooled_probs.append(probability)
        for epoch in history:
            histories.append({"fold": fold_id, **epoch})

    pooled_probs = np.asarray(pooled_probs, dtype=np.float32)
    pooled_predictions = pooled_probs.argmax(axis=1)
    metrics = compute_metrics(pooled_labels, pooled_predictions, num_classes)
    elapsed = time.time() - start_time
    complete = args.role == "audit" or args.max_folds == 0
    log(f"FINAL {args.role}: UF1={metrics['UF1']:.4f} "
        f"UAR={metrics['UAR']:.4f} ACC={metrics['ACC']:.4f}")

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)
    pd.DataFrame(histories).to_json(
        os.path.join(output_dir, "epoch_history.jsonl"), orient="records", lines=True)
    pd.DataFrame(per_fold).to_csv(
        os.path.join(output_dir, "per_fold.csv"), index=False)
    pd.DataFrame({
        "key": pooled_keys, "subject": pooled_subjects, "label": pooled_labels,
        "pred": pooled_predictions,
        "correct": np.asarray(pooled_labels) == pooled_predictions,
    }).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    np.savez(os.path.join(output_dir, "probs.npz"),
             keys=np.asarray(pooled_keys), subject=np.asarray(pooled_subjects),
             label=np.asarray(pooled_labels), probs=pooled_probs)
    pd.DataFrame(metrics["confusion_matrix"], index=cfg["class_names"],
                 columns=cfg["class_names"]).to_csv(
                     os.path.join(output_dir, "confusion_matrix.csv"))
    plot_confusion(metrics["confusion_matrix"], cfg["class_names"],
                   os.path.join(output_dir, "confusion_matrix.png"))
    summary = {
        "name": name, "role": args.role, "complete": complete,
        "time": datetime.now().isoformat(), "elapsed_sec": elapsed,
        "config_sha256": cfg_hash,
        "protocol_samples_sha256": protocol["samples_sha256"],
        "split": args.split,
        "n_folds": len(folds), "n_subjects": len(set(pooled_subjects)),
        "fold_checkpoints_saved": bool(args.save_fold_checkpoints),
        "num_classes": num_classes, **metrics, "config": cfg,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    ledger_path = os.path.join(REPO, "results_protocol_v2.csv")
    ledger_row = {
        "name": name, "role": args.role, "complete": complete,
        "UF1": metrics["UF1"], "UAR": metrics["UAR"], "ACC": metrics["ACC"],
        "split": args.split,
        "n_folds": len(folds), "n_subjects": len(set(pooled_subjects)),
        "elapsed_sec": elapsed, "config_sha256": cfg_hash,
        "time": datetime.now().isoformat(),
    }
    pd.DataFrame([ledger_row]).to_csv(
        ledger_path, mode="a", header=not os.path.exists(ledger_path), index=False)
    log(f"saved reproducible artifacts to {output_dir} in {elapsed:.1f}s")
    log_file.close()


if __name__ == "__main__":
    main()
