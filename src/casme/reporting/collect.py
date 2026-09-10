"""Collect all experiment metrics across runs into results/results.csv.

Modeled after fer.reporting.collect:
- Reads every run directory in runs/
- Uses the directory name (NNN_name) as the single source of truth for `run`
- Eliminates any legacy iter_* naming
- Sorts runs in strict numerical order (000_, 001_, ... 106_)
- Saves to results/results.csv as the single authoritative results table
"""
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pandas as pd

from casme.paths import RESULTS_CSV, RESULTS_DIR, RUNS_DIR


COLUMNS = [
    "run", "stem", "protocol", "split", "role", "complete",
    "UF1", "UAR", "ACC", "n_folds", "n_subjects", "n_samples",
    "backbone", "input_mode", "img_size", "T", "lr", "seed",
    "epochs", "elapsed_sec", "time"
]


def _detect_split(name: str, n_folds: int, n_subjects: int) -> str:
    name_lower = name.lower()
    if "loso_all" in name_lower or (n_folds == 26 and n_subjects == 26):
        return "LOSO 26"
    if "loso_dev" in name_lower or (n_folds == 21 and n_subjects == 21):
        return "LOSO 21"
    if "audit" in name_lower:
        return "audit"
    if n_folds == 4 or "grouped" in name_lower or "_p5" in name_lower:
        return "grouped 4-fold"
    if n_folds == 26:
        return "LOSO 26"
    if n_folds == 0:
        return "0-fold"
    if n_folds == 1:
        return "1-fold"
    if n_folds == 2:
        return "2-fold"
    return f"{n_folds}-fold"


def _clean_stem(run_name: str, cfg_stem: Optional[str] = None) -> str:
    """Generate a clean functional stem without iter_ or run-number prefix."""
    stem = cfg_stem or run_name
    # Strip leading number prefix (e.g. 014_, 001_)
    stem = re.sub(r"^\d{3}[a-z]?_", "", stem)
    # Strip iter_ prefix (including letter suffixes like 47b, 48b)
    stem = re.sub(r"^iter_\d+[a-z]?_", "", stem)
    stem = re.sub(r"^iter_\d+[a-z]?$", "", stem)
    # Strip protocol suffixes for clean grouping
    stem = re.sub(r"_v2_dev_p5ck$", "", stem)
    stem = re.sub(r"_v2_dev_p5$", "", stem)
    stem = re.sub(r"_v2_dev_loso_dev_p5$", "", stem)
    stem = re.sub(r"_v2_dev_loso_all_p5$", "", stem)
    stem = re.sub(r"_v2_dev$", "", stem)
    return stem if stem else run_name


def read_run(run_dir: str) -> Optional[Dict]:
    """Read summary.json and config.json from a run directory."""
    name = os.path.basename(run_dir)
    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(summary_path):
        return None

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    cfg = data.get("config", {})
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = {**cfg, **json.load(f)}
        except Exception:
            pass

    role = data.get("role", "dev")
    complete = data.get("complete", True)
    time_str = data.get("time", "")
    elapsed = data.get("elapsed_sec")
    n_folds = data.get("n_folds", 0)
    n_subjects = data.get("n_subjects", 0)
    n_samples = data.get("n", 0)
    uf1 = data.get("UF1")
    uar = data.get("UAR")
    acc = data.get("ACC")

    split = _detect_split(name, n_folds, n_subjects)
    protocol = "protocol_v2" if ("_v2_" in name or "protocol_v2" in summary_path or "_p5" in name) else "protocol_v1"
    if "fusion" in name:
        protocol = "fusion"

    raw_cfg_name = cfg.get("name", "")
    stem = _clean_stem(name, raw_cfg_name if raw_cfg_name else None)

    return {
        "run": name,
        "stem": stem,
        "protocol": protocol,
        "split": split,
        "role": role,
        "complete": complete,
        "UF1": uf1,
        "UAR": uar,
        "ACC": acc,
        "n_folds": n_folds,
        "n_subjects": n_subjects,
        "n_samples": n_samples,
        "backbone": cfg.get("backbone", ""),
        "input_mode": cfg.get("input_mode", cfg.get("modality", "")),
        "img_size": cfg.get("img_size"),
        "T": cfg.get("T"),
        "lr": cfg.get("lr"),
        "seed": cfg.get("seed"),
        "epochs": cfg.get("epochs"),
        "elapsed_sec": elapsed,
        "time": time_str,
    }


def collect(verbose: bool = True) -> pd.DataFrame:
    """Collect all runs in runs/ into results/results.csv."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = []

    for item in sorted(os.listdir(RUNS_DIR)):
        run_path = os.path.join(RUNS_DIR, item)
        if not os.path.isdir(run_path):
            continue
        rec = read_run(run_path)
        if rec:
            records.append(rec)

    df = pd.DataFrame(records, columns=COLUMNS)
    if not df.empty:
        # Sort in natural numerical order of run name
        df.sort_values(by=["run"], inplace=True)
        df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
        if verbose:
            print(f"Collected {len(df)} runs into {RESULTS_CSV}")
    return df


if __name__ == "__main__":
    collect(verbose=True)
