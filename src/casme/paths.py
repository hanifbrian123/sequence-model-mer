"""Single source of truth for all filesystem paths in the CASME II project.

Directory contract:
    configs/            NNN_name.json, one per experiment
    runs/               NNN_name/, per-run self-contained artifacts
    results/            results.csv, all experiment metrics in one table
    reports/            LAPORAN.md, master generated report
    logs/               throwaway queue & execution logs (gitignored)
    docs/               narasi.md, architecture notes, handoffs
    scripts/            automation bash scripts (run_queue.sh, check.sh, etc.)
    deploy/             app & serving code
    cache/              precomputed flow & face caches
    protocols/          split specifications (accuracy_v*.json)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_DIR = os.path.join(ROOT, "configs")
RUNS_DIR = os.path.join(ROOT, "runs")
EXP_DIR = os.path.join(ROOT, "experiments")
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
LOGS_DIR = os.path.join(ROOT, "logs")
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
DOCS_DIR = os.path.join(ROOT, "docs")
DEPLOY_DIR = os.path.join(ROOT, "deploy")
CACHE_DIR = os.path.join(ROOT, "cache")
PROTOCOLS_DIR = os.path.join(ROOT, "protocols")
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")

RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
REPORT_MD = os.path.join(REPORTS_DIR, "LAPORAN.md")
NARRATIVE_MD = os.path.join(DOCS_DIR, "narasi.md")

# CASME II 5 canonical classes
CLASSES = ("happiness", "disgust", "repression", "surprise", "others")
NUM_CLASSES = len(CLASSES)
LABEL_OF = {name: i for i, name in enumerate(CLASSES)}


def run_dir(name: str, create: bool = False) -> str:
    path = os.path.join(RUNS_DIR, name)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def config_path(name: str) -> str:
    stem = name[:-5] if name.endswith(".json") else name
    return os.path.join(CONFIG_DIR, f"{stem}.json")
