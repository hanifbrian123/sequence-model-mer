"""Structural integrity tests guarding the CASME II repository layout.

Modeled after fer/tests/test_structure.py:
- Prevents loose scripts or CSVs in the root
- Ensures results live in results/results.csv and reports in reports/LAPORAN.md
- Enforces NNN_nama naming for runs and configs
- Ensures logs belong in logs/ or run folders
- Enforces dataset privacy rules
"""
import glob
import json
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from casme.paths import (CONFIG_DIR, EXP_DIR, LOGS_DIR, REPORTS_DIR,
                         RESULTS_CSV, RESULTS_DIR, ROOT, RUNS_DIR)

CONFIG_RE = re.compile(r"^\d{3}_[a-z0-9_]+$")


class TestRootCleanliness(unittest.TestCase):
    """Keep the repository root clean, professional, and maintainable."""

    def test_no_loose_shell_scripts_in_root(self):
        scripts = glob.glob(os.path.join(ROOT, "*.sh"))
        self.assertEqual(
            scripts, [],
            f"Script shell tidak boleh ditaruh di root repo! "
            f"Pindahkan ke folder scripts/. Ditemukan: {scripts}"
        )

    def test_no_loose_csv_in_root(self):
        csvs = glob.glob(os.path.join(ROOT, "*.csv"))
        self.assertEqual(
            csvs, [],
            f"File CSV tidak boleh ditaruh di root repo! "
            f"Semua angka harus masuk ke results/results.csv. Ditemukan: {csvs}"
        )

    def test_no_loose_python_scripts_in_root(self):
        pys = glob.glob(os.path.join(ROOT, "*.py"))
        self.assertEqual(
            pys, [],
            f"Script Python tidak boleh ditaruh di root repo! "
            f"Pindahkan ke src/casme/ atau deploy/. Ditemukan: {pys}"
        )


class TestSingleSourceOfTruth(unittest.TestCase):
    """Results live in results/results.csv, report in reports/LAPORAN.md."""

    def test_results_csv_exists(self):
        self.assertTrue(
            os.path.exists(RESULTS_CSV),
            f"results/results.csv wajib ada sebagai single source of truth untuk seluruh angka!"
        )

    def test_report_md_exists(self):
        report_path = os.path.join(REPORTS_DIR, "LAPORAN.md")
        self.assertTrue(
            os.path.exists(report_path),
            f"reports/LAPORAN.md wajib ada sebagai laporan terpadu seluruh eksperimen!"
        )


class TestNamingConventions(unittest.TestCase):
    """Configs and runs follow NNN_nama convention."""

    def test_standardized_configs_exist(self):
        numbered_configs = [
            f for f in glob.glob(os.path.join(CONFIG_DIR, "*.json"))
            if re.match(r"^\d{3}_", os.path.basename(f))
        ]
        self.assertGreater(
            len(numbered_configs), 50,
            f"Minimal harus ada configs bernomor NNN_*.json di configs/. "
            f"Ditemukan {len(numbered_configs)}"
        )

    def test_runs_are_numbered(self):
        runs = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d)) and not d.startswith(("_", "."))]
        numbered_runs = [d for d in runs if re.match(r"^\d{3}_", d)]
        self.assertGreater(
            len(numbered_runs), 50,
            f"Runs di runs/ wajib menggunakan format bernomor NNN_nama. "
            f"Ditemukan {len(numbered_runs)} dari {len(runs)}"
        )

    def test_runs_and_configs_have_strictly_unique_numbers(self):
        runs = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d)) and not d.startswith(("_", "."))]
        run_prefixes = [d[:3] for d in runs if re.match(r"^\d{3}_", d)]
        self.assertEqual(
            len(run_prefixes), len(set(run_prefixes)),
            f"Setiap run di runs/ wajib memiliki nomor unik tanpa duplikat! Ditemukan duplikat pada run."
        )

        cfgs = [os.path.basename(f)[:-5] for f in glob.glob(os.path.join(CONFIG_DIR, "*.json"))]
        cfg_prefixes = [c[:3] for c in cfgs if re.match(r"^\d{3}_", c)]
        self.assertEqual(
            len(cfg_prefixes), len(set(cfg_prefixes)),
            f"Setiap config di configs/ wajib memiliki nomor unik tanpa duplikat! Ditemukan duplikat pada configs."
        )


class TestLoggingDiscipline(unittest.TestCase):
    """Logs belong in logs/ or inside run directories, never loose in experiments/."""

    def test_no_stray_log_files_in_experiments(self):
        stray_logs = glob.glob(os.path.join(EXP_DIR, "*.log"))
        self.assertEqual(
            stray_logs, [],
            f"File log lepas tidak boleh ada di experiments/! "
            f"Pindahkan ke logs/ (sekali pakai, gitignored). Ditemukan: {stray_logs}"
        )


class TestDatasetStaysPrivate(unittest.TestCase):
    """Licensed dataset must never be tracked by git."""

    def test_gitignore_protects_logs(self):
        with open(os.path.join(ROOT, ".gitignore"), "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("logs/", content, "Folder logs/ harus di-gitignore")


if __name__ == "__main__":
    unittest.main(verbosity=2)
