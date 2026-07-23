import json
import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import compare_protocol  # noqa: E402
import lock_protocol_champion  # noqa: E402
import protocol_v2  # noqa: E402
import run_protocol_v2  # noqa: E402


def grouped_samples():
    samples = []
    for subject in range(15):
        for index in range(5):
            samples.append({
                "key": f"subject_{subject}_sample_{index}",
                "subject": subject,
                "label": (subject + index) % 5,
            })
    return samples


class ProtocolV2Tests(unittest.TestCase):
    def test_protocol_is_deterministic_disjoint_and_reusable(self):
        samples = grouped_samples()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "protocol.json")
            first = protocol_v2.load_or_create_protocol(
                samples, 5, path, n_folds=5, seed=123)
            second = protocol_v2.load_or_create_protocol(
                list(reversed(samples)), 5, path, n_folds=5, seed=123)
        self.assertEqual(first, second)
        audit, development, folds = protocol_v2.protocol_subject_sets(first)
        self.assertFalse(audit & development)
        self.assertEqual(audit | development, set(range(15)))
        self.assertEqual(len(folds), 4)
        all_fold_subjects = [subject for _, fold in folds for subject in fold]
        self.assertEqual(len(all_fold_subjects), len(set(all_fold_subjects)))

    def test_fingerprint_rejects_changed_metadata(self):
        samples = grouped_samples()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "protocol.json")
            protocol_v2.load_or_create_protocol(samples, 5, path)
            changed = [dict(row) for row in samples]
            changed[0]["label"] = 4
            with self.assertRaises(ValueError):
                protocol_v2.load_or_create_protocol(changed, 5, path)

    def test_paired_subject_bootstrap_detects_clear_improvement(self):
        labels = np.tile(np.arange(5), 6)
        subjects = np.repeat(np.arange(6), 5)
        baseline = np.full((30, 5), 0.025, dtype=np.float32)
        baseline[:, 0] = 0.9
        candidate = np.full((30, 5), 0.025, dtype=np.float32)
        candidate[np.arange(30), labels] = 0.9
        result = compare_protocol.paired_subject_bootstrap(
            labels, subjects, baseline, candidate, 5,
            iterations=200, seed=7, interval=0.80)
        self.assertGreater(result["UF1"]["delta"], 0.5)
        self.assertEqual(result["UF1"]["probability_improved"], 1.0)
        self.assertGreater(result["ACC"]["ci"][0], 0.5)

    def test_nested_fusion_lock_flattens_and_authorizes_leaf_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            leaves = []
            for index in range(2):
                leaf = os.path.join(directory, f"leaf{index}")
                os.makedirs(leaf)
                with open(os.path.join(leaf, "summary.json"), "w",
                          encoding="utf-8") as handle:
                    json.dump({"config_sha256": f"hash{index}"}, handle)
                leaves.append(leaf)
            nested = os.path.join(directory, "nested")
            os.makedirs(nested)
            with open(os.path.join(nested, "summary.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"fusion_recipe": {
                    "members": [leaves[0], leaves[1]],
                    "weights": [0.25, 0.75]}}, handle)
            flattened = lock_protocol_champion._combine_duplicate_leaves(
                lock_protocol_champion._flatten_members(nested))
            self.assertEqual([row["weight"] for row in flattened], [0.25, 0.75])

            lock_path = os.path.join(directory, "lock.json")
            with open(lock_path, "w", encoding="utf-8") as handle:
                json.dump({"gate_passed": True,
                           "allowed_config_sha256": ["hash0", "hash1"],
                           "protocol_samples_sha256": "protocol"}, handle)
            protocol = {"samples_sha256": "protocol"}
            run_protocol_v2._verify_audit_lock(lock_path, "hash1", protocol)
            with self.assertRaises(ValueError):
                run_protocol_v2._verify_audit_lock(lock_path, "other", protocol)


if __name__ == "__main__":
    unittest.main()
