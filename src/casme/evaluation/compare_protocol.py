"""Paired subject-bootstrap gate for protocol-v2 candidates."""
import argparse
import json
import os
from datetime import datetime

import numpy as np

from casme.evaluation.metrics import compute_metrics


def _load_predictions(directory):
    summary_path = os.path.join(directory, "summary.json")
    probability_path = os.path.join(directory, "probs.npz")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("complete", False):
        raise ValueError(f"incomplete protocol run: {directory}")
    data = np.load(probability_path)
    return summary, {key: data[key] for key in data.files}


def _align(base, candidate):
    base_index = {str(key): index for index, key in enumerate(base["keys"])}
    candidate_index = {str(key): index for index, key in enumerate(candidate["keys"])}
    if set(base_index) != set(candidate_index):
        raise ValueError("baseline and candidate sample keys differ")
    keys = sorted(base_index)
    bi = np.asarray([base_index[key] for key in keys])
    ci = np.asarray([candidate_index[key] for key in keys])
    labels = base["label"][bi]
    subjects = base["subject"][bi]
    if not np.array_equal(labels, candidate["label"][ci]):
        raise ValueError("baseline and candidate labels differ")
    if not np.array_equal(subjects, candidate["subject"][ci]):
        raise ValueError("baseline and candidate subject alignment differs")
    return labels, subjects, base["probs"][bi], candidate["probs"][ci]


def paired_subject_bootstrap(labels, subjects, baseline_probs, candidate_probs,
                             num_classes, iterations=10000, seed=20260722,
                             interval=0.80):
    unique_subjects = np.unique(subjects)
    rng = np.random.default_rng(seed)
    metric_names = ("UF1", "UAR", "ACC")
    observed_baseline = compute_metrics(
        labels, baseline_probs.argmax(1), num_classes)
    observed_candidate = compute_metrics(
        labels, candidate_probs.argmax(1), num_classes)
    baseline_predictions = baseline_probs.argmax(1)
    candidate_predictions = candidate_probs.argmax(1)

    def subject_confusions(predictions):
        confusions = np.zeros(
            (len(unique_subjects), num_classes, num_classes), dtype=np.int64)
        for subject_index, subject in enumerate(unique_subjects):
            indices = np.flatnonzero(subjects == subject)
            np.add.at(confusions[subject_index],
                      (labels[indices], predictions[indices]), 1)
        return confusions

    def vectorized_metrics(confusions):
        true_positive = np.diagonal(confusions, axis1=1, axis2=2)
        support = confusions.sum(axis=2)
        predicted = confusions.sum(axis=1)
        recall = np.divide(true_positive, support,
                           out=np.zeros_like(true_positive, dtype=np.float64),
                           where=support != 0)
        f1 = np.divide(2.0 * true_positive, support + predicted,
                       out=np.zeros_like(true_positive, dtype=np.float64),
                       where=(support + predicted) != 0)
        total = confusions.sum(axis=(1, 2))
        accuracy = np.divide(true_positive.sum(axis=1), total,
                             out=np.zeros_like(total, dtype=np.float64),
                             where=total != 0)
        return {"UF1": f1.mean(axis=1), "UAR": recall.mean(axis=1),
                "ACC": accuracy}

    # Sampling subjects with replacement is equivalent to multinomial subject
    # multiplicities. Aggregate precomputed per-subject confusion matrices in
    # one vectorized operation; metrics are exactly the fixed-label sklearn
    # definitions but several orders of magnitude faster.
    multiplicities = rng.multinomial(
        len(unique_subjects), np.full(len(unique_subjects), 1.0 / len(unique_subjects)),
        size=iterations)
    baseline_confusions = np.einsum(
        "bs,sij->bij", multiplicities, subject_confusions(baseline_predictions))
    candidate_confusions = np.einsum(
        "bs,sij->bij", multiplicities, subject_confusions(candidate_predictions))
    baseline_bootstrap = vectorized_metrics(baseline_confusions)
    candidate_bootstrap = vectorized_metrics(candidate_confusions)
    differences = {metric: candidate_bootstrap[metric] - baseline_bootstrap[metric]
                   for metric in metric_names}

    tail = (1.0 - interval) / 2.0
    result = {}
    for metric in metric_names:
        delta = differences[metric]
        result[metric] = {
            "baseline": observed_baseline[metric],
            "candidate": observed_candidate[metric],
            "delta": observed_candidate[metric] - observed_baseline[metric],
            "ci": [float(np.quantile(delta, tail)),
                   float(np.quantile(delta, 1.0 - tail))],
            "probability_improved": float(np.mean(delta > 0.0)),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lock_out", default="")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--interval", type=float, default=0.80)
    parser.add_argument("--min_uf1_delta", type=float, default=0.005)
    parser.add_argument("--min_probability", type=float, default=0.80)
    parser.add_argument("--max_secondary_drop", type=float, default=0.01)
    args = parser.parse_args()

    baseline_summary, baseline = _load_predictions(args.baseline)
    candidate_summary, candidate = _load_predictions(args.candidate)
    if baseline_summary["role"] != "dev" or candidate_summary["role"] != "dev":
        raise ValueError("the tuning gate only compares development runs")
    if baseline_summary["protocol_samples_sha256"] != \
            candidate_summary["protocol_samples_sha256"]:
        raise ValueError("protocol fingerprints differ")
    if baseline_summary["num_classes"] != candidate_summary["num_classes"]:
        raise ValueError("class counts differ")
    labels, subjects, base_probs, candidate_probs = _align(baseline, candidate)
    comparison = paired_subject_bootstrap(
        labels, subjects, base_probs, candidate_probs,
        candidate_summary["num_classes"], iterations=args.iterations,
        seed=args.seed, interval=args.interval)
    passed = (
        comparison["UF1"]["delta"] >= args.min_uf1_delta
        and comparison["UF1"]["probability_improved"] >= args.min_probability
        and comparison["UAR"]["delta"] >= -args.max_secondary_drop
        and comparison["ACC"]["delta"] >= -args.max_secondary_drop
    )
    report = {
        "time": datetime.now().isoformat(),
        "baseline": os.path.abspath(args.baseline),
        "candidate": os.path.abspath(args.candidate),
        "iterations": args.iterations, "interval": args.interval,
        "gate": {
            "passed": passed, "min_uf1_delta": args.min_uf1_delta,
            "min_probability": args.min_probability,
            "max_secondary_drop": args.max_secondary_drop,
        },
        "metrics": comparison,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if args.lock_out:
        if not passed:
            raise SystemExit("candidate did not pass; no audit lock written")
        lock = {
            "time": datetime.now().isoformat(), "gate_passed": True,
            "config_sha256": candidate_summary["config_sha256"],
            "protocol_samples_sha256": candidate_summary["protocol_samples_sha256"],
            "candidate_experiment": os.path.abspath(args.candidate),
            "comparison_report": os.path.abspath(args.out),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.lock_out)), exist_ok=True)
        with open(args.lock_out, "w", encoding="utf-8") as handle:
            json.dump(lock, handle, indent=2)


if __name__ == "__main__":
    main()
