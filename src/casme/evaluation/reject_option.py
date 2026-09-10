"""Let the deployed model answer "tidak yakin" instead of guessing.

For an uploaded-video app a confident wrong answer is worse than an honest
abstention, and the model already carries the information needed to tell the
difference — it just was not being used.

Three confidence scores are compared rather than assuming max-probability is
best: the top probability, the margin between the top two classes, and negative
entropy. The margin is often better on an imbalanced problem because a model
can be highly confident and still be choosing between two classes it confuses.

Thresholds are reported per fold as well as pooled. Reading a threshold off the
pooled curve and then quoting the accuracy at that threshold is mildly
optimistic — the same data chose the threshold and scored it — so the per-fold
column is the honest estimate of what a new subject would see.
"""
import argparse
import json
import os

import numpy as np

from casme.evaluation.metrics import compute_metrics


def confidence_scores(probs):
    """Return the three candidate confidence measures for each sample."""
    ordered = np.sort(probs, axis=1)
    return {
        "max_prob": probs.max(axis=1),
        "margin": ordered[:, -1] - ordered[:, -2],
        "neg_entropy": (probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1),
    }


def coverage_curve(labels, predictions, confidence, num_classes, steps=21):
    """Accuracy and UF1 as a function of how many samples we agree to answer."""
    order = np.argsort(-confidence)
    rows = []
    for fraction in np.linspace(1.0, 0.3, steps):
        keep = max(num_classes, int(round(len(labels) * fraction)))
        selected = order[:keep]
        metrics = compute_metrics(
            labels[selected], predictions[selected], num_classes)
        rows.append({
            "coverage": keep / len(labels),
            "n_answered": int(keep),
            "threshold": float(confidence[order[keep - 1]]),
            "ACC": metrics["ACC"],
            "UF1": metrics["UF1"],
        })
    return rows


def threshold_for_target(rows, target_accuracy):
    """Highest coverage whose accuracy still reaches the target."""
    feasible = [row for row in rows if row["ACC"] >= target_accuracy]
    if not feasible:
        return None
    return max(feasible, key=lambda row: row["coverage"])


def per_fold_honest_estimate(labels, predictions, confidence, subjects,
                             num_classes, target_accuracy):
    """Leave-one-subject-out threshold fitting, so scoring never sees its own fit.

    For each subject, the threshold is chosen on every OTHER subject and then
    applied to that subject alone. This is what a deployed threshold actually
    faces: it was tuned on people the model has already seen.
    """
    unique = np.unique(subjects)
    answered = correct = 0
    for subject in unique:
        held = subjects == subject
        rest = ~held
        rows = coverage_curve(labels[rest], predictions[rest],
                              confidence[rest], num_classes)
        chosen = threshold_for_target(rows, target_accuracy)
        if chosen is None:
            continue
        mask = held & (confidence >= chosen["threshold"])
        answered += int(mask.sum())
        correct += int((predictions[mask] == labels[mask]).sum())
    return {
        "n_answered": answered,
        "coverage": answered / len(labels),
        "ACC": (correct / answered) if answered else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", default="experiments/protocol_v2/fusions/deployable_50full_50auto47")
    parser.add_argument("--out", default="")
    parser.add_argument("--target_accuracy", type=float, default=0.80)
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = args.run if os.path.isabs(args.run) else os.path.join(repo, args.run)
    data = np.load(os.path.join(run_dir, "probs.npz"))
    probs = data["probs"].astype(np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = data["label"]
    subjects = data["subject"]
    predictions = probs.argmax(axis=1)
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    num_classes = int(summary["num_classes"])

    base = compute_metrics(labels, predictions, num_classes)
    print(f"model      : {summary['name']}")
    print(f"tanpa reject: ACC {base['ACC']:.4f}  UF1 {base['UF1']:.4f}  "
          f"n={len(labels)}\n")

    scores = confidence_scores(probs)
    report = {"run": summary["name"], "n": int(len(labels)),
              "baseline": {"ACC": base["ACC"], "UF1": base["UF1"]},
              "measures": {}}

    for measure, confidence in scores.items():
        rows = coverage_curve(labels, predictions, confidence, num_classes)
        chosen = threshold_for_target(rows, args.target_accuracy)
        honest = per_fold_honest_estimate(
            labels, predictions, confidence, subjects, num_classes,
            args.target_accuracy)
        report["measures"][measure] = {
            "curve": rows, "pooled_choice": chosen, "honest_estimate": honest}
        print(f"--- ukuran keyakinan: {measure} ---")
        print(f'{"dijawab":>9s} {"ACC":>7s} {"UF1":>7s}   ambang')
        for row in rows[::4]:
            print(f'{row["coverage"]*100:8.0f}% {row["ACC"]:7.4f} '
                  f'{row["UF1"]:7.4f}   {row["threshold"]:.4f}')
        if chosen:
            print(f'  target ACC {args.target_accuracy:.2f} tercapai saat '
                  f'menjawab {chosen["coverage"]*100:.0f}% '
                  f'(ambang {chosen["threshold"]:.4f})')
        else:
            print(f"  target ACC {args.target_accuracy:.2f} tidak tercapai")
        print(f'  estimasi JUJUR (ambang dipilih dari subject lain): '
              f'menjawab {honest["coverage"]*100:.0f}%, ACC {honest["ACC"]:.4f}')
        print()

    out = args.out or os.path.join(run_dir, "reject_option.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"laporan disimpan: {out}")


if __name__ == "__main__":
    main()
