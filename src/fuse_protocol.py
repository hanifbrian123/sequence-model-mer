"""Create a reproducible fixed-weight fusion from protocol-v2 OOF outputs."""
import argparse
import hashlib
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from metrics import compute_metrics


def _load(directory, role):
    with open(os.path.join(directory, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("complete", False) or summary.get("role") != role:
        raise ValueError(
            f"fusion member must be a complete {role} run: {directory}")
    raw = np.load(os.path.join(directory, "probs.npz"))
    data = {key: raw[key] for key in raw.files}
    return summary, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="fixed_weight_fusion")
    parser.add_argument("--role", choices=("dev", "audit"), default="dev")
    args = parser.parse_args()
    if len(args.members) != len(args.weights):
        raise ValueError("--members and --weights lengths must match")
    weights = np.asarray(args.weights, dtype=np.float64)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("fusion weights must be non-negative with positive sum")
    weights /= weights.sum()

    loaded = [_load(os.path.abspath(member), args.role) for member in args.members]
    summaries = [item[0] for item in loaded]
    protocol_hash = summaries[0]["protocol_samples_sha256"]
    num_classes = summaries[0]["num_classes"]
    if any(summary["protocol_samples_sha256"] != protocol_hash
           for summary in summaries):
        raise ValueError("fusion member protocols differ")
    if any(summary["num_classes"] != num_classes for summary in summaries):
        raise ValueError("fusion member class counts differ")

    reference = loaded[0][1]
    reference_index = {str(key): index for index, key in enumerate(reference["keys"])}
    keys = sorted(reference_index)
    ref_indices = np.asarray([reference_index[key] for key in keys])
    labels = reference["label"][ref_indices]
    subjects = reference["subject"][ref_indices]
    fused = np.zeros((len(keys), num_classes), dtype=np.float64)
    for weight, (_, data) in zip(weights, loaded):
        index = {str(key): position for position, key in enumerate(data["keys"])}
        if set(index) != set(keys):
            raise ValueError("fusion member sample keys differ")
        positions = np.asarray([index[key] for key in keys])
        if not np.array_equal(data["label"][positions], labels) or not np.array_equal(
                data["subject"][positions], subjects):
            raise ValueError("fusion member labels/subjects differ")
        fused += weight * data["probs"][positions]
    predictions = fused.argmax(axis=1)
    metrics = compute_metrics(labels, predictions, num_classes)
    recipe = {
        "members": [os.path.abspath(path) for path in args.members],
        "weights": weights.tolist(),
    }
    recipe_hash = hashlib.sha256(json.dumps(
        recipe, sort_keys=True).encode("utf-8")).hexdigest()
    output = os.path.abspath(args.out)
    os.makedirs(output, exist_ok=True)
    np.savez(os.path.join(output, "probs.npz"), keys=np.asarray(keys),
             subject=subjects, label=labels, probs=fused.astype(np.float32))
    pd.DataFrame({
        "key": keys, "subject": subjects, "label": labels,
        "pred": predictions, "correct": predictions == labels,
    }).to_csv(os.path.join(output, "predictions.csv"), index=False)
    summary = {
        "name": args.name, "role": args.role, "complete": True,
        "time": datetime.now().isoformat(), "num_classes": num_classes,
        "protocol_samples_sha256": protocol_hash,
        "config_sha256": recipe_hash, "fusion_recipe": recipe,
        **metrics,
    }
    with open(os.path.join(output, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"output": output, "UF1": metrics["UF1"],
                      "UAR": metrics["UAR"], "ACC": metrics["ACC"]}, indent=2))


if __name__ == "__main__":
    main()
