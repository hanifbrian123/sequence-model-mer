"""Seal a development champion before the one-time audit evaluation."""
import argparse
import json
import os
from datetime import datetime


def _flatten_members(experiment, weight=1.0):
    """Resolve a possibly nested fusion into weighted leaf experiments."""
    with open(os.path.join(experiment, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    recipe = summary.get("fusion_recipe")
    if not recipe:
        return [{
            "experiment": os.path.abspath(experiment),
            "config_sha256": summary["config_sha256"],
            "weight": float(weight),
        }]
    members = recipe.get("members", [])
    weights = recipe.get("weights", [])
    if len(members) != len(weights) or not members:
        raise ValueError(f"invalid fusion recipe in {experiment}")
    leaves = []
    for member, member_weight in zip(members, weights):
        leaves.extend(_flatten_members(
            os.path.abspath(member), weight * float(member_weight)))
    return leaves


def _combine_duplicate_leaves(leaves):
    combined = {}
    for leaf in leaves:
        key = (leaf["experiment"], leaf["config_sha256"])
        combined[key] = combined.get(key, 0.0) + leaf["weight"]
    total = sum(combined.values())
    if total <= 0:
        raise ValueError("fusion leaf weights must sum to a positive value")
    return [{"experiment": key[0], "config_sha256": key[1],
             "weight": value / total}
            for key, value in sorted(combined.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--comparison", default="",
                        help="passed compare_protocol report; optional only for initial baseline")
    parser.add_argument("--initial_baseline", action="store_true")
    args = parser.parse_args()
    experiment = os.path.abspath(args.experiment)
    with open(os.path.join(experiment, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("complete", False) or summary.get("role") != "dev":
        raise ValueError("only a complete development experiment can be locked")
    selection = "initial_baseline" if args.initial_baseline else "passed_gate"
    comparison_path = ""
    if args.initial_baseline:
        if args.comparison:
            raise ValueError("initial baseline lock must not claim a comparison")
    else:
        if not args.comparison:
            raise ValueError("non-baseline champion requires --comparison")
        comparison_path = os.path.abspath(args.comparison)
        with open(comparison_path, encoding="utf-8") as handle:
            comparison = json.load(handle)
        if not comparison.get("gate", {}).get("passed", False):
            raise ValueError("comparison gate did not pass")
        if os.path.abspath(comparison.get("candidate", "")) != experiment:
            raise ValueError("comparison candidate is not the requested experiment")
    lock = {
        "time": datetime.now().isoformat(), "gate_passed": True,
        "selection": selection, "config_sha256": summary["config_sha256"],
        "protocol_samples_sha256": summary["protocol_samples_sha256"],
        "candidate_experiment": experiment,
        "comparison_report": comparison_path,
        "audit_policy": "single use; do not tune after observing audit metrics",
    }
    leaves = _combine_duplicate_leaves(_flatten_members(experiment))
    lock["fusion_members"] = leaves
    lock["allowed_config_sha256"] = sorted({
        leaf["config_sha256"] for leaf in leaves})
    output = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
