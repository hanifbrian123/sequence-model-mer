"""Extract Action Unit multi-labels from the coding sheet, keyed by cache key.

Why this exists: 76% of the champion's errors involve the `others` class, and
`others` is defined negatively — "not one of the four named emotions" — so it
has no consistent emotional signature to learn. It does have a consistent AU
signature. Training the same backbone to predict AUs alongside the emotion
gives a dense, physically grounded target on a 192-sample problem, and the AU
head is dropped at inference so nothing about deployment changes.

Written as a side file rather than a manifest rebuild on purpose: the protocol
fingerprint covers key/subject/label, and regenerating the cache manifest would
risk disturbing a split that must stay byte-identical across every experiment.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

CODING = r"D:/AI-Projects/casmeII-facesleuth-r/dataset/CASME2-coding-20140508.xlsx"
# AUs with at least 8 positives in the 246-sample index. Rarer AUs cannot be
# learned from single-digit positives and only add label noise.
AU_CODES = [1, 2, 4, 6, 7, 9, 10, 12, 14, 15, 17]


def parse_units(value):
    """'4+7' / '12' / 'R14' -> {4, 7} / {12} / {14}."""
    return {int(token) for token in re.findall(r"\d+", str(value))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="cache/flow144/manifest.csv")
    parser.add_argument("--out", default="cache/au_labels.csv")
    parser.add_argument("--coding", default=CODING)
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = pd.read_csv(os.path.join(repo, args.manifest))
    coding = pd.read_excel(args.coding)

    lookup = {}
    for _, row in coding.iterrows():
        key = f"sub{int(row['Subject']):02d}__{str(row['Filename']).strip()}"
        lookup[key] = parse_units(row["Action Units"])

    rows = []
    missing = []
    for _, row in manifest.iterrows():
        key = row["key"]
        units = lookup.get(key)
        if units is None:
            missing.append(key)
            units = set()
        entry = {"key": key}
        for code in AU_CODES:
            entry[f"au{code}"] = int(code in units)
        entry["n_au"] = len(units)
        rows.append(entry)

    table = pd.DataFrame(rows)
    out_path = os.path.join(repo, args.out)
    table.to_csv(out_path, index=False)

    counts = {f"au{c}": int(table[f"au{c}"].sum()) for c in AU_CODES}
    positives = table[[f"au{c}" for c in AU_CODES]].values
    print(f"written {len(table)} rows -> {out_path}")
    print(f"unmatched keys: {len(missing)}" + (f" {missing[:5]}" if missing else ""))
    print("positif per AU:", counts)
    print(f"rata-rata AU aktif per sampel: {positives.sum(axis=1).mean():.2f}")
    print(f"sampel tanpa satupun AU terdaftar: "
          f"{int((positives.sum(axis=1) == 0).sum())}")
    # pos_weight for BCE: each AU is rare, so unweighted BCE would collapse to
    # predicting all-zero. Report the weights the trainer will use.
    counts_array = positives.sum(axis=0).astype(np.float64)
    pos_weight = (len(table) - counts_array) / np.maximum(counts_array, 1.0)
    print("pos_weight BCE:",
          {f"au{c}": round(float(w), 2) for c, w in zip(AU_CODES, pos_weight)})


if __name__ == "__main__":
    main()
