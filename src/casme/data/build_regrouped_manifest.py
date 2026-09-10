"""Build alternative label groupings without disturbing the sealed audit split.

Two artifacts per grouping:

  cache/<cache_dir>/manifest_<name>.csv   labels remapped, referenced from a
                                          config via "manifest_name"
  protocols/accuracy_v5_<name>.json       same subject->fold assignment as the
                                          5-class protocol, only the class count
                                          and sample fingerprint updated

The protocol is *derived*, never regenerated. ``load_or_create_protocol`` would
otherwise refuse the existing file (the class count and sample list both change)
and build a fresh split, which could move audit subjects into development. The
audit fold is asserted identical before anything is written.

    python src/build_regrouped_manifest.py --grouping g2_disgust_others
    python src/build_regrouped_manifest.py --list
"""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from casme.evaluation.protocol_v2 import PROTOCOL_VERSION, samples_fingerprint

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PROTOCOL = os.path.join(REPO, "protocols", "accuracy_v5.json")

# Every grouping is expressed over the raw "Estimated Emotion" strings so the
# mapping is auditable at a glance. Emotions absent from a grouping are dropped.
GROUPINGS = {
    # As proposed by the user: merges disgust with surprise, and folds the two
    # emotions the 5-class protocol discards (fear, sadness) into repression.
    "g1_user": {
        "classes": ["happiness", "disgust_surprise", "fear_serious", "others"],
        "map": {"happiness": 0, "disgust": 1, "surprise": 1,
                "repression": 2, "fear": 2, "sadness": 2, "others": 3},
    },
    # What the confusion matrix and the AU coding both point at: disgust and
    # others share objective class 2 and account for 19 of 25 mutual errors.
    # Surprise is kept intact because it is the one class with a unique AU
    # signature (AU1, purity 1.00) and the best F1 in the 5-class model.
    "g2_disgust_others": {
        "classes": ["happiness", "disgust_others", "repression", "surprise"],
        "map": {"happiness": 0, "disgust": 1, "others": 1,
                "repression": 2, "surprise": 3},
    },
    # g2 plus the nine fear/sadness samples, to test whether recovering them
    # helps or simply adds noise to an already weak class.
    "g3_disgust_others_plus": {
        "classes": ["happiness", "disgust_others", "repression_fear_sadness",
                    "surprise"],
        "map": {"happiness": 0, "disgust": 1, "others": 1,
                "repression": 2, "fear": 2, "sadness": 2, "surprise": 3},
    },
}


def build(name, cache_dir, source_manifest):
    spec = GROUPINGS[name]
    mapping = spec["map"]
    classes = spec["classes"]

    source_path = os.path.join(REPO, cache_dir, source_manifest)
    frame = pd.read_csv(source_path)
    frame["emotion"] = frame["emotion"].astype(str).str.strip().str.lower()
    print(f"sumber: {source_path} ({len(frame)} baris)")

    unknown = sorted(set(frame["emotion"]) - set(mapping))
    kept = frame[frame["emotion"].isin(mapping)].copy()
    kept["label"] = kept["emotion"].map(mapping).astype(int)
    print(f"emosi dibuang oleh grouping ini: {unknown if unknown else 'tidak ada'}")

    missing = [row["npy"] for _, row in kept.iterrows()
               if not os.path.exists(os.path.join(REPO, cache_dir, row["npy"]))]
    if missing:
        raise SystemExit(f"{len(missing)} array cache tidak ditemukan, contoh: {missing[:3]}")

    print(f"total sampel: {len(kept)}")
    for index, class_name in enumerate(classes):
        members = sorted(kept.loc[kept["label"] == index, "emotion"].unique())
        print(f"  [{index}] {class_name:26s} n={int((kept['label'] == index).sum()):4d} "
              f"<- {', '.join(members)}")

    out_manifest = os.path.join(REPO, cache_dir, f"manifest_{name}.csv")
    kept.to_csv(out_manifest, index=False)
    print(f"ditulis: {out_manifest}")

    # --- derive the protocol, preserving the subject partition exactly --------
    with open(BASE_PROTOCOL, encoding="utf-8") as handle:
        base = json.load(handle)
    samples = [{"key": row["key"], "subject": int(row["subject"]),
                "label": int(row["label"])} for _, row in kept.iterrows()]
    protocol = dict(base)
    protocol["num_classes"] = len(classes)
    protocol["samples_sha256"] = samples_fingerprint(samples)
    protocol["derived_from"] = os.path.relpath(BASE_PROTOCOL, REPO).replace("\\", "/")
    protocol["grouping"] = {"name": name, "classes": classes, "map": mapping}
    protocol["version"] = PROTOCOL_VERSION

    def audit_subjects(spec):
        return next(fold["subjects"] for fold in spec["folds"]
                    if fold["fold"] == spec["audit_fold"])

    base_audit = audit_subjects(base)
    new_audit = audit_subjects(protocol)
    if sorted(base_audit) != sorted(new_audit):
        raise SystemExit("audit fold berubah — dibatalkan")
    if [f["subjects"] for f in base["folds"]] != [f["subjects"] for f in protocol["folds"]]:
        raise SystemExit("pembagian subject berubah — dibatalkan")
    present = set(kept["subject"].astype(int))
    audit_present = sorted(set(new_audit) & present)
    print(f"audit fold (tetap tersegel): {sorted(new_audit)}")
    print(f"  subject audit yang muncul di manifest: {audit_present} "
          f"(dipisahkan saat runtime oleh protocol_subject_sets)")

    out_protocol = os.path.join(REPO, "protocols", f"accuracy_v5_{name}.json")
    with open(out_protocol, "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    print(f"ditulis: {out_protocol}")
    return out_manifest, out_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouping", choices=sorted(GROUPINGS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cache_dir", default="cache/flow144")
    parser.add_argument("--source_manifest", default="manifest_objective.csv")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, spec in GROUPINGS.items():
            print(f"{name}: {spec['classes']}")
        return
    targets = sorted(GROUPINGS) if args.all else [args.grouping]
    if not targets or targets == [None]:
        raise SystemExit("pilih --grouping <nama> atau --all")
    for name in targets:
        print(f"\n=== {name} ===")
        build(name, args.cache_dir, args.source_manifest)


if __name__ == "__main__":
    main()
