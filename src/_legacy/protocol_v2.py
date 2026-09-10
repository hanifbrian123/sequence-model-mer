"""Deterministic subject-group protocol utilities for accuracy experiments."""
import hashlib
import json
import os

import numpy as np


PROTOCOL_VERSION = 5


def samples_fingerprint(samples):
    """Hash identity/split metadata without serializing licensed sample content."""
    digest = hashlib.sha256()
    for sample in sorted(samples, key=lambda row: str(row["key"])):
        line = f'{sample["key"]}\t{sample["subject"]}\t{sample["label"]}\n'
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def config_fingerprint(cfg):
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _balanced_subject_folds(samples, num_classes, n_folds, seed):
    """Globally balance label mass, sample count, and subject capacities."""
    subjects = sorted({int(row["subject"]) for row in samples})
    if len(subjects) < n_folds:
        raise ValueError(f"need at least {n_folds} subjects, found {len(subjects)}")
    vectors = {subject: np.zeros(num_classes, dtype=np.float64)
               for subject in subjects}
    for row in samples:
        vectors[int(row["subject"])][int(row["label"])] += 1.0

    total_labels = sum(vectors.values(), np.zeros(num_classes, dtype=np.float64))
    target_labels = np.maximum(total_labels / n_folds, 1.0)
    target_samples = float(total_labels.sum()) / n_folds
    rng = np.random.default_rng(seed)
    fold_ties = rng.permutation(n_folds).tolist()
    base_capacity, remainder = divmod(len(subjects), n_folds)
    capacities = np.full(n_folds, base_capacity, dtype=np.int64)
    # Fold zero is the sealed audit set. Keep it at floor(N/K) and distribute
    # any remainder over development folds first so audit does not become an
    # outsized fraction of this tiny dataset.
    extra_order = [fold for fold in fold_ties if fold != 0] + [0]
    for fold in extra_order[:remainder]:
        capacities[fold] += 1

    def assignment_from_permutation(permutation):
        result = []
        cursor = 0
        for capacity in capacities:
            result.append(list(permutation[cursor:cursor + capacity]))
            cursor += capacity
        return result

    def score(folds):
        label_counts = np.stack([
            sum((vectors[subject] for subject in fold),
                np.zeros(num_classes, dtype=np.float64))
            for fold in folds
        ])
        # Pearson-style discrepancy gives rare classes equal leverage while
        # total sample imbalance is already reflected in the label-count sum.
        discrepancy = float(np.square(
            (label_counts - target_labels) / np.sqrt(target_labels)).sum())
        sample_counts = label_counts.sum(axis=1)
        sample_discrepancy = float(np.square(
            (sample_counts - target_samples) / np.sqrt(target_samples)).sum())
        feasible_classes = total_labels >= n_folds
        missing = (label_counts[:, feasible_classes] == 0).sum()
        return discrepancy + 4.0 * sample_discrepancy + 25.0 * float(missing)

    subject_array = np.asarray(subjects, dtype=np.int64)
    best_folds = None
    best_score = float("inf")
    # Randomized global search is cheap here (26 subjects) and avoids the
    # path-dependence that allowed a greedy assignment to produce 35-vs-72
    # sample folds even though every fold had five subjects.
    for _ in range(50000):
        candidate = assignment_from_permutation(rng.permutation(subject_array))
        candidate_score = score(candidate)
        if candidate_score < best_score:
            best_score, best_folds = candidate_score, candidate
    folds = best_folds
    if not all(len(fold) == int(capacities[index])
               for index, fold in enumerate(folds)):
        raise RuntimeError("failed to satisfy fixed subject-fold capacities")
    return [sorted(int(subject) for subject in fold) for fold in folds]


def load_or_create_protocol(samples, num_classes, path, n_folds=5,
                            seed=20260722):
    """Create once, then verify and reuse a fixed audit/development split."""
    fingerprint = samples_fingerprint(samples)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            protocol = json.load(handle)
        if protocol.get("version") != PROTOCOL_VERSION:
            raise ValueError("protocol version mismatch")
        if protocol.get("samples_sha256") != fingerprint:
            raise ValueError("dataset metadata changed; refusing to reuse protocol")
        if protocol.get("num_classes") != num_classes:
            raise ValueError("class count changed; refusing to reuse protocol")
        return protocol

    folds = _balanced_subject_folds(
        samples, num_classes=num_classes, n_folds=n_folds, seed=seed)
    protocol = {
        "version": PROTOCOL_VERSION,
        "seed": int(seed),
        "num_classes": int(num_classes),
        "samples_sha256": fingerprint,
        "audit_fold": 0,
        "folds": [{"fold": index, "subjects": subjects}
                  for index, subjects in enumerate(folds)],
        "policy": {
            "development": "group CV over every non-audit fold",
            "audit": "fold 0 is evaluated only after a candidate is locked",
            "outer_loso": "reserved for one final locked literature benchmark",
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    return protocol


def protocol_subject_sets(protocol):
    audit_fold = int(protocol["audit_fold"])
    audit = set()
    development_folds = []
    for fold in protocol["folds"]:
        subjects = {int(value) for value in fold["subjects"]}
        if int(fold["fold"]) == audit_fold:
            audit |= subjects
        else:
            development_folds.append((int(fold["fold"]), subjects))
    development = set().union(*(subjects for _, subjects in development_folds))
    if audit & development:
        raise ValueError("audit and development subjects overlap")
    return audit, development, development_folds
