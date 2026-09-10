"""Summarise the label-grouping comparison across seeds.

Two rules are enforced here, because both are easy to violate by accident:

1. Numbers from different taxonomies are NOT comparable. Merging two classes
   removes the errors between them from the metric, so a 4-class score is
   arithmetically higher than a 5-class score on the same model. Only the
   spread across seeds within one grouping is a fair internal comparison.

2. The one genuinely paired comparison available is g3 against g2, restricted
   to the 246 samples they share. On those samples the two label spaces are
   identical, so the promotion gate applies and answers a real question: does
   recovering the nine fear/sadness samples help or hurt?
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from casme.evaluation.compare_protocol import paired_subject_bootstrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(REPO, "experiments", "protocol_v2")
GROUPINGS = ["g1_user", "g2_disgust_others", "g3_disgust_others_plus"]
SEEDS = [42, 123, 2024]


def run_dir(grouping, seed):
    return os.path.join(P, f"iter_56_{grouping}_s{seed}_v2_dev_p5")


def load(directory):
    summary_path = os.path.join(directory, "summary.json")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("complete", False):
        return None
    data = np.load(os.path.join(directory, "probs.npz"), allow_pickle=True)
    order = np.argsort([str(k) for k in data["keys"]])
    return {
        "summary": summary,
        "keys": np.asarray([str(k) for k in data["keys"]])[order],
        "subject": data["subject"][order],
        "label": data["label"][order],
        "probs": data["probs"][order],
    }


print("=" * 74)
print("PERBANDINGAN PENGELOMPOKAN LABEL - 3 grouping x 3 seed")
print("=" * 74)
print()
print("PERINGATAN: angka antar-taksonomi TIDAK sebanding. Menggabungkan dua kelas")
print("menghapus kesalahan di antara keduanya dari metrik, jadi skor 4 kelas selalu")
print("lebih tinggi daripada 5 kelas pada model yang sama. Referensi 5 kelas:")
print("  baseline full-span UF1 0,6816 / ACC 0,6823   champion deployable UF1 0,7021")
print()

loaded = {}
for grouping in GROUPINGS:
    print(f"--- {grouping} ---")
    rows = []
    for seed in SEEDS:
        item = load(run_dir(grouping, seed))
        if item is None:
            print(f"  seed {seed:>4d}: belum selesai")
            continue
        loaded[(grouping, seed)] = item
        s = item["summary"]
        rows.append((s["UF1"], s["UAR"], s["ACC"]))
        print(f"  seed {seed:>4d}: UF1={s['UF1']:.4f}  UAR={s['UAR']:.4f}  "
              f"ACC={s['ACC']:.4f}  (n={s['n']})")
    if len(rows) >= 2:
        arr = np.asarray(rows)
        print(f"  {'rata-rata':>9s}: UF1={arr[:,0].mean():.4f}+-{arr[:,0].std(ddof=1):.4f}  "
              f"UAR={arr[:,1].mean():.4f}  ACC={arr[:,2].mean():.4f}")
    if rows:
        best_seed = SEEDS[int(np.argmax([r[0] for r in rows]))]
        item = loaded.get((grouping, best_seed))
        if item is not None:
            names = item["summary"]["config"]["class_names"]
            print("  F1 per kelas (seed terbaik):")
            for name, f1 in zip(names, item["summary"]["per_class_f1"]):
                support = int((item["label"] == names.index(name)).sum())
                print(f"    {name:26s} F1={f1:.4f}  (n={support})")
    print()

print("=" * 74)
print("SATU-SATUNYA GATE YANG SAH: g3 vs g2 pada 246 sampel yang sama")
print("pertanyaan: apakah menambahkan 9 sampel fear/sadness menolong?")
print("=" * 74)
deltas = []
for seed in SEEDS:
    a = loaded.get(("g2_disgust_others", seed))
    b = loaded.get(("g3_disgust_others_plus", seed))
    if a is None or b is None:
        print(f"  seed {seed}: belum lengkap")
        continue
    common = np.intersect1d(a["keys"], b["keys"])
    ia = np.searchsorted(a["keys"], common)
    ib = np.searchsorted(b["keys"], common)
    if not np.array_equal(a["label"][ia], b["label"][ib]):
        print(f"  seed {seed}: label berbeda pada sampel bersama - dilewati")
        continue
    result = paired_subject_bootstrap(
        a["label"][ia], a["subject"][ia], a["probs"][ia], b["probs"][ib],
        len(a["summary"]["config"]["class_names"]))
    u = result["UF1"]
    deltas.append(u["delta"])
    passed = (u["delta"] >= 0.005 and u["probability_improved"] >= 0.80
              and result["UAR"]["delta"] >= -0.01 and result["ACC"]["delta"] >= -0.01)
    print(f"  seed {seed:>4d} (n={len(common)}): g2={u['baseline']:.4f} "
          f"g3={u['candidate']:.4f} dUF1={u['delta']:+.4f} "
          f"CI80=[{u['ci'][0]:+.4f},{u['ci'][1]:+.4f}] "
          f"P={u['probability_improved']:.3f} -> {'LOLOS' if passed else 'GAGAL'}")
if len(deltas) >= 2:
    arr = np.asarray(deltas)
    print(f"  efek lintas seed: rata-rata {arr.mean():+.4f}, sd {arr.std(ddof=1):.4f}, "
          f"lolos di {sum(d >= 0.005 for d in arr)} dari {len(arr)} seed")

print()
print("=" * 74)
print("CARA MEMBACA")
print("=" * 74)
print("Bandingkan g2 dengan g3 lewat gate di atas - itu perbandingan yang adil.")
print("Bandingkan g1 dengan g2/g3 hanya secara deskriptif: keduanya 4 kelas, tetapi")
print("isi kelasnya berbeda, sehingga tugasnya memang tidak sama beratnya.")
print("Kelas gabungan yang besar menaikkan ACC secara otomatis; lihat F1 per kelas")
print("untuk menilai apakah ada kelas yang dikorbankan.")
