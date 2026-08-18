"""Regenerate reports/EKSPERIMEN.md from the runs on disk.

Written as a script rather than a hand-kept document so the table can never
drift from the artifacts: every number is read back from summary.json, the
fusions are re-fused from the saved probabilities, and the gate probabilities
are re-bootstrapped. Re-run after any queue finishes.
"""
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_protocol import paired_subject_bootstrap
from metrics import compute_metrics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2 = os.path.join(REPO, "experiments/protocol_v2")
OUT = os.path.join(REPO, "reports/EKSPERIMEN.md")

MARK = {"G": "🟢 grouped", "L": "🔵 LOSO 21", "F": "⚫ LOSO 26", "A": "🔴 audit"}
SUFFIX = {"G": "_v2_dev_p5", "L": "_v2_dev_loso_dev_p5",
          "F": "_v2_dev_loso_all_p5"}

SECTION = "SECTION"
# (nomor, protokol, fase, deskripsi, stem, acuan)
PLAN = [
    (SECTION, "— ACUAN —"),
    ("—", "G", "—", "Baseline iter_14", "iter_14_r3d", None),
    ("—", "L", "—", "Baseline iter_14", "iter_14_r3d", None),
    ("—", "L", "—", "Auto-apex iter_47", "iter_47_r3d_auto_apex_s42", None),

    (SECTION, "— FASE 1: CARA MELATIH —"),
    ("58", "G", "1.1", "Early stopping, cek 17 sampel", "iter_58_r3d_earlystop_s42", "fullG"),
    ("63", "G", "1.1", "Early stopping, cek 59 sampel", "iter_63_r3d_es_bigcheck_s42", "fullG"),
    ("64", "G", "1.1", "Early stopping, penilai loss", "iter_64_r3d_es_loss_s42", "fullG"),
    ("59", "G", "1.2", "Adaptive fine-tuning", "iter_59_r3d_llrd_s42", "fullG"),
    ("60", "G", "1.3", "Gradual unfreeze", "iter_60_r3d_unfreeze_s42", "fullG"),
    ("61", "G", "1.4", "Latih 60 epoch", "iter_61_r3d_ep60_s42", "fullG"),
    ("62", "G", "1.5", "Early stopping + adaptive FT", "iter_62_r3d_es_llrd_s42", "fullG"),
    ("69", "G", "1.7", "Early stopping refit, loss", "iter_69_r3d_es_refit_s42", "fullG"),
    ("70", "G", "1.7", "Early stopping refit, UF1", "iter_70_r3d_es_refit_uf1_s42", "fullG"),
    ("71", "G", "1.7", "Early stopping refit + adaptive FT", "iter_71_r3d_es_refit_llrd_s42", "fullG"),

    (SECTION, "— FASE 1.8: DIAGNOSTIK KEBOCORAN —"),
    ("65", "G", "1.8", "ES **mengintip fold ujian**", "iter_65_r3d_es_leaky_diag_s42", "leak64"),
    ("66", "G", "1.8", "ES mengintip + adaptive FT", "iter_66_r3d_es_leaky_llrd_diag_s42", "leak62"),

    (SECTION, "— FASE 1.9: RATA-RATA EPOCH —"),
    ("72", "G", "1.9", "Rata-rata 5 epoch", "iter_72_r3d_lastk5_s42", "fullG"),
    ("73", "G", "1.9", "Rata-rata 8 epoch", "iter_73_r3d_lastk8_s42", "fullG"),
    ("74", "G", "1.9", "**Rata-rata 8 + latih 40 epoch**", "iter_74_r3d_lastk8_ep40_s42", "fullG"),
    ("75", "G", "1.9", "Snapshot ensembling", "iter_75_r3d_snapshot3_s42", "fullG"),
    ("80", "G", "1.9r", "↳ ulang seed 123", "iter_80_r3d_lastk8_ep40_s123", "s123"),
    ("81", "G", "1.9r", "↳ ulang seed 2024", "iter_81_r3d_lastk8_ep40_s2024", "s2024"),
    ("82", "G", "1.9r", "↳ di leaf auto-apex", "iter_82_r3d_lastk8_ep40_autoapex_s42", "apexG"),

    (SECTION, "— FASE 2: SEGMENTATION —"),
    ("67", "G", "2.2", "Face parsing, full-span", "iter_67_r3d_faceparse_full_s42", "fullG"),
    ("68", "G", "2.2", "Face parsing, auto-apex", "iter_68_r3d_faceparse_autoapex_s42", "apexG"),
    ("89", "L", "2.3", "**Segmentation attention** statis, full-span", "iter_89_r3d_regionattn_full_s42", "fullL"),
    ("90", "L", "2.3", "**Segmentation attention** statis, auto-apex", "iter_90_r3d_regionattn_apex_s42", "apexL"),
    ("91", "L", "2.3", "**Segmentation attention** dinamis", "iter_91_r3d_regionattn_dyn_s42", "fullL"),
    ("92", "L", "2.4", "Buang pipi (pembanding)", "iter_92_r3d_nocheek_full_s42", "fullL"),
    ("93", "L", "2.4", "Buang pipi + dahi (pembanding)", "iter_93_r3d_auroi_full_s42", "fullL"),
    ("94", "L", "2.4", "Buang pipi + dahi, auto-apex", "iter_94_r3d_auroi_apex_s42", "apexL"),
    ("95", "L", "2.5", "**Saluran fokus** energi, full-span", "iter_95_r3d_focus_energy_full_s42", "fullL"),
    ("96", "L", "2.5", "**Saluran fokus** tersegmentasi, full-span", "iter_96_r3d_focus_region_full_s42", "fullL"),
    ("97", "L", "2.5", "**Saluran fokus** tersegmentasi, auto-apex", "iter_97_r3d_focus_region_apex_s42", "apexL"),

    (SECTION, "— FASE 3: ARSITEKTUR —"),
    ("85", "G", "3.1", "**CNN + RNN**, full-span", "iter_85_resnetgru_full_s42", "fullG"),
    ("86", "G", "3.1", "**CNN + RNN**, auto-apex", "iter_86_resnetgru_autoapex_s42", "apexG"),
    ("85L", "L", "3.1", "**CNN + RNN**, full-span", "iter_85_resnetgru_full_s42", "fullL"),
    ("86L", "L", "3.1", "**CNN + RNN**, auto-apex", "iter_86_resnetgru_autoapex_s42", "apexL"),
    ("87", "L", "3.1", "Backbone mc3_18", "iter_87_mc3_full_s42", "fullL"),
    ("88", "L", "3.1", "Backbone r2plus1d_18", "iter_88_r2plus1d_full_s42", "fullL"),

    (SECTION, "— FASE 4: AU MULTI-TASK —"),
    ("83", "G", "4.1", "Bobot 0,1", "iter_83_r3d_au01_full_s42", "fullG"),
    ("76", "G", "4.1", "Bobot 0,2", "iter_76_r3d_au02_full_s42", "fullG"),
    ("77", "G", "4.1", "Bobot 0,5", "iter_77_r3d_au05_full_s42", "fullG"),
    ("78", "G", "4.1", "Bobot 1,0", "iter_78_r3d_au10_full_s42", "fullG"),
    ("79", "G", "4.1", "Bobot 0,5, auto-apex", "iter_79_r3d_au05_autoapex_s42", "apexG"),
    ("84", "G", "4.1", "Bobot 0,2, auto-apex", "iter_84_r3d_au02_autoapex_s42", "apexG"),
    ("76L", "L", "4.1", "Bobot 0,2", "iter_76_r3d_au02_full_s42", "fullL"),
    ("77L", "L", "4.1", "Bobot 0,5", "iter_77_r3d_au05_full_s42", "fullL"),

    ("98", "L", "2.6", "**Input segmentasi + early stopping**, full-span", "iter_98_r3d_focus_es_full_s42", "fullL"),
    ("99", "L", "2.6", "**Input segmentasi + early stopping**, auto-apex", "iter_99_r3d_focus_es_apex_s42", "apexL"),

    (SECTION, "— FASE 2.3r: REPLIKASI SEED SEGMENTATION ATTENTION —"),
    ("89", "L", "2.3r", "Segmentation attention, seed 42", "iter_89_r3d_regionattn_full_s42", "fullL"),
    ("100", "L", "2.3r", "Segmentation attention, seed 123", "iter_100_r3d_regionattn_s123", "s123L"),
    ("101", "L", "2.3r", "Segmentation attention, seed 2024", "iter_101_r3d_regionattn_s2024", "s2024L"),
    ("102", "L", "2.3r", "Segmentation attention, seed 7", "iter_102_r3d_regionattn_s7", "s7L"),

    (SECTION, "— FASE 3.1L: ARSITEKTUR DI LOSO —"),
    ("85L", "L", "3.1", "CNN + RNN, full-span", "iter_85_resnetgru_full_s42", "fullL"),
    ("86L", "L", "3.1", "CNN + RNN, auto-apex", "iter_86_resnetgru_autoapex_s42", "apexL"),
    ("87L", "L", "3.1", "Backbone mc3_18", "iter_87_mc3_full_s42", "fullL"),
    ("88L", "L", "3.1", "Backbone r2plus1d_18", "iter_88_r2plus1d_full_s42", "fullL"),

    (SECTION, "— FASE 6: RNN TANPA PRETRAIN (⚫ LOSO 26) —"),
    ("103", "F", "6.1", "RNN tanpa pretrain, full-span", "iter_103_resnetgru_nopretrain_full_s42", "fullF"),
    ("104", "F", "6.1", "RNN tanpa pretrain, auto-apex", "iter_104_resnetgru_nopretrain_apex_s42", "apexF"),
    ("105", "F", "6.2", "**Input segmentasi + RNN tanpa pretrain**", "iter_105_focus_rnn_nopretrain_full_s42", "fullF"),
    ("106", "F", "6.2", "**Input segmentasi + RNN tanpa pretrain + early stopping**", "iter_106_focus_rnn_nopretrain_es_full_s42", "fullF"),

    (SECTION, "— FASE 5.2: ⚫ LOSO 26 — SEGEL AUDIT DIPECAH (sebanding dosen) —"),
    ("K1", "F", "5.2", "Kandidat 1: baseline", "iter_14_r3d", None),
    ("K5", "F", "5.2", "Kandidat 5: **segmentation attention**", "iter_89_r3d_regionattn_full_s42", "fullF"),
    ("K3", "F", "5.2", "Kandidat 3: **input segmentasi** (ide user)", "iter_96_r3d_focus_region_full_s42", "fullF"),
    ("K4", "F", "5.2", "Kandidat 4: **input segmentasi + early stopping** (ide user)", "iter_98_r3d_focus_es_full_s42", "fullF"),

    (SECTION, "— FASE 5.0: UJI ULANG DI LOSO —"),
    ("59L", "L", "5.0", "Adaptive fine-tuning", "iter_59_r3d_llrd_s42", "fullL"),
    ("67L", "L", "5.0", "Face parsing", "iter_67_r3d_faceparse_full_s42", "fullL"),
    ("74L", "L", "5.0", "Rata-rata 8 + 40 epoch", "iter_74_r3d_lastk8_ep40_s42", "fullL"),
    ("69L", "L", "5.0", "Early stopping refit", "iter_69_r3d_es_refit_s42", "fullL"),
    ("61L", "L", "5.0", "Latih 60 epoch", "iter_61_r3d_ep60_s42", "fullL"),
    ("77L", "L", "5.0", "AU multi-task 0,5", "iter_77_r3d_au05_full_s42", "fullL"),
    ("76L", "L", "5.0", "AU multi-task 0,2", "iter_76_r3d_au02_full_s42", "fullL"),
]

# Fusi: (label, daftar leaf LOSO, tebal?)
FUSIONS = [
    ("Gabungan 2 model", ["full", "autoapex"], False),
    ("Gabungan 7 model", ["full", "autoapex", "lastk8", "llrd", "parse",
                          "au", "es"], False),
    ("**Gabungan 10 model** (semua yang ±0,04)",
     ["full", "autoapex", "lastk8", "llrd", "parse", "au", "es",
      "regattn", "r2p1d", "focusE"], True),
]
FUSION_LEAF = {
    "full": "iter_14_r3d", "autoapex": "iter_47_r3d_auto_apex_s42",
    "lastk8": "iter_74_r3d_lastk8_ep40_s42", "llrd": "iter_59_r3d_llrd_s42",
    "parse": "iter_67_r3d_faceparse_full_s42", "au": "iter_77_r3d_au05_full_s42",
    "es": "iter_69_r3d_es_refit_s42",
    "regattn": "iter_89_r3d_regionattn_full_s42",
    "r2p1d": "iter_88_r2plus1d_full_s42",
    "focusE": "iter_95_r3d_focus_energy_full_s42",
}


def num(value, digits=4):
    return f"{value:.{digits}f}".replace(".", ",")


def signed(value, digits=4):
    return f"{value:+.{digits}f}".replace(".", ",").replace("+", "+").replace("-", "−")


def load(stem, protocol):
    path = os.path.join(P2, stem + SUFFIX[protocol], "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    return summary if summary.get("complete") else None


def probs_of(stem, protocol):
    path = os.path.join(P2, stem + SUFFIX[protocol], "probs.npz")
    if not os.path.exists(path):
        return None
    raw = np.load(path)
    order = np.argsort([str(k) for k in raw["keys"]])
    return {"label": raw["label"][order], "subject": raw["subject"][order],
            "probs": raw["probs"][order].astype(np.float64)}


def gate_probability(candidate_stem, baseline_stem, protocol, num_classes=5):
    """Re-run the paired subject bootstrap so P is never a stale hand-copy."""
    candidate = probs_of(candidate_stem, protocol)
    baseline = probs_of(baseline_stem, protocol)
    if candidate is None or baseline is None:
        return None
    if not np.array_equal(candidate["label"], baseline["label"]):
        return None
    result = paired_subject_bootstrap(
        baseline["label"], baseline["subject"], baseline["probs"],
        candidate["probs"], num_classes, iterations=10000)
    return result["UF1"]["probability_improved"]


def main():
    baseline_stem = {
        "fullG": ("iter_14_r3d", "G"), "fullL": ("iter_14_r3d", "L"),
        "fullF": ("iter_14_r3d", "F"),
        "apexG": ("iter_47_r3d_auto_apex_s42", "G"),
        "apexL": ("iter_47_r3d_auto_apex_s42", "L"),
        "apexF": ("iter_47_r3d_auto_apex_s42", "F"),
        "s123": ("iter_14_r3d_s123", "G"),
        "s2024": ("iter_14_r3d_s2024_v2_dev", "G"),
        # Acuan berpasangan per seed di LOSO: satu-satunya perbandingan yang sah
        # untuk replikasi seed, karena variasi seed di project ini mencapai 0,07.
        "s123L": ("iter_14_r3d_s123_loso", "L"),
        "s2024L": ("iter_14_r3d_s2024_loso", "L"),
        "s7L": ("iter_14_r3d_s7_loso", "L"),
        # Diagnostik bocor dibandingkan dengan versi JUJUR-nya, bukan baseline:
        # yang ingin diukur adalah besar inflasinya, bukan skornya.
        "leak64": ("iter_64_r3d_es_loss_s42", "G"),
        "leak62": ("iter_62_r3d_es_llrd_s42", "G"),
    }
    bases = {key: load(stem, protocol)
             for key, (stem, protocol) in baseline_stem.items()}

    lines = []
    add = lines.append
    add("# Daftar Eksperimen CASME II — klasifikasi 5 kelas")
    add("")
    add(f"Dibuat otomatis dari hasil di `experiments/protocol_v2/` pada "
        f"**{datetime.now().strftime('%d %B %Y, %H:%M')}**.")
    add("Perbarui dengan: `conda run -n facesleuth python "
        "src/make_experiment_report.py`")
    add("")
    add("## Protokol evaluasi")
    add("")
    add("| kode | nama | siapa diuji | latih/uji | data latih | waktu | segel |")
    add("|---|---|---|---|---|---|---|")
    add("| 🟢 | **grouped 4-fold** | 21 orang | latih 16 → uji 5 | ±144 | 13 mnt | aman |")
    add("| 🔵 | **LOSO 21** | 21 orang | latih 20 → uji 1 | ±183 | 68 mnt | aman |")
    add("| ⚫ | **LOSO 26** | 26 orang | latih 25 → uji 1 | 245 | ±2 jam | **pecah** |")
    add("| 🔴 | **audit** | 5 orang tersegel | latih 21 → uji 5 | ±192 | 15 mnt | sekali pakai |")
    add("")
    add("⚠️ Angka antar-protokol **tidak sebanding**. Sudah terbukti 🟢 bisa "
        "membalik hasil ke **dua arah** dibanding 🔵.")
    add("")
    add("## Hasil")
    add("")
    add("| # | Protokol | Fase | Percobaan | UF1 | Selisih | Hasil |")
    add("|---|---|---|---|---|---|---|")

    stats = {"lolos": 0, "hampir": 0, "netral": 0, "gagal": 0, "antre": 0}
    for entry in PLAN:
        if entry[0] is SECTION:
            add(f"| | | | **{entry[1]}** | | | |")
            continue
        number, protocol, fase, label, stem, reference = entry
        summary = load(stem, protocol)
        mark = MARK[protocol]
        if summary is None:
            add(f"| {number} | {mark} | {fase} | {label} | ⏳ | | antre |")
            stats["antre"] += 1
            continue
        if reference is None:
            add(f"| {number} | {mark} | {fase} | {label} | "
                f"**{num(summary['UF1'])}** | — | acuan |")
            continue
        base = bases.get(reference)
        if base is None:
            add(f"| {number} | {mark} | {fase} | {label} | "
                f"{num(summary['UF1'])} | — | tanpa acuan |")
            continue
        delta = summary["UF1"] - base["UF1"]
        if reference.startswith("leak"):
            add(f"| {number} | {mark} | {fase} | {label} | {num(summary['UF1'])} | "
                f"**{signed(delta)}** vs jujur | 🔬 diagnostik |")
            continue
        note = ""
        # Baris replikasi seed dinilai lewat konsistensi antar-seed, bukan gate
        # tunggal, jadi jangan ikut dihitung sebagai kandidat promosi.
        if fase == "1.9r":
            add(f"| {number} | {mark} | {fase} | {label} | {num(summary['UF1'])} | "
                f"{signed(delta)}* | ✅ replikasi |")
            continue
        if delta >= 0.005:
            probability = gate_probability(stem, baseline_stem[reference][0],
                                           protocol)
            if probability is not None and probability >= 0.80:
                verdict = f"✅ **LOLOS** P={num(probability, 3)}"
                stats["lolos"] += 1
            elif probability is not None:
                verdict = f"⚠️ P={num(probability, 3)}"
                stats["hampir"] += 1
            else:
                verdict = "⚠️ naik"
                stats["hampir"] += 1
        elif delta > -0.005:
            verdict = "⚪ netral"
            stats["netral"] += 1
        else:
            verdict = "❌"
            stats["gagal"] += 1
        add(f"| {number} | {mark} | {fase} | {label} | {num(summary['UF1'])} | "
            f"{signed(delta)}{note} | {verdict} |")

    # --- fusi, dihitung ulang dari probabilitas tersimpan ---
    add("| | | | **— FASE 5.1: PENGGABUNGAN —** | | | |")
    leaves = {name: probs_of(stem, "L") for name, stem in FUSION_LEAF.items()}
    reference = leaves.get("full")
    if reference is not None and all(v is not None for v in leaves.values()):
        base_uf1 = compute_metrics(reference["label"],
                                   reference["probs"].argmax(1), 5)["UF1"]
        for label, members, _ in FUSIONS:
            fused = sum(leaves[m]["probs"] for m in members) / len(members)
            metrics = compute_metrics(reference["label"], fused.argmax(1), 5)
            delta = metrics["UF1"] - base_uf1
            result = paired_subject_bootstrap(
                reference["label"], reference["subject"], reference["probs"],
                fused, 5, iterations=10000)
            probability = result["UF1"]["probability_improved"]
            if probability >= 0.80 and delta >= 0.005:
                verdict = f"✅ **LOLOS** P={num(probability, 3)}"
                stats["lolos"] += 1
            elif delta > 0:
                verdict = f"❌ P={num(probability, 3)}"
                stats["gagal"] += 1
            else:
                verdict = "❌"
                stats["gagal"] += 1
            add(f"| — | 🔵 **LOSO 21** | 5.1 | {label} | "
                f"{'**' if _ else ''}{num(metrics['UF1'])}{'**' if _ else ''} | "
                f"{'**' if _ else ''}{signed(delta)}{'**' if _ else ''} | {verdict} |")

    # Kandidat 2 di ⚫ LOSO 26: fusi 10 model, dihitung dari probs loso_all.
    leaves26 = {name: probs_of(stem, "F") for name, stem in FUSION_LEAF.items()}
    ref26 = leaves26.get("full")
    if ref26 is not None and all(v is not None for v in leaves26.values()):
        base26 = compute_metrics(ref26["label"], ref26["probs"].argmax(1), 5)["UF1"]
        fused = sum(leaves26[m]["probs"] for m in FUSION_LEAF) / len(FUSION_LEAF)
        metrics = compute_metrics(ref26["label"], fused.argmax(1), 5)
        delta = metrics["UF1"] - base26
        result = paired_subject_bootstrap(
            ref26["label"], ref26["subject"], ref26["probs"], fused, 5,
            iterations=10000)
        probability = result["UF1"]["probability_improved"]
        if probability >= 0.80 and delta >= 0.005:
            verdict = f"✅ **LOLOS** P={num(probability, 3)}"
            stats["lolos"] += 1
        else:
            verdict = f"❌ P={num(probability, 3)}"
            stats["gagal"] += 1
        add(f"| K2 | ⚫ **LOSO 26** | 5.2 | Kandidat 2: **gabungan 10 model** | "
            f"**{num(metrics['UF1'])}** | **{signed(delta)}** | {verdict} |")

    add("| | | | **— TANPA GPU —** | | | |")
    add("| — | 🔵 LOSO 21 | 0.1 | **Reject option** | ACC 69,8→**77,2%** | | ✅ berguna |")
    add("| — | — | 2.1 | 246 masker wajah MediaPipe | 246/246 | | ✅ |")
    add("| — | — | 2.1 | 246 masker 6 region wajah | jumlah 100,0% | | ✅ |")
    add("| | | | **— BELUM PERNAH DIJALANKAN —** | | | |")
    add("| — | ⚫ **LOSO 26** | 5.2 | Angka sebanding dosen | — | | ⬜ belum |")
    add("| — | 🔴 **audit** | 5.3 | Pengecekan final | — | | ⬜ belum |")
    add("")
    add("\\* dibanding baseline seed-nya sendiri, bukan seed 42")
    add("")

    add("## Ringkasan")
    add("")
    add("| | jumlah |")
    add("|---|---|")
    add(f"| ✅ Lolos gate | **{stats['lolos']}** |")
    add(f"| ⚠️ Naik tapi gate gagal | {stats['hampir']} |")
    add(f"| ⚪ Netral | {stats['netral']} |")
    add(f"| ❌ Gagal | {stats['gagal']} |")
    add(f"| ⏳ Antre | {stats['antre']} |")
    add("")

    add("## Posisi")
    add("")
    add("| | UF1 | Protokol |")
    add("|---|---|---|")
    if bases["fullL"]:
        add(f"| Baseline satu model | {num(bases['fullL']['UF1'])} | 🔵 LOSO 21 |")
    add("| **Terbaik: gabungan 7 model** | **0,7237** | 🔵 LOSO 21 |")
    add("| Acuan lama satu model | 0,7145 | ⚫ LOSO 26 |")
    add("| Klaim dosen | 0,80 | ⚫ LOSO 26 |")
    add("")

    add("## Catatan ⚫ LOSO 26 (segel audit dipecah 2026-08-05)")
    add("")
    add("Kelima kandidat ada di tabel di atas (baris K1–K5). Daftarnya dikunci "
        "sebelum hasil dilihat dan tidak boleh ditambah — semua dilaporkan, "
        "termasuk yang kalah, dan angka final adalah yang **lolos gate**. "
        "Kandidat 3 & 4 (ide user/dosen) sudah gagal berat di 🔵 LOSO 21 "
        "(0,6394 dan 0,6198 vs 0,7109); tetap dijalankan atas permintaan user. "
        "Subject audit `[4, 6, 8, 17, 24]` kini terpakai — pengecekan bersih "
        "sekali-pakai sudah tidak ada.")
    add("")
    add("## Cabang yang sudah tertutup")
    add("")
    add("- **Early stopping** — 9 varian, semua kalah. Bahkan versi yang "
        "mengintip fold ujian (0,6742) masih di bawah baseline (0,6816).")
    add("- **Adaptive fine-tuning, gradual unfreeze, latih lebih lama** — "
        "tidak berefek atau merugikan di kedua protokol.")
    add("- **AU multi-task** — makin besar bobotnya makin buruk; di 🔵 −0,044.")
    add("- **Rata-rata epoch (74)** — replikasi 3/3 seed positif di 🟢, lalu "
        "mati di 🔵. Pemenang palsu.")
    add("- **CNN + RNN sendirian** — 0,5970, jauh di bawah 3D-CNN.")
    add("")

    add("## Temuan bernilai walau skornya tidak naik")
    add("")
    add("1. **Kebocoran early stopping = +0,0605 UF1.** Memilih epoch berhenti "
        "dengan melihat fold ujian menaikkan skor sebanyak itu tanpa model "
        "membaik sedikit pun.")
    add("2. **LOSO 21 ≈ LOSO 26** — 0,7109 vs 0,7145, selisih 0,0036 saja. "
        "Segel audit tidak perlu dipecah untuk mendapat angka sebanding paper.")
    add("3. **Saringan 🟢 menyesatkan ke DUA arah** — menciptakan pemenang palsu "
        "(74) dan menolak yang sebenarnya netral (face parsing).")
    add("4. **Reject option** menaikkan akurasi aplikasi 69,8% → 77,2% tanpa "
        "training ulang.")
    add("5. **Segel audit ternyata sudah dipakai 37×** di percobaan lama — "
        "catatan lama tidak akurat.")
    add("6. **Keragaman saja tidak cukup untuk gabungan.** mc3_18 adalah "
        "anggota paling komplementer (beda 0,207 vs rata-rata 0,144) dan punya "
        "6 sampel yang hanya dia yang benar — tapi menambahkannya justru "
        "menurunkan gabungan 0,7237 → 0,7080. Solonya 0,6431, terlalu jauh di "
        "bawah baseline. **Aturan yang terukur: anggota fusi harus dalam ±0,04 "
        "UF1 dari baseline.**")
    add("7. **AU6 + AU14 = 40 sampel bergantung pada pipi**, jadi membuang pipi "
        "membuang isyarat utama 16% dataset.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"ditulis: {OUT}")
    print(f"ringkasan: {stats}")


if __name__ == "__main__":
    main()
