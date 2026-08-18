"""
Eksplorasi lanjutan CASME II — lima pertanyaan yang muncul dari tahap 1 & 2.

  1. Apakah `disgust` dan `others` benar-benar terbedakan? (keduanya didominasi AU4)
  2. Berapa akurasi yang bisa didapat TANPA melihat gerakan sama sekali? (uji jalan pintas)
  3. Apakah peta gerak per kelas itu ciri kelas, atau ciri wajah beberapa orang saja?
  4. Apakah "offset" berarti wajah sudah kembali netral? (kontrol pinggir vs tengah)
  5. Apakah BENTUK kurva waktu membawa informasi, terlepas dari besarnya gerakan?

Sama seperti tahap 2: gambar didekode jadi angka, tidak pernah ditampilkan atau disimpan.

Jalankan:
    conda run -n facesleuth python eda/explore_deep.py
    conda run -n facesleuth python eda/explore_deep.py --replot   (pakai cache)
"""

from __future__ import annotations

import os
import re
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATASET = Path(r"D:\AI-Projects\casmeII-facesleuth-r\dataset")
CROPPED = DATASET / "Cropped"
OUT = Path(__file__).resolve().parent / "out"
META = OUT / "metadata.csv"
CACHE = OUT / "deep_cache.npz"

FIVE = ["happiness", "disgust", "repression", "surprise", "others"]
GRID_N, BLOCK_N, CURVE_N = 32, 8, 40

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDC, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE, RED = SERIES[0], "#e34948"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec",
                 "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_br", ["#0d366b", "#256abf", "#9ec5f4", "#f0efec",
               "#f0a3a2", "#d03b3b", "#7d1f1f"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 130,
})


def style_axes(ax, ygrid=True, xgrid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1)
    if ygrid:
        ax.grid(True, axis="y", color=GRIDC, linewidth=1)
    if xgrid:
        ax.grid(True, axis="x", color=GRIDC, linewidth=1)
    ax.set_axisbelow(True)


def title(ax, main, sub=None, pad=10):
    if not sub:
        ax.set_title(main, loc="left", pad=pad, fontsize=13,
                     fontweight="600", color=INK)
        return
    lines = sub.count("\n") + 1
    ax.set_title(main, loc="left", pad=pad + 14 * lines, fontsize=13,
                 fontweight="600", color=INK)
    ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5),
                textcoords="offset points", fontsize=9.5, color=INK2,
                va="bottom", ha="left")


def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"  tersimpan: {p.name}")
    return p


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRIDC)


FACTS: dict[str, str] = {}

# --------------------------------------------------------------------------
# Pemindaian
# --------------------------------------------------------------------------
FRAME_RE = re.compile(r"(\d+)")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Bagian tengah = wilayah mata/hidung/mulut. Pinggir = cincin luar (rambut,
# latar, tepi potongan) — dipakai sebagai KONTROL: kalau pinggir ikut bergerak,
# yang terukur adalah geser kepala / perubahan cahaya, bukan ekspresi.
CENTER = (slice(8, 24), slice(8, 24))
BORDER_MASK = np.ones((GRID_N, GRID_N), bool)
BORDER_MASK[4:GRID_N - 4, 4:GRID_N - 4] = False


def frame_paths(folder: Path):
    items = []
    for e in os.scandir(folder):
        if not e.is_file():
            continue
        stem, ext = os.path.splitext(e.name)
        if ext.lower() in IMG_EXT:
            m = FRAME_RE.findall(stem)
            if m:
                items.append((int(m[-1]), e.path))
    items.sort()
    return items


def load_gray(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im.draft("L", (GRID_N * 2, GRID_N * 2))
        return np.asarray(im.convert("L").resize((GRID_N, GRID_N), Image.BILINEAR),
                          dtype=np.float32)


def blockify(mat, n=BLOCK_N):
    k = mat.shape[0] // n
    return mat[:n * k, :n * k].reshape(n, k, n, k).mean(axis=(1, 3))


def resample(t_norm, y):
    return np.interp(np.linspace(0, 1, CURVE_N), t_norm, y)


def deep_scan(meta: pd.DataFrame):
    rows, C = [], {k: [] for k in ("full", "center", "border", "nolight")}
    bmaps, block_curves = [], []
    t0 = time.time()
    for i, r in enumerate(meta.itertuples(), start=1):
        folder = CROPPED / r.subject / r.clip
        items = frame_paths(folder)
        if len(items) < 3:
            continue
        nums = np.array([n for n, _ in items])
        stack = np.stack([load_gray(p) for _, p in items])
        t_norm = (nums - nums[0]) / max(nums[-1] - nums[0], 1)

        # versi "cahaya diratakan": tiap frame dikurangi kecerahan rata-ratanya
        flat = stack - stack.mean(axis=(1, 2), keepdims=True)

        d_full = np.abs(stack - stack[0]).mean(axis=(1, 2))
        d_cen = np.abs(stack[:, CENTER[0], CENTER[1]]
                       - stack[0, CENTER[0], CENTER[1]]).mean(axis=(1, 2))
        d_bor = np.abs((stack - stack[0])[:, BORDER_MASK]).mean(axis=1)
        d_nol = np.abs(flat - flat[0]).mean(axis=(1, 2))
        d_cen_nol = np.abs(flat[:, CENTER[0], CENTER[1]]
                           - flat[0, CENTER[0], CENTER[1]]).mean(axis=(1, 2))

        for k, v in zip(("full", "center", "border", "nolight"),
                        (d_full, d_cen, d_bor, d_nol)):
            C[k].append(resample(t_norm, v))

        # kurva per blok: gerakan di tiap kotak 8x8 sepanjang waktu.
        # Ini yang membedakan "model urutan" dari "satu frame puncak".
        db = np.abs(stack - stack[0])                       # (T,32,32)
        db = db.reshape(len(db), BLOCK_N, GRID_N // BLOCK_N,
                        BLOCK_N, GRID_N // BLOCK_N).mean(axis=(2, 4))   # (T,8,8)
        bc = np.stack([resample(t_norm, db[:, a, b])
                       for a in range(BLOCK_N) for b in range(BLOCK_N)], axis=1)
        block_curves.append(bc)                             # (CURVE_N, 64)

        def first_cross(y, frac):
            thr = y.max() * frac
            idx = np.argmax(y >= thr)
            return float(t_norm[idx])

        apex_idx = int(np.argmin(np.abs(nums - r.apex))) if not np.isnan(r.apex) else None
        j = apex_idx if apex_idx is not None else int(np.argmax(d_full))
        bm = np.abs(stack[j] - stack[0])
        bm = blockify(bm)
        bmaps.append(bm / max(bm.max(), 1e-6))

        rows.append(dict(
            subject=r.subject, clip=r.clip, emotion=r.emotion, aus=str(r.aus),
            n_frame=len(items), lebar=0, tinggi=0,
            bright_mean=float(stack.mean()), bright_std=float(stack.mean(axis=(1, 2)).std()),
            contrast_mean=float(stack.std(axis=(1, 2)).mean()),
            energi_max=float(d_full.max()), energi_mean=float(d_full.mean()),
            gerak_sesaat=float(np.abs(np.diff(stack, axis=0)).mean()),
            border_max=float(d_bor.max()), center_max=float(d_cen.max()),
            end_over_peak=float(d_full[-1] / max(d_full.max(), 1e-6)),
            e_argmax_raw=float(t_norm[int(np.argmax(d_full))]),
            e_argmax_nolight=float(t_norm[int(np.argmax(d_nol))]),
            e_argmax_center=float(t_norm[int(np.argmax(d_cen))]),
            e_argmax_center_nolight=float(t_norm[int(np.argmax(d_cen_nol))]),
            e_cross80=first_cross(d_full, 0.80),
            e_cross90=first_cross(d_full, 0.90),
            e_cross95=first_cross(d_full, 0.95),
        ))
        if i % 40 == 0 or i == len(meta):
            print(f"  {i}/{len(meta)} klip  ({time.time() - t0:.0f}s)")

    px = pd.DataFrame(rows)
    curves = {k: np.array(v) for k, v in C.items()}
    return px, curves, np.stack(bmaps), np.stack(block_curves)


# --------------------------------------------------------------------------
# Alat bantu evaluasi (selalu subject-independent)
# --------------------------------------------------------------------------
def loso_eval(X, y, groups, model=None):
    """Latih-uji Leave-One-Subject-Out, lalu gabungkan semua prediksi baru diukur."""
    model = model or make_pipeline(StandardScaler(),
                                   LogisticRegression(max_iter=2000, C=1.0))
    pred = np.empty(len(y), dtype=object)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            pred[te] = pd.Series(y[tr]).mode()[0]
            continue
        m = model.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")


def majority_baseline(y):
    vc = pd.Series(y).value_counts()
    return vc.iloc[0] / len(y), f1_score(y, [vc.index[0]] * len(y), average="macro")


# --------------------------------------------------------------------------
# 1. disgust vs others
# --------------------------------------------------------------------------
def fig_disgust_vs_others(px, bmaps, curves):
    au = px["aus"].str.upper().str.replace("[RL]", "", regex=True).str.strip()
    core = au == "4"                       # klip yang HANYA AU4
    sub = px[core & px["emotion"].isin(["disgust", "others"])]
    idx = sub.index.values
    n_d = int((sub["emotion"] == "disgust").sum())
    n_o = int((sub["emotion"] == "others").sum())

    md = bmaps[sub.index[sub["emotion"] == "disgust"]].mean(axis=0)
    mo = bmaps[sub.index[sub["emotion"] == "others"]].mean(axis=0)
    diff = md - mo
    lim = float(np.abs(diff).max())

    # bisakah dibedakan? peta gerak 8x8 -> pengklasifikasi, uji per-subjek
    X = bmaps[idx].reshape(len(idx), -1)
    y = sub["emotion"].values
    g = sub["subject"].values
    acc, uf1 = loso_eval(X, y, g)
    b_acc, b_uf1 = majority_baseline(y)

    # pembanding: seluruh disgust vs seluruh others (bukan hanya AU4-saja)
    allm = px["emotion"].isin(["disgust", "others"])
    Xa = bmaps[px.index[allm]].reshape(int(allm.sum()), -1)
    ya = px.loc[allm, "emotion"].values
    ga = px.loc[allm, "subject"].values
    acc_a, uf1_a = loso_eval(Xa, ya, ga)
    b_acc_a, b_uf1_a = majority_baseline(ya)

    fig = plt.figure(figsize=(12.6, 4.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.9], wspace=0.28)
    for k, (M, lab) in enumerate([(md, f"disgust (n={n_d})"), (mo, f"others (n={n_o})")]):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(M, cmap=SEQ, vmin=0, vmax=max(md.max(), mo.max()))
        bare(ax)
        ax.set_title(lab, fontsize=11, color=INK, fontweight="600", pad=8)
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(diff, cmap=DIV, vmin=-lim, vmax=lim)
    bare(ax)
    ax.set_title("selisih", fontsize=11, color=INK, fontweight="600", pad=8)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.055, pad=0.06)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED, labelcolor=MUTED, length=0, labelsize=8)
    cb.set_label("← others lebih besar   |   disgust lebih besar →",
                 color=MUTED, fontsize=8)

    ax = fig.add_subplot(gs[0, 3])
    style_axes(ax)
    xs = np.arange(2)
    w = 0.34
    ax.bar(xs - w / 2, [b_acc * 100, b_acc_a * 100], w, color=GRIDC,
           label="tebak kelas terbanyak", zorder=3)
    ax.bar(xs + w / 2, [acc * 100, acc_a * 100], w, color=BLUE,
           label="pakai peta gerak 8×8", zorder=3)
    for xi, (bv, mv, bf, mf) in enumerate([(b_acc, acc, b_uf1, uf1),
                                           (b_acc_a, acc_a, b_uf1_a, uf1_a)]):
        ax.text(xi - w / 2, bv * 100 + 1.5, f"{bv * 100:.0f}%\nUF1 {bf:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.text(xi + w / 2, mv * 100 + 1.5, f"{mv * 100:.0f}%\nUF1 {mf:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"AU4 saja\n(n={n_d + n_o})",
                        f"semua klip\n(n={int(allm.sum())})"], fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_ylabel("akurasi (uji per-subjek)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", labelcolor=INK2, ncol=1)
    title(ax, "Bisa dibedakan?")

    fig.suptitle("Klip yang HANYA memakai AU4: apakah `disgust` beda dari `others`?",
                 x=0.012, ha="left", fontsize=13.5, fontweight="600", color=INK, y=1.09)
    fig.text(0.012, 1.005, f"{n_d + n_o} klip dengan kode otot yang identik, "
                           "tapi diberi dua label berbeda.",
             ha="left", fontsize=9.5, color=INK2)
    FACTS["au4"] = (
        f"{n_d} disgust + {n_o} others memakai kode otot yang persis sama (AU4 saja). "
        f"Peta gerak 8×8 + uji per-subjek: {acc * 100:.1f}% / UF1 {uf1:.3f}, "
        f"sedangkan tebak-terbanyak {b_acc * 100:.1f}% / UF1 {b_uf1:.3f} — "
        f"tidak ada perbaikan. Pada seluruh {int(allm.sum())} klip disgust+others "
        f"pun sama: {acc_a * 100:.1f}% / UF1 {uf1_a:.3f} vs {b_acc_a * 100:.1f}% / "
        f"UF1 {b_uf1_a:.3f}.")
    return save(fig, "17_disgust_vs_others.png")


# --------------------------------------------------------------------------
# 2. uji jalan pintas
# --------------------------------------------------------------------------
SHORTCUT_FEATS = ["bright_mean", "bright_std", "contrast_mean", "n_frame",
                  "energi_max", "energi_mean", "gerak_sesaat"]


def fig_shortcut(px):
    y = px["emotion"].values
    g = px["subject"].values
    b_acc, b_uf1 = majority_baseline(y)

    sets = {
        "tebak kelas terbanyak": None,
        "hanya kecerahan\n& kontras": ["bright_mean", "bright_std", "contrast_mean"],
        "hanya panjang klip": ["n_frame"],
        "hanya besar gerakan": ["energi_max", "energi_mean", "gerak_sesaat"],
        "semua angka sederhana\n(tanpa lihat gerak)": SHORTCUT_FEATS,
    }
    res = []
    for name, feats in sets.items():
        if feats is None:
            res.append((name, b_acc, b_uf1))
            continue
        X = px[feats].values.astype(float)
        rf = make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=-1))
        a, f = loso_eval(X, y, g, rf)
        res.append((name, a, f))

    # apakah identitas orang bisa ditebak dari kecerahan saja?
    Xb = px[["bright_mean", "contrast_mean"]].values
    ys = px["subject"].values
    sid = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(
            Xb, ys if pd.Series(ys).value_counts().min() >= 3 else np.zeros(len(ys))):
        m = make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1)).fit(Xb[tr], ys[tr])
        sid.append(accuracy_score(ys[te], m.predict(Xb[te])))
    sid_acc = float(np.mean(sid))
    chance = 1 / px["subject"].nunique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                   gridspec_kw={"width_ratios": [2.1, 1]})
    style_axes(ax1)
    names = [r[0] for r in res]
    accs = [r[1] * 100 for r in res]
    cols = [GRIDC] + [BLUE] * (len(res) - 1)
    bars = ax1.bar(range(len(res)), accs, color=cols, width=0.58, zorder=3)
    for b, r in zip(bars, res):
        ax1.text(b.get_x() + b.get_width() / 2, r[1] * 100 + 1,
                 f"{r[1] * 100:.1f}%\nUF1 {r[2]:.2f}", ha="center", va="bottom",
                 fontsize=9, color=INK)
    ax1.set_xticks(range(len(res)))
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("akurasi (uji per-subjek)")
    ax1.set_ylim(0, max(accs) * 1.35)
    title(ax1, "Berapa yang bisa didapat tanpa melihat gerakan wajah?",
          "Batang abu-abu = menebak kelas terbanyak. Semua diuji per-subjek.")

    style_axes(ax2)
    bars = ax2.bar(["tebak acak", "dari kecerahan\n& kontras saja"],
                   [chance * 100, sid_acc * 100], color=[GRIDC, RED], width=0.5, zorder=3)
    for b, v in zip(bars, [chance * 100, sid_acc * 100]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center",
                 va="bottom", fontsize=11, color=INK, fontweight="600")
    ax2.set_ylabel("akurasi menebak SIAPA orangnya")
    ax2.set_ylim(0, 100)
    title(ax2, "Seberapa kentara identitas orang?",
          f"{px['subject'].nunique()} orang.")

    best = max(res[1:], key=lambda r: r[1])
    FACTS["shortcut"] = (
        f"Tanpa melihat gerakan wajah sama sekali, akurasi terbaik {best[1] * 100:.1f}% "
        f"(UF1 {best[2]:.3f}) dari '{best[0].replace(chr(10), ' ')}' — dibanding "
        f"tebak-terbanyak {b_acc * 100:.1f}% (UF1 {b_uf1:.3f}). "
        f"Identitas orang bisa ditebak {sid_acc * 100:.0f}% hanya dari kecerahan "
        f"dan kontras (tebak acak {chance * 100:.1f}%).")
    fig.tight_layout()
    return save(fig, "18_uji_jalan_pintas.png")


# --------------------------------------------------------------------------
# 3. peta gerak dengan bobot per-subjek
# --------------------------------------------------------------------------
def fig_blockmaps_fair(px, bmaps):
    fig, axes = plt.subplots(3, 5, figsize=(13.2, 8.4))
    naive_all, fair_all = [], []
    for j, c in enumerate(FIVE):
        m = px["emotion"] == c
        naive = bmaps[px.index[m]].mean(axis=0)
        per_sub = [bmaps[px.index[m & (px["subject"] == s)]].mean(axis=0)
                   for s in px.loc[m, "subject"].unique()]
        fair = np.mean(per_sub, axis=0)
        naive_all.append(naive)
        fair_all.append(fair)

        def norm(M):
            return (M - M.min()) / max(M.max() - M.min(), 1e-9)

        axes[0, j].imshow(norm(naive), cmap=SEQ, vmin=0, vmax=1)
        axes[1, j].imshow(norm(fair), cmap=SEQ, vmin=0, vmax=1)
        d = norm(fair) - norm(naive)
        lim = max(float(np.abs(d).max()), 1e-6)
        im = axes[2, j].imshow(d, cmap=DIV, vmin=-lim, vmax=lim)
        for ax in axes[:, j]:
            bare(ax)
        axes[0, j].set_title(f"{c}\n({len(per_sub)} orang, n={int(m.sum())})",
                             fontsize=10.5, color=INK, fontweight="600", pad=8)
    for row, lab in enumerate(["rata-rata\nper KLIP\n(cara lama)",
                               "rata-rata\nper ORANG\n(adil)",
                               "selisih"]):
        axes[row, 0].text(-0.22, 0.5, lab, transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=9.5, color=INK2)

    corrs = [float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
             for a, b in zip(naive_all, fair_all)]
    # seberapa mirip antar kelas setelah dibuat adil?
    cross = {}
    for a in range(5):
        for b in range(a + 1, 5):
            cross[f"{FIVE[a]}↔{FIVE[b]}"] = float(
                np.corrcoef(fair_all[a].ravel(), fair_all[b].ravel())[0, 1])
    hi = max(cross, key=cross.get)
    lo = min(cross, key=cross.get)
    fig.suptitle("Peta gerak: ciri kelas, atau ciri wajah beberapa orang saja?",
                 x=0.012, ha="left", fontsize=13.5, fontweight="600", color=INK, y=1.015)
    fig.text(0.012, 0.968, "Baris atas menghitung tiap klip sama berat — orang dengan "
                           "banyak klip mendominasi. Baris tengah menyamakan bobot tiap "
                           "orang dulu. Baris bawah = selisihnya.",
             ha="left", fontsize=9.5, color=INK2)
    FACTS["fair"] = (
        f"Kekhawatiran perancu subjek ternyata kecil: menyamakan bobot tiap orang "
        f"hampir tidak mengubah peta gerak (korelasi {min(corrs):.2f}–{max(corrs):.2f} "
        f"dengan versi lama). Antar kelas, peta paling mirip {hi} "
        f"({cross[hi]:+.2f}) dan paling berbeda {lo} ({cross[lo]:+.2f}).")
    fig.tight_layout(rect=[0.03, 0, 1, 0.945])
    return save(fig, "19_peta_gerak_adil.png")


# --------------------------------------------------------------------------
# 4. kontrol offset: tengah vs pinggir
# --------------------------------------------------------------------------
def fig_offset_control(px, curves):
    x = np.linspace(0, 1, CURVE_N)
    cen, bor = curves["center"], curves["border"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    style_axes(ax1)
    for arr, col, lab in [(cen, BLUE, "tengah wajah (mata/hidung/mulut)"),
                          (bor, SERIES[1], "pinggir potongan (rambut/latar)")]:
        med = np.median(arr, axis=0)
        q1, q3 = np.percentile(arr, [25, 75], axis=0)
        ax1.fill_between(x, q1, q3, color=col, alpha=0.15, linewidth=0)
        ax1.plot(x, med, color=col, linewidth=2.4, label=lab)
    ax1.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=INK2)
    ax1.set_xlabel("waktu di dalam klip  (0 = onset, 1 = offset)")
    ax1.set_ylabel("beda piksel terhadap frame awal")
    ax1.set_xlim(0, 1)
    title(ax1, "Kontrol: apakah pinggir ikut bergerak?",
          "Kalau pinggir ikut naik, yang terukur adalah geser kepala / cahaya.")

    style_axes(ax2)
    ratio_c = cen[:, -1] / np.maximum(cen.max(axis=1), 1e-6)
    ratio_b = bor[:, -1] / np.maximum(bor.max(axis=1), 1e-6)
    share = bor.max(axis=1) / np.maximum(cen.max(axis=1), 1e-6)
    ax2.hist(share, bins=np.arange(0, min(share.max(), 3) + 0.1, 0.1),
             color=BLUE, edgecolor=SURFACE, linewidth=1.2)
    ax2.axvline(1.0, color=RED, linewidth=2)
    ax2.text(1.03, ax2.get_ylim()[1] * 0.9, "pinggir = tengah", color=RED,
             fontsize=9.5, fontweight="600")
    ax2.set_xlabel("gerak pinggir ÷ gerak tengah, per klip")
    ax2.set_ylabel("jumlah klip")
    dom = float((share > 1).mean() * 100)
    title(ax2, "Klip mana yang didominasi gerak non-ekspresi?",
          f"{dom:.0f}% klip punya gerak pinggir lebih besar dari gerak tengah.")

    FACTS["offset"] = (
        f"Di frame offset, gerak di tengah wajah masih {np.median(ratio_c) * 100:.0f}% "
        f"dari puncaknya (pinggir {np.median(ratio_b) * 100:.0f}%). "
        f"Gerak pinggir sebesar {np.median(share):.2f}× gerak tengah (median); "
        f"pada {dom:.0f}% klip pinggir bahkan lebih besar.")
    fig.tight_layout()
    return save(fig, "20_kontrol_offset.png")


def fig_apex_estimators(px, meta):
    d = px.merge(meta[["subject", "clip", "apex_relatif"]], on=["subject", "clip"])
    d = d.dropna(subset=["apex_relatif"])
    t = d["apex_relatif"].clip(0, 1)
    cand = {
        "konstanta 0,50": pd.Series(0.5, index=d.index),
        "puncak, gambar penuh": d["e_argmax_raw"],
        "puncak, cahaya diratakan": d["e_argmax_nolight"],
        "puncak, tengah wajah": d["e_argmax_center"],
        "puncak, tengah + cahaya\ndiratakan": d["e_argmax_center_nolight"],
        "pertama capai 80%\ndari puncak": d["e_cross80"],
        "pertama capai 90%\ndari puncak": d["e_cross90"],
    }
    mae = {k: float((t - v).abs().mean()) for k, v in cand.items()}
    order = sorted(mae, key=mae.get)

    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    style_axes(ax)
    vals = [mae[k] for k in order]
    base = mae["konstanta 0,50"]
    cols = [SERIES[2] if v < base else (GRIDC if k == "konstanta 0,50" else RED)
            for k, v in zip(order, vals)]
    bars = ax.bar(range(len(order)), vals, color=cols, width=0.58, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center",
                va="bottom", fontsize=10, color=INK, fontweight="600")
    ax.axhline(base, color=GRIDC, linewidth=1.6, linestyle="--", zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8.5)
    ax.set_ylabel("rata-rata meleset (0 = sempurna)")
    ax.set_ylim(0, max(vals) * 1.2)
    best = order[0]
    title(ax, "Cara mana yang paling baik menebak apex tanpa label manusia?",
          f"Hijau = mengalahkan konstanta 0,50. Terbaik: {best.replace(chr(10), ' ')} "
          f"({mae[best]:.3f}).")
    FACTS["apex"] = (
        f"Penebak apex terbaik: {best.replace(chr(10), ' ')}, meleset rata-rata "
        f"{mae[best]:.3f}; konstanta 0,50 meleset {base:.3f}; "
        f"puncak piksel mentah {mae['puncak, gambar penuh']:.3f}.")
    return save(fig, "21_penebak_apex.png")


# --------------------------------------------------------------------------
# 5. bentuk kurva vs besar gerakan
# --------------------------------------------------------------------------
def fig_shape(px, curves):
    x = np.linspace(0, 1, CURVE_N)
    raw = curves["full"]
    shape = raw / np.maximum(raw.max(axis=1, keepdims=True), 1e-6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    style_axes(ax1)
    for c, col in zip(FIVE, SERIES):
        m = (px["emotion"] == c).values
        med = np.median(shape[m], axis=0)
        ax1.plot(x, med, color=col, linewidth=2.2, label=f"{c} (n={m.sum()})")
    ax1.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK2)
    ax1.set_xlabel("waktu di dalam klip  (0 = onset, 1 = offset)")
    ax1.set_ylabel("gerak, dibagi puncaknya sendiri")
    ax1.set_xlim(0, 1)
    title(ax1, "Bentuk kurva setelah besar gerakan dinormalkan",
          "Kalau semua garis bertumpuk, bentuk waktu tidak membedakan kelas.")

    y = px["emotion"].values
    g = px["subject"].values
    b_acc, b_uf1 = majority_baseline(y)
    a_amp, f_amp = loso_eval(px[["energi_max", "energi_mean"]].values, y, g)
    a_shp, f_shp = loso_eval(shape, y, g)
    a_raw, f_raw = loso_eval(raw, y, g)

    style_axes(ax2)
    labs = ["tebak\nterbanyak", "besar\ngerakan", "BENTUK\nsaja", "kurva\nmentah"]
    accs = [b_acc, a_amp, a_shp, a_raw]
    uf1s = [b_uf1, f_amp, f_shp, f_raw]
    cols = [GRIDC, SERIES[1], BLUE, SERIES[2]]
    bars = ax2.bar(range(4), [a * 100 for a in accs], color=cols, width=0.55, zorder=3)
    for b, a, f in zip(bars, accs, uf1s):
        ax2.text(b.get_x() + b.get_width() / 2, a * 100 + 1,
                 f"{a * 100:.0f}%\nUF1 {f:.2f}", ha="center", va="bottom",
                 fontsize=9, color=INK)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labs, fontsize=9)
    ax2.set_ylabel("akurasi (uji per-subjek)")
    ax2.set_ylim(0, max(accs) * 145)
    title(ax2, "Apa yang benar-benar terpakai?")

    FACTS["shape"] = (
        f"Bentuk kurva saja: akurasi {a_shp * 100:.1f}% / UF1 {f_shp:.3f}. "
        f"Besar gerakan saja: {a_amp * 100:.1f}% / UF1 {f_amp:.3f}. "
        f"Kurva mentah: {a_raw * 100:.1f}% / UF1 {f_raw:.3f}. "
        f"Tebak-terbanyak: {b_acc * 100:.1f}% / UF1 {b_uf1:.3f}.")
    fig.tight_layout()
    return save(fig, "22_bentuk_vs_amplitudo.png")


def fig_spatiotemporal(px, bmaps, curves, block_curves):
    """Apakah WAKTU menambah sesuatu di atas satu frame puncak?

    Pembandingnya sengaja dibuat setara: semua memakai kotak 8x8 yang sama,
    yang berbeda hanya ada-tidaknya sumbu waktu.
    """
    y = px["emotion"].values
    g = px["subject"].values
    b_acc, b_uf1 = majority_baseline(y)

    sel = np.linspace(0, block_curves.shape[1] - 1, 10).astype(int)  # 10 titik waktu
    st10 = block_curves[:, sel, :].reshape(len(px), -1)   # 10 x 64 = 640 fitur
    static = bmaps.reshape(len(px), -1)                # 64 fitur, hanya frame apex
    glob = curves["full"]                              # 40 fitur, tanpa info ruang

    tests = [
        ("tebak\nterbanyak", None, GRIDC),
        ("kurva global\n(waktu, tanpa ruang)", glob, SERIES[1]),
        ("peta apex\n(ruang, tanpa waktu)", static, SERIES[3]),
        ("ruang × waktu\n(10 titik × 64 kotak)", st10, BLUE),
    ]
    res = []
    for name, X, col in tests:
        if X is None:
            res.append((name, b_acc, b_uf1, col))
            continue
        a, f = loso_eval(X, y, g)
        res.append((name, a, f, col))

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    style_axes(ax)
    bars = ax.bar(range(len(res)), [r[1] * 100 for r in res],
                  color=[r[3] for r in res], width=0.55, zorder=3)
    for b, r in zip(bars, res):
        ax.text(b.get_x() + b.get_width() / 2, r[1] * 100 + 1,
                f"{r[1] * 100:.1f}%\nUF1 {r[2]:.3f}", ha="center", va="bottom",
                fontsize=9.5, color=INK)
    ax.set_xticks(range(len(res)))
    ax.set_xticklabels([r[0] for r in res], fontsize=9)
    ax.set_ylabel("akurasi (uji per-subjek)")
    ax.set_ylim(0, max(r[1] for r in res) * 145)
    title(ax, "Apakah sumbu waktu menambah sesuatu di atas satu frame puncak?",
          "Semua memakai kotak 8×8 yang sama. Yang berbeda hanya informasi yang diberikan.")
    FACTS["spatiotemporal"] = (
        f"Ruang saja (peta apex): {res[2][1] * 100:.1f}% / UF1 {res[2][2]:.3f}. "
        f"Waktu saja (kurva global): {res[1][1] * 100:.1f}% / UF1 {res[1][2]:.3f}. "
        f"Ruang × waktu: {res[3][1] * 100:.1f}% / UF1 {res[3][2]:.3f}. "
        f"Tebak-terbanyak: {b_acc * 100:.1f}% / UF1 {b_uf1:.3f}.")
    return save(fig, "23_ruang_vs_waktu.png")


# --------------------------------------------------------------------------
def build_html(figs):
    teks = {
        "17_disgust_vs_others.png": (
            "Ini adalah pemeriksaan paling penting di halaman ini. Diambil hanya klip yang "
            "kode ototnya <b>persis sama</b> (AU4 saja), lalu ditanya: bisakah dibedakan "
            "mana yang dilabeli <code>disgust</code> dan mana <code>others</code>?<br><br>"
            "<b>Kenapa ini penting:</b> kalau jawabannya tidak, maka sebagian dari "
            "'kesulitan' dataset ini <b>ada di labelnya</b>, bukan di modelnya. Tidak ada "
            "arsitektur yang bisa memperbaiki dua nama berbeda untuk gerakan yang sama. "
            "Itu mengubah cara membaca setiap angka akurasi setelah ini."
        ),
        "18_uji_jalan_pintas.png": (
            "Kiri: pengklasifikasi yang <b>tidak melihat gerakan wajah sama sekali</b> — "
            "hanya angka ringkasan seperti kecerahan, kontras, panjang klip. Kanan: "
            "bisakah ditebak <i>siapa orangnya</i> hanya dari kecerahan?<br><br>"
            "<b>Kenapa ini penting:</b> panel kiri memberi <b>garis dasar yang jujur</b>. "
            "Model apa pun yang hasilnya tidak jauh dari batang-batang ini sebenarnya "
            "belum belajar apa-apa. Panel kanan mengukur seberapa kentara identitas "
            "orang — makin tinggi, makin besar godaan model untuk menghafal wajah alih-alih "
            "mengenali emosi."
        ),
        "19_peta_gerak_adil.png": (
            "Baris atas: rata-rata semua klip (orang dengan banyak klip mendominasi). "
            "Baris tengah: tiap orang diberi bobot sama dulu. Baris bawah: selisihnya.<br><br>"
            "<b>Kenapa ini penting:</b> <code>repression</code> 70% klipnya dari 3 orang saja, "
            "dan sub17 sendirian menyumbang sepertiga. Jadi 'pola gerak repression' pada "
            "baris atas bisa jadi sebenarnya 'bentuk wajah sub17'. Baris tengah memperbaiki "
            "itu. Kalau kelima peta pada baris tengah tetap mirip satu sama lain, memotong "
            "wilayah wajah (ROI) tidak akan menolong."
        ),
        "20_kontrol_offset.png": (
            "Wilayah tengah (mata/hidung/mulut) dibandingkan dengan pinggir potongan "
            "(rambut, latar, tepi). Pinggir dipakai sebagai <b>kontrol</b>: di sana tidak "
            "ada otot ekspresi.<br><br>"
            "<b>Kenapa ini penting:</b> di tahap 2 terlihat gerakan tidak pernah kembali "
            "ke nol di frame offset. Ada dua kemungkinan: ekspresinya memang belum selesai, "
            "atau kepala/cahaya yang bergeser. Grafik ini memisahkan keduanya. Kalau "
            "pinggir ikut naik sebesar tengah, sebagian besar yang selama ini diukur "
            "sebagai 'gerak ekspresi' sebenarnya <b>gerak kepala</b>."
        ),
        "21_penebak_apex.png": (
            "Tujuh cara menebak posisi apex tanpa label manusia, diadu dengan anotasi "
            "asli. Batang lebih pendek = lebih baik.<br><br>"
            "<b>Kenapa ini penting:</b> di video upload tidak ada apex dari manusia. "
            "Grafik ini menjawab dengan angka: cara mana yang layak dipakai, dan apakah "
            "usaha mendeteksi apex secara otomatis lebih baik daripada sekadar mengambil "
            "frame tengah. Kalau tidak ada yang mengalahkan konstanta 0,50, pakai saja "
            "konstanta — lebih sederhana dan tidak bisa gagal."
        ),
        "22_bentuk_vs_amplitudo.png": (
            "Tiap kurva dibagi puncaknya sendiri, sehingga yang tersisa hanya "
            "<b>bentuk waktunya</b> — cepat atau lambat naiknya, kapan mendatar. "
            "Kanan: seberapa jauh tiap jenis informasi bisa membawa.<br><br>"
            "<b>Kenapa ini penting:</b> ini keputusan arsitektur. Kalau 'bentuk saja' "
            "hampir tidak lebih baik dari tebak-terbanyak, maka informasi waktu tipis, dan "
            "model urutan yang rumit sulit dibenarkan atas model satu-frame yang sederhana. "
            "Kalau bentuk membawa sinyal nyata, model urutan memang tepat.<br><br>"
            "<b>Hati-hati membacanya:</b> yang diuji di sini adalah <i>satu</i> kurva "
            "rata-rata seluruh wajah. Itu ringkasan yang sangat kasar. Grafik berikutnya "
            "menguji versi yang adil."
        ),
        "23_ruang_vs_waktu.png": (
            "Tiga jenis informasi diadu dengan bahan yang sama persis (kotak 8×8): "
            "<b>hanya waktu</b> (satu kurva, ruangnya dibuang), <b>hanya ruang</b> "
            "(peta di frame puncak, waktunya dibuang), dan <b>ruang × waktu</b> "
            "(10 titik waktu × 64 kotak).<br><br>"
            "<b>Kenapa ini penting:</b> ini pertanyaan arsitektur yang sebenarnya. "
            "Kalau 'ruang × waktu' tidak lebih baik dari 'ruang saja', maka satu frame "
            "puncak sudah memuat hampir semua informasi, dan model urutan yang mahal "
            "sulit dibenarkan. Kalau lebih baik, model urutan memang membawa sesuatu "
            "yang nyata. Catatan jujur: kotak 8×8 itu kasar — hasil di sini adalah "
            "<b>batas bawah</b>, bukan batas atas untuk model sungguhan."
        ),
    }
    cards = "".join(f"""
    <section class="card"><div class="num">{i}</div>
      <img src="{Path(f).name}" alt="">
      <div class="body">{teks.get(Path(f).name, '')}</div></section>"""
                    for i, f in enumerate(figs, start=17))
    facts = "".join(f"<li>{v}</li>" for v in FACTS.values())
    return f"""<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eksplorasi Lanjutan CASME II</title><style>
:root{{--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
 --muted:#898781;--line:#e1e0d9;--blue:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--surface:#1a1a19;--plane:#0d0d0d;
 --ink:#fff;--ink2:#c3c2b7;--line:#2c2c2a;--blue:#3987e5}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--plane);color:var(--ink);
 font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:1000px;margin:0 auto;padding:48px 20px 80px}}
h1{{font-size:30px;line-height:1.2;margin:0 0 8px;letter-spacing:-.02em}}
h2{{font-size:19px;margin:40px 0 12px}}
.lede{{color:var(--ink2);margin:0 0 24px;max-width:70ch}}
.nav{{font-size:14px;color:var(--muted);margin:0 0 4px}}
.nav a{{color:var(--blue);text-decoration:none;font-weight:600}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:14px;
 padding:22px;margin:0 0 22px;position:relative}}
.num{{position:absolute;top:18px;left:-14px;width:28px;height:28px;border-radius:50%;
 background:var(--blue);color:#fff;font-size:13px;font-weight:700;display:grid;place-items:center}}
.card img{{width:100%;height:auto;display:block;border-radius:8px;background:#fcfcfb}}
.body{{color:var(--ink2);font-size:15px;margin-top:16px;padding-top:16px;
 border-top:1px solid var(--line);max-width:78ch}}
b{{color:var(--ink);font-weight:600}}
code{{background:var(--plane);border:1px solid var(--line);padding:1px 5px;
 border-radius:5px;font-size:13.5px}}
.warn{{background:var(--surface);border:1px solid var(--line);border-left:3px solid #eda100;
 border-radius:12px;padding:16px 20px;margin:0 0 32px}}
.warn ul{{margin:8px 0 0;padding-left:20px;color:var(--ink2);font-size:14.5px}}
.warn li{{margin-bottom:8px}}
footer{{color:var(--muted);font-size:13px;margin-top:44px;border-top:1px solid var(--line);padding-top:18px}}
</style></head><body><div class="wrap">
<h1>Eksplorasi Lanjutan CASME II</h1>
<p class="lede">Lima pertanyaan yang muncul setelah membaca ulang grafik tahap 1 dan 2.
Semua pengujian di halaman ini <b>subject-independent</b>: model tidak pernah diuji
pada orang yang pernah dilihatnya saat latihan.</p>
<p class="nav">Bagian 3 dari 3 &nbsp;·&nbsp;
<a href="laporan.html">← Metadata</a> &nbsp;·&nbsp;
<a href="laporan_piksel.html">← Piksel</a></p>
<div class="warn"><b>Angka utama</b><ul>{facts}</ul></div>
<h2>Tujuh pemeriksaan</h2>
{cards}
<footer>Dibuat oleh <code>eda/explore_deep.py</code>.</footer></div></body></html>"""


def main():
    meta = pd.read_csv(META)
    meta = meta[meta["dipakai_5kelas"]].reset_index(drop=True)

    if "--replot" in sys.argv and CACHE.exists():
        print("Memakai hasil pindaian tersimpan (--replot).")
        px = pd.read_csv(OUT / "deep_stats.csv")
        z = np.load(CACHE)
        curves = {k: z[k] for k in ("full", "center", "border", "nolight")}
        bmaps, block_curves = z["bmaps"], z["block_curves"]
    else:
        print(f"Memindai {len(meta)} klip...")
        px, curves, bmaps, block_curves = deep_scan(meta)
        px.to_csv(OUT / "deep_stats.csv", index=False)
        np.savez_compressed(CACHE, bmaps=bmaps, block_curves=block_curves, **curves)
        print("  tersimpan: deep_stats.csv")

    px = px.reset_index(drop=True)
    print("Membuat grafik (beberapa melatih model, mohon tunggu)...")
    figs = [
        fig_disgust_vs_others(px, bmaps, curves),
        fig_shortcut(px),
        fig_blockmaps_fair(px, bmaps),
        fig_offset_control(px, curves),
        fig_apex_estimators(px, meta),
        fig_shape(px, curves),
        fig_spatiotemporal(px, bmaps, curves, block_curves),
    ]
    (OUT / "laporan_lanjutan.html").write_text(build_html(figs), encoding="utf-8")
    print("\n=== ANGKA UTAMA ===")
    for k, v in FACTS.items():
        print(f"[{k}] {v}")
    print(f"\nSelesai. Buka: {OUT / 'laporan_lanjutan.html'}")


if __name__ == "__main__":
    main()
