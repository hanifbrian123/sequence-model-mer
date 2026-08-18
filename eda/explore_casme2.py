"""
Eksplorasi data CASME II — TANPA membuka satu gambar pun.

Yang dibaca skrip ini:
  1. CASME2-coding-20140508.xlsx   -> anotasi (subject, klip, onset/apex/offset, AU, emosi)
  2. CASME2-ObjectiveClasses.xlsx  -> label alternatif berbasis AU
  3. NAMA FILE di dalam folder Cropped (os.scandir) -> untuk cek integritas

Yang TIDAK dilakukan: membuka, mendekode, atau menampilkan piksel gambar.

Jalankan:
    conda run -n facesleuth python eda/explore_casme2.py
Hasil:
    eda/out/*.png            grafik
    eda/out/metadata.csv     tabel gabungan yang bisa dibuka di Excel
    eda/out/laporan.html     laporan lengkap (buka dengan browser)
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------
# Konfigurasi
# --------------------------------------------------------------------------
DATASET = Path(r"D:\AI-Projects\casmeII-facesleuth-r\dataset")
CODING_XLSX = DATASET / "CASME2-coding-20140508.xlsx"
OBJECTIVE_XLSX = DATASET / "CASME2-ObjectiveClasses.xlsx"
CROPPED = DATASET / "Cropped"

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

FPS = 200.0  # CASME II direkam dengan kamera 200 frame per detik
FIVE_CLASSES = ["happiness", "disgust", "repression", "surprise", "others"]

# Palet tervalidasi (light mode)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
BLUE = SERIES[0]
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 130,
})


def style_axes(ax, ygrid=True, xgrid=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    if ygrid:
        ax.grid(True, axis="y", color=GRID, linewidth=1, zorder=0)
    if xgrid:
        ax.grid(True, axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def title(ax, main, sub=None):
    """Judul + subjudul, diukur dalam poin supaya tidak pernah bertumpuk."""
    if not sub:
        ax.set_title(main, loc="left", pad=10, fontsize=13,
                     fontweight="600", color=INK)
        return
    lines = sub.count("\n") + 1
    ax.set_title(main, loc="left", pad=10 + 14 * lines, fontsize=13,
                 fontweight="600", color=INK)
    ax.annotate(sub, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 5), textcoords="offset points",
                fontsize=9.5, color=INK2, va="bottom", ha="left")


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"  tersimpan: {path.name}")
    return path


NOTES: list[str] = []


def note(text):
    NOTES.append(text)
    print("  ! " + text)


# --------------------------------------------------------------------------
# 1. Muat metadata
# --------------------------------------------------------------------------
def load_metadata() -> pd.DataFrame:
    df = pd.read_excel(CODING_XLSX, sheet_name="Sheet1")
    df = df[["Subject", "Filename", "OnsetFrame", "ApexFrame", "OffsetFrame",
             "Action Units", "Estimated Emotion"]].copy()
    df.columns = ["subject_id", "clip", "onset", "apex", "offset", "aus", "emotion"]

    obj = pd.read_excel(OBJECTIVE_XLSX, sheet_name="Sheet1")
    obj.columns = ["subject_id", "clip", "objective_class"]
    df = df.merge(obj, on=["subject_id", "clip"], how="left")

    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df["subject"] = df["subject_id"].apply(lambda s: f"sub{int(s):02d}")

    # Kolom frame kadang berisi '/' (nilai hilang) -> jadikan NaN, bukan crash.
    for col in ("onset", "apex", "offset"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_apex = df["apex"].isna().sum()
    if bad_apex:
        note(f"{bad_apex} klip tidak punya ApexFrame yang sah (isi '/' di Excel).")

    zero_offset = (df["offset"] == 0).sum()
    if zero_offset:
        note(f"{zero_offset} klip punya OffsetFrame = 0 (anotasi cacat di file asli).")

    df["durasi_frame"] = df["offset"] - df["onset"] + 1
    df["durasi_ms"] = df["durasi_frame"] / FPS * 1000.0
    # posisi apex relatif: 0 = tepat di onset, 1 = tepat di offset
    span = (df["offset"] - df["onset"]).replace(0, np.nan)
    df["apex_relatif"] = (df["apex"] - df["onset"]) / span

    df["dipakai_5kelas"] = df["emotion"].isin(FIVE_CLASSES)
    return df


# --------------------------------------------------------------------------
# 2. Cek folder (hanya nama file, tidak dibuka)
# --------------------------------------------------------------------------
FRAME_RE = re.compile(r"(\d+)")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_folders(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        d = CROPPED / r.subject / r.clip
        if not d.is_dir():
            rows.append((None, None, None, None))
            continue
        nums = []
        n_files = 0
        for e in os.scandir(d):
            if not e.is_file():
                continue
            stem, ext = os.path.splitext(e.name)
            if ext.lower() not in IMG_EXT:
                continue
            n_files += 1
            m = FRAME_RE.findall(stem)
            if m:
                nums.append(int(m[-1]))
        if not nums:
            rows.append((n_files, None, None, None))
            continue
        nums.sort()
        gaps = np.diff(nums) if len(nums) > 1 else np.array([], dtype=int)
        rows.append((n_files, nums[0], nums[-1], int((gaps != 1).sum())))

    out = pd.DataFrame(rows, columns=["n_file", "frame_min", "frame_max", "n_lompatan"],
                       index=df.index)
    df = pd.concat([df, out], axis=1)

    hilang = df["n_file"].isna().sum()
    if hilang:
        note(f"{hilang} klip ada di Excel tapi foldernya tidak ada di disk.")

    ada = df["n_file"].notna()
    beda = ada & (df["n_file"] != df["durasi_frame"])
    if beda.any():
        note(f"{int(beda.sum())} klip: jumlah file != (offset - onset + 1).")
    lompat = df["n_lompatan"].fillna(0) > 0
    if lompat.any():
        note(f"{int(lompat.sum())} klip punya penomoran frame yang bolong.")
    else:
        note("Penomoran frame rapat di SEMUA klip — tidak ada frame yang dilewat.")
    return df


# --------------------------------------------------------------------------
# 3. Grafik
# --------------------------------------------------------------------------
def fig_kelas(df):
    counts = df["emotion"].value_counts()
    order = [c for c in FIVE_CLASSES if c in counts.index] + \
            [c for c in counts.index if c not in FIVE_CLASSES]
    vals = [counts[c] for c in order]
    colors = [BLUE if c in FIVE_CLASSES else "#d8d7d1" for c in order]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    style_axes(ax)
    bars = ax.bar(range(len(order)), vals, color=colors, width=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, str(v),
                ha="center", va="bottom", fontsize=10, color=INK, fontweight="600")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylabel("jumlah klip")
    ax.set_ylim(0, max(vals) * 1.16)
    total5 = sum(v for c, v in zip(order, vals) if c in FIVE_CLASSES)
    title(ax, "Berapa banyak contoh untuk tiap emosi?",
          f"Biru = 5 kelas yang dipakai ({total5} klip). Abu-abu = dibuang karena "
          f"contohnya terlalu sedikit.")
    return save(fig, "01_distribusi_kelas.png")


def fig_per_subjek(df):
    d = df[df["dipakai_5kelas"]]
    counts = d.groupby("subject").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    style_axes(ax)
    bars = ax.bar(range(len(counts)), counts.values, color=BLUE, width=0.66, zorder=3)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, str(v), ha="center",
                va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=90)
    ax.set_ylabel("jumlah klip")
    ax.set_ylim(0, counts.max() * 1.14)
    title(ax, "Tiap orang menyumbang berapa klip?",
          f"{len(counts)} orang, total {int(counts.sum())} klip. Terbanyak "
          f"{counts.iloc[0]}, tersedikit {counts.iloc[-1]}.")
    return save(fig, "02_klip_per_subjek.png")


def fig_matriks(df):
    d = df[df["dipakai_5kelas"]]
    piv = d.pivot_table(index="subject", columns="emotion", values="clip",
                        aggfunc="count").reindex(columns=FIVE_CLASSES).fillna(0)
    piv = piv.sort_index()
    M = piv.values

    fig, ax = plt.subplots(figsize=(6.6, 8.6))
    im = ax.imshow(M, cmap=SEQ, aspect="auto", vmin=0, vmax=M.max())
    ax.set_xticks(range(len(FIVE_CLASSES)))
    ax.set_xticklabels(FIVE_CLASSES, rotation=35, ha="right", color=INK2)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8.5, color=INK2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = int(M[i, j])
            ax.text(j, i, "·" if v == 0 else str(v), ha="center", va="center",
                    fontsize=8.5,
                    color=("#c3c2b7" if v == 0 else ("white" if v > M.max() * 0.55 else INK)))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(FIVE_CLASSES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(piv.index), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    kosong = int((M == 0).sum())
    title(ax, "Siapa punya emosi apa?",
          f"Titik = orang itu tidak punya contoh emosi tsb. {kosong} dari {M.size} sel kosong.")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED, labelcolor=MUTED, length=0)
    cb.set_label("jumlah klip", color=INK2, fontsize=9)
    return save(fig, "03_matriks_subjek_kelas.png")


def fig_durasi(df):
    d = df[df["dipakai_5kelas"] & df["durasi_frame"].notna() & (df["durasi_frame"] > 0)]
    v = d["durasi_frame"].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))

    style_axes(ax1)
    ax1.hist(v, bins=np.arange(0, v.max() + 10, 8), color=BLUE, zorder=3,
             edgecolor=SURFACE, linewidth=1.2)
    med = float(np.median(v))
    ax1.axvline(med, color="#e34948", linewidth=2, zorder=4)
    ax1.text(med + 3, ax1.get_ylim()[1] * 0.92, f"median {med:.0f} frame",
             color="#e34948", fontsize=9.5, fontweight="600")
    ax1.set_xlabel("panjang klip (frame)")
    ax1.set_ylabel("jumlah klip")
    title(ax1, "Panjang ekspresi dalam frame")

    style_axes(ax2)
    ms = d["durasi_ms"].values
    ax2.hist(ms, bins=np.arange(0, ms.max() + 40, 40), color=BLUE, zorder=3,
             edgecolor=SURFACE, linewidth=1.2)
    ax2.axvline(500, color="#e34948", linewidth=2, zorder=4)
    ax2.text(510, ax2.get_ylim()[1] * 0.92, "batas 500 ms", color="#e34948",
             fontsize=9.5, fontweight="600")
    ax2.set_xlabel("panjang klip (milidetik, pada 200 fps)")
    ax2.set_ylabel("jumlah klip")
    over = int((ms > 500).sum())
    title(ax2, "Panjang ekspresi dalam waktu nyata",
          f"{over} dari {len(ms)} klip lebih panjang dari 500 ms.")
    fig.tight_layout()
    return save(fig, "04_durasi.png")


def fig_durasi_per_kelas(df):
    d = df[df["dipakai_5kelas"] & (df["durasi_frame"] > 0)]
    data = [d.loc[d["emotion"] == c, "durasi_frame"].values for c in FIVE_CLASSES]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    style_axes(ax)
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False,
                    medianprops=dict(color=INK, linewidth=2),
                    whiskerprops=dict(color=AXIS, linewidth=1.4),
                    capprops=dict(color=AXIS, linewidth=1.4),
                    boxprops=dict(edgecolor=SURFACE, linewidth=1.5))
    for patch, c in zip(bp["boxes"], SERIES):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    rng = np.random.default_rng(0)
    for i, arr in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(arr)), arr, s=9,
                   color=INK2, alpha=0.35, zorder=4, linewidths=0)
    ax.set_xticklabels(FIVE_CLASSES)
    ax.set_ylabel("panjang klip (frame)")
    meds = [float(np.median(a)) for a in data]
    title(ax, "Apakah tiap emosi punya panjang yang berbeda?",
          f"Median per kelas: {', '.join(f'{c}={m:.0f}' for c, m in zip(FIVE_CLASSES, meds))} frame. "
          f"Kalau bedanya besar, model bisa 'menebak' dari panjang klip saja.")
    return save(fig, "05_durasi_per_kelas.png")


def fig_apex(df):
    d = df[df["dipakai_5kelas"] & df["apex_relatif"].notna()]
    v = d["apex_relatif"].clip(0, 1).values
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    style_axes(ax)
    ax.hist(v, bins=np.arange(0, 1.05, 0.05), color=BLUE, zorder=3,
            edgecolor=SURFACE, linewidth=1.2)
    med = float(np.median(v))
    ax.axvline(med, color="#e34948", linewidth=2, zorder=4)
    ax.text(med + 0.012, ax.get_ylim()[1] * 0.92, f"median {med:.2f}",
            color="#e34948", fontsize=9.5, fontweight="600")
    ax.set_xlabel("posisi puncak ekspresi  (0 = awal klip, 1 = akhir klip)")
    ax.set_ylabel("jumlah klip")
    title(ax, "Di mana letak puncak ekspresi (apex)?",
          "Kalau menumpuk di satu titik, puncak bisa ditebak tanpa label. "
          "Kalau menyebar rata, harus dideteksi otomatis.")
    return save(fig, "06_posisi_apex.png")


def fig_aus(df):
    d = df[df["dipakai_5kelas"]]
    per_class = {c: Counter() for c in FIVE_CLASSES}
    all_au = Counter()
    for r in d.itertuples():
        aus = re.findall(r"\d+[A-Za-z]*", str(r.aus))
        aus = {"AU" + a for a in aus}
        per_class[r.emotion].update(aus)
        all_au.update(aus)
    top = [a for a, _ in all_au.most_common(12)]
    M = np.array([[per_class[c][a] for a in top] for c in FIVE_CLASSES], dtype=float)
    frac = M / np.maximum(M.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    im = ax.imshow(frac, cmap=SEQ, aspect="auto", vmin=0, vmax=frac.max())
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top, color=INK2)
    ax.set_yticks(range(len(FIVE_CLASSES)))
    ax.set_yticklabels(FIVE_CLASSES, color=INK2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = int(M[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                        color="white" if frac[i, j] > frac.max() * 0.55 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(top), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(FIVE_CLASSES), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    title(ax, "Otot wajah mana yang menyala di tiap emosi?",
          "AU = Action Unit, kode gerakan otot wajah. Angka = berapa klip memakainya. "
          "Warna = seberapa dominan AU itu di dalam kelasnya.")
    return save(fig, "07_action_units.png")


def fig_integritas(df):
    d = df[df["n_file"].notna()].copy()
    exp = d["durasi_frame"].values.astype(float)
    act = d["n_file"].values.astype(float)
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    style_axes(ax, ygrid=True, xgrid=True)
    lim = max(exp.max(), act.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=AXIS, linewidth=1.5, linestyle="--", zorder=2)
    ax.scatter(exp, act, s=26, color=BLUE, alpha=0.55, linewidths=0, zorder=3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("dari Excel:  offset − onset + 1")
    ax.set_ylabel("dari disk:  jumlah file gambar")
    cocok = int((exp == act).sum())
    title(ax, "Apakah isi folder cocok dengan anotasi?",
          f"{cocok} dari {len(d)} klip persis di garis putus-putus. "
          f"Titik di luar garis = folder tidak sesuai anotasi.")
    return save(fig, "08_integritas_folder.png")


def fig_loso(df):
    """Simulasi: kalau 1 orang dijadikan test, berapa sampel test-nya?"""
    d = df[df["dipakai_5kelas"]]
    per_sub = d.groupby("subject").size().sort_values()
    n_kelas = d.groupby("subject")["emotion"].nunique().reindex(per_sub.index)
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    style_axes(ax)
    colors = [SERIES[7] if n <= 2 else (SERIES[3] if n == 3 else BLUE) for n in n_kelas]
    ax.bar(range(len(per_sub)), per_sub.values, color=colors, width=0.66, zorder=3)
    for i, (v, n) in enumerate(zip(per_sub.values, n_kelas.values)):
        ax.text(i, v + 0.4, str(n), ha="center", va="bottom", fontsize=8,
                color=INK2)
    ax.set_xticks(range(len(per_sub)))
    ax.set_xticklabels(per_sub.index, rotation=90)
    ax.set_ylabel("jumlah klip kalau orang ini jadi data uji")
    ax.set_ylim(0, per_sub.max() * 1.16)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[7], label="hanya 1–2 kelas hadir"),
                       Patch(color=SERIES[3], label="3 kelas hadir"),
                       Patch(color=BLUE, label="4–5 kelas hadir")],
              frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    kecil = int((per_sub <= 5).sum())
    title(ax, "Kenapa hasil per-orang tidak bisa dipercaya sendirian",
          f"Angka di atas batang = berapa kelas yang hadir. {kecil} orang punya ≤5 klip; "
          f"satu tebakan salah menggeser akurasinya belasan persen.")
    return save(fig, "09_kerapuhan_loso.png")


# --------------------------------------------------------------------------
# 4. Laporan HTML
# --------------------------------------------------------------------------
def build_html(df, figs):
    d5 = df[df["dipakai_5kelas"]]
    counts = d5["emotion"].value_counts().reindex(FIVE_CLASSES)
    imbalance = counts.max() / counts.min()
    med_dur = d5["durasi_frame"].median()
    med_apex = d5["apex_relatif"].median()

    penjelasan = {
        "01_distribusi_kelas.png": (
            "Datanya <b>tidak seimbang</b>. Kelas terbanyak (<code>others</code>, "
            f"{int(counts['others'])} klip) punya <b>{imbalance:.1f}× lipat</b> lebih banyak "
            f"contoh dibanding yang tersedikit (<code>{counts.idxmin()}</code>, "
            f"{int(counts.min())} klip).<br><br>"
            "<b>Kenapa ini penting:</b> kalau model asal menebak <code>others</code> untuk "
            f"semua input, dia sudah dapat akurasi {counts['others'] / counts.sum() * 100:.0f}% "
            "tanpa belajar apa pun. Jadi <b>akurasi biasa itu menyesatkan</b> — laporkan juga "
            "UF1/UAR yang menghitung tiap kelas dengan bobot sama."
        ),
        "02_klip_per_subjek.png": (
            "Tiap batang = satu orang. Tingginya sangat timpang. "
            "Beberapa orang menyumbang belasan klip, beberapa cuma segelintir.<br><br>"
            "<b>Kenapa ini penting:</b> wajah tiap orang beda. Kalau data satu orang muncul "
            "di latihan <i>dan</i> di ujian, model bisa lulus dengan cara menghafal wajah, "
            "bukan mengenali emosi. Karena itu pembagian data harus "
            "<b>per-orang</b> (subject-independent), bukan acak per-klip."
        ),
        "03_matriks_subjek_kelas.png": (
            "Baris = orang, kolom = emosi, angka = jumlah klip. Titik berarti kosong.<br><br>"
            "<b>Kenapa ini penting:</b> banyak sel kosong. Artinya kalau satu orang dipakai "
            "sebagai data uji, sering kali <b>tidak semua kelas ada</b> di ujian itu. "
            "Skor dari satu orang saja jadi tidak bisa dibandingkan dengan skor orang lain — "
            "hasilnya harus <b>digabung dulu</b> dari semua fold, baru dihitung metriknya."
        ),
        "04_durasi.png": (
            f"Panjang klip sangat bervariasi; mediannya {med_dur:.0f} frame. "
            "Karena kamera merekam 200 gambar per detik, itu setara "
            f"{med_dur / FPS * 1000:.0f} milidetik — memang sekejap, sesuai definisi "
            "<i>micro-expression</i>.<br><br>"
            "<b>Kenapa ini penting:</b> model urutan (sequence) butuh jumlah frame yang "
            "<b>tetap</b>, misal 16. Klip pendek harus diregangkan, klip panjang "
            "diperjarang. Cara memilih 16 frame itu sendiri adalah keputusan desain yang "
            "berpengaruh besar."
        ),
        "05_durasi_per_kelas.png": (
            "Kotak = rentang tengah (50% data), garis di dalamnya = median, titik = klip "
            "satuan.<br><br>"
            "<b>Kenapa ini penting:</b> ini pemeriksaan <b>kecurangan tak sengaja</b>. "
            "Kalau satu kelas jelas lebih panjang dari yang lain, model bisa menebak benar "
            "hanya dari panjang klip — bukan dari gerak wajah. Kalau kotak-kotaknya banyak "
            "bertumpang tindih, panjang klip bukan jalan pintas, dan itu kabar baik."
        ),
        "06_posisi_apex.png": (
            f"Apex = frame saat ekspresi paling kuat. Posisinya dinormalkan: 0 = tepat di awal "
            f"klip, 1 = tepat di akhir. Mediannya {med_apex:.2f}.<br><br>"
            "<b>Kenapa ini penting:</b> apex ini <b>ditandai manusia</b>. Di video upload "
            "dari HP, label itu tidak ada. Kalau sebarannya lebar (bukan menumpuk di satu "
            "titik), berarti apex tidak bisa ditebak dengan aturan tetap seperti 'ambil "
            "tengah' — harus dideteksi otomatis, dan itu sumber error tersendiri."
        ),
        "07_action_units.png": (
            "AU (Action Unit) adalah kode standar gerakan otot wajah — misalnya AU12 = "
            "sudut bibir tertarik ke atas, AU4 = alis mengerut.<br><br>"
            "<b>Kenapa ini penting:</b> ini memberi tahu <b>di mana di wajah</b> sinyalnya "
            "berada untuk tiap emosi, dan kelas mana yang berbagi otot yang sama. "
            "Dua kelas yang memakai AU yang mirip akan sulit dibedakan model — kotak ini "
            "memprediksi di mana nanti muncul kebingungan di confusion matrix."
        ),
        "08_integritas_folder.png": (
            "Sumbu X = jumlah frame menurut Excel, sumbu Y = jumlah file yang benar-benar "
            "ada di folder. Garis putus-putus = cocok sempurna.<br><br>"
            "<b>Kenapa ini penting:</b> ini uji kejujuran data sebelum apa pun dilatih. "
            "Titik yang meleset dari garis berarti anotasi dan isi disk tidak sepakat — "
            "klip seperti itu harus diperiksa atau dibuang, bukan diam-diam ikut dilatih."
        ),
        "09_kerapuhan_loso.png": (
            "Simulasi: <i>kalau</i> satu orang dijadikan data uji, sebanyak apa ujiannya?<br><br>"
            "<b>Kenapa ini penting:</b> untuk orang dengan 4 klip, satu tebakan salah "
            "mengubah akurasi sebesar 25 poin. Itu <b>derau, bukan sinyal</b>. "
            "Kesimpulannya: jangan pernah mengambil keputusan dari skor satu fold; "
            "kumpulkan prediksi semua fold dulu, baru hitung satu metrik gabungan."
        ),
    }

    cards = []
    for i, f in enumerate(figs, start=1):
        name = Path(f).name
        cards.append(f"""
    <section class="card">
      <div class="num">{i}</div>
      <img src="{name}" alt="{name}">
      <div class="body">{penjelasan.get(name, "")}</div>
    </section>""")

    notes_html = "".join(f"<li>{n}</li>" for n in NOTES) or "<li>Tidak ada anomali.</li>"

    return f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eksplorasi Data CASME II</title>
<style>
  :root {{
    --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
    --muted:#898781; --line:#e1e0d9; --blue:#2a78d6;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7;
             --muted:#898781; --line:#2c2c2a; --blue:#3987e5; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--plane); color:var(--ink);
    font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif; }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:48px 20px 80px; }}
  h1 {{ font-size:30px; line-height:1.2; margin:0 0 8px; letter-spacing:-.02em; }}
  .lede {{ color:var(--ink2); margin:0 0 28px; max-width:70ch; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:10px; margin:0 0 32px; }}
  .stat {{ background:var(--surface); border:1px solid var(--line);
    border-radius:12px; padding:14px 16px; }}
  .stat .v {{ font-size:26px; font-weight:650; letter-spacing:-.02em; }}
  .stat .k {{ font-size:12.5px; color:var(--muted); }}
  .card {{ background:var(--surface); border:1px solid var(--line);
    border-radius:14px; padding:22px; margin:0 0 22px; position:relative; }}
  .num {{ position:absolute; top:18px; left:-14px; width:28px; height:28px;
    border-radius:50%; background:var(--blue); color:#fff; font-size:13px;
    font-weight:700; display:grid; place-items:center; }}
  .card img {{ width:100%; height:auto; display:block; border-radius:8px;
    background:#fcfcfb; }}
  .body {{ color:var(--ink2); font-size:15px; margin-top:16px;
    padding-top:16px; border-top:1px solid var(--line); max-width:78ch; }}
  code {{ background:var(--plane); border:1px solid var(--line);
    padding:1px 5px; border-radius:5px; font-size:13.5px; }}
  b {{ color:var(--ink); font-weight:600; }}
  .warn {{ background:var(--surface); border:1px solid var(--line);
    border-left:3px solid #eda100; border-radius:12px; padding:16px 20px; margin:0 0 32px; }}
  .warn ul {{ margin:8px 0 0; padding-left:20px; color:var(--ink2); font-size:14.5px; }}
  h2 {{ font-size:19px; margin:40px 0 12px; }}
  .nav {{ font-size:14px; color:var(--muted); margin:0 0 4px; }}
  .nav a {{ color:var(--blue); text-decoration:none; font-weight:600; }}
  footer {{ color:var(--muted); font-size:13px; margin-top:44px;
    border-top:1px solid var(--line); padding-top:18px; }}
</style></head><body><div class="wrap">
<h1>Eksplorasi Data CASME II</h1>
<p class="lede">Semua angka dan grafik di halaman ini dihitung <b>hanya dari file Excel
anotasi dan dari nama-nama file</b> di folder <code>Cropped</code>.
Tidak ada satu pun gambar yang dibuka, didekode, atau ditampilkan.</p>

<div class="stats">
  <div class="stat"><div class="v">{len(df)}</div><div class="k">klip di anotasi</div></div>
  <div class="stat"><div class="v">{len(d5)}</div><div class="k">klip dipakai (5 kelas)</div></div>
  <div class="stat"><div class="v">{d5['subject'].nunique()}</div><div class="k">orang</div></div>
  <div class="stat"><div class="v">{imbalance:.1f}×</div><div class="k">ketimpangan kelas</div></div>
  <div class="stat"><div class="v">{med_dur:.0f}</div><div class="k">median panjang (frame)</div></div>
  <div class="stat"><div class="v">{int(df['n_file'].sum()):,}</div><div class="k">total frame di disk</div></div>
</div>

<div class="warn"><b>Catatan pemeriksaan otomatis</b><ul>{notes_html}</ul></div>

<p class="nav">Bagian 1 dari 2 &nbsp;·&nbsp; lanjut ke
<a href="laporan_piksel.html">Eksplorasi Piksel →</a></p>

<h2>Sembilan hal yang perlu diketahui sebelum melatih model</h2>
{''.join(cards)}

<footer>Dibuat oleh <code>eda/explore_casme2.py</code>. Tabel mentah:
<code>metadata.csv</code> di folder yang sama — bisa dibuka dengan Excel.</footer>
</div></body></html>
"""


# --------------------------------------------------------------------------
def main():
    print("Membaca metadata...")
    df = load_metadata()
    print(f"  {len(df)} baris anotasi, {df['subject'].nunique()} subjek")

    print("Memindai folder (hanya nama file)...")
    df = scan_folders(df)

    csv_path = OUT / "metadata.csv"
    df.drop(columns=["subject_id"]).to_csv(csv_path, index=False)
    print(f"  tersimpan: {csv_path.name}")

    print("Membuat grafik...")
    figs = [
        fig_kelas(df), fig_per_subjek(df), fig_matriks(df), fig_durasi(df),
        fig_durasi_per_kelas(df), fig_apex(df), fig_aus(df),
        fig_integritas(df), fig_loso(df),
    ]

    html = OUT / "laporan.html"
    html.write_text(build_html(df, figs), encoding="utf-8")
    print(f"  tersimpan: {html.name}")
    print(f"\nSelesai. Buka: {html}")


if __name__ == "__main__":
    main()
