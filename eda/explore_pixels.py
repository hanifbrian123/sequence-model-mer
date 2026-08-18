"""
Eksplorasi PIKSEL CASME II — gambar diproses secara matematis, tidak pernah ditampilkan.

Yang dilakukan skrip ini pada tiap gambar:
  - didekode pada resolusi rendah (grayscale 32x32) hanya untuk menghitung angka
  - dihitung: kecerahan, kontras, selisih antar-frame, energi gerak per blok
Yang TIDAK dilakukan: menyimpan, menampilkan, atau mengirim gambar apa pun.
Semua keluaran berupa angka agregat dan grafik statistik.

Jalankan:
    conda run -n facesleuth python eda/explore_pixels.py
Hasil:
    eda/out/1x_*.png            grafik
    eda/out/pixel_stats.csv     ringkasan per klip
    eda/out/laporan_piksel.html laporan (buka dengan browser)
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

warnings.filterwarnings("ignore", category=UserWarning)

DATASET = Path(r"D:\AI-Projects\casmeII-facesleuth-r\dataset")
CROPPED = DATASET / "Cropped"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
META = OUT / "metadata.csv"

FPS = 200.0
FIVE = ["happiness", "disgust", "repression", "surprise", "others"]
GRID_N = 32       # tiap frame diringkas jadi 32x32 angka
BLOCK_N = 8       # peta energi gerak 8x8 blok
CURVE_N = 40      # kurva waktu dinormalkan ke 40 titik

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDC, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE = SERIES[0]
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec",
                 "#3987e5", "#256abf", "#184f95", "#0d366b"])

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
    """Judul + subjudul, diukur dalam poin supaya tidak pernah bertumpuk."""
    if not sub:
        ax.set_title(main, loc="left", pad=pad, fontsize=13,
                     fontweight="600", color=INK)
        return
    lines = sub.count("\n") + 1
    ax.set_title(main, loc="left", pad=pad + 14 * lines, fontsize=13,
                 fontweight="600", color=INK)
    ax.annotate(sub, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 5), textcoords="offset points",
                fontsize=9.5, color=INK2, va="bottom", ha="left")


def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"  tersimpan: {p.name}")
    return p


# --------------------------------------------------------------------------
# Pemindaian piksel
# --------------------------------------------------------------------------
FRAME_RE = re.compile(r"(\d+)")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def frame_paths(folder: Path):
    items = []
    for e in os.scandir(folder):
        if not e.is_file():
            continue
        stem, ext = os.path.splitext(e.name)
        if ext.lower() not in IMG_EXT:
            continue
        m = FRAME_RE.findall(stem)
        if m:
            items.append((int(m[-1]), e.path))
    items.sort()
    return items


def load_gray_grid(path: str) -> np.ndarray:
    """Dekode JPEG pada skala kecil, kembalikan matriks 32x32 angka 0..255.

    `draft()` membuat libjpeg mendekode langsung pada 1/8 ukuran — cepat, dan
    hasilnya memang hanya dipakai sebagai angka.
    """
    with Image.open(path) as im:
        im.draft("L", (GRID_N * 2, GRID_N * 2))
        g = im.convert("L").resize((GRID_N, GRID_N), Image.BILINEAR)
        return np.asarray(g, dtype=np.float32)


def blockify(mat: np.ndarray, n: int) -> np.ndarray:
    k = mat.shape[0] // n
    return mat[:n * k, :n * k].reshape(n, k, n, k).mean(axis=(1, 3))


def scan(meta: pd.DataFrame):
    rows, curves, blockmaps = [], [], []
    t0 = time.time()
    total = len(meta)
    for i, r in enumerate(meta.itertuples(), start=1):
        folder = CROPPED / r.subject / r.clip
        if not folder.is_dir():
            continue
        items = frame_paths(folder)
        if len(items) < 3:
            continue
        nums = np.array([n for n, _ in items])
        stack = np.stack([load_gray_grid(p) for _, p in items])   # (T,32,32)

        bright = stack.mean(axis=(1, 2))
        contrast = stack.std(axis=(1, 2))
        d_onset = np.abs(stack - stack[0]).mean(axis=(1, 2))       # energi vs frame awal
        d_prev = np.abs(np.diff(stack, axis=0)).mean(axis=(1, 2))  # gerak sesaat

        # kurva energi pada sumbu waktu ternormalkan 0..1
        t_norm = (nums - nums[0]) / max(nums[-1] - nums[0], 1)
        curve = np.interp(np.linspace(0, 1, CURVE_N), t_norm, d_onset)
        peak_rel = float(t_norm[int(np.argmax(d_onset))])

        # peta blok: |apex - onset|, dinormalkan agar tiap klip berbobot sama
        apex_idx = None
        if not np.isnan(r.apex):
            apex_idx = int(np.argmin(np.abs(nums - r.apex)))
        j = apex_idx if apex_idx is not None else int(np.argmax(d_onset))
        diff = np.abs(stack[j] - stack[0])
        bm = blockify(diff, BLOCK_N)
        bm = bm / max(bm.max(), 1e-6)

        rows.append(dict(
            subject=r.subject, clip=r.clip, emotion=r.emotion,
            n_frame=len(items),
            bright_mean=float(bright.mean()), bright_std=float(bright.std()),
            contrast_mean=float(contrast.mean()),
            energi_max=float(d_onset.max()), energi_mean=float(d_onset.mean()),
            gerak_sesaat_mean=float(d_prev.mean()), gerak_sesaat_max=float(d_prev.max()),
            frame_beku=int((d_prev < 0.35).sum()),
            puncak_energi_rel=peak_rel,
        ))
        curves.append(curve)
        blockmaps.append((r.emotion, bm))

        if i % 25 == 0 or i == total:
            print(f"  {i}/{total} klip  ({time.time() - t0:.0f}s)")

    return pd.DataFrame(rows), np.array(curves), blockmaps


# --------------------------------------------------------------------------
# Grafik
# --------------------------------------------------------------------------
def fig_resolusi(meta):
    sizes = []
    for r in meta.itertuples():
        f = CROPPED / r.subject / r.clip
        if not f.is_dir():
            continue
        items = frame_paths(f)
        if not items:
            continue
        with Image.open(items[0][1]) as im:      # header saja, tidak didekode
            sizes.append(im.size)
    w = np.array([s[0] for s in sizes], float)
    h = np.array([s[1] for s in sizes], float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    style_axes(ax1, xgrid=True)
    ax1.scatter(w, h, s=30, color=BLUE, alpha=0.5, linewidths=0)
    ax1.set_xlabel("lebar gambar (piksel)")
    ax1.set_ylabel("tinggi gambar (piksel)")
    title(ax1, "Ukuran potongan wajah",
          f"{len(set(sizes))} ukuran berbeda pada {len(sizes)} klip.")

    style_axes(ax2)
    ar = h / w
    ax2.hist(ar, bins=30, color=BLUE, edgecolor=SURFACE, linewidth=1.2)
    ax2.axvline(float(np.median(ar)), color="#e34948", linewidth=2)
    ax2.set_xlabel("rasio tinggi ÷ lebar")
    ax2.set_ylabel("jumlah klip")
    title(ax2, "Bentuk potongan wajah",
          f"median {np.median(ar):.2f}, rentang {ar.min():.2f}–{ar.max():.2f}.")
    fig.tight_layout()
    return save(fig, "10_ukuran_crop.png")


def fig_kecerahan(px):
    order = px.groupby("subject")["bright_mean"].median().sort_values().index
    data = [px.loc[px.subject == s, "bright_mean"].values for s in order]
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    style_axes(ax)
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.6),
                    whiskerprops=dict(color=AXIS, linewidth=1.2),
                    capprops=dict(color=AXIS, linewidth=1.2),
                    boxprops=dict(edgecolor=SURFACE, linewidth=1.2))
    for b in bp["boxes"]:
        b.set_facecolor(BLUE)
        b.set_alpha(0.8)
    rng = np.random.default_rng(0)
    for i, a in enumerate(data, 1):
        ax.scatter(i + rng.uniform(-0.14, 0.14, len(a)), a, s=8, color=INK2,
                   alpha=0.4, linewidths=0, zorder=4)
    ax.set_xticklabels(order, rotation=90)
    ax.set_ylabel("kecerahan rata-rata (0 = hitam, 255 = putih)")
    med = px.groupby("subject")["bright_mean"].median()
    title(ax, "Pencahayaan tiap orang berbeda-beda",
          f"Median antar-orang berkisar {med.min():.0f} sampai {med.max():.0f} — "
          f"selisih {med.max() - med.min():.0f} tingkat keabuan.")
    return save(fig, "11_kecerahan_per_subjek.png")


def fig_kurva(px, curves):
    x = np.linspace(0, 1, CURVE_N)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    style_axes(ax1)
    for c, col in zip(FIVE, SERIES):
        m = (px["emotion"] == c).values
        if m.sum() == 0:
            continue
        med = np.median(curves[m], axis=0)
        ax1.plot(x, med, color=col, linewidth=2, label=f"{c} (n={m.sum()})")
        ax1.text(1.005, med[-1], c, color=col, fontsize=9, va="center",
                 fontweight="600")
    ax1.set_xlabel("waktu di dalam klip  (0 = onset, 1 = offset)")
    ax1.set_ylabel("beda piksel terhadap frame awal")
    ax1.set_xlim(0, 1.02)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)
    title(ax1, "Bagaimana bentuk gerakan sepanjang klip?",
          "Garis = median tiap kelas.")

    style_axes(ax2)
    allc = np.median(curves, axis=0)
    q1, q3 = np.percentile(curves, [25, 75], axis=0)
    ax2.fill_between(x, q1, q3, color=BLUE, alpha=0.18, linewidth=0)
    ax2.plot(x, allc, color=BLUE, linewidth=2.4)
    pk = x[int(np.argmax(allc))]
    ax2.axvline(pk, color="#e34948", linewidth=2)
    ax2.text(pk + 0.015, ax2.get_ylim()[1] * 0.1, f"puncak di {pk:.2f}",
             color="#e34948", fontsize=9.5, fontweight="600")
    ax2.set_xlabel("waktu di dalam klip  (0 = onset, 1 = offset)")
    ax2.set_ylabel("beda piksel terhadap frame awal")
    ax2.set_xlim(0, 1)
    title(ax2, "Gabungan semua klip",
          "Pita = 50% klip di tengah. Naik lalu melandai, bukan naik-turun.")
    fig.tight_layout()
    return save(fig, "12_kurva_gerak.png")


def fig_apex_vs_puncak(px, meta):
    m = meta[["subject", "clip", "apex_relatif"]]
    d = px.merge(m, on=["subject", "clip"], how="left").dropna(subset=["apex_relatif"])
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    style_axes(ax, xgrid=True)
    ax.plot([0, 1], [0, 1], "--", color=AXIS, linewidth=1.5)
    ax.scatter(d["apex_relatif"].clip(0, 1), d["puncak_energi_rel"],
               s=26, color=BLUE, alpha=0.5, linewidths=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("apex menurut anotasi manusia")
    ax.set_ylabel("puncak gerakan menurut hitungan piksel")
    err = (d["puncak_energi_rel"] - d["apex_relatif"].clip(0, 1))
    dekat = float((err.abs() <= 0.10).mean() * 100)
    corr = float(np.corrcoef(d["apex_relatif"].clip(0, 1), d["puncak_energi_rel"])[0, 1])
    title(ax, "Bisakah apex ditebak tanpa label manusia?",
          f"Korelasi {corr:.2f}. Hanya {dekat:.0f}% klip yang tebakan otomatisnya "
          f"meleset ≤ 0,10.\nBias rata-rata {err.mean():+.2f} (positif = tebakan terlalu akhir).")
    return save(fig, "13_apex_vs_puncak_gerak.png")


def fig_peta_blok(blockmaps):
    fig, axes = plt.subplots(1, 5, figsize=(13.2, 3.5))
    for ax, c, col in zip(axes, FIVE, SERIES):
        maps = [b for e, b in blockmaps if e == c]
        M = np.mean(maps, axis=0)
        M = (M - M.min()) / max(M.max() - M.min(), 1e-6)
        ax.imshow(M, cmap=SEQ, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRIDC)
        ax.set_title(f"{c}  (n={len(maps)})", fontsize=11, color=INK,
                     fontweight="600", pad=8)
        ax.text(0.5, -0.09, "← lebar wajah →", transform=ax.transAxes,
                ha="center", fontsize=8, color=MUTED)
    axes[0].text(-0.12, 0.5, "atas kepala\n↓\ndagu", transform=axes[0].transAxes,
                 ha="right", va="center", fontsize=8, color=MUTED)
    fig.suptitle("Di bagian wajah mana gerakan terjadi? (rata-rata 8×8 blok, "
                 "bukan gambar wajah)", x=0.012, ha="left", fontsize=13,
                 fontweight="600", color=INK, y=1.10)
    fig.text(0.012, 1.02, "Gelap = banyak perubahan piksel dari onset ke apex. "
                          "Tiap kotak = rata-rata puluhan klip dari orang berbeda.",
             ha="left", fontsize=9.5, color=INK2)
    fig.tight_layout()
    return save(fig, "14_peta_gerak_blok.png")


def fig_amplitudo(px):
    data = [px.loc[px.emotion == c, "energi_max"].values for c in FIVE]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    style_axes(ax)
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False,
                    medianprops=dict(color=INK, linewidth=2),
                    whiskerprops=dict(color=AXIS, linewidth=1.4),
                    capprops=dict(color=AXIS, linewidth=1.4),
                    boxprops=dict(edgecolor=SURFACE, linewidth=1.5))
    for b, col in zip(bp["boxes"], SERIES):
        b.set_facecolor(col)
        b.set_alpha(0.85)
    rng = np.random.default_rng(1)
    for i, a in enumerate(data, 1):
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(a)), a, s=9, color=INK2,
                   alpha=0.35, linewidths=0, zorder=4)
    ax.set_xticklabels(FIVE)
    ax.set_ylabel("perubahan piksel terbesar dalam klip")
    meds = [float(np.median(a)) for a in data]
    title(ax, "Seberapa besar gerakannya, per emosi?",
          f"Median: {', '.join(f'{c}={m:.1f}' for c, m in zip(FIVE, meds))}. "
          "Kotak yang saling tumpang tindih = besar gerakan bukan jalan pintas.")
    return save(fig, "15_amplitudo_per_kelas.png")


def fig_beku(px):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))
    style_axes(ax1)
    ratio = px["energi_max"] / px["gerak_sesaat_mean"]
    ax1.hist(ratio, bins=np.arange(0, ratio.max() + 0.5, 0.5), color=BLUE,
             edgecolor=SURFACE, linewidth=1.2)
    med = float(ratio.median())
    ax1.axvline(med, color="#e34948", linewidth=2)
    ax1.text(med + 0.15, ax1.get_ylim()[1] * 0.9, f"median {med:.1f}×",
             color="#e34948", fontsize=9.5, fontweight="600")
    ax1.set_xlabel("gerak seluruh ekspresi ÷ gerak antar dua frame berurutan")
    ax1.set_ylabel("jumlah klip")
    title(ax1, "Seberapa mubazir frame yang berdekatan?",
          f"Beda antar dua frame berurutan hanya {px['gerak_sesaat_mean'].median():.2f} "
          f"tingkat keabuan — {med:.1f}× lebih kecil dari gerak seluruh ekspresi.")

    style_axes(ax2, xgrid=True)
    ax2.scatter(px["n_frame"], px["energi_max"], s=26, color=BLUE, alpha=0.5,
                linewidths=0)
    c = float(np.corrcoef(px["n_frame"], px["energi_max"])[0, 1])
    ax2.set_xlabel("panjang klip (frame)")
    ax2.set_ylabel("perubahan piksel terbesar")
    title(ax2, "Klip panjang = gerakan besar?",
          f"Korelasi {c:.2f}. Kalau mendekati 0, panjang klip tidak menyandi kekuatan ekspresi.")
    fig.tight_layout()
    return save(fig, "16_frame_diam.png")


# --------------------------------------------------------------------------
def build_html(px, figs, meta):
    med_b = px.groupby("subject")["bright_mean"].median()
    penjelasan = {
        "10_ukuran_crop.png": (
            "Ukuran piksel tiap klip berbeda-beda (sekitar 225–300 lebarnya), tapi "
            "<b>bentuknya hampir seragam</b>: rasio tinggi÷lebar berkumpul rapat di sekitar "
            "1,21.<br><br>"
            "<b>Kenapa ini penting:</b> model butuh masukan berukuran tetap, jadi semua "
            "gambar akan diregangkan. Kalau rasionya bervariasi lebar, wajah akan melar "
            "berbeda-beda dan itu jadi derau. Di sini <b>hasilnya kabar baik</b>: "
            "rasionya sudah konsisten, jadi resize ke ukuran tetap hampir tidak "
            "menimbulkan distorsi tambahan. Satu sumber masalah yang bisa dicoret "
            "dari daftar."
        ),
        "11_kecerahan_per_subjek.png": (
            f"Kecerahan rata-rata tiap orang berbeda jauh: dari {med_b.min():.0f} sampai "
            f"{med_b.max():.0f} dari skala 0–255.<br><br>"
            "<b>Kenapa ini penting:</b> ini disebut <i>domain shift</i>. Model bisa "
            "diam-diam belajar 'gambar terang = orang A', dan kemampuannya runtuh saat "
            "diuji pada orang yang belum pernah dilihat. Ini salah satu penyebab hasil "
            "antar-orang naik-turun ekstrem. Penangkalnya: <b>normalisasi per klip</b> "
            "(kurangi rata-rata, bagi simpangan baku) atau pakai selisih antar-frame, "
            "bukan piksel mentah."
        ),
        "12_kurva_gerak.png": (
            "Sumbu tegak = seberapa jauh sebuah frame berbeda dari frame pertama, "
            "dihitung langsung dari piksel. Sumbu datar = posisi waktu di dalam klip.<br><br>"
            "<b>Kenapa ini penting:</b> kurvanya <b>naik lalu melandai</b>, bukan naik-turun "
            "kacau. Artinya sinyalnya mulus terhadap waktu — model urutan memang cocok, "
            "dan mengambil 16 frame dari klip tidak akan merusak bentuk kurva ini. "
            "Kalau kurva tiap kelas berbeda bentuknya, itu berarti <b>pola waktu</b> "
            "membawa informasi kelas, bukan cuma satu frame puncak."
        ),
        "13_apex_vs_puncak_gerak.png": (
            "Sumbu datar = apex yang ditandai manusia. Sumbu tegak = frame dengan "
            "perubahan piksel terbesar, dihitung otomatis. Garis putus-putus = tebakan "
            "otomatis persis sama dengan manusia.<br><br>"
            "<b>Kenapa ini penting:</b> di video upload nanti tidak ada apex dari manusia. "
            "Grafik ini mengukur <b>seberapa jauh</b> penggantinya meleset. Titik yang "
            "berkumpul di atas garis berarti tebakan otomatis cenderung terlalu akhir — "
            "itu bias yang bisa dikoreksi, dan koreksinya harus dihitung dari data latih "
            "saja, jangan dari data uji."
        ),
        "14_peta_gerak_blok.png": (
            "Wajah dibagi jadi kotak 8×8. Warna tiap kotak = rata-rata besarnya perubahan "
            "piksel dari onset ke apex, dirata-ratakan atas puluhan klip dari orang "
            "berbeda. Ini <b>peta angka</b>, bukan foto.<br><br>"
            "<b>Kenapa ini penting:</b> ini menunjukkan <b>di mana</b> sinyal berada. "
            "Kalau kotak gelapnya berkumpul di daerah mulut untuk satu kelas dan di daerah "
            "alis untuk kelas lain, memotong wilayah wajah (ROI) masuk akal. "
            "Kalau petanya mirip semua, ROI tidak akan menolong — dan itu menghemat "
            "berhari-hari percobaan."
        ),
        "15_amplitudo_per_kelas.png": (
            "Seberapa besar perubahan piksel maksimum dalam satu klip, dipisah per emosi."
            "<br><br><b>Kenapa ini penting:</b> pemeriksaan jalan pintas kedua. Kalau satu "
            "kelas jelas bergerak lebih besar, model bisa menang hanya dengan mengukur "
            "'seberapa ramai'. Kotak yang bertumpang tindih berarti model <b>terpaksa</b> "
            "belajar bentuk gerakan, bukan sekadar besarnya — itu yang kita mau."
        ),
        "16_frame_diam.png": (
            "Kiri: berapa kali lipat gerakan seluruh ekspresi dibanding gerakan antara dua "
            "frame yang berurutan. Kanan: apakah klip yang panjang cenderung punya gerakan "
            "besar.<br><br>"
            "<b>Kenapa ini penting:</b> pada 200 frame/detik, dua frame berurutan nyaris "
            "identik — bedanya kurang dari setengah tingkat keabuan, setipis derau kamera. "
            "Artinya <b>informasinya jauh lebih sedikit daripada jumlah framenya</b>. "
            "Ini alasan teknis kenapa memilih 16 frame dari 65, atau menurunkan ke 30 "
            "frame/detik, tidak otomatis merusak sinyal. Panel kanan memastikan panjang "
            "klip tidak diam-diam menyandi kekuatan ekspresi."
        ),
    }
    cards = "".join(f"""
    <section class="card"><div class="num">{i}</div>
      <img src="{Path(f).name}" alt="">
      <div class="body">{penjelasan.get(Path(f).name, '')}</div></section>"""
                    for i, f in enumerate(figs, start=10))

    n_img = int(px["n_frame"].sum())
    return f"""<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eksplorasi Piksel CASME II</title><style>
:root {{ --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
 --muted:#898781; --line:#e1e0d9; --blue:#2a78d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --surface:#1a1a19; --plane:#0d0d0d;
 --ink:#fff; --ink2:#c3c2b7; --line:#2c2c2a; --blue:#3987e5; }} }}
*{{box-sizing:border-box}} body{{margin:0;background:var(--plane);color:var(--ink);
 font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:1000px;margin:0 auto;padding:48px 20px 80px}}
h1{{font-size:30px;line-height:1.2;margin:0 0 8px;letter-spacing:-.02em}}
.lede{{color:var(--ink2);margin:0 0 28px;max-width:70ch}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:0 0 32px}}
.stat{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
.stat .v{{font-size:26px;font-weight:650;letter-spacing:-.02em}}
.stat .k{{font-size:12.5px;color:var(--muted)}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:14px;
 padding:22px;margin:0 0 22px;position:relative}}
.num{{position:absolute;top:18px;left:-14px;width:28px;height:28px;border-radius:50%;
 background:var(--blue);color:#fff;font-size:13px;font-weight:700;display:grid;place-items:center}}
.card img{{width:100%;height:auto;display:block;border-radius:8px;background:#fcfcfb}}
.body{{color:var(--ink2);font-size:15px;margin-top:16px;padding-top:16px;
 border-top:1px solid var(--line);max-width:78ch}}
b{{color:var(--ink);font-weight:600}} h2{{font-size:19px;margin:40px 0 12px}}
.nav{{font-size:14px;color:var(--muted);margin:0 0 4px}}
.nav a{{color:var(--blue);text-decoration:none;font-weight:600}}
code{{background:var(--plane);border:1px solid var(--line);padding:1px 5px;border-radius:5px;font-size:13.5px}}
footer{{color:var(--muted);font-size:13px;margin-top:44px;border-top:1px solid var(--line);padding-top:18px}}
</style></head><body><div class="wrap">
<h1>Eksplorasi Piksel CASME II</h1>
<p class="lede">Bagian ini <b>menghitung</b> dari isi gambar — bukan menampilkannya.
Tiap frame didekode sebentar menjadi matriks angka 32×32, diambil statistiknya,
lalu dibuang. Tidak ada gambar yang disimpan, ditampilkan, atau dikirim ke mana pun.</p>
<div class="stats">
 <div class="stat"><div class="v">{len(px)}</div><div class="k">klip diproses</div></div>
 <div class="stat"><div class="v">{n_img:,}</div><div class="k">frame dihitung</div></div>
 <div class="stat"><div class="v">{px['bright_mean'].mean():.0f}</div><div class="k">kecerahan rata-rata</div></div>
 <div class="stat"><div class="v">{px['n_frame'].median():.0f}</div><div class="k">median frame/klip</div></div>
</div>
<p class="nav">Bagian 2 dari 2 &nbsp;·&nbsp;
<a href="laporan.html">← kembali ke Eksplorasi Metadata</a></p>

<h2>Tujuh hal yang hanya kelihatan dari piksel</h2>
{cards}
<footer>Dibuat oleh <code>eda/explore_pixels.py</code>. Angka mentah per klip:
<code>pixel_stats.csv</code>.</footer></div></body></html>"""


CACHE = OUT / "pixel_cache.npz"


def main():
    if not META.exists():
        raise SystemExit("Jalankan dulu: python eda/explore_casme2.py")
    meta = pd.read_csv(META)
    meta = meta[meta["dipakai_5kelas"]].reset_index(drop=True)

    replot = "--replot" in sys.argv
    if replot and CACHE.exists():
        print("Memakai hasil pindaian yang tersimpan (--replot).")
        px = pd.read_csv(OUT / "pixel_stats.csv")
        z = np.load(CACHE, allow_pickle=True)
        curves = z["curves"]
        blockmaps = list(zip(z["bm_emotion"].tolist(), list(z["bm_data"])))
    else:
        print(f"Memproses {len(meta)} klip (5 kelas)...")
        px, curves, blockmaps = scan(meta)
        px.to_csv(OUT / "pixel_stats.csv", index=False)
        np.savez_compressed(
            CACHE, curves=curves,
            bm_emotion=np.array([e for e, _ in blockmaps]),
            bm_data=np.stack([b for _, b in blockmaps]))
        print(f"  tersimpan: pixel_stats.csv ({len(px)} klip)")

    print("Membuat grafik...")
    figs = [
        fig_resolusi(meta), fig_kecerahan(px), fig_kurva(px, curves),
        fig_apex_vs_puncak(px, meta), fig_peta_blok(blockmaps),
        fig_amplitudo(px), fig_beku(px),
    ]
    (OUT / "laporan_piksel.html").write_text(build_html(px, figs, meta), encoding="utf-8")
    print(f"\nSelesai. Buka: {OUT / 'laporan_piksel.html'}")


if __name__ == "__main__":
    main()
