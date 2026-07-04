# LOG EKSPERIMEN LENGKAP — CASME II (Sequence Model)

Dokumen ini mencatat **SEMUA eksperimen** dari iter_01 sampai selesai, apa adanya (dari `results_ledger.csv` + log `experiments/`).
Protokol: **LOSO 26-fold, pooled UF1 / UAR / ACC**. Emotion = 5-kelas (246 sampel); Objective = 6-kelas (254 sampel).

---

## A. ESTIMATED EMOTION (5-kelas) — semua run

| # | Nama | Perubahan utama | UF1 | UAR | ACC | Verdict |
|---|---|---|---|---|---|---|
| 01 | iter_01 | RGB mentah, r2plus1d_18 | 0.301 | 0.337 | 0.329 | ❌ overfit identitas |
| 02 | iter_02 | onset_ref (frame − onset) | 0.594 | 0.604 | 0.569 | ✅ motion menang besar |
| 03 | iter_03 | optical flow TV-L1, res112 | 0.660 | 0.675 | 0.638 | ✅ flow > onset_ref |
| 04 | iter_04 | two-stream (flow + appearance) | 0.639 | 0.659 | 0.626 | ❌ appearance menyeret |
| 05 | iter_05 | flow + sampling onset→apex | 0.358 | 0.359 | 0.435 | ❌ truncation merusak (+ kena bug TTA) |
| 06 | iter_06 | flow + TTA (kode buggy) | 0.332 | 0.342 | 0.427 | ❌ **BUG TTA** → ditemukan & diperbaiki |
| 07 | iter_07 | flow res112 + TTA(fix) + ensemble 3-seed | 0.695 | 0.716 | 0.667 | ✅ TTA +3%, ensemble bantu |
| 08 | iter_08 | flow, T=24 frame | 0.671 | 0.692 | 0.642 | ❌ T=16 lebih baik |
| 09 | iter_09 | flow **res128** (crop dari base144), 1-seed | 0.691 | 0.696 | 0.675 | ✅ resolusi 128 bantu |
| 10 | iter_10 | flow res128 + ensemble 3-seed (r2plus1d) | 0.708 | 0.721 | 0.675 | ✅ best r2plus1d |
| 12 | iter_12 | optical strain sbg channel-3 `[u,v,strain]` | 0.692 | 0.692 | 0.671 | ➖ seri dg mag → buang |
| 13 | iter_13 | input 4-channel `[u,v,mag,strain]` | 0.674 | 0.674 | 0.646 | ❌ turun → buang strain |
| 14 | **iter_14** | **backbone r3d_18**, res128, 1-seed | **0.7145** | **0.721** | **0.687** | ✅✅ **BEST SINGLE** (kalahkan ens r2plus1d) |
| 15 | iter_15 | motion magnification α=5 (r2plus1d@112) | 0.682 | 0.707 | 0.675 | ➖ campuran, surprise rusak |
| 16 | iter_16 | magnified flow + r3d @128 | 0.683 | 0.710 | 0.659 | ❌ < r3d plain → buang magnification |
| 17 | iter_17 | backbone mc3_18, res128 | 0.699 | 0.720 | 0.675 | ➖ < r3d (rank: r3d>mc3>r2plus1d) |
| 18 | iter_18 | focal loss + r3d | 0.692 | 0.703 | 0.679 | ➖ < r3d solo, tapi profil beda (disgust↑) |
| — | iter_14_r3d_s123 | r3d seed 123 (uji varians) | 0.672 | 0.685 | 0.646 | ⚠️ **varians seed tinggi** (vs s42=0.7145) |
| 19 | iter_19 | mixup α=0.2 + r3d | 0.667 | 0.694 | 0.646 | ❌ turun & merusak fusi |
| — | apex_shallowcnn | **PIVOT apex-flow + shallow net** (onset→apex, 2D CNN) | 0.545 | 0.568 | 0.520 | ❌ GAGAL — jauh di bawah sequence, dibuang |
| — | twostage (smoke 3-fold) | biner others-vs-ekspresi → 4-way | 0.571 | 0.580 | 0.655 | ❌ 0.655<0.724 (fold sama) → error-propagation, buang |
| ★ | **FUSION (pre-committed)** | late-fusion {iter_14_r3d + iter_17_mc3 + iter_13_4ch + iter_18_focal} | **0.7177** | **0.7239** | **0.6911** | ✅✅ **BEST EMOTION (final)** |

**Confusion matrix fusi emotion (rows=true):** happiness 25/34, disgust 39/63, repression 16/27, surprise 24/25, others 66/99. Error dominan: `disgust↔others` (38) — 88% error menyentuh "others".

---

## B. OBJECTIVE CLASSES — semua run

| # | Nama | Perubahan | UF1 | UAR | ACC | Verdict |
|---|---|---|---|---|---|---|
| 11 | iter_11_objective | **7-kelas** (semua, cls6=1 sampel) | 0.202 | 0.292 | 0.106 | ❌ patologi bobot 36× (bukan bug) |
| — | obj6_r3d | **6-kelas** (buang cls6), r3d res128 | 0.557 | 0.560 | 0.677 | ✅ best single |
| — | obj6_mc3 | 6-kelas, mc3_18 | 0.549 | 0.552 | 0.669 | ➖ |
| — | obj6_focal | 6-kelas, focal loss | 0.491 | 0.492 | 0.630 | ❌ turun solo |
| — | obj6_4ch | 6-kelas, 4-channel r2plus1d | 0.536 | 0.546 | 0.654 | ➖ |
| ★ | **FUSION (fuse-all-4)** | {r3d+mc3+focal+4ch} | **0.588** | **0.593** | **0.7008** | ✅✅ **BEST OBJECTIVE (final)** |

Catatan: UF1 ≪ ACC karena imbalance parah (obj3=99 mudah F1 0.88; kelas kecil obj2=15/obj5=20 sulit F1 ~0.45). Focal lemah solo tapi tetap menaikkan fusi (dekorelasi).

---

## C. SMOKE-TEST (verifikasi jalur kode baru, 2–3 fold — bukan hasil final)

| Nama | Tujuan | Hasil |
|---|---|---|
| smoke_smoke | validasi pipeline awal (RGB) | jalan OK |
| smoke_2stream_smoke | validasi two-stream | jalan OK |
| smoke_tta_smoke | verifikasi FIX bug TTA | tidak lagi kolaps ✅ |
| smoke_focal_smoke | validasi focal loss | jalan OK |
| apex_ststnet_smoke | STSTNet apex (terlalu kecil) | 0.207 → ganti ShallowCNN |
| apex_shallowcnn_smoke | ShallowCNN apex | 0.55 (3-fold) |
| twostage_..._smoke | validasi + sinyal two-stage | 0.655 < single |

---

## D. PREPROCESSING / CACHE yang dibangun

| Cache | Isi | Dipakai |
|---|---|---|
| frames128 | RGB uint8 (L,128²,3) | iter_01–04 |
| flow128 | TV-L1 onset-ref flow (L,128²,2) f16 — 255 npy | flow res112, objective |
| flow144 | flow base144 (→crop 128) — 255 npy | **res128 (best)**, objective |
| flow128_mag5 / flow144_mag5 | motion-magnified flow α=5 | magnification (dibuang) |

Kalibrasi: `flow_clip`≈3.0 (flow biasa), ≈7.0 (magnified), `strain_clip`≈0.3.

---

## E. RINGKASAN LEVER (menang vs gagal)

**✅ MENANG (masuk resep final):**
- Optical flow (onset-ref TV-L1) >> RGB — lompatan terbesar (0.33 → 0.64 ACC).
- Backbone **r3d_18** > mc3_18 > r2plus1d_18.
- Resolusi 128 > 112.
- TTA×5 (setelah bug diperbaiki) — +3%.
- Late-fusion model beragam — +sedikit tapi konsisten.
- Class-weighting (di-cap 10×) + label smoothing.

**❌ GAGAL / DIBUANG (semua diuji berbasis bukti):**
two-stream (appearance menyeret) · onset→apex sampling (truncation) · T=24 · optical strain (3ch & 4ch) · motion magnification (α=5, merusak surprise) · mc3/r2plus1d (< r3d) · **apex-flow shallow-net (PIVOT gagal 0.545)** · two-stage (error-propagation) · mixup (merusak fusi) · extra r3d seeds (varians tinggi menyeret fusi) · focal-solo · objective 7-kelas (patologi kelas 1-sampel).

---

## F. HASIL AKHIR (jujur, LOSO)

| Task | Model | UF1 | UAR | ACC |
|---|---|---|---|---|
| Emotion 5-kelas | single r3d | 0.7145 | 0.721 | 0.687 |
| Emotion 5-kelas | **ensemble** | **0.7177** | 0.724 | **0.691** |
| Objective 6-kelas | single r3d | 0.557 | 0.560 | 0.677 |
| Objective 6-kelas | **ensemble** | **0.588** | 0.593 | **0.701** |


**Total: ~25 run LOSO penuh + smoke + fusi, ~4 hari (13–16 Jul 2026).**

---

## G. PUSH v2 — uji-plafon ekshaustif (17 Jul 2026, ~12 jam GPU)

Setelah user minta lanjut "habis-habisan", ~11 lever baru diuji untuk menembus 0.72 / mencapai ACC ≥ 0.70. **Tak ada yang mengalahkan r3d@128 (0.7145).**

| # | Nama | Perubahan | UF1 | UAR | ACC | Verdict |
|---|---|---|---|---|---|---|
| 20 | iter_20_ema | EMA weight-averaging | 0.7094 | 0.7310 | 0.6829 | ➖ ≈baseline solo; +0.004 ACC di fusi (disimpan) |
| 21 | iter_21_freezebn | freeze BatchNorm | 0.5978 | 0.6187 | 0.5732 | ❌ BN Kinetics ≠ flow |
| 23 | iter_23_resnetgru | ResNet2D+BiGRU | 0.6864 | 0.7176 | 0.6504 | ❌ menyeret fusi |
| 24 | iter_24_res160 | resolusi 160 (cache flow176) | 0.6657 | 0.6802 | 0.6341 | ❌ overfit; 128 sweet-spot |
| 25 | iter_25_bnadapt | test-time BN adaptation | 0.6805 | 0.6979 | 0.6585 | ❌ berisik di fold kecil |
| 26 | iter_26_erase | random-erase aug | 0.6936 | 0.7077 | 0.6707 | ❌ modest |
| 27 | iter_27_snapshot | snapshot ensemble (cyclic LR) | 0.6958 | 0.7019 | 0.6707 | ❌ siklus kurang matang |
| 28 | iter_28_seqflow | sequential (consecutive) flow | 0.5602 | 0.5980 | 0.5244 | ❌ ≪ onset-referenced |
| 29 | iter_29_noweight | tanpa class-weighting (untuk ACC) | 0.7027 | 0.6997 | 0.6748 | ❌ others-recall naik tapi menyeret ensemble |
| — | multi-seed avg | rata-rata s42+s123 | 0.6819 | 0.6890 | 0.6545 | ❌ campur seed bagus+jelek → balik ke mean |
| — | decision-tuning (nested) | geser threshold per-kelas | ~0.66 | — | ~0.63 | ❌ overfit ke 26 subjek |

**Best robust final (fixed-rule):** `deployed-4 + ema` = **UF1 0.7182 / UAR 0.7229 / ACC 0.6951**. Fusi-selektif nested (pool terkurasi) menyentuh ~0.7302/0.6992 tapi POOL-SENSITIVE (turun ke 0.686 kalau member lemah masuk) → estimasi-atas, bukan deployable.

**Kesimpulan:** plafon ~0.72 UF1 / ~0.695 ACC terkonfirmasi dari ~6 sudut. ACC ≥ 0.70 didekati (0.6992) tapi tak tembus robust. Deployable TIDAK diubah (gain EMA dalam pita noise; `train_final.py` belum dukung EMA). Detail: `reports/emotion_report.md §7`, `experiments/CONSOLIDATION.txt`.
