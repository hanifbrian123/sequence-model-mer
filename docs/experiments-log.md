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

---

## H. PUSH v3 — ViViT video-transformer (18–19 Jul 2026, loop otonom multi-hari)

User minta arah baru: **coba ViViT** (`google/vivit-b-16x2-kinetics400`, 88.7M param, 32 frame/224/tubelet[2,16,16]) + optimasi **adaptive-finetuning** (gradual unfreeze + layer-wise LR decay/LLRD + warmup) + **Focal Loss** + **Cosine Annealing**. Tujuan akhir = software inference realtime kamera (GPU). Input dipilih user = **RGB mentah ONSET-FREE** (deploy tanpa spotting onset). ~17 jam/run LOSO (RTX 3060, batch8, grad-checkpoint).

**Temuan onset-gap (dari user):** benchmark LOSO onset-aligned (clip mulai di onset) → semua angka historis optimistik vs realtime. Diukur via `rand_start`/`eval_rand_start` (Track C).

| # | Nama | Perubahan | UF1 | UAR | ACC | Verdict |
|---|---|---|---|---|---|---|
| 30 | iter_30_vivit_rgb | ViViT-B, RGB onset-free, adaptive-FT+focal+cosine | 0.2922 | 0.3094 | 0.3577 | ❌ **overfit identitas** — train_acc 0.93 tapi pooled 0.358; surprise F1 kolaps 0.19 (flow≈0.92). RGB appearance ≠ micro-motion subject-independent. ViViT+adaptive-FT TAK menyelamatkan RGB. |

**Diagnosis iter_30:** mengulang kegagalan iter_01 (RGB mentah 0.30) dengan model lebih besar. Fitur Kinetics-RGB menghafal wajah/identitas, bukan gerak halus. Gap train(0.93)→val(0.358) = overfit klasik. per-class F1: happy 0.38 / disgust 0.33 / repress 0.11 / surprise 0.19 / others 0.45. Konfirmasi ulang: **representasi gerak (flow/motion) wajib**, bukan appearance. Kode ViViT + adaptive-FT + resize + LLRD tervalidasi (smoke OK end-to-end); jadi hasil buruk = representasi, bukan bug.

**Lanjut (gated):** iter_33 ViViT-**flow** (modalitas terbukti; uji apakah arsitektur transformer bisa saingi r3d di representasi yang menang; kandidat fusi terbaik) → iter_32 ViViT-**diff** (onset-free motion, jalur deploy) → **fusi** dgn pool r3d/mc3/focal. Best masih **r3d flow 0.7145**.

| 33 | iter_33_vivit_flow | ViViT-B, flow [u,v,mag] (modalitas terbukti) | 0.4905 | 0.4947 | 0.5000 | ❌ jauh < r3d-flow 0.7145. Surprise pulih (F1 0.77) tapi happiness kolaps (0.29). Transformer 88M < CNN 33M di 246 sampel. |

**Fusi (jujur, fixed-add vs r3d+mc3+4ch+focal=0.7177):** + iter_33 → **0.6800 (TURUN)**; tak ada subset ber-iter_33 di top search. + iter_30 juga tak membantu. **ViViT tak berguna solo MAUPUN fusi** (RGB & flow). Konfirmasi tesis proyek: di data kecil, representasi+regularisasi > kapasitas; transformer butuh data jauh lebih banyak (senada Micron-BERT 0.36). **iter_32 ViViT-diff (onset-free) menyusul untuk melengkapi gambar.** Best tetap r3d 0.7145 / ensemble 0.7177.

| 32 | iter_32_vivit_diff | ViViT-B, diff onset-free motion | — | — | — | ⏹️ **DIBATALKAN di fold-1 (acc 0.222)**. Tren ViViT sudah konklusif (RGB 0.36 + flow 0.50 + diff-fold1 0.22 semua ≪ r3d). GPU dialihkan ke eksperimen r3d onset-free (~1.7 jam/run vs ViViT 17.6 jam) yang lebih tinggi EV untuk tujuan deploy realtime user. Keputusan evidence-driven, hemat ~15 jam. |

### H.1 Pasca-ViViT — r3d onset-free (untuk deploy realtime, arsitektur terbukti)
ViViT ditutup: **transformer under-perform di 246 sampel, solo & fusi.** Pivot ke pertanyaan software user: *seberapa bagus model ONSET-FREE (tanpa spotting) memakai arsitektur juara r3d_18?* r3d ~10× lebih cepat → banyak probe muat. Onset-free reps: `diff` (selisih antar-frame dalam window), `seq-flow` (historis iter_28 = 0.56). Target: model onset-free deployable terbaik + ukur onset-gap (`eval_rand_start`).

| 34 | iter_34_r3d_diff | r3d_18, diff onset-free (img112, T16) | **0.5897** | 0.6077 | **0.5610** | ✅ **onset-free terbaik sejauh ini** > seq-flow 0.56 > semua ViViT. Balanced (surprise F1 0.89). Hanya 1.4 jam. Deployable tanpa spotting onset. |

**iter_34 signifikan:** membuktikan pivot benar — dalam waktu yang dihemat dari batal ViViT-diff, dapat model onset-free lebih baik. Onset-free ranking: **r3d-diff 0.59** > seq-flow 0.56 > ViViT-flow 0.50 > ViViT-rgb 0.36. Trade-off deploy jujur: onset-free 0.59 vs onset-ref 0.71 (r3d-flow). Per-class F1: happy 0.47 / disgust 0.54 / repress 0.53 / surprise 0.89 / others 0.53.

**★ FUSI ONSET-FREE (diff + seq-flow, fixed 2-member): UF1 0.6260 / UAR 0.6476 / ACC 0.5894** — mengalahkan kedua anggota (diff 0.59, seqflow 0.56). Dua representasi gerak onset-free yang dekorelasi → ensemble deployable TANPA spotting onset. **Ini kandidat deliverable onset-free terbaik.** Trade-off deploy jujur: onset-free ensemble 0.626 vs onset-ref r3d-flow 0.7145 (butuh spotting). Untuk software realtime user: 0.626 tanpa beban spotting = pilihan praktis.

| 35 | iter_35_r3d_diff_randstart | r3d diff + rand_start (train window tergeser) | 0.4555 | 0.4856 | 0.4472 | ❌ TURUN dari iter_34 (0.59). **Kuantifikasi onset-gap:** latih pada window tergeser (deploy-realistic) merugikan benchmark onset-aligned ~0.13 UF1. Benchmark memang menguntungkan asumsi onset-alignment. Shift-robust training TAK sepadan; ukur gap lewat eval saja. |

| 36 | iter_36_r3d_diff_res128 | r3d diff onset-free, res128 (img128) | **0.6012** | 0.5928 | 0.5813 | ✅ res128 > img112 (0.59). **Best onset-free single.** |

**★★ FUSI ONSET-FREE TERBAIK — diff128 + seq-flow (fixed 2-member): UF1 0.6574 / UAR 0.6689 / ACC 0.6423.** Naik dari 0.626 (diff112+seqflow). Menambah diff112 tak membantu (korelasi dg diff128). **Ini DELIVERABLE onset-free final** — deployable realtime TANPA spotting onset.

**Trade-off deploy (jujur):**
| Model | Butuh spotting onset? | UF1 | ACC |
|---|---|---|---|
| r3d flow ensemble (juara akurasi) | ✅ ya | 0.7177 | 0.691 |
| **diff128 + seq-flow (onset-free)** | ❌ tidak | **0.6574** | **0.6423** |

Hanya ~0.06 UF1 untuk melepas syarat spotting onset → sangat layak untuk kamera realtime. Ini hasil konkret dari pivot pasca-ViViT.

| 37 | iter_37_r3d_diff_evalshift | r3d diff, EVAL pada window tergeser (eval_rand_start) | 0.5773 | 0.5884 | 0.5569 | ✅ **ONSET-GAP KECIL** — vs iter_34 onset-aligned 0.5897 hanya turun ~0.012 UF1 (~2%). |

**★ Kesimpulan onset-gap (jawaban untuk kekhawatiran user):** model **diff onset-free bersifat SHIFT-ROBUST** — dilatih onset-aligned, dievaluasi pada window sembarang (non-onset) hanya turun ~2%. Sebab: diff = selisih antar-frame DALAM window (gerak relatif), tak bergantung di mana window mulai. **Resep deploy: latih onset-aligned (terbaik), deploy pada window apa pun (robust).** Ini membuat diff ideal untuk realtime kamera: onset-free + shift-robust + murah (tanpa hitung flow). (Bandingkan iter_35: menggeser window TRAINING merusak 0.46 — jangan; cukup geser eval.)

| 38 | iter_38_mc3_diff_res128 | mc3_18, diff onset-free res128 | **0.6086** | 0.6012 | 0.5935 | ✅ > r3d-diff128 (0.60). Anggota onset-free beragam untuk fusi. |

**Fusi onset-free (dgn mc3-diff):** mc3-diff+seqflow=**0.6582**/ACC0.6341 ≈ diff128+seqflow=0.6574/ACC**0.6423** (SERI). 3-arah (r3d-diff+mc3-diff+seqflow)=0.6408 (TURUN — dua diff-model korelasi, saling meniadakan). **Plafon onset-free ~0.657**: hanya seqflow yang ortogonal terhadap diff; ragam arsitektur pada diff tak menambah (redundant). Deployed tetap diff128 (murah+robust). iter_39 r2plus1d-diff menyusul (diperkirakan konfirmasi pola).

| 39 | iter_39_r2plus1d_diff | r2plus1d_18, diff onset-free res128 | 0.5785 | 0.5738 | 0.5569 | ➖ anggota diff terlemah; korelasi dg r3d/mc3-diff → tak menambah fusi. |
