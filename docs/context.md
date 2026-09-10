# CONTEXT HANDOFF — CASME II Micro-Expression (sequence model)

Dokumen ini berisi SEMUA yang perlu diketahui agar chat baru melanjutkan pekerjaan dengan BENAR. Bahasa kerja: **Indonesian**.

> ## ✅ STATUS: PROJECT SELESAI (2026-07-16)
> Kedua task terkirim (single+ensemble model + inferensi tervalidasi + laporan):
> - **Emotion 5-kelas:** ensemble **UF1 0.7177 / ACC 0.691**; single r3d 0.7145/0.687. → `models/emotion_single`, `models/emotion_ensemble`, `reports/emotion_report.md`.
> - **Objective 6-kelas:** ensemble **UF1 0.588 / ACC 0.701**; single r3d 0.557/0.677. → `models/objective6_single`, `models/objective6_ensemble`, `reports/objective6_report.md`.
> - Resep: onset-ref TV-L1 flow (res128) + **r3d_18** + TTA×5 + late-fusion. **Target 0.80 TIDAK tercapai** (plateau jujur; "others" + 246 sampel = struktural).
> - Log lengkap semua eksperimen: [experiments-log.md](experiments-log.md). Proses berpikir: [tought-process.md](tought-process.md).
> - Inferensi app: `src/infer.py` (single, realtime-friendly), `src/infer_ensemble.py` (ensemble). apex = DEAD-END (jangan diulang).
>
> **UPDATE 17 Jul 2026 — PUSH v2 (uji-plafon ekshaustif, ~12 jam GPU):** ~11 lever lagi diuji (EMA, freeze-BN, resnet_gru, res160, snapshot-ensemble, test-BN-adapt, seq-flow, random-erase, no-class-weight, multi-seed, decision-tuning). **TAK ADA yang mengalahkan r3d@128.** Best robust final = `deployed-4 + ema` **UF1 0.7182 / ACC 0.6951** (gain EMA marginal, dalam noise). Plafon ~0.72 UF1 / ~0.695 ACC kini terkonfirmasi dari ~6 sudut; ACC ≥ 0.70 didekati (0.6992 selektif) tapi tak tembus robust. **Deployable TIDAK diubah** (masih yang tervalidasi). Detail: [experiments-log.md §G](experiments-log.md), `reports/emotion_report.md §7`, `experiments/CONSOLIDATION.txt`. Untuk naik lagi HANYA lewat data tambahan (SAMM/SMIC composite — belum ada) atau apex (langgar realtime). **Semua lever within-family sudah habis; jangan diulang.**
>
> Jika user minta lanjut: semua cache & tool utuh. Untuk naik butuh data baru (composite), bukan lever baru. Fokus jaga integritas & jangan strategy-hop.

---

## 0. MISI & ARAHAN USER

- Bangun model micro-expression **CASME II** dengan **metode sequence** (tensor `(B, C, T, H, W)`) untuk dipakai di aplikasi user.
- **Loop OTONOM**: selesai satu iterasi → analisis → langsung lanjut perbaikan berikutnya, sendiri. **JANGAN bertanya di tengah loop** (user sering pergi; pertanyaan memblok loop). Ambil keputusan sendiri dari metrik.
- **Push estimated-emotion (5-kelas) HABIS-HABISAN dengan segala cara legitim.** Berhenti push emotion HANYA jika benar-benar sudah tak mungkin naik lagi (definisi plateau di bawah). Baru setelah itu pindah ke objective.
- Di akhir emotion: berikan **deployable model (SINGLE + ENSEMBLE, keduanya) + laporan akhir**. Lalu **objective 6-kelas** diperlakukan sama (push → deploy single+ensemble → laporan).
- **Integritas wajib**: LOSO subject-independent, metrik pooled, tanpa tuning-on-test. Tetap **5-kelas standar** (user menolak varian 3-kelas).
- Logging lengkap tiap training. Metrik: **UF1, UAR, Accuracy**.
- **JANGAN pernah membuka/melihat gambar dataset** (dataset berlisensi). Proses piksel secara programatik boleh; menampilkan/Read gambar TIDAK boleh. Boleh eksplор struktur folder & file non-gambar.
- Plan file yang sudah disetujui: `C:\Users\OWNER\.claude\plans\sebelum-ke-objective-6-kelas-parallel-mochi.md`.

---

## 1. ENVIRONMENT & GOTCHAS

- Working dir / repo: `d:\AI-Projects\casmeII-new-from-sequence-model` (BUKAN git repo).
- Conda env: **`facesleuth`** (Python 3.11, torch 2.6+cu124, torchvision 0.21, CUDA aktif **RTX 3060 12GB**, pandas/cv2 4.10/sklearn/matplotlib ada).
- Jalankan script: `conda run -n facesleuth python <file.py>`.
  - **GOTCHA**: `conda run ... python -c "<multiline>"` GAGAL (NotImplementedError newline). SELALU tulis ke file `.py` lalu jalankan. Untuk snippet cepat, tulis ke scratchpad dulu.
- Dataset (JANGAN lihat gambar): `D:\AI-Projects\casmeII-facesleuth-r\dataset\`
  - `Cropped\sub01..sub26\<EPxx_yy>\reg_imgN.jpg` (frame; N selaras Onset/Offset).
  - `CASME2-coding-20140508.xlsx` (Estimated Emotion), `CASME2-ObjectiveClasses.xlsx`.
  - Ada file sampah `Desktop - Shortcut.lnk` di sub01 → sudah difilter oleh build_index (hanya baca `reg_imgN.jpg`).
- Shell: Bash tersedia. Hindari `rm -rf` (pernah kena permission-deny). Foreground `sleep` diblok; gunakan `run_in_background` + notifikasi.
- Output training panjang → selalu jalankan `run_experiment.py` / `run_ensemble.py` dengan `run_in_background: true`, redirect ke `experiments/<name>_stdout.log`. Runner juga menulis `experiments/<name>/run.log` (flush per baris).

---

## 2. DATASET & PROTOKOL

- **Estimated Emotion 5-kelas (PRIMARY):** `{happiness, disgust, repression, surprise, others}` = **246 sampel** (buang sadness=7, fear=2). Label 0..4 sesuai urutan itu.
  - Distribusi: happiness=32, disgust=63, repression=27, surprise=25, others=99.
- **Objective Classes (fallback, SETELAH emotion tuntas):** 7 kelas, 255 sampel. Class 6 hanya **1 sampel** (patologis). **User memutuskan pakai 6-kelas** (buang HANYA Class 6) → 254 sampel. Distribusi 6-kelas: {25,15,99,26,20,69}. JANGAN jalankan varian 5-kelas (buang cls2&6) — user menolak.
- **Evaluasi:** LOSO 26-fold (test = 1 subjek), kumpulkan prediksi semua fold → **pooled** UF1 (macro-F1) / UAR (macro-recall) / ACC. Ini implementasi standar & jujur.
- **Target user:** ACC ≥ 0.80. Kelas "others" (40% data, semantik campur) adalah penghambat struktural. Kejar setinggi mungkin dengan jujur; jangan memalsukan angka.

---

## 3. STRUKTUR REPO & FUNGSI TIAP FILE

```
src/
  build_index.py       parse xlsx -> cache/index_emotion.csv & index_objective.csv (validasi frame)
  preprocess.py        cache frame RGB uint8 (L,128,128,3) -> cache/frames128/  (+manifest.csv)
  flow_preprocess.py   TV-L1 onset-ref flow (L,H,W,2) float16 -> cache/flow128/ (+manifest.csv)
  magnify_preprocess.py motion-magnified flow: g_amp=g0+alpha*blur(g_i-g0), lalu TV-L1 -> cache/flow*_mag5/
  dataset.py           SeqDataset + TwoStreamDataset; sample_indices, build_rgb, build_flow, _optical_strain
  models.py            build_model (r2plus1d_18/r3d_18/mc3_18/resnet_gru), TwoStream, _adapt_in_channels (4ch)
  engine.py            train_fold (LOSO 1 fold), predict (TTA), evaluate_val (per-epoch val), class_weights (capped 10x)
  metrics.py           compute_metrics -> UF1/UAR/ACC + per-class + confusion
  run_experiment.py    LOSO orchestrator (1 config, 1 seed). Full logging. Menyimpan probs.npz, per_fold, predictions, confusion, summary, ledger row.
  run_ensemble.py      LOSO multi-seed, rata-rata probs antar-seed -> pooled metrics. (juga simpan probs.npz)
  train_final.py       latih model FINAL di SEMUA data (multi-seed) -> models/<dir>/final_seed*.pt + deploy.json
  infer.py             inferensi aplikasi: folder frame -> onset-ref TV-L1 flow -> ensemble+TTA -> prediksi
configs/               *.json per iterasi (lihat DEFAULTS di run_experiment.py)
experiments/<name>/    run.log, epoch_history.jsonl, per_fold.csv, predictions.csv, probs.npz, confusion_matrix.csv/png, summary.json
cache/                 index_*.csv + frames128/ flow128/ flow144/ flow128_mag5/ flow144_mag5/
results_ledger.csv     ringkasan semua run (root repo)
private/               dokumen context (ini)
```

### Config JSON — field penting (default di `run_experiment.py::DEFAULTS`)
`seed, T=16, img_size, base_size, modality("rgb"|"flow"|"two_stream"), input_mode("rgb"|"onset_ref"|"diff"), backbone,
cache_dir, manifest_name(default "manifest.csv"), flow_clip, flow_third("mag"|"strain"|"both"), in_channels(3|4),
temporal_span("onset_offset"|"onset_apex"), epochs, lr=1e-4, weight_decay=1e-3, batch_size, dropout=0.5,
class_weighting, label_smoothing=0.1, hflip, color_jitter, temporal_jitter, amp, eval_last_k, tta, class_names[]`
- Two-stream pakai `cache_dir_a` (flow) & `cache_dir_b` (rgb) + `stream_b_input_mode`.
- Objective: `manifest_name="manifest_obj6.csv"`, `cache_dir="cache/flow128"`, `index="cache/index_objective.csv"`, `class_names`=6 entri.

### Cara jalankan
```
conda run -n facesleuth python src/run_experiment.py --config configs/X.json            # 1 seed LOSO
conda run -n facesleuth python src/run_experiment.py --config configs/X.json --max_folds 2 --tag smoke  # smoke
conda run -n facesleuth python src/run_ensemble.py --config configs/X.json --seeds 42 123 2024          # ensemble
```
Selalu background + redirect ke experiments/<name>_stdout.log. Cek hasil: `grep -E "UF1|UAR|ACC" experiments/<name>/run.log | tail`.

---

## 4. CACHE YANG SUDAH ADA (siap pakai)

| cache dir | isi | base | catatan |
|---|---|---|---|
| cache/frames128 | RGB uint8 (L,128,128,3) | 128 | 246 (emotion) |
| cache/flow128   | TV-L1 onset-ref flow (L,128,128,2) f16 | 128 | 255 npy; manifest.csv(246 emo), manifest_objective.csv(255), manifest_obj6.csv(254), manifest_obj5.csv(239, JANGAN dipakai) |
| cache/flow144   | flow | 144 | 246 (emotion) — untuk res128 |
| cache/flow128_mag5 | magnified flow alpha=5 blur=1 | 128 | 246, **flow_clip≈7.0** |
| cache/flow144_mag5 | magnified flow alpha=5 blur=1 | 144 | SEDANG dibuat (untuk mag+res128) |

**Kalibrasi flow_clip (p99 magnitudo):** flow biasa ≈ **3.0**; flow magnified alpha=5 ≈ **7.0**. strain_clip ≈ **0.3** (p99 strain).

---

## 5. RIWAYAT ITERASI & PEMBELAJARAN (Estimated Emotion 5-kelas LOSO, pooled)

| iter | perubahan | UF1 | UAR | ACC | verdict |
|---|---|---|---|---|---|
| 01 | RGB mentah r2plus1d | 0.301 | 0.337 | 0.329 | overfit identitas |
| 02 | onset_ref (residual gerak) | 0.594 | 0.604 | 0.569 | motion menang besar |
| 03 | optical flow TV-L1 res112 | 0.660 | 0.675 | 0.638 | flow > onset_ref |
| 04 | two-stream flow+onset_ref | 0.639 | 0.659 | 0.626 | **appearance MENYERET → buang two-stream(appearance)** |
| 05 | flow + onset→apex sampling | 0.358 | — | 0.435 | **apex truncation MERUSAK → buang** |
| 06 | flow + TTA (BUGGY) | 0.332 | — | 0.427 | **BUG TTA (lihat §7) → sudah diperbaiki** |
| 07 | flow res112 + TTA + ens 3-seed | 0.695 | 0.716 | 0.667 | TTA (fixed) +3%; ensemble bantu |
| 08 | flow T=24 | 0.671 | 0.692 | 0.642 | **T=24 < T=16 → tetap T=16** |
| 09 | flow **res128** (flow144) 1-seed | 0.691 | 0.696 | 0.675 | resolusi 128 bantu ACC |
| 10 | flow res128 + ens 3-seed | 0.708 | 0.721 | 0.675 | (best r2plus1d) |
| 11 | **objective 7-kelas** | 0.202 | 0.292 | 0.106 | patologi cls6 1-sampel (bobot 36×). BUKAN bug. |
| 12 | optical strain 3ch (res128) | 0.692 | 0.692 | 0.671 | seri dg mag → buang strain |
| 13 | 4ch [u,v,mag,strain] | 0.674 | 0.674 | 0.646 | turun → buang strain |
| 14 | **backbone r3d_18** res128 1-seed | **0.7145** | 0.721 | 0.687 | **BEST SINGLE** |
| 15–19 | magnification/mc3/focal/mixup/two-stage | 0.55–0.70 | | | semua < r3d → dibuang |
| ★ | **FUSION {r3d+mc3+4ch+focal}** | **0.7177** | 0.724 | **0.691** | **BEST EMOTION (final)** |

**Riwayat LENGKAP semua ~25 run (emotion + objective + apex + fusi): [experiments-log.md](experiments-log.md).**

**Lever TERBUKTI (resep final):** flow>>rgb; **r3d_18**; res128; TTA(fixed); late-fusion beragam; class-weight cap.
**Lever GAGAL (JANGAN diulang):** two-stream(appearance), onset→apex, T=24, strain(3/4ch), magnification, mc3<r3d, **apex-shallow-net (0.545)**, two-stage(error-prop), mixup, extra-seeds(varians), focal-solo, objective-7kelas(patologi).

---

## 6. PROSES BERJALAN

**PUSH v3 — ViViT (iter_30+), AKTIF sejak 2026-07-18 (loop otonom multi-hari, user pergi 2–5 hari).**
- Arah user: coba **ViViT-B** (`google/vivit-b-16x2-kinetics400`, HF `transformers` sudah di-install) + **adaptive-finetuning** (gradual unfreeze + LLRD + warmup) + **Focal Loss** + **Cosine Annealing**. Input = **RGB mentah onset-free** (pilihan user demi deploy realtime GPU tanpa spotting).
- Kode baru (ter-gate cfg, jalur CNN/flow tak berubah): `models.py::ViViTWrapper` (+LLRD, gradual unfreeze); `dataset.py` `resize_to`/`rand_start`/`eval_rand_start`; `engine.py` LLRD-optimizer + gradual-unfreeze + warmup→cosine (`SequentialLR`) + `grad_accum`.
- **iter_30_vivit_rgb SELESAI (17.6 jam): UF1 0.2922 / UAR 0.3094 / ACC 0.3577 = GAGAL.** train_acc 0.93 tapi pooled 0.358 → overfit identitas (ulangi iter_01). RGB appearance ≠ micro-motion; ViViT+adaptive-FT tak menyelamatkan. Kode tervalidasi (bukan bug).
- **iter_33_vivit_flow SELESAI (17.8 jam): UF1 0.4905 / UAR 0.4947 / ACC 0.5000.** Jauh < r3d-flow 0.7145. Surprise pulih (F1 0.77), happiness kolaps (0.29). **Fusi fixed-add ke 0.7177 → TURUN 0.6800.** ViViT tak berguna solo maupun fusi (RGB & flow).
- **Sedang jalan:** `iter_32_vivit_diff` FULL LOSO (diff onset-free motion, ~17 jam) — melengkapi gambar ViViT (3 representasi: rgb/flow/diff).
- **KESIMPULAN SEMENTARA ViViT:** transformer 88M < CNN 33M di 246 sampel (senada Micron-BERT 0.36). Untuk naik butuh DATA, bukan arsitektur. Laporan: `reports/vivit_report.md`.
- **PIVOT pasca-ViViT SUKSES (r3d onset-free, ~1.5 jam/run):**
  - iter_34 diff112=0.5897 · iter_35 diff+randstart=0.4555 (❌ geser train merusak) · iter_36 **diff128=0.6012** (best single).
  - **★ FUSI onset-free diff128+seqflow = UF1 0.6574 / ACC 0.6423** (fixed 2-member) — DELIVERABLE onset-free, TANPA spotting. Trade-off: vs r3d-flow onset-ref 0.7177 hanya ~0.06 UF1 untuk melepas spotting.
  - iter_37 (eval_rand_start): **onset-gap hanya ~2%** (0.5897→0.5773) — diff **shift-robust** (gerak relatif dalam window). Resep: latih onset-aligned, deploy window apa pun.
- **Deliverable onset-free (sedang dibangun):** `train_final iter_36 → models/onsetfree_diff` (3-seed) + `src/infer_onsetfree.py` (sliding-window diff, onset-free, murah realtime). train_final.py deploy.json sudah + `input_mode`/`resize_to`. Uji: `infer_onsetfree.py --frames <Cropped/subXX/EPxx> --models models/onsetfree_diff`.
- iter_38 mc3-diff=0.6086 · iter_39 r2plus1d-diff=0.5785 — anggota diff saling korelasi; **plafon onset-free ~0.657 terkonfirmasi** (hanya seqflow ortogonal; ragam arsitektur diff tak menambah).
- **Standings final:** juara akurasi = r3d-flow ensemble **0.7177** (butuh spotting); juara ONSET-FREE realtime = diff128+seqflow **0.6574 / ACC 0.6423** (tanpa spotting). ViViT = gagal semua (0.36–0.50). **Deliverable onset-free dibangun+tervalidasi** (`models/onsetfree_diff` + `src/infer_onsetfree.py`).
- **LOOP DIKONSOLIDASI (2026-07-20):** ruang lever tereksplor tuntas (ViViT family + onset-free r3d family). Kedua plafon (0.72 onset-ref, 0.657 onset-free) DATA-limited, bukan arsitektur. Berhenti spinning run low-EV (jujur > busywork). Laporan lengkap: `reports/vivit_report.md`. Jika user minta lanjut spesifik: lever tersisa EV-rendah/butuh preprocessing (magnified-diff, two-stream diff+seqflow butuh cache res-cocok, better seqflow).
- **Catatan onset-gap (temuan user):** benchmark LOSO onset-aligned → angka historis optimistik vs realtime. Ukur & kecilkan (rand_start + shifted-eval).
- **~1 run ViViT ≈ 17 jam** di RTX 3060 (32×224, batch8, grad-checkpoint). Detail rencana: `C:\Users\OWNER\.claude\plans\lihat-yang-ada-di-starry-graham.md` + memory [[casme2-push-v3]].
- Best masih **iter_14_r3d UF1 0.7145** — belum ada ViViT yang mengalahkan (pending hasil).

Tool lama tetap: `src/fuse.py`, `src/run_ensemble.py`, `src/train_final.py`, `src/infer_ensemble.py`. Apex (`run_apex.py`, `apex_dataset.py`, `shallow_models.py`) = jalur mati, abaikan.

---

## 7. BUG & FIX PENTING (jangan sampai regресi)

1. **TTA shuffle-misalignment (FIXED):** dulu `predict()` membuat val-loader dgn `shuffle=train` saat `train=(tta>1)` → probs teracak, misalign sampel → metrik jadi sampah (iter_05/06 kolaps). **Fix:** `make_loader(..., shuffle=False)` untuk TTA/eval; augmentasi (train=True) DIPISAH dari shuffle loader. Jangan gabungkan lagi.
2. **Class-weight explosion (FIXED):** kelas 1-sampel → bobot inverse-freq 36× merusak training. **Fix:** `class_weights(..., cap=10.0)` clip ke [1/10,10]. Tak memengaruhi emotion (bobot <2.5×).
3. **Prinsip verifikasi:** setiap JALUR KODE BARU → smoke-test 2 fold (`--max_folds 2 --tag smoke`) SEBELUM LOSO penuh. (Pelajaran dari bug TTA.)
4. Audit objective (7-kelas) menunjukkan 0 label-mismatch, semua npy ada → pipeline inti BENAR; kejelekan objective murni patologi kelas.

---

## 8. RENCANA LANJUTAN — ✅ SUDAH DIEKSEKUSI SEMUA (hasil di [experiments-log.md](experiments-log.md)); backlog historis di bawah.

## 8. RENCANA LANJUTAN (backlog lever, result-gated). Tiap probe 1-seed ~1.5-2 jam; smoke dulu jika kode baru.

**Segera (GPU) setelah iter_15:**
1. **r3d_18 ENSEMBLE res128** (3-5 seed) via run_ensemble → kunci best baru (~0.72+). Config: copy iter_14_r3d.json.
2. **mag + r3d @128** (pakai cache/flow144_mag5, flow_clip=7, backbone r3d_18, img128) — HANYA jika iter_15 menunjukkan magnification membantu (mag@112 > plain@112≈0.69). Jika iter_15 turun jelas → buang magnification.
3. **mc3_18** backbone probe (res128).

**Fase C (regularisasi):** augmentasi lebih kuat (rotasi/skala/random-erasing), **mixup/cutmix** temporal, **focal loss / class-balanced** (serang "others"), warmup+cosine, ensemble 5-7 seed.
**Fase D ("others" bottleneck):** two-stage (biner others-vs-ekspresi → 4-kelas) lalu gabung.
**Fase E konsolidasi:** late-fusion probs lintas config terbaik (r3d + mag + dst; `probs.npz` sudah disimpan tiap run).

**Definisi PLATEAU (baru boleh stop push emotion):** semua lever Fase A-D EV tinggi/menengah sudah dicoba; DAN ensemble multi-config/seed terbaik tak naik >0.5% UF1 selama ~5 lever/kombinasi baru terakhir; DAN pendekatan "others" sudah dicoba.

**Setelah plateau → DELIVERABLE EMOTION:**
- `train_final.py --config <best> --seeds 42 --out models/emotion_single` (SINGLE) dan `--seeds 42 123 2024 --out models/emotion` (ENSEMBLE). Menghasilkan `final_seed*.pt` + `deploy.json`.
- Validasi `infer.py --frames <satu folder sekuens> --models models/emotion` → cek output masuk akal.
- Tulis `reports/emotion_report.md` (+ opsional Artifact): metodologi, protokol LOSO, tabel riwayat iterasi, UF1/UAR/ACC terbaik, per-kelas, confusion matrix, trade-off single vs ensemble, positioning jujur vs literatur, cara pakai infer.py.

**Lalu OBJECTIVE 6-kelas (perlakuan sama):** push (manifest_obj6.csv, cap bobot aktif) → deliverable single+ensemble (models/objective6/) + reports/objective6_report.md.

---

## 9. LANGKAH PERTAMA UNTUK CHAT BARU (jika project dilanjut/direvisi)

1. Baca file ini (banner atas) + memory (MEMORY.md, casme2-project.md, casme2-user-prefs.md) + [experiments-log.md](experiments-log.md).
2. Project SELESAI — deliverable siap di `models/` + `reports/`. Uji: `conda run -n facesleuth python src/infer.py --frames <folder> --models models/emotion_single`.
3. Jika user minta perbaikan: hormati komitmen SEQUENCE (jangan apex/strategy-hop), integritas LOSO, smoke-test kode baru. Best config = `configs/iter_14_r3d.json` (r3d res128). Ide tersisa EV-rendah: random-erase aug, class-balanced sampling, tuning HP hati-hati.
4. Objective 6-kelas juga sudah tuntas (`obj6_*` configs, `manifest_obj6.csv`).
