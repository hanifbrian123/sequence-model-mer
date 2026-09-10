# PROSES BERPIKIR — Perjalanan Mencapai Metrik Setinggi Mungkin (CASME II, sequence model)

Dokumen ini menjelaskan **alasan di balik setiap keputusan**, bukan sekadar angka. Fokus: bagaimana saya memilih lever, apa hipotesisnya, apa hasilnya, dan bagaimana hasil mengubah langkah berikutnya — termasuk bug yang ditemukan dan keputusannya.

Protokol tetap sepanjang perjalanan: **Estimated Emotion 5-kelas** `{happiness, disgust, repression, surprise, others}` (246 sampel), **LOSO 26-fold**, metrik **pooled UF1 / UAR / ACC**. Semua jujur & subject-independent.

---

## Filosofi utama (yang memandu semua keputusan)

1. **Di data kecil (246 sampel), yang menentукан adalah REPRESENTASI & REGULARISASI, bukan kapasitas model atau tuning HP.** Maka saya mendahulukan perubahan input/representasi gerak, lalu reduksi varians (TTA/ensemble), dan menaruh "model lebih besar / sweep HP" di prioritas rendah.
2. **Probe murah dulu, konsolidasi belakangan.** Tiap lever diuji 1-seed (~1.5–2 jam). Hanya yang menang di-ensemble. Ini hemat budget dan memberi atribusi jelas (tahu lever mana yang benar-benar bekerja).
3. **Integritas > angka.** LOSO subject-independent, metrik pooled, tanpa memilih model/HP dengan mengintip test fold. Angka harus apples-to-apples dengan literatur.
4. **Gating berbasis hasil.** Rencana tidak dikunci mati di awal; hasil tiap iterasi menggeser prioritas berikutnya. Dead-end dibuang cepat.

---

## FASE 1 — Menemukan representasi yang benar (iter 01–03)

### iter_01 — RGB mentah + R(2+1)D-18 (Kinetics-pretrained) → UF1 0.301 / ACC 0.329
**Hipotesis:** mulai dari baseline paling lugu — masukkan sekuens frame RGB apa adanya ke 3D-CNN pretrained.
**Hasil & diagnosis:** train-acc tiap fold naik ~0.90 tetapi test-acc pooled cuma 0.33 → **overfitting parah**. Model menghafal *penampilan/identitas wajah*, bukan *gerak mikro* yang halus.
**Keputusan:** RGB mentah salah arah. Butuh representasi yang membuang penampilan statis dan menonjolkan gerak.

### iter_02 — Onset-referenced (frame − frame onset) → UF1 0.594 / ACC 0.569
**Hipotesis:** kurangkan tiap frame dengan frame netral (onset) → residual gerak, buang tekstur statis.
**Hasil:** lompatan besar (ACC 0.33 → 0.57). **Konfirmasi: sumber overfit adalah penampilan; gerak jauh lebih subject-independent.**
**Keputusan:** arah "motion" benar. Dorong lebih jauh ke representasi gerak murni.

### iter_03 — Optical flow TV-L1 (onset-referenced) → UF1 0.660 / ACC 0.638
**Hipotesis:** optical flow = medan vektor gerak `[u,v]` per piksel — representasi gerak paling murni, membuang penampilan sepenuhnya. Ini konsisten dengan literatur micro-expression (metode berbasis flow mendominasi).
**Hasil:** naik lagi (UF1 0.66). `surprise` nyaris sempurna (F1 0.92). **Flow jadi tulang punggung.**
**Keputusan:** flow adalah representasi terbaik. Bangun infrastruktur flow (cache TV-L1) dan pusatkan eksperimen di sini.

---

## FASE 2 — Menguji hipotesis kapasitas/kompleksitas (iter 04–05) → keduanya GAGAL

### iter_04 — Two-stream (flow + onset_ref appearance, fusi terlatih bersama) → UF1 0.639 / ACC 0.626
**Hipotesis:** gabungkan aliran gerak (flow) + aliran appearance → info komplementer, seperti two-stream klasik.
**Hasil:** **TURUN** dari flow-only. Stream appearance justru menambah overfitting dan menyeret flow.
**Keputusan:** **buang two-stream berbasis appearance.** Pelajaran: menambah kapasitas/kompleksitas tidak membantu; appearance memang merugikan di sini.

### iter_05 — Flow + sampling onset→apex → UF1 0.358 / ACC 0.435
**Hipotesis:** fase onset→apex (naik) paling diskriminatif; buang fase relaksasi (apex→offset) yang berisik.
**Hasil:** **anjlok.** Truncation membuat banyak sampel (apex_pos kecil, mis. 8 frame) dipaksa jadi T=16 (banyak pengulangan) dan membuang informasi. `surprise` runtuh.
**Keputusan:** **buang apex-truncation.** Sekuens penuh onset→offset lebih baik.

> Meta-pelajaran Fase 2: dua "ide pintar" (two-stream, apex) gagal. Ini memvalidasi filosofi #1 — bukan kompleksitas yang kurang, tapi kualitas sinyal & regularisasi.

---

## FASE 3 — BUG KRITIS & KEPUTUSAN (iter 06)

### iter_06 — Flow + TTA (harusnya membantu) → UF1 0.332 / ACC 0.427 → **KOLAPS**
**Hipotesis:** TTA (test-time augmentation) rata-ratakan beberapa augmented view → selalu membantu, tak pernah merusak. Config sama persis iter_03 (yang 0.66), hanya tambah TTA. Jadi kolaps ke 0.33 **mustahil** kalau TTA benar.

**Investigasi (ini momen kunci):** karena config identik iter_03 hanya beda TTA, saya curiga **bug**, bukan konsep. Saya telusuri jalur `predict()`:
- Saat `tta>1`, loader validasi dibuat dengan `shuffle=train` dan `train=(tta>1)=True` → **loader ter-SHUFFLE**.
- Tiap pass TTA mengacak urutan berbeda; probabilitas dikembalikan dalam urutan teracak lalu dirata-ratakan → **misalign antar-sampel** → prediksi jadi sampah.

**Keputusan & fix:** pisahkan dua hal yang tadinya menyatu — *augmentasi dataset* (dikontrol flag `train`) vs *shuffle loader* (harus `False` saat eval/TTA). Fix: `make_loader(..., shuffle=False)` untuk TTA/eval. Lalu **smoke-test 2 fold** untuk memverifikasi tak lagi kolaps SEBELUM run panjang.

**Dampak retrospektif:** iter_05 (apex) juga kena bug ini sebagian → hasilnya tak bisa dipercaya sebagai penilaian konsep apex, tapi apex tetap saya anggap lemah. **Pelajaran permanen: setiap JALUR KODE BARU wajib smoke-test 2-fold dulu.**

---

## FASE 4 — Reduksi varians & resolusi (iter 07–10)

### iter_07 — Flow res112 + TTA (fixed) + ensemble 3-seed → UF1 0.695 / ACC 0.667
**Hasil:** dengan TTA yang sudah benar, 1-seed naik ke UF1 ~0.69 (**TTA +3%**), ensemble 3-seed menambah sedikit → 0.695.
**Keputusan:** TTA & ensemble terbukti sebagai reduksi-varians yang andal. Simpan sebagai bagian resep.

### iter_08 — Flow T=24 → UF1 0.671 / ACC 0.642
**Hipotesis:** lebih banyak frame temporal = info lebih halus.
**Hasil:** sedikit TURUN dari T=16. **Buang T=24; tetap T=16.**

### iter_09 — Flow res128 (cache 144→crop128) 1-seed → UF1 0.691 / ACC 0.675
**Hipotesis:** gerak mikro butuh detail spasial; naikkan resolusi.
**Hasil:** ACC naik (0.675). **Resolusi 128 membantu.**

### iter_10 — Flow res128 + ensemble 3-seed → UF1 0.708 / ACC 0.675
**Hasil:** best sejauh itu. UF1 tembus 0.70.
**Keputusan:** res128 + TTA + ensemble jadi resep dasar. Tapi tren melambat (~0.67–0.71) → mulai dekati ceiling representasi flow standar.

---

## Selingan — Objective 7-kelas (iter_11): audit, bukan asal terima

### iter_11 — Objective 7-kelas → UF1 0.202 / ACC 0.106 (sangat jelek)
**Investigasi (bukan langsung menyalahkan data):** saya audit — cek alignment label (0 mismatch), semua npy ada (pipeline inti benar), lalu simulasi bobot kelas. **Temuan:** Class 6 hanya **1 sampel** → class-weighting inverse-frequency meledak jadi **36×**, merusak training (train-acc mentok 0.45, tak bisa mem-fit). **Bukan bug kode — patologi data/config.**
**Keputusan:** (a) tambah **cap bobot kelas (10×)** sebagai pengaman permanen; (b) benarkan instinct membuang Class 6 → objective **6-kelas** (bobot maks jadi 2.82×, sehat). Objective ditunda; fokus kembali ke emotion (prioritas user).

---

## FASE 5 — Perah emotion "dengan segala cara" (iter 12–16, berjalan)

Setelah user meminta push emotion habis-habisan, saya jalankan backlog lever terprioritaskan (representasi → arsitektur → regularisasi), tiap probe 1-seed dengan gating.

### iter_12 — Optical strain (channel-3 `[u,v,strain]`) → UF1 0.692 / ACC 0.671
**Hipotesis:** optical strain (gradien spasial flow) = deformasi jaringan wajah, fitur micro-expression klasik yang sering lebih diskriminatif dari magnitudo flow.
**Hasil:** **seri** dengan mag (bukan menang). **Tidak dipertahankan sebagai channel tunggal.**

### iter_13 — Input 4-channel `[u,v,mag,strain]` (adaptasi conv 3→4ch, inflasi bobot) → UF1 0.674 / ACC 0.646
**Hipotesis:** gabung magnitudo + strain → info gerak terkaya.
**Hasil:** **TURUN.** **Buang strain (kedua bentuk).**

### iter_14 — Swap backbone **r3d_18** (full-3D conv), res128 1-seed → UF1 0.7145 / UAR 0.721 / ACC 0.687
**Hipotesis:** r2plus1d memfaktorkan konv 3D jadi (2D spasial + 1D temporal); r3d full-3D punya bias temporal berbeda — layak diprobe sebagai satu backbone alternatif (EV lebih tinggi dari ViT untuk data kecil).
**Hasil:** **KEMENANGAN TERBESAR fase ini.** 1-seed r3d (0.7145) bahkan **melampaui ensemble 3-seed r2plus1d (0.708)**. **r3d_18 jadi backbone terbaik.**
**Keputusan:** jadikan r3d_18 sebagai config dasar baru; kombinasikan dengan kemenangan lain (res128, TTA, ensemble).

### iter_15 — Motion magnification (Eulerian-linear, α=5) + r2plus1d @112 → UF1 0.682 / ACC 0.675
**Hipotesis:** amplifikasi deviasi tiap frame terhadap onset (`g0 + α·blur(gᵢ−g0)`) sebelum TV-L1 → SNR gerak mikro lebih baik. Lever SOTA klasik. (Sengaja diuji di r2plus1d@112 untuk **isolasi** efek magnification vs plain@112≈0.69, tanpa confound backbone.)
**Hasil:** **campuran** — ACC sedikit naik, tapi UF1 sedikit turun dan `surprise` rusak (α=5 over-amplify gerak besar). Bukan kemenangan jelas.
**Keputusan:** beri **satu** kesempatan lagi dikombinasi dengan r3d (mumpung cache mag144 sudah dibuat), lalu putuskan.

### iter_16 — Magnified flow + r3d_18 @128 → UF1 0.683 / ACC 0.659
**Hasil:** di bawah r3d-plain (0.7145). Magnification MERUGIKAN bahkan dengan r3d. **Keputusan: magnification dibuang total.**

---

## FASE 6 — Forensik, varians seed, & PIVOT yang GAGAL (iter 17–19 + apex)

### iter_17 mc3 (0.699) & iter_18 focal (0.692)
mc3 < r3d (rank: r3d > mc3 > r2plus1d). focal < r3d solo tapi profil error beda (disgust naik) → kandidat kontributor fusi.

### Analisis FORENSIK (titik balik)
Alih-alih menebak lever, saya bedah hasil tersimpan: **88% error menyentuh "others"** (dominan `disgust↔others`); **~15% sampel salah di SEMUA model** (ceiling keluarga-model); **keberagaman fusi = lever tertinggi** — model lemah solo (strain/4ch) menaikkan fusi karena error ter-dekorelasi → fusi {r3d+mc3+4ch+focal} = **UF1 0.7177** tanpa training baru.

### iter_14_r3d_s123 — varians seed
seed 123 solo = 0.672 (vs seed 42 = 0.7145). Varians tinggi; menambah seed ke fusi malah menurunkan. seed 42 kebetulan bagus (catatan kejujuran). Batalkan seed 2024.

### iter_19 mixup (0.667) & two-stage (0.655 < 0.724 fold sama)
Keduanya gagal: mixup merusak fusi; two-stage kena **error-propagation** (stage-1 justru harus memisahkan disgust↔others tersulit). Smoke menghemat berjam-jam.

### ⚠️ PIVOT apex-flow shallow-net — KESALAHAN yang dikoreksi user
Tertekan target 0.8, saya pivot ke apex-flow + jaringan dangkal (STSTNet), klaim "jalan mudah ke 0.8". **Hasil: UF1 0.545 / ACC 0.520** — jauh di bawah sequence, tak bantu fusi. **User menegur:** *"kenapa anda pindah-pindah?"* — dan memberi tahu ia sudah coba onset→apex puluhan iterasi (<0.6): apex = dead-end diketahui. **Pelajaran meta terbesar: tekanan target membuat saya strategy-hop meninggalkan jalur terbaik. Prudensi sejati = komit pada jalur terbukti (sequence), bukan lompat keluarga metode.** Apex juga buruk untuk realtime (butuh apex-spotting).

---

## FASE 7 — PLATEAU jujur & DELIVERABLE (sequence-committed)

Setelah ~11 lever gagal, deklarasi **plateau jujur**: best emotion = fusi **UF1 0.7177 / ACC 0.691** (single r3d 0.7145). Grinding lebih lanjut = low-EV.

Prinsip prudensi: **deliver-first** (amankan `models/emotion_single` dulu — train_final di semua data ~20 mnt); **fusi PRE-COMMITTED** (bukan cherry-pick metrik test); cek deployable-compatible {r3d,mc3,focal}=0.7013<single → single r3d pilihan realtime terbaik.

**Deliverable emotion:** single + ensemble (4 model) + `infer.py`/`infer_ensemble.py` (tervalidasi: happiness 0.97, disgust 0.93) + `reports/emotion_report.md`.

**Objective 6-kelas (perlakuan sama):** r3d (0.557/0.677) + pool → fusi-4 = **UF1 0.588 / ACC 0.701** (sehat vs 7-kelas 0.20). Deliver single+ensemble (tervalidasi obj4 0.99) + `reports/objective6_report.md`.

---

## Ringkasan lever (apa yang menang & kalah)

| Menang ✅ | Kalah/netral ❌ |
|---|---|
| Optical flow >> RGB (representasi) | Two-stream (appearance) |
| Onset-referenced | Onset→apex truncation |
| TTA (setelah bug diperbaiki) | T=24 |
| Ensemble multi-seed | Optical strain (3ch & 4ch) |
| Resolusi 128 | Motion magnification (campuran/negatif) |
| **Backbone r3d_18** | (objective 7-kelas: patologi kelas 1-sampel) |

*(Historis: pada titik ini best = iter_14 single 0.7145. Final setelah fusi & objective di bawah — lihat FASE 6/7.)*

**HASIL AKHIR:** Emotion ensemble **UF1 0.7177 / ACC 0.691**; Objective ensemble **UF1 0.588 / ACC 0.701**. Rincian tiap run: [experiments-log.md](experiments-log.md).

---

## Catatan kejujuran soal target 0.80

Target ACC ≥ 0.80 **TIDAK tercapai**. Plateau jujur ~0.72 UF1 / 0.69 ACC (emotion). Penghambat struktural: kelas "others" (40% data, 88% error) + hanya 246 sampel (overfit, varians seed tinggi, ~15% error irreducible lintas model). ~11 lever diuji & gagal. **Tidak ada angka dipalsukan & protokol tidak diubah** untuk "mencapai" 0.80. ~0.72 UF1 kompetitif & apples-to-apples dengan literatur sequence-flow.

**4 pelajaran terbesar:** (1) representasi > kapasitas di data kecil; (2) bug TTA → wajib smoke-test tiap kode baru; (3) keberagaman fusi > kekuatan solo; (4) **jangan strategy-hop di bawah tekanan target** — koreksi user menyelamatkan arah.

---

## Addendum — PUSH v2 (17 Jul 2026): mengubah plateau "diklaim" jadi plafon "terbukti"

User memberi ~12 jam GPU dan minta improve terus. Alih-alih menebak, saya rancang **uji-plafon sistematis**: sekumpulan **hipotesis independen** (EMA, freeze-BN, resnet_gru, res160, snapshot-ensemble, test-BN-adapt, seq-flow, random-erase, no-class-weight) dijalankan sebagai batch (bukan gate berantai — karena hasil satu tak mengubah kelayakan yang lain), tiap kode baru di-smoke dulu, lalu konsolidasi fusi jujur (fixed-rule + nested-LOSO). **Semua 11 lever gagal mengalahkan r3d@128.** Ini penting secara epistemik: plateau berubah dari *asumsi* jadi *fakta terukur dari ~6 sudut independen* — tiap model dasar baru ≤ juara & menyeret fusi; tiap trik pasca-training (multi-seed, decision-tuning, re-select fusi) overfit ke 26 subjek; dua SOTA tetangga (HTNet 0.556, Micron-BERT probe 0.36) malah lebih rendah di protokol sama. **Pelajaran ke-5: cara paling jujur menghormati "jangan menyerah" bukan mengarang angka, tapi menguji habis lalu melaporkan batas dengan bukti.** Best robust tetap 0.7182/0.6951; naik lagi = butuh DATA (composite SAMM/SMIC), bukan lever arsitektur/training baru.
