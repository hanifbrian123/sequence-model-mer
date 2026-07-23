# Laporan ViViT — CASME II Micro-Expression (Estimated Emotion 5-kelas)

**Tanggal:** 18–20 Jul 2026 · **Protokol:** LOSO 26-fold subject-independent, metrik pooled (UF1/UAR/ACC) · **Status:** RGB & flow selesai; diff berjalan.

> **TL;DR (jujur):** ViViT-B (video transformer, 88.7M param, Kinetics-pretrained) dengan **adaptive-finetuning + Focal Loss + Cosine Annealing** **TIDAK mengalahkan** CNN 3D `r3d_18` (33M) yang jadi juara proyek (UF1 0.7145 / ACC 0.687). Pada 246 sampel, transformer besar overfit; representasi gerak (flow) + arsitektur kompak tetap unggul. ViViT juga **tidak membantu fusi**. Kesimpulan sejalan dengan tesis proyek (di data kecil: representasi + regularisasi > kapasitas).

---

## 1. Motivasi & permintaan

User meminta arah baru: **coba ViViT**, lalu optimasi dengan **adaptive finetuning**, **Focal Loss**, dan **Cosine Annealing** untuk learning rate. Tujuan akhir = software inference **realtime kamera (GPU)**. Untuk deploy realtime, user memilih input **RGB mentah onset-free** (tanpa perlu mencari frame onset / spotting).

## 2. Metode — dari input Cropped sampai prediksi

```
dataset/Cropped/subXX/EPxx_yy/reg_imgN.jpg   (frame wajah teregistrasi)
  → cache/frames128 (RGB uint8 L×128×128×3) atau cache/flow144 (TV-L1 onset-ref flow)
  → SeqDataset: sampling T=32 frame + augmentasi (hflip, color-jitter, temporal-jitter, random-erase)
  → resize 128→224, normalisasi Kinetics
  → ViViT-B (google/vivit-b-16x2-kinetics400), pixel_values (B,32,3,224,224)
  → ADAPTIVE FINETUNING per fold LOSO:
      • Gradual unfreeze: 2 epoch head-only (linear-probe) → buka backbone
      • Layer-wise LR decay (LLRD, decay 0.85): layer awal LR kecil, head LR besar (3e-3)
      • Linear warmup (3 ep) → Cosine Annealing ke 0
      • Focal Loss (γ=1.5) + class-weight capped → serang kelas "others"
  → TTA×5 → pooled LOSO (UF1/UAR/ACC)
```

**T=32 & 224²** dipilih agar cocok dengan positional-embedding ViViT-B pretrained (32 frame, tubelet 2 → 16 token temporal; 224/16 → 14×14 spasial) — transfer tanpa interpolasi PE. Runtime ≈ **17.6 jam/run** (RTX 3060 12GB, batch 8, gradient-checkpointing). Semua kode ter-gate config; jalur CNN/flow proyek tak berubah. Pipeline tervalidasi smoke end-to-end (bukan bug).

## 3. Hasil (LOSO pooled)

| Model | Input | UF1 | UAR | ACC | Catatan |
|---|---|---|---|---|---|
| **r3d_18 (juara)** | flow onset-ref | **0.7145** | 0.721 | **0.687** | CNN 3D 33M — baseline yang harus dikalahkan |
| ViViT-B | RGB mentah (onset-free) | 0.2922 | 0.3094 | 0.3577 | ❌ overfit identitas (train 0.93 → val 0.36); surprise F1 kolaps 0.19 |
| ViViT-B | flow [u,v,mag] | 0.4905 | 0.4947 | 0.5000 | ❌ surprise pulih (F1 0.77), happiness kolaps (0.29); tetap ≪ r3d |
| ViViT-B | diff (onset-free) | _dibatalkan_ | | | fold-1 acc 0.222 → tren konklusif, GPU dialihkan (lihat §7) |

**Fusi (late-fusion probs, jujur — fixed-add ke ensemble pre-committed r3d+mc3+4ch+focal = 0.7177):**
- + ViViT-RGB → tidak naik (subset terbaik dengannya 0.7108 < baseline)
- + ViViT-flow → **0.6800 (TURUN)** — anggota lemah menyeret ensemble
- Tidak ada subset yang mengandung ViViT muncul di top hasil pencarian.

→ **ViViT tidak berguna solo maupun sebagai anggota fusi.**

## 4. Diagnosis — kenapa ViViT gagal di sini

1. **Data terlalu kecil (246 sampel).** Transformer 88.7M punya induktif-bias lemah (tak ada lokalitas/translation-equivariance konvolusi) → butuh data besar. Di 246 sampel ia menghafal, bukan menggeneralisasi. ViViT-RGB: train_acc 0.93 tapi pooled 0.358 = gap overfit klasik — mengulang kegagalan iter_01 (RGB mentah + CNN = 0.30) dengan model lebih besar.
2. **RGB appearance ≠ micro-motion.** Fitur Kinetics-RGB menonjolkan penampilan/identitas; ekspresi mikro adalah gerak halus sub-pixel. Flow (gerak murni) memperbaiki surprise (F1 0.19→0.77) tapi tetap kalah dari r3d.
3. **Adaptive-FT/Focal/Cosine sudah diterapkan penuh** (gradual unfreeze + LLRD + warmup + focal γ1.5) — mereka membantu kestabilan tapi tak mengubah kesimpulan: kapasitas besar bukan penghambatnya, melainkan kurangnya data + representasi.

Konsisten dengan probe tetangga (Micron-BERT 0.36, HTNet 0.556 di protokol sama) — pendekatan transformer/berat under-perform di CASME II 5-kelas LOSO.

## 5. Catatan onset-gap & deploy realtime (temuan penting user)

Benchmark LOSO ini **onset-aligned** (tiap clip mulai di frame onset/netral). Realtime kamera tidak begitu → ada **train/deploy gap**: SEMUA angka historis (termasuk r3d 0.72) sedikit **optimistik** vs realtime. Ini diukur via `rand_start` (training window tergeser) + `eval_rand_start` (evaluasi window tergeser) — lihat Track C.

Implikasi deploy:
- **RGB onset-free** paling mudah deploy (tanpa spotting) TAPI paling lemah akurasi (0.358).
- **flow onset-ref** paling akurat TAPI butuh spotting onset saat deploy (beban nyata).
- Alternatif onset-free ber-motion (diff / sequential-flow) = kompromi (seq-flow r3d historis ≈ 0.56).

## 7. PIVOT pasca-ViViT — model ONSET-FREE deployable (arsitektur terbukti r3d)

Setelah ViViT terbukti gagal pada RGB (0.36) & flow (0.50), dan fold-1 ViViT-diff = 0.22, GPU dialihkan dari melengkapi dead-end (17 jam) ke pertanyaan software user: **seberapa bagus model ONSET-FREE (tanpa spotting) memakai r3d_18?** r3d ~10× lebih cepat (~1.5 jam/run vs 17.6 jam), jadi banyak probe muat.

| iter | Model onset-free | UF1 | UAR | ACC |
|---|---|---|---|---|
| 28 | r3d seq-flow | 0.5602 | 0.5980 | 0.5244 |
| 34 | r3d diff (img112) | 0.5897 | 0.6077 | 0.5610 |
| 36 | r3d diff (res128) | 0.6012 | 0.5928 | 0.5813 |
| **★** | **diff128 + seq-flow (fusi 2-member)** | **0.6574** | **0.6689** | **0.6423** |

**Trade-off deploy (jujur):**
| Model | Butuh spotting onset? | UF1 | ACC |
|---|---|---|---|
| r3d flow ensemble (juara akurasi) | ✅ ya (spotting = beban realtime) | 0.7177 | 0.691 |
| **diff128 + seq-flow (onset-free)** | ❌ **tidak** | **0.6574** | **0.6423** |

Hanya **~0.06 UF1** untuk melepas syarat spotting onset sepenuhnya → sangat layak untuk kamera realtime. Inilah hasil konkret dari pivot.

**Onset-gap (kuantifikasi temuan user) — HASIL:** model diff dilatih onset-aligned, **dievaluasi pada window tergeser (non-onset) hanya turun ~2%** (iter_37: 0.5897→0.5773 UF1). Diff = selisih antar-frame DALAM window (gerak relatif) → **inheren shift-robust**, tak bergantung di mana window mulai. Sebaliknya, menggeser window TRAINING (`rand_start`, iter_35) merusak (0.59→0.46) — jadi jangan. **Resep deploy: latih onset-aligned, deploy pada window apa pun.** Diff = ideal realtime: onset-free + shift-robust + murah (tanpa hitung flow).

## 8. Kesimpulan & rekomendasi

- **ViViT TIDAK direkomendasikan** untuk CASME II 5-kelas dengan data saat ini (transformer 88M overfit di 246 sampel; senada Micron-BERT 0.36). Juara akurasi tetap **r3d_18 flow ensemble 0.7177** (butuh spotting onset).
- **Untuk deploy realtime onset-free (tujuan user): pakai ensemble diff128 + seq-flow (UF1 0.657 / ACC 0.642)** — tanpa spotting, hanya ~0.06 di bawah juara. Ini rekomendasi deploy praktis.
- Untuk **menembus 0.72** perlu **DATA tambahan** (komposit SAMM/SMIC), bukan arsitektur/tuning baru — plateau terkonfirmasi ulang dari sudut transformer.

## 9. Deliverable onset-free (dibangun & tervalidasi)

- **`models/onsetfree_diff/`** — r3d_18 diff128, 3-seed (train_final di SEMUA data) + `deploy.json` (`input_mode:diff`, onset-free).
- **`src/infer_onsetfree.py`** — inferensi sliding-window T-frame → frame-diff → r3d ensemble + TTA. **TANPA onset/spotting.** Tervalidasi: `sub01/EP02_01f` (happiness) → prediksi **happiness p=0.812** ✓ (in-sample, memastikan pipeline; angka generalisasi = LOSO 0.60 single / 0.657 ensemble).
- Cara pakai: `python src/infer_onsetfree.py --frames <folder-frame> --models models/onsetfree_diff`.
- **Opsi akurasi lebih tinggi (0.657):** tambah model seq-flow + rata-ratakan (butuh hitung TV-L1 flow sekuensial saat inferensi — lebih berat; diff-only lebih murah & ~shift-robust untuk realtime).

*Catatan: diff-single (0.60) dipilih sebagai jalur realtime utama (murah + shift-robust ~2%); ensemble diff+seqflow (0.657) = opsi akurasi-lebih-tinggi.*
