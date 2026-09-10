# Handoff detail — session Codex 22 Juli 2026

Dokumen pendamping `CLAUDE.md`. Berisi detail yang tidak perlu dimuat setiap sesi.
Baca bagian yang relevan saja sesuai kebutuhan tugas.

Sumber: transkrip session Codex `rollout-2026-07-22T12-43-21-019f8859`, 1620 event,
05:43–13:07 WIB, berakhir karena `usage_limit_exceeded`.

---

## 1. Inventaris file yang dibuat/diubah

### Infrastruktur protokol (baru)

| File | Fungsi |
|---|---|
| `src/protocol_v2.py` | Definisi split development/audit subject-independent + fold generator |
| `src/run_protocol_v2.py` | Runner eksperimen: training 4 fold dev, logging, checkpoint, artifact |
| `src/compare_protocol.py` | Gate statistik: paired subject-bootstrap, ΔUF1/ΔUAR/ΔACC + CI + P(improve) |
| `src/fuse_protocol.py` | Fusion probabilitas OOF antar-run dengan bobot tetap |
| `src/reevaluate_protocol.py` | Evaluasi ulang dari snapshot checkpoint tanpa training ulang |
| `src/lock_protocol_champion.py` | Kunci champion (termasuk ensemble: leaf model + bobot fusion) sebelum audit |
| `src/analyze_apex_estimator.py` | Validasi agregat estimator apex label-free pada dev split |
| `src/inference_utils.py` | Resolusi sample untuk deployment (apex label-free vs annotation) |
| `src/flow_preprocess_hq.py` | Preprocessing TV-L1 high-quality + ECC stabilization |
| `src/flow_pipeline.py` | Pipeline flow |

### File yang diperbaiki

| File | Perubahan |
|---|---|
| `src/engine.py` | TTA deterministik (`deterministic_tta_views`), evaluasi tidak lagi memanggil augmentasi training, monitoring tidak mengubah RNG training |
| `src/dataset.py` | `eval_rand_start` berfungsi di jalur TTA; `build_flow(..., roi=)` untuk soft face-ellipse mask; `resolve_apex_position()`; `estimate_apex_from_flow()` |
| `src/train_final.py` | Parity dengan evaluator: menghormati focal loss, EMA, snapshot, last-k averaging |
| `src/infer.py`, `src/infer_onsetfree.py`, `src/infer_ensemble.py`, `app_predict.py` | Parity preprocessing + TTA dengan evaluasi |

### Test (semua harus tetap hijau)

- `tests/test_eval_pipeline.py` — TTA deterministik, shifted-eval, no random erase
- `tests/test_training_parity.py` — train_final identik dengan resep LOSO
- `tests/test_protocol_v2.py` — split, fold, bootstrap gate (12 test)
- `tests/test_flow_pipeline.py` — pipeline flow

---

## 2. Cara menjalankan

Semua dari root project, env conda `facesleuth`.

```bash
# Smoke satu fold dulu sebelum run penuh
python src/run_protocol_v2.py --config configs/<cfg>.json --role dev --max_folds 1 --tag smoke

# Cek split tanpa training
python src/run_protocol_v2.py --config configs/<cfg>.json --role dev --dry_run --tag splitcheck

# Run development penuh 4 fold (tag p5 = protokol accuracy_v5)
python src/run_protocol_v2.py --config configs/<cfg>.json --role dev --tag p5

# Gate statistik kandidat vs baseline/champion
python src/compare_protocol.py \
  --baseline  experiments/protocol_v2/iter_14_r3d_v2_dev_p5 \
  --candidate experiments/protocol_v2/<kandidat>_v2_dev_p5 \
  --out       experiments/protocol_v2/comparisons/<nama>.json

# Fusion probabilitas
python src/fuse_protocol.py \
  --members experiments/protocol_v2/<run_a> experiments/protocol_v2/<run_b> \
  --weights 0.5 0.5 \
  --out experiments/protocol_v2/fusions/<nama> --name <nama>

# Validasi estimator apex label-free
python src/analyze_apex_estimator.py --cache_dir cache/flow144 \
  --protocol protocols/accuracy_v5.json \
  --out experiments/protocol_v2/apex_estimator_dev.json

# Preprocessing HQ + stabilization (long running)
python src/flow_preprocess_hq.py --index cache/index_emotion.csv --out_dir cache/flow144_hq_stab

# Test
python -m pytest tests/ -q
```

Catatan: Windows kadang mengunci `__pycache__` saat `py_compile`. Jalankan test
dengan bytecode dinonaktifkan bila terjadi.

### Lokasi artifact

```
experiments/protocol_v2/
  <config_name>_v2_dev_p5/          run.log, epoch_history.jsonl, snapshot, prob OOF
  fusions/<nama>/                   hasil fuse_protocol
  comparisons/<nama>.json           hasil gate bootstrap
protocols/accuracy_v5.json          definisi split dev/audit
cache/flow144/                      cache TV-L1 baseline
cache/flow144_hq_stab/              cache HQ + ECC (belum selesai, ~110/246)
```

Baseline pembanding tetap: `experiments/protocol_v2/iter_14_r3d_v2_dev_p5`
Champion saat ini: `experiments/protocol_v2/fusions/baseline_iter43_50full_50apex`

---

## 3. Hasil per fold (development split)

Fold 2 adalah subject group tersulit secara konsisten di semua kandidat.

| Run | fold 1 | fold 2 | fold 3 | fold 4 | pooled UF1 |
|---|---|---|---|---|---|
| Baseline v2 (`iter_14`) | 0,7824 | 0,4746 | 0,6700 | — | 0,6816 |
| Flow translation comp (`iter_40`) | 0,6447 | 0,4806 | 0,4978 | — | 0,5987 |
| Onset→apex s42 (`iter_43`) | 0,7166 | 0,5433 | 0,6158 | — | 0,6995 |
| Res160 (`iter_44`) | 0,7152 | 0,4999 | 0,4907 | — | 0,6254 |
| Seqflow (`iter_28`) | 0,4165 | 0,3056 | 0,3354 | — | 0,3684 |
| EMA 0,998 (`iter_20`) | 0,7109 | 0,4806 | 0,6270 | 0,8235 | 0,6725 |
| Full-span s123 | 0,7321 | 0,4687 | 0,5625 | — | 0,6163 |
| Apex s123 (`iter_45`) | 0,7166 | 0,5152 | — | — | 0,7119 |
| Auto-apex s42 (`iter_47`) | 0,7148 | — | terputus | — | — |

Rentang 0,47–0,82 pada baseline yang sama adalah alasan semua keputusan harus
memakai pooled + paired bootstrap, bukan fold tunggal.

### Detail gate yang sudah dijalankan

| Perbandingan | ΔUF1 | P(improve) | Putusan |
|---|---|---|---|
| Translation comp vs baseline | −0,0829 | 0,84% | tolak |
| Fusion translation 10/20/30% | −0,0254 dan lebih buruk | — | tolak |
| Onset→apex single vs baseline | +0,0179 | 69,3% | tolak (di bawah 80%) |
| **Fusion temporal 50:50 vs baseline** | **+0,0288** | **83,9%** | **terima → champion** |
| Res160 single | −0,0562 | — | tolak |
| Res160 fusion 10% vs champion | +0,0037 | 59,4% | tolak |
| Seqflow fusion 5% vs champion | negatif | — | tolak |
| EMA fusion 10% vs champion | +0,0037 | — | tolak (< min effect 0,005) |
| Ensemble 2-seed apex vs champion | positif | 76,1% | tolak (di bawah 80%) |

---

## 4. Akar masalah plateau (audit awal, 05:43–06:08)

Analisis dilakukan tanpa membuka dataset — hanya dari log, config, probabilitas
OOF, kode, dan artifact model.

**Statistik:**

- Baseline flow r3d lama: UF1 0,7145, tapi subject-cluster bootstrap 95% CI 0,616–0,799.
- Akurasi per subjek 0,375–1,000, SD 0,184.
- Train acc ~0,969 vs val ~0,682 → overfit jelas.
- Fusi flow lama: 0,7145 → 0,7177, paired CI −0,036…+0,045 → tidak beda dari noise.
- Seed 42 (0,7145) vs seed 123 (0,6718), beda prediksi pada 16,3% sampel.
  Memilih seed 42 karena terbaik di LOSO = winner's curse.
- Satu-satunya gain onset-free yang konsisten: diff+seqflow +0,056 (CI +0,010…+0,110),
  tapi model fusi itu tidak pernah ada di deployment.

**Masalah deployment yang ditemukan:**

- `models/onsetfree_diff` hanya berisi 3 checkpoint r3d-diff, tidak ada sequential-flow.
  Jadi angka UF1 0,657 bukan angka model yang dikirim.
- Generalisasi ensemble 3 seed tersebut belum pernah diukur dengan LOSO.
- Anggota deploy bernama `focal` sebenarnya dilatih dengan CrossEntropy.

**Masalah konseptual "onset-free":**

`preprocess.py` membangun seluruh training clip dari onset sampai offset. Jadi model
hanya "onset-free" dalam arti tidak mengurangi terhadap frame onset — ia tetap selalu
menerima window yang berisi satu micro-expression, sudah dipotong onset→offset,
sudah cropped/aligned, dan tidak pernah berisi window netral. Pada video nyata,
mayoritas window justru tidak berisi ekspresi. Kelas `others` bukan kelas netral.

---

## 5. Roadmap yang direncanakan (E0–E9)

| Tahap | Eksperimen | Gate keberhasilan | Status |
|---|---|---|---|
| E0 | Perbaiki evaluator, TTA, train-final parity | Test parity lulus | **selesai** |
| E1 | Re-run r3d flow baseline, 3 seed | Baseline v2 + CI | **selesai** |
| E2 | Global-motion compensation | Paired UF1 naik ≥0,01 | **ditolak** |
| E3 | Alignment + ROI flow | Kalahkan E2 atau error ortogonal | `iter_49` siap, belum jalan |
| E4 | High-res TV-L1 → downsample 128 | Paired improvement atas E2 | belum (beda dari res160 yang gagal) |
| E5 | Inner-CV TV-L1 tuning | Parameter terkunci sebelum outer eval | belum |
| E6 | Multi-hypothesis onset/window | Event recall naik, class UF1 tidak turun | `iter_48/48b` siap, belum jalan |
| E7 | r3d onset-flow + mc3 seqflow + ROI fusion | Delta CI atas best single positif | belum |
| E8 | Soft hierarchy/contrastive untuk `others` | Disgust↔others turun | belum |
| E9 | Self-supervised / cross-dataset pretraining | Final paired improvement | belum |
| Final | 3–5 seed + TTA deterministik + calibrated fusion | Locked audit + pilot video upload | belum |

**Penting soal E4:** eksperimen `res160` yang gagal hanya menguji "masukkan tensor
160² ke model". Yang belum diuji adalah menghitung TV-L1 pada wajah aligned resolusi
224/256, mempertahankan motion subpixel, lalu downsample flow ke 128 dengan
penyesuaian magnitude vector. Ini strategi berbeda dan belum tertutup.

---

## 6. Arsitektur target untuk video upload

```
Uploaded video
  → normalisasi FPS
  → face tracking + canonical alignment
  → kompensasi gerakan kepala / global affine
  → offline spotting: top-K kandidat event/window
  → multi-scale TV-L1 bank per kandidat
  → beberapa classifier 5-kelas
  → fusion + kalibrasi
  → pilih event/prediksi terbaik
```

Output ke pengguna tetap 5 kelas, tapi secara internal tetap perlu skor `no-event`
karena video upload bisa berisi bagian netral.

### Multi-hypothesis spotting (E6)

1. Hitung motion-energy murah pada facial ROI.
2. Temukan beberapa peak kandidat.
3. Untuk tiap peak buat beberapa kandidat onset/offset/durasi.
4. Jalankan classifier terhadap semua kandidat.
5. Gabungkan kandidat overlap dengan temporal NMS.
6. Pilih event dari gabungan event score + confidence + temporal consistency.

### Rencana untuk kelas `others` (E8)

Hard two-stage sudah pernah gagal. Rencananya soft hierarchy: shared motion encoder,
head A (expressive subtype vs heterogeneous), head B (5 kelas), auxiliary AU head bila
anotasi tersedia, supervised contrastive agar 4 kelas ekspresif membentuk cluster
rapat, dan prototype subcluster internal untuk `others`. Output akhir tetap 5 kelas.

---

## 7. Ekspektasi realistis

Dengan data yang sama (246 sampel, 26 subject), target improvement robust yang masuk
akal adalah **+0,02–0,05 UF1**, bukan lompat ke 0,80. Sumber peluang terbesar menurut
urutan: motion stabilization → multi-hypothesis window → high-resolution residual
TV-L1 → ROI/multi-stream fusion → tambahan data pretraining.

Rekomendasi eksplisit dari audit: **jangan** langsung kembali ke ViViT atau
transformer besar. Data berlabel terlalu kecil.
