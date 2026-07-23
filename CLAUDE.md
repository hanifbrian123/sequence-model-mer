# CLAUDE.md

Project: `casmeII-new-from-sequence-model` — klasifikasi micro-expression CASME II,
5 kelas (`happiness`, `disgust`, `repression`, `surprise`, `others`).

Detail lengkap (inventaris file, perintah CLI, hasil per-fold, roadmap E0–E9,
arsitektur target) ada di `docs/handoff-codex-2026-07-22.md`. Baca file itu saat
butuh konteks lebih dalam — jangan diasumsikan sudah termuat di sini.

## Aturan keras

- **JANGAN membuka, membaca, atau menampilkan isi dataset** — dataset berlisensi.
  Pemrosesan hanya lewat pipeline programatik yang sudah ada di project.
- Gunakan conda env **`facesleuth`** untuk semua eksekusi Python.
- Shell PowerShell, GPU RTX 3060 (~9,7 GB VRAM bebas).
- **Audit split (5 subject) masih tersegel dan belum pernah disentuh.** Jangan
  mengevaluasi apa pun di sana sampai champion final terkunci.

## Sasaran

Akurasi maksimal untuk **inference offline dari video upload** (bukan realtime).
Latency bukan lagi batasan — keputusan ini diambil setelah terbukti TV-L1 butuh
~2,1 detik per window, tidak mungkin per-frame.

## Status: session sebelumnya terputus karena usage limit

Session Codex berhenti **22 Juli 2026, 13:07 WIB** (`usage_limit_exceeded`).

- Edit kode terakhir (`src/inference_utils.py`, 13:00:37) **selesai sukses** —
  tidak ada file yang rusak setengah jalan.
- Yang terputus: **training `iter_47_r3d_auto_apex_s42` sedang di fold 3/4**,
  berjalan di background cell. Proses itu hampir pasti sudah mati.
  Cek `experiments/protocol_v2/iter_47_r3d_auto_apex_s42_v2_dev_p5/run.log`
  untuk melihat sampai fold/epoch berapa yang sempat tersimpan.
- Preprocessing cache HQ+ECC juga berjalan paralel (posisi terakhir ~110/246).

## Protokol eksperimen (protocol v2) — jangan diubah tanpa persetujuan

Dibangun karena protokol lama sudah tidak valid (LOSO yang sama dipakai berulang
untuk memilih 37+ eksperimen, jadi bukan lagi estimasi independen).

- **Development split**: 21 subject, 192 sampel, 4 fold subject-independent.
- **Audit split**: 5 subject, terkunci, sekali pakai di akhir.
- Semua keputusan pakai **pooled 4 fold**, bukan fold tunggal.
- Gate promosi kandidat: **paired subject-bootstrap 10.000 iterasi**,
  minimum effect **ΔUF1 ≥ 0,005**, probabilitas improvement **≥ 80%**.
- Kandidat baru harus mengalahkan **champion**, bukan baseline lama.
- Champion lock mengunci seluruh leaf-model + bobot fusion sekaligus.

## Bug yang sudah diperbaiki — jangan sampai kembali

Ada 12+ regression test yang menjaga ini. Jalankan test dulu sebelum mengubah
`engine.py` / `dataset.py` / `train_final.py`.

1. `eval_rand_start` tidak aktif di jalur TTA (TTA memanggil dataset dengan
   `train=True`) — akibatnya hasil `iter_37` dulu keliru.
2. TTA mengaktifkan augmentasi training termasuk random erase. Sekarang TTA
   memakai daftar view deterministik.
3. Outer LOSO subject dievaluasi tiap epoch → bocor ke pemilihan eksperimen.
4. `train_final.py` mengabaikan `loss="focal"`, EMA, snapshot, last-k averaging.
   Anggota deploy bernama `focal` dulu sebenarnya dilatih dengan CrossEntropy.
5. `infer_onsetfree.py` pakai 2 flip deterministik sementara evaluasi pakai
   TTA×5 stokastik; field `tta: 5` di `deploy.json` tidak terpakai.

## Hasil sejauh ini (development split, pooled 4 fold)

| kandidat | UF1 | UAR | ACC | status |
|---|---|---|---|---|
| Baseline v2 (`r3d_18` + onset-referenced TV-L1) | 0,6816 | 0,7120 | 0,6823 | referensi |
| Flow translation compensation | 0,5987 | 0,6217 | 0,6042 | **ditolak** (P=0,84%) |
| Resolusi 160 | 0,6254 | 0,6562 | 0,6250 | **ditolak** (ΔUF1 −0,0562) |
| Sequential flow (t→t+1) | 0,3684 | 0,4215 | 0,3698 | **ditolak** |
| EMA 0,998 | 0,6725 | — | — | **ditolak** (efek < 0,005) |
| Onset→apex single (seed 42) | 0,6995 | — | 0,6927 | point naik, gate gagal (P=69,3%) |
| **Fusion temporal 50:50 full+apex (seed 42)** | **0,7104** | **0,7345** | **0,7083** | **CHAMPION** (P=83,9%) |
| Ensemble 2-seed apex (42+123) | 0,7356 | — | — | point tertinggi, gate gagal (P=0,761) |
| Automatic-apex (`iter_47`) | fold 1: 0,7148 | — | — | **terputus di fold 3/4** |

Replikasi seed untuk efek temporal onset→apex > full-span:
seed 42 ✓, seed 123 (0,7119 vs 0,6163) ✓, seed 2024 (0,7210 vs 0,6895) ✓.
Menambahkan seed ketiga ke ensemble apex justru **menurunkan** skor titik.

## Cabang yang sudah ditutup — jangan diulang

- Kompensasi translasi global pada flow (single maupun fusion 10/20/30%).
- Resolusi input 160 sebagai default (memperbesar variance pada data kecil).
- Consecutive/sequential TV-L1 sebagai expert (fusion 5% pun menurunkan champion).
- EMA weight averaging.
- Grid search bobot fusion lebih halus — sengaja dihindari agar tidak overfit
  development split.

## Temuan penting lain

- Estimator apex label-free (flow energy) awalnya bias **+11 frame** (memilih
  frame terlalu akhir). Sudah dibatasi dengan `search_max_fraction` yang
  ditentukan hanya dari development split.
- Model onset→apex sebelumnya memakai apex **beranotasi** saat evaluasi —
  tidak tersedia di video upload. Karena itu dibuat jalur deployment label-free.
- Snapshot per fold sekarang disimpan (3 epoch terakhir), sehingga satu training
  bisa dievaluasi ulang dengan beberapa strategi endpoint tanpa training ulang.
  Sudah diverifikasi: trajectory training identik, hanya validation yang berubah.
- Variasi antar-fold sangat besar (UF1 0,47–0,78 pada baseline yang sama).
  Satu angka fold tunggal tidak berarti apa-apa di project ini.

## Config yang sudah dibuat tapi BELUM dijalankan

- `iter_47b_r3d_auto_apex_cap65_s42.json` — energy cap 65%
- `iter_47c_r3d_oracle_apex_reuse_s42.json` — kontrol oracle apex dari snapshot
- `iter_48_r3d_multihyp_apex_s42.json`, `iter_48b_r3d_multihyp15_apex_s42.json`
  — multi-hypothesis 3 endpoint × 5 view TTA
- `iter_49_r3d_face_roi_s42.json` — soft face-ellipse ROI mask pada flow

## Rencana yang belum selesai

1. Selesaikan/ulang `iter_47` automatic-apex 4 fold; verifikasi bahwa
   re-evaluation oracle dari snapshot mereproduksi hasil onset→apex lama.
2. Re-evaluate snapshot yang sama untuk: energy-cap 55%, cap 65% (`47b`),
   oracle reuse (`47c`), multi-hypothesis 15 view (`48b`).
3. Jalankan cabang ROI (`iter_49`) terpisah agar tidak tercampur efek temporal.
4. Eksperimen HQ+ECC setelah cache preprocessing selesai.
5. Kunci champion final → audit split sekali → laporan reproducible.

## Cara kerja yang saya harapkan

- Jangan ubah file apa pun sebelum menjelaskan rencana dan saya setujui.
- Jangan menjalankan training GPU tanpa memberi tahu saya dulu.
- Laporkan hasil dengan pooled metrics + gate statistik, bukan skor fold tunggal.
- Kalau sebuah kandidat gagal gate, tutup cabangnya; jangan cari-cari bobot
  sampai kelihatan menang.
