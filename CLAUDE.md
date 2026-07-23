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

## Status per 23 Juli 2026 — pemulihan session Codex selesai

Session Codex berhenti 22 Juli 2026 (`usage_limit_exceeded`). Ternyata **yang mati
hanya session-nya, bukan pekerjaannya** — proses background terus jalan:

- `iter_47_r3d_auto_apex_s42` **selesai penuh 4 fold** (`complete: true`, 20:20:38),
  bukan terputus di fold 3/4. Pooled UF1 0,6707.
- Cache HQ+ECC **selesai 246/246**, 0 error, 0 gagal stabilisasi (~8,8 jam).
- Audit integritas: 33 `probs.npz` semua utuh tanpa NaN, 12 checkpoint semua
  loadable, tidak ada file setengah jadi. 23 regression test hijau.

Catatan: transkrip Codex memakai UTC, jam mesin memakai WIB (UTC+7). "13:07"
di transkrip = 20:07 WIB — itu sebabnya kelihatan seperti terputus.

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
| Fusion temporal 50:50 full+apex (seed 42) | 0,7104 | 0,7345 | 0,7083 | champion **oracle** (P=83,9%) — lihat peringatan di bawah |
| Ensemble 2-seed apex (42+123) | 0,7356 | — | — | point tertinggi, gate gagal (P=0,761) |
| Automatic-apex label-free (`iter_47`) | 0,6707 | 0,7008 | 0,6719 | selesai 4 fold |
| Energy cap 65% (`47b`) | 0,6574 | 0,6891 | 0,6563 | **ditolak** |
| Multi-hipotesis 6 view (`48`) | 0,6654 | 0,7070 | 0,6563 | **ditolak** |
| Multi-hipotesis 15 view (`48b`) | 0,6781 | 0,7091 | 0,6719 | **ditolak** — lihat catatan fusi |
| Oracle reuse dari snapshot (`47c`) | 0,6995 | 0,7088 | 0,6927 | kontrol: reproduksi `iter_43` persis |
| **Fusion deployable 50:50 full+auto-apex** | **0,7021** | **0,7418** | **0,6979** | **CHAMPION yang bisa dipakai di app** |

### ⚠️ Champion 0,7104 tidak bisa direproduksi saat inference

Leaf `iter_43` dievaluasi memakai **apex beranotasi** dari dataset — tidak ada di
video upload. Penggantinya yang label-free (`fusions/deployable_50full_50auto47`)
memberi **0,7021**, lolos gate atas baseline (ΔUF1 +0,0205, P=0,832) dan tidak
bisa dibedakan dari champion oracle (−0,0084, P=0,308).

**Angka jujur untuk dilaporkan adalah 0,7021, bukan 0,7104.**

Replikasi seed untuk efek temporal onset→apex > full-span:
seed 42 ✓, seed 123 (0,7119 vs 0,6163) ✓, seed 2024 (0,7210 vs 0,6895) ✓.
Menambahkan seed ketiga ke ensemble apex justru **menurunkan** skor titik.

## Cabang yang sudah ditutup — jangan diulang

- **Seluruh keluarga stabilisasi gerakan.** Dua pendekatan independen sama-sama
  jatuh ~0,09 UF1: kompensasi translasi global (`iter_40`, 0,5987) dan
  HQ TV-L1 + ECC euclidean (`iter_41`, 0,5906) di atas `cache/flow144_hq_stab`
  yang sudah jadi. Baseline 0,6816. Gerakan kepala yang dibuang tampaknya justru
  membawa sinyal kelas. **Jangan habiskan GPU lagi di sumbu ini.**
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
- **Skor leaf tidak memprediksi nilainya dalam fusi.** `48b` menang sendirian atas
  `iter_47` (0,6781 vs 0,6707) tapi fusinya justru jatuh (0,6789 vs 0,7021,
  ΔUF1 −0,0232, P=0,057). TTA multi-crop + multi-fase membuatnya berkorelasi
  dengan model full-span, jadi tidak menambah sinyal komplementer.
  **Jangan pernah memilih anggota fusi dari skor solonya.**
- **Folder `Cropped` dipotong di ujung, bukan dijarangkan.** Diverifikasi dengan
  menghitung nama file pada 246 sampel: setiap folder berisi persis
  `offset − onset + 1` file, penomoran sepenuhnya berurutan (histogram jarak
  antar frame = `{1: 16477}`, tidak ada satu pun frame terlewat). Jadi resolusi
  waktu tetap penuh 200 fps — tapi model **belum pernah melihat satu frame pun
  di luar jendela ekspresi berlabel**.
- **`CASME2-RAW` tidak ada di komputer ini** (dikonfirmasi user). Tanpa itu tidak
  ada ground truth untuk spotting, tidak ada wajah non-crop untuk menguji
  alignment, dan tidak ada bahan membuat video uji yang realistis.

## Robustness untuk video upload — sudah diukur 23 Juli 2026

Semua dari snapshot `iter_47`, tanpa training ulang.

**Error jendela onset tidak berbahaya.** Lewat `eval_start_fraction`:

| onset terlambat | 0% | 10% | 20% | 30% |
|---|---|---|---|---|
| UF1 | 0,6707 | 0,6837 | 0,6680 | 0,6757 |

Tidak ada keruntuhan sistematis. Keterbatasan jujur: hanya menguji onset yang
**terlambat**; cache tidak menyimpan frame sebelum onset, jadi onset yang
**kedahuluan** dan error offset belum terukur.

**FPS bukan ancaman besar — dugaan awal saya (30 fps merusak model) keliru.**
`src/eval_fps_robustness.py`:

| fps | 200 | 120 | 60 | 30 |
|---|---|---|---|---|
| leaf auto-apex (`iter_47`) | 0,6707 | 0,6866 | 0,7113 | 0,6537 |
| gate leaf vs 200 fps | — | P=0,923 | P=0,965 | P=0,246 |
| **fusion deployable** | **0,7021** | 0,6729 | 0,6881 | **0,6953** |
| gate fusion vs 200 fps | — | **P=0,016 (lebih buruk)** | P=0,241 | **P=0,372 (tidak beda)** |

**Kesimpulan untuk app: video HP 30 fps layak dipakai.** Di level sistem, 30 fps
hanya −0,0068 UF1 dengan ACC identik (0,6979), dan gate menyatakan tidak berbeda
dari 200 fps. App **tidak perlu** mensyaratkan rekaman high-speed.

⚠️ **Jangan baca baris leaf sebagai peningkatan.** Kenaikan +0,04 pada leaf
**tidak menular ke fusi** — di fusi semua fps rendah justru di bawah 200 fps.
Pengulangan pelajaran `48b`: perbaikan satu leaf ≠ perbaikan sistem.

Dugaan mekanisme ketahanan: flow onset-referenced itu **kumulatif** (`arr[t]` =
TV-L1 dari onset ke frame t), jadi kurvanya mulus terhadap waktu dan tahan
penjarangan — berbeda dari flow sequential (t→t+1) yang memang sudah ditolak.
Faktor yang belum dimodelkan: motion blur dan rolling shutter pada rekaman
30 fps sungguhan.

## Masalah terbesar yang belum tersentuh: semua angka mengasumsikan jendela sempurna

`cache/flow144` dibangun dari `[onset, offset]`, jadi **setiap model protocol v2
selalu menerima jendela ekspresi yang sudah dipotong tepat**. Di video upload,
onset maupun offset sama-sama tidak diketahui.

Masalah apex sudah ditutup dengan estimator label-free. Masalah **batas jendela
belum**. Artinya label "deployable" pada 0,7021 pun masih optimistis.

Bisa diukur murah lewat `eval_rand_start` + `eval_start_fraction`
([dataset.py:133](src/dataset.py:133)) — keduanya `EVAL_ONLY_KEYS`, jadi cukup
re-evaluasi dari snapshot tanpa training. Keterbatasan jujur: hanya bisa meniru
onset yang **ketinggalan** (mulai terlalu lambat), karena cache tidak menyimpan
frame sebelum onset.

Risiko kedua yang belum diukur: **FPS**. CASME II 200 fps; video HP ~30 fps.
Ekspresi 0,2 detik = 41 frame di 200 fps, tapi hanya ~6 frame di 30 fps,
sedangkan model butuh T=16. Kata "fps" tidak muncul di mana pun dalam repo.

## Rencana

**Fase 1 — selesai 23 Juli 2026.** Cabang apex label-free ditutup; champion
deployable 0,7021 terkunci angkanya.

**Fase 2 — robustness untuk video upload (prioritas utama).**

1. Uji degradasi FPS (200 → 120 / 60 / 30) dari snapshot yang ada.
2. Uji error jendela onset lewat `eval_start_fraction`.
3. Pembaca video + normalisasi FPS + deteksi/pelurusan wajah, lalu **uji
   kesetaraan**: hasil lewat jalur app harus sama dengan jalur training.
4. Spotting — **terblokir**, butuh `CASME2-RAW` yang tidak ada di mesin ini.
5. Skor "tidak ada ekspresi" (reject option) supaya app tidak asal menebak.

**Fase 3 — akurasi.** HQ+ECC (`iter_41`, cache sudah siap), ROI (`iter_49`),
replikasi seed label-free (`iter_50`/`iter_51`). Antreannya di `run_queue_v2.sh`.

**Fase 4 —** kunci champion → audit split sekali → laporan reproducible.

Ekspektasi jujur: Fase 2 kemungkinan besar **menurunkan** 0,7021, dan itu sehat —
0,7021 adalah angka kondisi ideal.

## Cara kerja yang saya harapkan

- Jangan ubah file apa pun sebelum menjelaskan rencana dan saya setujui.
- Jangan menjalankan training GPU tanpa memberi tahu saya dulu.
- Laporkan hasil dengan pooled metrics + gate statistik, bukan skor fold tunggal.
- Kalau sebuah kandidat gagal gate, tutup cabangnya; jangan cari-cari bobot
  sampai kelihatan menang.
