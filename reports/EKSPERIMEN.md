# Daftar Eksperimen CASME II — klasifikasi 5 kelas

Dibuat otomatis dari hasil di `experiments/protocol_v2/` pada **07 August 2026, 06:25**.
Perbarui dengan: `conda run -n facesleuth python src/make_experiment_report.py`

## Protokol evaluasi

| kode | nama | siapa diuji | latih/uji | data latih | waktu | segel |
|---|---|---|---|---|---|---|
| 🟢 | **grouped 4-fold** | 21 orang | latih 16 → uji 5 | ±144 | 13 mnt | aman |
| 🔵 | **LOSO 21** | 21 orang | latih 20 → uji 1 | ±183 | 68 mnt | aman |
| ⚫ | **LOSO 26** | 26 orang | latih 25 → uji 1 | 245 | ±2 jam | **pecah** |
| 🔴 | **audit** | 5 orang tersegel | latih 21 → uji 5 | ±192 | 15 mnt | sekali pakai |

⚠️ Angka antar-protokol **tidak sebanding**. Sudah terbukti 🟢 bisa membalik hasil ke **dua arah** dibanding 🔵.

## Hasil

| # | Protokol | Fase | Percobaan | UF1 | Selisih | Hasil |
|---|---|---|---|---|---|---|
| | | | **— ACUAN —** | | | |
| — | 🟢 grouped | — | Baseline iter_14 | **0,6816** | — | acuan |
| — | 🔵 LOSO 21 | — | Baseline iter_14 | **0,7109** | — | acuan |
| — | 🔵 LOSO 21 | — | Auto-apex iter_47 | **0,6817** | — | acuan |
| | | | **— FASE 1: CARA MELATIH —** | | | |
| 58 | 🟢 grouped | 1.1 | Early stopping, cek 17 sampel | 0,5755 | −0,1062 | ❌ |
| 63 | 🟢 grouped | 1.1 | Early stopping, cek 59 sampel | 0,5664 | −0,1152 | ❌ |
| 64 | 🟢 grouped | 1.1 | Early stopping, penilai loss | 0,6137 | −0,0679 | ❌ |
| 59 | 🟢 grouped | 1.2 | Adaptive fine-tuning | 0,6832 | +0,0015 | ⚪ netral |
| 60 | 🟢 grouped | 1.3 | Gradual unfreeze | 0,6233 | −0,0584 | ❌ |
| 61 | 🟢 grouped | 1.4 | Latih 60 epoch | 0,6644 | −0,0172 | ❌ |
| 62 | 🟢 grouped | 1.5 | Early stopping + adaptive FT | 0,6194 | −0,0623 | ❌ |
| 69 | 🟢 grouped | 1.7 | Early stopping refit, loss | 0,6509 | −0,0307 | ❌ |
| 70 | 🟢 grouped | 1.7 | Early stopping refit, UF1 | 0,6518 | −0,0298 | ❌ |
| 71 | 🟢 grouped | 1.7 | Early stopping refit + adaptive FT | 0,6212 | −0,0604 | ❌ |
| | | | **— FASE 1.8: DIAGNOSTIK KEBOCORAN —** | | | |
| 65 | 🟢 grouped | 1.8 | ES **mengintip fold ujian** | 0,6742 | **+0,0605** vs jujur | 🔬 diagnostik |
| 66 | 🟢 grouped | 1.8 | ES mengintip + adaptive FT | 0,6694 | **+0,0500** vs jujur | 🔬 diagnostik |
| | | | **— FASE 1.9: RATA-RATA EPOCH —** | | | |
| 72 | 🟢 grouped | 1.9 | Rata-rata 5 epoch | 0,6592 | −0,0225 | ❌ |
| 73 | 🟢 grouped | 1.9 | Rata-rata 8 epoch | 0,6702 | −0,0114 | ❌ |
| 74 | 🟢 grouped | 1.9 | **Rata-rata 8 + latih 40 epoch** | 0,6983 | +0,0166 | ⚠️ P=0,798 |
| 75 | 🟢 grouped | 1.9 | Snapshot ensembling | 0,6651 | −0,0165 | ❌ |
| 80 | 🟢 grouped | 1.9r | ↳ ulang seed 123 | 0,6483 | +0,0320* | ✅ replikasi |
| 81 | 🟢 grouped | 1.9r | ↳ ulang seed 2024 | 0,7088 | +0,0193* | ✅ replikasi |
| 82 | 🟢 grouped | 1.9r | ↳ di leaf auto-apex | 0,7060 | +0,0353* | ✅ replikasi |
| | | | **— FASE 2: SEGMENTATION —** | | | |
| 67 | 🟢 grouped | 2.2 | Face parsing, full-span | 0,6637 | −0,0179 | ❌ |
| 68 | 🟢 grouped | 2.2 | Face parsing, auto-apex | 0,6698 | −0,0009 | ⚪ netral |
| 89 | 🔵 LOSO 21 | 2.3 | **Segmentation attention** statis, full-span | 0,7283 | +0,0174 | ✅ **LOLOS** P=0,801 |
| 90 | 🔵 LOSO 21 | 2.3 | **Segmentation attention** statis, auto-apex | 0,6667 | −0,0149 | ❌ |
| 91 | 🔵 LOSO 21 | 2.3 | **Segmentation attention** dinamis | 0,6758 | −0,0352 | ❌ |
| 92 | 🔵 LOSO 21 | 2.4 | Buang pipi (pembanding) | 0,6458 | −0,0652 | ❌ |
| 93 | 🔵 LOSO 21 | 2.4 | Buang pipi + dahi (pembanding) | 0,6479 | −0,0630 | ❌ |
| 94 | 🔵 LOSO 21 | 2.4 | Buang pipi + dahi, auto-apex | 0,6351 | −0,0465 | ❌ |
| 95 | 🔵 LOSO 21 | 2.5 | **Saluran fokus** energi, full-span | 0,6850 | −0,0259 | ❌ |
| 96 | 🔵 LOSO 21 | 2.5 | **Saluran fokus** tersegmentasi, full-span | 0,6394 | −0,0715 | ❌ |
| 97 | 🔵 LOSO 21 | 2.5 | **Saluran fokus** tersegmentasi, auto-apex | 0,6642 | −0,0174 | ❌ |
| | | | **— FASE 3: ARSITEKTUR —** | | | |
| 85 | 🟢 grouped | 3.1 | **CNN + RNN**, full-span | 0,5970 | −0,0847 | ❌ |
| 86 | 🟢 grouped | 3.1 | **CNN + RNN**, auto-apex | 0,6656 | −0,0051 | ❌ |
| 85L | 🔵 LOSO 21 | 3.1 | **CNN + RNN**, full-span | 0,6129 | −0,0981 | ❌ |
| 86L | 🔵 LOSO 21 | 3.1 | **CNN + RNN**, auto-apex | 0,6579 | −0,0238 | ❌ |
| 87 | 🔵 LOSO 21 | 3.1 | Backbone mc3_18 | 0,6431 | −0,0678 | ❌ |
| 88 | 🔵 LOSO 21 | 3.1 | Backbone r2plus1d_18 | 0,7059 | −0,0051 | ❌ |
| | | | **— FASE 4: AU MULTI-TASK —** | | | |
| 83 | 🟢 grouped | 4.1 | Bobot 0,1 | 0,6682 | −0,0134 | ❌ |
| 76 | 🟢 grouped | 4.1 | Bobot 0,2 | 0,6894 | +0,0077 | ⚠️ P=0,627 |
| 77 | 🟢 grouped | 4.1 | Bobot 0,5 | 0,6690 | −0,0126 | ❌ |
| 78 | 🟢 grouped | 4.1 | Bobot 1,0 | 0,6690 | −0,0126 | ❌ |
| 79 | 🟢 grouped | 4.1 | Bobot 0,5, auto-apex | 0,6714 | +0,0007 | ⚪ netral |
| 84 | 🟢 grouped | 4.1 | Bobot 0,2, auto-apex | 0,6595 | −0,0112 | ❌ |
| 76L | 🔵 LOSO 21 | 4.1 | Bobot 0,2 | 0,6671 | −0,0438 | ❌ |
| 77L | 🔵 LOSO 21 | 4.1 | Bobot 0,5 | 0,6743 | −0,0366 | ❌ |
| 98 | 🔵 LOSO 21 | 2.6 | **Input segmentasi + early stopping**, full-span | 0,6198 | −0,0911 | ❌ |
| 99 | 🔵 LOSO 21 | 2.6 | **Input segmentasi + early stopping**, auto-apex | 0,6466 | −0,0350 | ❌ |
| | | | **— FASE 2.3r: REPLIKASI SEED SEGMENTATION ATTENTION —** | | | |
| 89 | 🔵 LOSO 21 | 2.3r | Segmentation attention, seed 42 | 0,7283 | +0,0174 | ✅ **LOLOS** P=0,801 |
| 100 | 🔵 LOSO 21 | 2.3r | Segmentation attention, seed 123 | 0,6865 | +0,0085 | ⚠️ P=0,643 |
| 101 | 🔵 LOSO 21 | 2.3r | Segmentation attention, seed 2024 | 0,6862 | +0,0009 | ⚪ netral |
| 102 | 🔵 LOSO 21 | 2.3r | Segmentation attention, seed 7 | 0,6953 | −0,0012 | ⚪ netral |
| | | | **— FASE 3.1L: ARSITEKTUR DI LOSO —** | | | |
| 85L | 🔵 LOSO 21 | 3.1 | CNN + RNN, full-span | 0,6129 | −0,0981 | ❌ |
| 86L | 🔵 LOSO 21 | 3.1 | CNN + RNN, auto-apex | 0,6579 | −0,0238 | ❌ |
| 87L | 🔵 LOSO 21 | 3.1 | Backbone mc3_18 | 0,6431 | −0,0678 | ❌ |
| 88L | 🔵 LOSO 21 | 3.1 | Backbone r2plus1d_18 | 0,7059 | −0,0051 | ❌ |
| | | | **— FASE 6: RNN TANPA PRETRAIN (⚫ LOSO 26) —** | | | |
| 103 | ⚫ LOSO 26 | 6.1 | RNN tanpa pretrain, full-span | 0,6612 | −0,0289 | ❌ |
| 104 | ⚫ LOSO 26 | 6.1 | RNN tanpa pretrain, auto-apex | 0,6904 | −0,0195 | ❌ |
| 105 | ⚫ LOSO 26 | 6.2 | **Input segmentasi + RNN tanpa pretrain** | 0,6148 | −0,0754 | ❌ |
| 106 | ⚫ LOSO 26 | 6.2 | **Input segmentasi + RNN tanpa pretrain + early stopping** | 0,5679 | −0,1223 | ❌ |
| | | | **— FASE 5.2: ⚫ LOSO 26 — SEGEL AUDIT DIPECAH (sebanding dosen) —** | | | |
| K1 | ⚫ LOSO 26 | 5.2 | Kandidat 1: baseline | **0,6902** | — | acuan |
| K5 | ⚫ LOSO 26 | 5.2 | Kandidat 5: **segmentation attention** | 0,7022 | +0,0120 | ⚠️ P=0,732 |
| K3 | ⚫ LOSO 26 | 5.2 | Kandidat 3: **input segmentasi** (ide user) | 0,7121 | +0,0219 | ✅ **LOLOS** P=0,805 |
| K4 | ⚫ LOSO 26 | 5.2 | Kandidat 4: **input segmentasi + early stopping** (ide user) | 0,6508 | −0,0393 | ❌ |
| | | | **— FASE 5.0: UJI ULANG DI LOSO —** | | | |
| 59L | 🔵 LOSO 21 | 5.0 | Adaptive fine-tuning | 0,7169 | +0,0059 | ⚠️ P=0,610 |
| 67L | 🔵 LOSO 21 | 5.0 | Face parsing | 0,7136 | +0,0027 | ⚪ netral |
| 74L | 🔵 LOSO 21 | 5.0 | Rata-rata 8 + 40 epoch | 0,7075 | −0,0034 | ⚪ netral |
| 69L | 🔵 LOSO 21 | 5.0 | Early stopping refit | 0,6930 | −0,0179 | ❌ |
| 61L | 🔵 LOSO 21 | 5.0 | Latih 60 epoch | 0,6874 | −0,0235 | ❌ |
| 77L | 🔵 LOSO 21 | 5.0 | AU multi-task 0,5 | 0,6743 | −0,0366 | ❌ |
| 76L | 🔵 LOSO 21 | 5.0 | AU multi-task 0,2 | 0,6671 | −0,0438 | ❌ |
| | | | **— FASE 5.1: PENGGABUNGAN —** | | | |
| — | 🔵 **LOSO 21** | 5.1 | Gabungan 2 model | 0,7088 | −0,0021 | ❌ |
| — | 🔵 **LOSO 21** | 5.1 | Gabungan 7 model | 0,7237 | +0,0128 | ✅ **LOLOS** P=0,858 |
| — | 🔵 **LOSO 21** | 5.1 | **Gabungan 10 model** (semua yang ±0,04) | **0,7251** | **+0,0141** | ✅ **LOLOS** P=0,853 |
| K2 | ⚫ **LOSO 26** | 5.2 | Kandidat 2: **gabungan 10 model** | **0,7118** | **+0,0216** | ✅ **LOLOS** P=0,986 |
| | | | **— TANPA GPU —** | | | |
| — | 🔵 LOSO 21 | 0.1 | **Reject option** | ACC 69,8→**77,2%** | | ✅ berguna |
| — | — | 2.1 | 246 masker wajah MediaPipe | 246/246 | | ✅ |
| — | — | 2.1 | 246 masker 6 region wajah | jumlah 100,0% | | ✅ |
| | | | **— BELUM PERNAH DIJALANKAN —** | | | |
| — | ⚫ **LOSO 26** | 5.2 | Angka sebanding dosen | — | | ⬜ belum |
| — | 🔴 **audit** | 5.3 | Pengecekan final | — | | ⬜ belum |

\* dibanding baseline seed-nya sendiri, bukan seed 42

## Ringkasan

| | jumlah |
|---|---|
| ✅ Lolos gate | **6** |
| ⚠️ Naik tapi gate gagal | 5 |
| ⚪ Netral | 7 |
| ❌ Gagal | 49 |
| ⏳ Antre | 0 |

## Posisi

| | UF1 | Protokol |
|---|---|---|
| Baseline satu model | 0,7109 | 🔵 LOSO 21 |
| **Terbaik: gabungan 7 model** | **0,7237** | 🔵 LOSO 21 |
| Acuan lama satu model | 0,7145 | ⚫ LOSO 26 |
| Klaim dosen | 0,80 | ⚫ LOSO 26 |

## Catatan ⚫ LOSO 26 (segel audit dipecah 2026-08-05)

Kelima kandidat ada di tabel di atas (baris K1–K5). Daftarnya dikunci sebelum hasil dilihat dan tidak boleh ditambah — semua dilaporkan, termasuk yang kalah, dan angka final adalah yang **lolos gate**. Kandidat 3 & 4 (ide user/dosen) sudah gagal berat di 🔵 LOSO 21 (0,6394 dan 0,6198 vs 0,7109); tetap dijalankan atas permintaan user. Subject audit `[4, 6, 8, 17, 24]` kini terpakai — pengecekan bersih sekali-pakai sudah tidak ada.

## Cabang yang sudah tertutup

- **Early stopping** — 9 varian, semua kalah. Bahkan versi yang mengintip fold ujian (0,6742) masih di bawah baseline (0,6816).
- **Adaptive fine-tuning, gradual unfreeze, latih lebih lama** — tidak berefek atau merugikan di kedua protokol.
- **AU multi-task** — makin besar bobotnya makin buruk; di 🔵 −0,044.
- **Rata-rata epoch (74)** — replikasi 3/3 seed positif di 🟢, lalu mati di 🔵. Pemenang palsu.
- **CNN + RNN sendirian** — 0,5970, jauh di bawah 3D-CNN.

## Temuan bernilai walau skornya tidak naik

1. **Kebocoran early stopping = +0,0605 UF1.** Memilih epoch berhenti dengan melihat fold ujian menaikkan skor sebanyak itu tanpa model membaik sedikit pun.
2. **LOSO 21 ≈ LOSO 26** — 0,7109 vs 0,7145, selisih 0,0036 saja. Segel audit tidak perlu dipecah untuk mendapat angka sebanding paper.
3. **Saringan 🟢 menyesatkan ke DUA arah** — menciptakan pemenang palsu (74) dan menolak yang sebenarnya netral (face parsing).
4. **Reject option** menaikkan akurasi aplikasi 69,8% → 77,2% tanpa training ulang.
5. **Segel audit ternyata sudah dipakai 37×** di percobaan lama — catatan lama tidak akurat.
6. **Keragaman saja tidak cukup untuk gabungan.** mc3_18 adalah anggota paling komplementer (beda 0,207 vs rata-rata 0,144) dan punya 6 sampel yang hanya dia yang benar — tapi menambahkannya justru menurunkan gabungan 0,7237 → 0,7080. Solonya 0,6431, terlalu jauh di bawah baseline. **Aturan yang terukur: anggota fusi harus dalam ±0,04 UF1 dari baseline.**
7. **AU6 + AU14 = 40 sampel bergantung pada pipi**, jadi membuang pipi membuang isyarat utama 16% dataset.
