# Audit robustness & penutupan cabang — 23 Juli 2026

Sesi otonom lanjutan setelah pemulihan session Codex. Semua angka pooled 4 fold
development split (192 sampel, 21 subject). Audit split **tidak disentuh**.

---

## Ringkasan untuk yang buru-buru

1. **Video HP 30 fps layak dipakai.** Di level sistem 30 fps hanya −0,0068 UF1
   dengan ACC identik, dan gate menyatakan tidak berbeda dari 200 fps.
   Kekhawatiran awal saya bahwa 30 fps akan merusak model **salah**.
2. **Error jendela onset tidak berbahaya** sampai 30% terlambat.
3. **Champion bukan noise.** Efeknya bereplikasi di ketiga seed (P = 0,839 /
   1,000 / 0,925). Kecurigaan saya bahwa gate memutuskan atas noise **salah**.
4. **Lima cabang ditutup** dengan bukti, termasuk dua yang paling menjanjikan
   (HQ+ECC dan multi-seed ensembling).
5. **Tidak ada peningkatan akurasi yang lolos gate.** Champion deployable tetap
   **UF1 0,7021**.

---

## Yang diverifikasi

| pemeriksaan | hasil |
|---|---|
| Reproduksibilitas training | `iter_14` dilatih ulang → UF1 0,6816 / UAR 0,7120 / ACC 0,6823 **persis sama** |
| Integritas artifact | 33 `probs.npz` utuh tanpa NaN; 12 checkpoint loadable; 0 file rusak |
| Regression test | 23/23 hijau |
| Jalur re-evaluasi | `47c` mereproduksi `iter_43` sampai 16 digit |

Karena training deterministik, semua perbandingan paired antar-run sah.

---

## Cabang yang ditutup

| kandidat | UF1 | vs pembanding | putusan |
|---|---|---|---|
| HQ TV-L1 + ECC (`iter_41`) | 0,5906 | baseline 0,6816 | **ditolak** |
| Soft face-ellipse ROI (`iter_49`) | 0,6370 | baseline 0,6816 | **ditolak** |
| Apex energy cap 65% (`47b`) | 0,6574 | `iter_47` 0,6707 | **ditolak** |
| Multi-hipotesis 6/15 view (`48`/`48b`) | 0,6654 / 0,6781 | fusi turun ke 0,6789 | **ditolak** |
| Apex smoothing radius 2/4/6/10 (`53`) | 0,6626–0,6697 | radius 0 = 0,6707 | **ditolak** |
| Multi-seed full-span dalam fusi | 0,6610 | deployable 0,7021 | **ditolak** (P=0,000) |

### Dua yang layak dicatat khusus

**Stabilisasi gerakan sudah mati sebagai keluarga.** Dua pendekatan independen —
kompensasi translasi global (`iter_40`, 0,5987) dan HQ TV-L1 + ECC euclidean
(`iter_41`, 0,5906) — sama-sama jatuh ~0,09 UF1 dari baseline. Cache
`flow144_hq_stab` yang butuh 8,8 jam preprocessing terpakai dan hasilnya negatif.
Gerakan kepala yang dibuang tampaknya membawa sinyal kelas. Jangan ulangi.

**Multi-seed ensembling gagal dengan cara yang aneh.** Rata-rata 3 seed full-span
**lebih baik sendirian** (0,6902 vs 0,6816), tapi difusikan dengan leaf apex
menghasilkan 0,6610 — **di bawah kedua komponennya sendiri**. Ini peringatan
tentang seberapa tidak stabil perilaku fusi pada 192 sampel.

---

## Robustness untuk video upload

Semua dari snapshot tersimpan, tanpa training ulang.

### Error jendela onset

| onset terlambat | 0% | 10% | 20% | 30% |
|---|---|---|---|---|
| UF1 | 0,6707 | 0,6837 | 0,6680 | 0,6757 |

Tidak ada keruntuhan. **Keterbatasan jujur:** hanya menguji onset yang terlambat.
Cache tidak menyimpan frame sebelum onset, jadi onset yang kedahuluan dan error
offset belum terukur — dan tidak bisa diukur tanpa `CASME2-RAW`.

### FPS

| fps | 200 | 120 | 60 | 30 |
|---|---|---|---|---|
| leaf auto-apex | 0,6707 | 0,6866 | 0,7113 | 0,6537 |
| **fusion deployable** | **0,7021** | 0,6729 | 0,6881 | 0,6953 |
| gate fusi vs 200 fps | — | P=0,016 (buruk) | P=0,241 | P=0,372 (tidak beda) |

**Kesimpulan: app tidak perlu mensyaratkan rekaman high-speed.**

Perhatikan baris leaf vs fusi: kenaikan +0,04 pada leaf di 60 fps **tidak menular
ke sistem**. Ini pengulangan ketiga dari pelajaran yang sama.

**Faktor yang belum dimodelkan:** motion blur dan rolling shutter pada rekaman
30 fps sungguhan. Simulasi ini menjarangkan flow yang dihitung dari rekaman
200 fps; nilai flow per frame identik, tapi karakteristik sensor tidak ditiru.

---

## Pelajaran metodologis: skor leaf ≠ nilai dalam sistem

Tiga kali terbukti dalam satu sesi:

| kasus | leaf | dalam fusi |
|---|---|---|
| `48b` multi-hipotesis 15 view | 0,6781 (**naik** dari 0,6707) | 0,6789 (**turun** dari 0,7021) |
| Downsampling 60 fps | 0,7113 (**naik** 0,04) | 0,6881 (**turun**) |
| Multi-seed full-span | 0,6902 (**naik**) | 0,6610 (**turun**) |

Penyebabnya sama: perubahan yang membuat sebuah leaf lebih baik sendirian sering
membuatnya **lebih mirip** leaf lain, sehingga kontribusi komplementernya hilang.

**Aturan praktis: selalu nilai kandidat di level fusi.**

---

## Replikasi champion lintas seed

Kecurigaan awal saya: gate mempromosikan noise. **Terbantah.**

| seed | full-span | fusion 50:50 | ΔUF1 | P(improve) | putusan |
|---|---|---|---|---|---|
| 42 | 0,6816 | 0,7104 | +0,0288 | 0,839 | lolos |
| 123 | 0,6163 | 0,7029 | +0,0866 | 1,000 | lolos |
| 2024 | 0,6895 | 0,7238 | +0,0343 | 0,925 | lolos |

Lolos di ketiganya, rata-rata +0,0499. **Cabang ini tidak perlu diperdebatkan lagi.**

Tapi datanya mengungkap hal lain: leaf full-span sangat tidak stabil terhadap seed
(rentang 0,0732, sd 0,0402), sedangkan leaf onset→apex stabil (rentang 0,0215,
sd 0,0108). Kemenangan terbesar fusi (+0,0866) terjadi tepat di seed tempat
baseline ambruk. **Fusi bekerja terutama sebagai peredam variance, bukan penambah
rata-rata.**

Catatan integritas: seed 2024 seragam lebih baik dari seed 42 (fusi 0,7238 vs
0,7104), tapi memilihnya sekarang adalah winner's curse. Alternatif principled-nya
— ensembling seed — sudah gagal dua kali.

---

## Status champion

**Tidak berubah: `fusions/deployable_50full_50auto47`, UF1 0,7021 / UAR 0,7418 /
ACC 0,6979.** Tidak ada kandidat yang lolos gate dalam sesi ini.

Itu hasil yang jujur, bukan kegagalan sesi: enam cabang ditutup dengan bukti,
dua risiko deployment terbesar terukur dan ternyata aman, dan satu kecurigaan
metodologis terbantah.

---

## Berikutnya

**Terblokir tanpa `CASME2-RAW`:** spotting, validasi deteksi wajah, error onset
arah kedahuluan, uji video end-to-end.

**Bisa jalan sekarang:**

1. Pembaca video + normalisasi FPS + deteksi/pelurusan wajah, lalu uji kesetaraan
   terhadap jalur training.
2. Skor "tidak ada ekspresi" (reject option) — model sekarang wajib menebak 1
   dari 5 walaupun tidak ada ekspresi sama sekali.
3. Kunci champion → audit split sekali → laporan reproducible.

Rekomendasi saya tetap: **jangan kejar akurasi lagi.** Enam cabang berturut-turut
gagal, dan sisa ruang pada 246 sampel sudah tipis. Nilai terbesar sekarang ada di
menutup jarak antara 0,7021 dan apa yang benar-benar terjadi di app.
