# Eksplorasi Data CASME II — mulai dari nol

Folder ini berdiri sendiri. Tidak memakai, tidak mengubah, dan tidak bergantung
pada apa pun dari eksperimen sebelumnya di repo ini.

## Cara melihat hasilnya (untuk pemula)

Buka **tiga file** ini dengan browser (klik dua kali, atau klik kanan → Open with → Chrome):

```
eda\out\laporan.html            <- Bagian 1: metadata
eda\out\laporan_piksel.html     <- Bagian 2: piksel
eda\out\laporan_lanjutan.html   <- Bagian 3: lima pertanyaan lanjutan
```

Ketiga halaman saling bertaut, jadi cukup buka yang pertama lalu ikuti tautan
di bagian atas tiap halaman.

Tiap grafik disertai kotak penjelasan: **apa yang ditunjukkan** dan
**kenapa itu penting untuk model**. Gambar mentahnya juga tersimpan sebagai
`.png` di folder yang sama kalau mau dipakai di laporan/skripsi.

## Soal lisensi dataset

Aturan yang dipegang: **tidak ada gambar yang dilihat, ditampilkan, atau disalin.**

- `explore_casme2.py` hanya membaca file Excel dan **nama-nama file**. Tidak satu
  gambar pun dibuka.
- `explore_pixels.py` **mendekode** gambar, tapi hanya untuk dijadikan matriks
  angka 32×32 di memori, diambil statistiknya, lalu dibuang. Tidak ada gambar
  yang disimpan atau ditampilkan. Semua keluarannya berupa angka agregat
  (rata-rata atas puluhan klip dari orang berbeda).
- Grafik "peta gerak" (`14_peta_gerak_blok.png`) adalah kisi **8×8 angka**
  hasil rata-rata puluhan klip — peta statistik, bukan foto wajah.

## Cara menjalankan ulang

```bash
conda run -n facesleuth python eda/explore_casme2.py
```

```bash
conda run -n facesleuth python eda/explore_pixels.py
```

```bash
conda run -n facesleuth python eda/explore_deep.py
```

Urutannya penting: `explore_pixels.py` dan `explore_deep.py` membaca
`out/metadata.csv` yang dihasilkan skrip pertama.

Bagian piksel butuh ~3 menit untuk 246 klip (17 ribu gambar). Hasil pindaiannya
disimpan, jadi kalau hanya ingin mengubah tampilan grafik, tambahkan `--replot`:

```bash
conda run -n facesleuth python eda/explore_deep.py --replot
```

## Isi folder `out/`

| file | isi |
|---|---|
| `laporan.html`, `laporan_piksel.html`, `laporan_lanjutan.html` | laporan lengkap, buka dengan browser |
| `01_*.png` … `23_*.png` | grafik satuan |
| `metadata.csv` | satu baris per klip: subjek, emosi, onset/apex/offset, durasi, hasil cek folder — bisa dibuka di Excel |
| `pixel_stats.csv` | satu baris per klip: kecerahan, kontras, energi gerak, posisi puncak gerak |
| `deep_stats.csv` | satu baris per klip: gerak tengah vs pinggir, tujuh penebak apex |
| `pixel_cache.npz`, `deep_cache.npz` | hasil pindaian piksel (untuk `--replot`) |

## Sumber data

```
D:\AI-Projects\casmeII-facesleuth-r\dataset\
  CASME2-coding-20140508.xlsx     255 klip beranotasi
  CASME2-ObjectiveClasses.xlsx    label alternatif berbasis Action Unit
  Cropped\subNN\EPxx_yy\*.jpg     frame wajah yang sudah dipotong
```

## Temuan yang perlu ditindaklanjuti

1. **`sub16/EP01_08` cacat.** Excel menyebut jendela frame 73–110 (38 frame),
   tapi di disk hanya ada frame 73–96 (24 file). Klip ini terpotong 14 frame.
   Apex-nya (frame 94) masih ada, tapi ekornya hilang. Putuskan: dibuang, atau
   offset-nya dikoreksi jadi 96.
2. **`sub04/EP12_01f` tidak punya ApexFrame** (isinya `/` di Excel). Perlu
   penanganan khusus di jalur yang memakai apex beranotasi.
3. Selain dua di atas, 244 klip lain **konsisten**: jumlah file persis
   `offset − onset + 1`, dan penomoran frame rapat tanpa satu pun lompatan.

## Angka penting dari tahap 3

Semua pengujian di bawah ini **subject-independent** (Leave-One-Subject-Out),
prediksi digabung dulu dari semua fold baru diukur. Baseline tebak-kelas-terbanyak
= 40,2% akurasi / UF1 0,115.

| pemeriksaan | hasil |
|---|---|
| `disgust` vs `others` pada klip AU4-saja (n=72) | 79,2% / UF1 0,442 — **di bawah** tebak-terbanyak 83,3% / UF1 0,455 |
| kemiripan peta gerak `disgust`↔`others` | korelasi **+0,91** (pasangan paling mirip dari 10 pasangan) |
| akurasi tanpa melihat gerakan sama sekali | maksimum 38,6% — tidak ada jalan pintas kasar |
| menebak identitas orang dari kecerahan+kontras | **60,2%** (tebak acak 3,8%) |
| gerak pinggir potongan ÷ gerak tengah wajah | **0,78×** (median); pada 24% klip pinggir lebih besar |
| gerak di tengah wajah saat frame offset | masih **96%** dari puncaknya |
| penebak apex terbaik dari 7 kandidat | konstanta 0,50 (meleset 0,125); puncak piksel mentah 0,284 |
| ruang saja / waktu saja / ruang×waktu | UF1 **0,411** / 0,204 / 0,371 |

Catatan kejujuran untuk baris terakhir: kotak 8×8 dan 10 titik waktu itu ringkasan
yang sangat kasar, dan 640 fitur untuk 246 sampel rawan overfit. Angka itu
**batas bawah**, bukan bukti bahwa model urutan tidak berguna.
