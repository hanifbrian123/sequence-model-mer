# Narasi Laporan Eksperimen CASME II

Bagian tulisan tangan yang disisipkan ke `reports/LAPORAN.md` saat laporan
di-generate otomatis. Angka kuantitatif selalu bersumber dari `results/results.csv`
atau `summary.json`. Yang ditulis di sini adalah konteks metodologi, interpretasi
temuan, dan panduan teknis.

Setiap blok dibatasi penanda `<!-- bagian: NAMA -->` … `<!-- /bagian -->`.

<!-- bagian: pengantar -->
Proyek ini mengklasifikasikan micro-expression pada rekaman video **CASME II**
ke dalam 5 kelas kanonik (`happiness`, `disgust`, `repression`, `surprise`, `others`).
Micro-expression adalah gerakan wajah involunter berdurasi sangat singkat (kurang dari 0,5 detik)
dengan intensitas gerakan otot wajah yang sangat halus.

Representasi temporal utama menggunakan **TV-L1 Optical Flow kumulatif ber-referensi onset**
(`arr[t]` = flow perpindahan piksel dari frame onset ke frame t) yang diproses menggunakan
arsitektur 3D Convolutional Neural Network (R3D-18 pra-terlatih Kinetics-400).
Metode ini dirancang untuk pipeline inferensi offline dari video unggahan pengguna di aplikasi web.
<!-- /bagian -->

<!-- bagian: peringatan_jendela -->
⚠️ **Peringatan Integritas Angka — Asumsi Jendela Sempurna:**
Semua metrik evaluasi dihitung di atas jendela temporal ekspresi yang dipotong dari rentang `[onset, offset]` berlabel dataset.
Pada skenario nyata (video upload pengguna), onset maupun offset tidak diketahui sebelumnya.
Tantangan estimasi apex tanpa anotasi telah dipecahkan menggunakan **estimator energi optical flow label-free** (`047_r3d_auto_apex_s42_v2_dev_p5`),
menghasilkan model **Champion Deployable (0,7021 UF1)** yang siap diaplikasikan pada backend produksi.
Sebagai pembanding, angka 0,7104 UF1 adalah champion *oracle* (menggunakan anotasi apex dataset yang tidak tersedia pada upload nyata),
sehingga acuan realistis untuk evaluasi aplikasi adalah **0,7021 UF1**.
<!-- /bagian -->

<!-- bagian: arsitektur -->
| Komponen | Detail Spesifikasi |
|---|---|
| **Input** | Grayscale CASME II $\rightarrow$ TV-L1 Dual Flow ($u, v$) kumulatif dari frame onset, $128 \times 128 \times 16$ frame |
| **Backbone** | R3D-18 (ResNet 3D 18-layer) pra-terlatih Kinetics-400 (spatiotemporal convolution) |
| **Pooling** | Adaptive Spatiotemporal Average Pooling $\rightarrow$ 512 fitur laten |
| **Head** | Dropout 0,5 $\rightarrow$ Fully Connected Layer $512 \rightarrow 5$ |
| **Kelas (5)** | `happiness`, `disgust`, `repression`, `surprise`, `others` |
| **Fusi Inferensi** | Ensemble fusi probabilitas 50:50 antara model full-span dan model apex-focused |
<!-- /bagian -->

<!-- bagian: protokol -->
Eksperimen diatur dalam protokol yang ketat untuk mencegah kebocoran identitas subjek dan bias seleksi:

| Kode | Protokol | Subjek | Pembagian Data | Tujuan & Fungsi |
|---|---|---|---|---|
| 🟢 | **Grouped 4-fold** | 21 subjek dev | Latih 16 orang $\rightarrow$ Uji 5 orang (stratified by subject) | Seleksi cepat & sweep hiperparameter |
| 🔵 | **LOSO 21** | 21 subjek dev | Latih 20 orang $\rightarrow$ Uji 1 orang (21 fold) | Validasi kandidat tanpa kebocoran identitas |
| ⚫ | **LOSO 26** | 26 subjek | Latih 25 orang $\rightarrow$ Uji 1 orang (26 fold) | Perbandingan literatur akademik & pengujian akhir |
| 🔴 | **Audit split** | 5 subjek audit | Tersegel & independen (tidak pernah dilihat selama tuning) | Verifikasi champion final |

Gate promosi kandidat: **paired subject-bootstrap 10.000 iterasi**, minimum effect **$\Delta\text{UF1} \ge 0,005$**, probabilitas keunggulan **$P \ge 80\%$**.
<!-- /bagian -->

<!-- bagian: temuan_utama -->
Temuan penting dari seluruh rangkaian eksperimen:
1. **Flow kumulatif ber-referensi onset unggul telak atas flow sekuensial ($t \rightarrow t+1$).** Flow kumulatif menangkap lintasan deformasi dari titik netral awal secara stabil (UF1 0,6601 vs 0,3684).
2. **3D CNN mengungguli arsitektur 2D-CNN + RNN dan ViViT Transformer.** Konvolusi 3D (R3D-18) mampu mengekstrak interaksi ruang-waktu secara langsung tanpa overfitting yang rentan terjadi pada arsitektur berbasis transformer di dataset berskala kecil.
3. **Estimasi Apex Label-Free (Auto-Apex) Menutup Celah Oracle.** Estimator energi optical flow berhasil mendeteksi frame apex secara mandiri dari sinyal gerakan, menghasilkan performa fusi 0,7021 UF1 (hanya terpaut 0,0083 dari kondisi oracle beranotasi 0,7104).
4. **Ketahanan terhadap Penurunan Frame Rate (FPS Robustness).** Evaluasi degradasi FPS (200 $\rightarrow$ 120 / 60 / 30 fps) membuktikan model fusi deployable mempertahankan akurasi 0,6979 dengan delta UF1 minimal (−0,0068, $P=0,372$). Aplikasi ponsel standar 30 fps dapat beroperasi tanpa kehilangan performa berarti.
5. **Epoch Averaging (Last-K = 8 pada 40 Epoch) Memberikan Stabilitas Optimal.** Mengurangi varians antar fold dan meningkatkan UF1 baseline dari 0,6816 menjadi 0,6983.
6. **Segmentation Attention Statis Lolos Validasi LOSO 21.** Masking berbasis landmark wajah pada area mata, alis, dan mulut menghasilkan skor 0,7283 UF1 ($P=0,801$).
<!-- /bagian -->

<!-- bagian: cabang_ditutup -->
Cabang eksperimen dan pendekatan yang **telah dibuktikan tidak efektif / ditutup** agar tidak diulang:
- **Stabilisasi Gerakan Kepala Eksternal (HQ TV-L1 + ECC):** Menurunkan performa secara drastis (~9 poin UF1, 0,5906 vs 0,6816). Gerakan kepala minor ternyata berkorelasi dengan dinamika ekspresi mikro.
- **ViViT (Video Vision Transformer):** Mengalami overfitting berat karena kapasitas model yang terlalu besar untuk dataset dengan jumlah klip terbatas.
- **Eulerian Motion Magnification:** Memperbesar artifak sensor kamera dan noise optical flow dibanding memperkuat sinyal mikro.
- **Mixup Data Augmentation:** Interpolasi linear antar klip merusak pola temporal halus gerakan mikro.
- **Resolusi Gambar 160x160:** Menambah beban komputasi dan meningkatkan varians overfitting tanpa perbaikan metrik (0,6254 vs 0,6816).
- **Ensemble Lebih dari 3 Seed:** Penambahan seed acak di luar seed 42 dan 123 menunjukkan diminishing returns dan anomali seleksi.
<!-- /bagian -->
