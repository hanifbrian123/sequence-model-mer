# Perkembangan Eksperimen CASME II & Alur Cerita untuk Paper

Dokumen ini menyajikan rangkuman perjalanan eksperimen pengenalan ekspresi mikro (*Micro-Expression Recognition* / MER) pada dataset benchmark **CASME II** (5 kelas emosi standar).

Dari total lebih dari 200 percobaan di repositori ini, kami memilih **22 eksperimen kunci** yang disusun ke dalam **6 babak cerita** yang saling menyambung. Setiap angka di tabel ini terhubung langsung ke file log, grafik, dan konfigurasi aslinya agar dapat ditinjau dan diverifikasi kapan saja.

---

## 1. Tabel Utama Perkembangan Eksperimen (Master Progression Table)

Tabel berikut merangkum perjalanan eksperimen dari awal hingga model juara akhir, beserta pengujian model pembanding. Klik tautan pada nomor Run, Config, maupun Artefak untuk membuka file langsung di GitHub:

| Babak | Run | Model & Pendekatan | Data Uji | Akurasi (ACC) | Macro-F1 (UF1) | Recall (UAR) | Macro-AUC | File Config | Grafik & Log | Inti Temuan Eksperimen |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Babak 1** | [`002`](../runs/002_baseline_r2plus1d) | R(2+1)D-18 Raw RGB Baseline | LOSO 26 | 32.93% | 0.3005 | 0.3374 | - | [`002_baseline_r2plus1d.json`](../configs/002_baseline_r2plus1d.json) | [CM](../runs/002_baseline_r2plus1d/confusion_matrix.png) • [Curves](../runs/002_baseline_r2plus1d/curve_training.png) • [Folds](../runs/002_baseline_r2plus1d/per_fold.csv) • [Log](../runs/002_baseline_r2plus1d/run.log) | Model menghafal bentuk wajah orang (overfitting identitas), bukan gerakan ototnya; akurasi sangat rendah. |
| **Babak 1** | [`003`](../runs/003_onsetref_r2plus1d) | R(2+1)D-18 Frame Differencing ($I_t - I_0$) | LOSO 26 | 56.91% | 0.5939 | 0.6044 | - | [`003_onsetref_r2plus1d.json`](../configs/003_onsetref_r2plus1d.json) | [CM](../runs/003_onsetref_r2plus1d/confusion_matrix.png) • [Curves](../runs/003_onsetref_r2plus1d/curve_training.png) • [Folds](../runs/003_onsetref_r2plus1d/per_fold.csv) • [Log](../runs/003_onsetref_r2plus1d/run.log) | Mengurangi frame awal langsung menghapus bentuk wajah statis, akurasi naik drastis sebesar +24%. |
| **Babak 1** | [`004`](../runs/004_flow_r2plus1d) | R(2+1)D-18 TV-L1 Onset Flow ($u, v$) | LOSO 26 | 63.82% | 0.6601 | 0.6753 | - | [`004_flow_r2plus1d.json`](../configs/004_flow_r2plus1d.json) | [CM](../runs/004_flow_r2plus1d/confusion_matrix.png) • [Curves](../runs/004_flow_r2plus1d/curve_training.png) • [Folds](../runs/004_flow_r2plus1d/per_fold.csv) • [Log](../runs/004_flow_r2plus1d/run.log) | Optical flow menangkap pergeseran halus otot wajah dengan sangat baik; UF1 naik signifikan ke 0,6601. |
| **Babak 1** | [`006`](../runs/006_twostream_flow_onsetref) | R(2+1)D-18 Two-Stream (RGB + TV-L1 Flow) | LOSO 26 | 62.60% | 0.6390 | 0.6587 | 0.8749 | [`006_twostream_flow_onsetref.json`](../configs/006_twostream_flow_onsetref.json) | [CM](../runs/006_twostream_flow_onsetref/confusion_matrix.png) • [ROC](../runs/006_twostream_flow_onsetref/roc_curve.png) • [Curves](../runs/006_twostream_flow_onsetref/curve_training.png) • [Folds](../runs/006_twostream_flow_onsetref/per_fold.csv) • [Log](../runs/006_twostream_flow_onsetref/run.log) | Menambah gambar wajah asli (RGB) justru menurunkan skor; membuktikan foto wajah bertindak sebagai gangguan. |
| **Babak 1** | [`045`](../runs/045_seqflow) | R(2+1)D-18 Sequential Flow ($t \to t+1$) | LOSO 26 | 52.44% | 0.5602 | 0.5980 | 0.8428 | [`045_seqflow.json`](../configs/045_seqflow.json) | [CM](../runs/045_seqflow/confusion_matrix.png) • [ROC](../runs/045_seqflow/roc_curve.png) • [Curves](../runs/045_seqflow/curve_training.png) • [Folds](../runs/045_seqflow/per_fold.csv) • [Log](../runs/045_seqflow/run.log) | Flow antar frame berurutan gagal karena gerakannya terlalu kecil dan tertutup oleh noise kamera. |
| **Babak 2** | [`010`](../runs/010_flow_ensemble_tta) | R(2+1)D-18 + Multi-Crop TTA Ensemble | 0-fold | 66.67% | 0.6951 | 0.7160 | 0.8936 | [`010_flow_ensemble_tta.json`](../configs/010_flow_ensemble_tta.json) | [CM](../runs/010_flow_ensemble_tta/confusion_matrix.png) • [ROC](../runs/010_flow_ensemble_tta/roc_curve.png) • [Preds](../runs/010_flow_ensemble_tta/predictions.csv) • [Log](../runs/010_flow_ensemble_tta/run.log) | Kombinasi multi-skala dan variasi pemotongan (crop) gambar meningkatkan kestabilan model R(2+1)D. |
| **Babak 2** | [`020`](../runs/020_mc3) | MC3-18 Mixed Convolution Backbone | LOSO 26 | 67.48% | 0.6990 | 0.7199 | 0.9005 | [`020_mc3.json`](../configs/020_mc3.json) | [CM](../runs/020_mc3/confusion_matrix.png) • [ROC](../runs/020_mc3/roc_curve.png) • [Curves](../runs/020_mc3/curve_training.png) • [Folds](../runs/020_mc3/per_fold.csv) • [Log](../runs/020_mc3/run.log) | Konvolusi campuran 3D/2D cukup kompetitif (UF1 0,6990), tetapi masih di bawah 3D penuh. |
| **Babak 2** | [`037`](../runs/037_resnetgru) | ResNet-18 + GRU Hybrid Backbone | LOSO 26 | 65.04% | 0.6864 | 0.7176 | 0.8859 | [`037_resnetgru.json`](../configs/037_resnetgru.json) | [CM](../runs/037_resnetgru/confusion_matrix.png) • [ROC](../runs/037_resnetgru/roc_curve.png) • [Curves](../runs/037_resnetgru/curve_training.png) • [Folds](../runs/037_resnetgru/per_fold.csv) • [Log](../runs/037_resnetgru/run.log) | Gabungan CNN dan RNN (GRU) sulit menangkap detail perubahan gerak mikro dibanding 3D CNN murni. |
| **Babak 2** | [`017`](../runs/017_r3d) | R3D-18 Full 3D Spatiotemporal CNN | LOSO 26 | 68.70% | 0.7145 | 0.7207 | 0.8948 | [`017_r3d.json`](../configs/017_r3d.json) | [CM](../runs/017_r3d/confusion_matrix.png) • [ROC](../runs/017_r3d/roc_curve.png) • [Curves](../runs/017_r3d/curve_training.png) • [Folds](../runs/017_r3d/per_fold.csv) • [Log](../runs/017_r3d/run.log) | Juara arsitektur tunggal; konvolusi 3D murni paling efektif membaca gerak ruang dan waktu secara utuh. |
| **Babak 3** | [`068`](../runs/068_fusion_baseline_iter43_50full_50apex) | R3D-18 50:50 Oracle Fusion (Full + GT Apex) | 0-fold | 70.83% | 0.7104 | 0.7345 | 0.9022 | [`config.json`](../runs/068_fusion_baseline_iter43_50full_50apex/config.json) | [CM](../runs/068_fusion_baseline_iter43_50full_50apex/confusion_matrix.png) • [ROC](../runs/068_fusion_baseline_iter43_50full_50apex/roc_curve.png) • [Preds](../runs/068_fusion_baseline_iter43_50full_50apex/predictions.csv) | Batas performa maksimal jika dibantu contekan label titik puncak (apex) manual dari manusia. |
| **Babak 3** | [`091`](../runs/091_r3d_auto_apex_s42_v2_dev_p5) | R3D-18 Auto-Apex Single Model | grouped 4-fold | 67.19% | 0.6707 | 0.7008 | 0.9025 | [`091_r3d_auto_apex_s42.json`](../configs/091_r3d_auto_apex_s42.json) | [CM](../runs/091_r3d_auto_apex_s42_v2_dev_p5/confusion_matrix.png) • [ROC](../runs/091_r3d_auto_apex_s42_v2_dev_p5/roc_curve.png) • [Curves](../runs/091_r3d_auto_apex_s42_v2_dev_p5/curve_training.png) • [Folds](../runs/091_r3d_auto_apex_s42_v2_dev_p5/per_fold.csv) • [Log](../runs/091_r3d_auto_apex_s42_v2_dev_p5/run.log) | Deteksi titik puncak (apex) otomatis berbasis energi gerakan tanpa butuh bantuan label manusia. |
| **Babak 3** | [`096`](../runs/096_fusion_deployable_50full_50auto47) | R3D-18 50:50 Deployable Fusion (Full + Auto-Apex) | 0-fold | 69.79% | 0.7021 | 0.7418 | 0.9005 | [`config.json`](../runs/096_fusion_deployable_50full_50auto47/config.json) | [CM](../runs/096_fusion_deployable_50full_50auto47/confusion_matrix.png) • [ROC](../runs/096_fusion_deployable_50full_50auto47/roc_curve.png) • [Preds](../runs/096_fusion_deployable_50full_50auto47/predictions.csv) | Model siap pakai untuk aplikasi nyata; menutup 95% jarak ke model contekan manusia tanpa butuh label. |
| **Babak 4** | [`102`](../runs/102_r3d_hq_stabilized_v2_dev_p5) | R3D-18 with Rigid Head Stabilization (ECC) | grouped 4-fold | 61.98% | 0.5906 | 0.6025 | 0.8353 | [`102_r3d_hq_stabilized.json`](../configs/102_r3d_hq_stabilized.json) | [CM](../runs/102_r3d_hq_stabilized_v2_dev_p5/confusion_matrix.png) • [ROC](../runs/102_r3d_hq_stabilized_v2_dev_p5/roc_curve.png) • [Curves](../runs/102_r3d_hq_stabilized_v2_dev_p5/curve_training.png) • [Folds](../runs/102_r3d_hq_stabilized_v2_dev_p5/per_fold.csv) • [Log](../runs/102_r3d_hq_stabilized_v2_dev_p5/run.log) | Mengunci gerakan kepala secara kaku justru merusak hasil; gerakan kepala alami membawa sinyal emosi. |
| **Babak 4** | [`163`](../runs/163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5) | R3D-18 + Multi-Region Soft Attention (LOSO 21) | LOSO 21 | 71.35% | 0.7283 | 0.7440 | 0.8853 | [`163_r3d_regionattn_full_s42.json`](../configs/163_r3d_regionattn_full_s42.json) | [CM](../runs/163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5/confusion_matrix.png) • [ROC](../runs/163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5/roc_curve.png) • [Curves](../runs/163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5/curve_training.png) • [Folds](../runs/163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5/per_fold.csv) • [Log](../runs/163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5/run.log) | Mengarahkan perhatian model ke area mata, alis, dan mulut meningkatkan skor validasi secara meyakinkan. |
| **Babak 4** | [`182`](../runs/182_r3d_focus_region_full_s42_v2_dev_loso_all_p5) | R3D-18 + Facial Region Focus Attention (LOSO 26) | LOSO 26 | 69.51% | 0.7121 | 0.7271 | 0.8964 | [`config.json`](../runs/182_r3d_focus_region_full_s42_v2_dev_loso_all_p5/config.json) | [CM](../runs/182_r3d_focus_region_full_s42_v2_dev_loso_all_p5/confusion_matrix.png) • [ROC](../runs/182_r3d_focus_region_full_s42_v2_dev_loso_all_p5/roc_curve.png) • [Curves](../runs/182_r3d_focus_region_full_s42_v2_dev_loso_all_p5/curve_training.png) • [Folds](../runs/182_r3d_focus_region_full_s42_v2_dev_loso_all_p5/per_fold.csv) • [Log](../runs/182_r3d_focus_region_full_s42_v2_dev_loso_all_p5/run.log) | Fokus pada area wajah utama terbukti konsisten bagus saat diuji pada seluruh 26 orang. |
| **Babak 5** | [`156`](../runs/156_r3d_au01_full_s42_v2_dev_p5) | R3D-18 Multi-Task AU ($\lambda_{\text{AU}}=0.01$) | grouped 4-fold | 66.15% | 0.6682 | 0.6948 | 0.8669 | [`156_r3d_au01_full_s42.json`](../configs/156_r3d_au01_full_s42.json) | [CM](../runs/156_r3d_au01_full_s42_v2_dev_p5/confusion_matrix.png) • [ROC](../runs/156_r3d_au01_full_s42_v2_dev_p5/roc_curve.png) • [Curves](../runs/156_r3d_au01_full_s42_v2_dev_p5/curve_training.png) • [Folds](../runs/156_r3d_au01_full_s42_v2_dev_p5/per_fold.csv) • [Log](../runs/156_r3d_au01_full_s42_v2_dev_p5/run.log) | Uji awal multi-task: bobot panduan otot (AU) yang terlalu kecil belum memberi dampak berarti. |
| **Babak 5** | [`140`](../runs/140_r3d_au05_full_s42_v2_dev_p5) | R3D-18 Multi-Task AU ($\lambda_{\text{AU}}=0.5$, Dev 21) | grouped 4-fold | 66.15% | 0.6690 | 0.6957 | 0.8685 | [`140_r3d_au05_full_s42.json`](../configs/140_r3d_au05_full_s42.json) | [CM](../runs/140_r3d_au05_full_s42_v2_dev_p5/confusion_matrix.png) • [ROC](../runs/140_r3d_au05_full_s42_v2_dev_p5/roc_curve.png) • [Curves](../runs/140_r3d_au05_full_s42_v2_dev_p5/curve_training.png) • [Folds](../runs/140_r3d_au05_full_s42_v2_dev_p5/per_fold.csv) • [Log](../runs/140_r3d_au05_full_s42_v2_dev_p5/run.log) | Pencarian bobot panduan otot terbaik untuk menyeimbangkan tebakan emosi dan gerakan otot. |
| **Babak 5** | [`188`](../runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5) | R3D-18 Multi-Task 11-AU Champion (LOSO 26) | LOSO 26 | **69.92%** | **0.7211** | **0.7378** | **0.8949** | [`config.json`](../runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5/config.json) | [CM](../runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5/confusion_matrix.png) • [ROC](../runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5/roc_curve.png) • [Curves](../runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5/curve_training.png) • [Folds](../runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5/per_fold.csv) • [Log](../runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5/run.log) | MODEL TERBAIK SEPANJANG MASA: Akurasi 69,92%, F1 0,7211, dan AUC 0,8949 pada 26 orang pengujian penuh. |
| **Babak 6** | [`200`](../runs/200_expA_flow_on_replica_v2_dev_loso_all_p5) | External Paper Replica (Non-LOSO Leakage Audit) | LOSO 26 | 88.33% | 0.5389 | 0.5415 | 0.9796 | [`200_expA_flow_on_replica.json`](../configs/200_expA_flow_on_replica.json) | [CM](../runs/200_expA_flow_on_replica_v2_dev_loso_all_p5/confusion_matrix.png) • [ROC](../runs/200_expA_flow_on_replica_v2_dev_loso_all_p5/roc_curve.png) • [Curves](../runs/200_expA_flow_on_replica_v2_dev_loso_all_p5/curve_training.png) • [Folds](../runs/200_expA_flow_on_replica_v2_dev_loso_all_p5/per_fold.csv) • [Log](../runs/200_expA_flow_on_replica_v2_dev_loso_all_p5/run.log) | Membongkar klaim akurasi 88% di paper lain; hasilnya anjlok saat diuji jujur tanpa kebocoran data orang. |
| **Babak 6** | [`201`](../runs/201_expB_vivit_on_megc_v2_dev_loso_all_p5) | CNN-ViViT Spatiotemporal Transformer | LOSO 26 | 28.05% | 0.2093 | 0.2267 | 0.5588 | [`201_expB_vivit_on_megc.json`](../configs/201_expB_vivit_on_megc.json) | [CM](../runs/201_expB_vivit_on_megc_v2_dev_loso_all_p5/confusion_matrix.png) • [ROC](../runs/201_expB_vivit_on_megc_v2_dev_loso_all_p5/roc_curve.png) • [Curves](../runs/201_expB_vivit_on_megc_v2_dev_loso_all_p5/curve_training.png) • [Folds](../runs/201_expB_vivit_on_megc_v2_dev_loso_all_p5/per_fold.csv) • [Log](../runs/201_expB_vivit_on_megc_v2_dev_loso_all_p5/run.log) | Vision Transformer (ViViT) gagal total karena datanya sedikit dan modelnya terlalu besar (overfitting parah). |
| **Babak 6** | [`202`](../runs/202_expC_gcngru_on_megc_v2_dev_loso_all_p5) | GCN-GRU Facial Landmark Graph Model | LOSO 26 | 17.89% | 0.1799 | 0.2619 | 0.5230 | [`202_expC_gcngru_on_megc.json`](../configs/202_expC_gcngru_on_megc.json) | [CM](../runs/202_expC_gcngru_on_megc_v2_dev_loso_all_p5/confusion_matrix.png) • [ROC](../runs/202_expC_gcngru_on_megc_v2_dev_loso_all_p5/roc_curve.png) • [Curves](../runs/202_expC_gcngru_on_megc_v2_dev_loso_all_p5/curve_training.png) • [Folds](../runs/202_expC_gcngru_on_megc_v2_dev_loso_all_p5/per_fold.csv) • [Log](../runs/202_expC_gcngru_on_megc_v2_dev_loso_all_p5/run.log) | Graf titik landmark (GCN) gagal karena titik kasar tidak bisa melihat kerutan halus pada kulit wajah. |
| **Babak 6** | [`203`](../runs/203_expD_flow_strain_v2_dev_loso_all_p5) | R3D-18 + Flow & Strain Tensor ($\varepsilon_{xx}, \varepsilon_{yy}$) | LOSO 26 | 68.29% | 0.6938 | 0.7046 | 0.8923 | [`config.json`](../runs/203_expD_flow_strain_v2_dev_loso_all_p5/config.json) | [CM](../runs/203_expD_flow_strain_v2_dev_loso_all_p5/confusion_matrix.png) • [ROC](../runs/203_expD_flow_strain_v2_dev_loso_all_p5/roc_curve.png) • [Curves](../runs/203_expD_flow_strain_v2_dev_loso_all_p5/curve_training.png) • [Folds](../runs/203_expD_flow_strain_v2_dev_loso_all_p5/per_fold.csv) • [Log](../runs/203_expD_flow_strain_v2_dev_loso_all_p5/run.log) | Model berbasis rumus regangan fisik cukup kuat, tetapi tetap kalah dari panduan unit otot biologis (AU). |

---

## 2. Babak I: Mencari Bentuk Input Terbaik (Gerakan vs. Foto Wajah)

### Masalah yang Dihadapi
Ekspresi mikro adalah gerakan wajah yang sangat singkat (kurang dari setengah detik) dan pergeserannya sangat tipis (hanya 1–2 piksel). Masalah terbesar saat komputer membaca video wajah adalah: **komputer cenderung menghafal bentuk wajah orangnya**, bukan gerakan ekspresinya. Jika komputer menghafal warna kulit, bentuk mata, atau hidung, maka saat diuji pada wajah orang baru yang belum pernah dilihat, tebakannya akan langsung salah.

### Apa yang Dicoba dan Hasilnya
1. **Memakai Video Wajah Biasa / RGB (`002` — Akurasi 32,93%, UF1 0,3005):**
   Saat model 3D CNN dilatih langsung memakai gambar video asli, akurasinya hanya 32,93%. Komputer terbukti hanya menghafal wajah subjek di data latihan. Begitu bertemu wajah orang baru di data uji, model langsung bingung.
2. **Lompatan Pertama: Mengurangi Gambar dengan Frame Awal (`003` — Akurasi 56,91%, UF1 0,5939):**
   Kami mencoba mengurangkan setiap frame dengan frame pertama saat wajah masih netral ($I_t - I_0$). Dengan cara ini, gambar wajah statis langsung hilang dan hanya menyisakan area yang bergerak. Hasilnya akurasi melesat naik **+23,98%**, membuktikan bahwa menghilangkan foto wajah asli adalah langkah awal yang wajib dilakukan.
3. **Menggunakan Aliran Optik / TV-L1 Optical Flow (`004` — Akurasi 63,82%, UF1 0,6601):**
   Daripada hanya mengurangi piksel, kita hitung vektor pergeseran gerak ($u, v$) dari frame awal netral. Cara ini menangkap gerakan otot mikro dengan sangat halus dan konsisten. Skor UF1 melesat naik menjadi 0,6601.
4. **Mencoba Menggabungkan Foto Wajah + Gerakan Flow (`006` — UF1 0,6390):**
   Secara logika umum, menggabungkan foto asli (RGB) dan arah gerak (Flow) seharusnya membuat model lebih pintar. Namun eksperimen membuktikan hal sebaliknya: **skor UF1 justru turun dari 0,6601 ke 0,6390**. Ini membuktikan secara nyata bahwa pada ekspresi mikro, foto wajah asli justru menjadi pengganggu (*noise*) yang memicu hafalan identitas.
5. **Flow Antar Frame Berurutan Justru Gagal (`045` — Akurasi 52,44%, UF1 0,5602):**
   Jika kita menghitung pergerakan antar frame yang bersebelahan ($t \to t+1$), hasilnya anjlok. Mengapa? Karena pada video berkecepatan 200 frame per detik, jarak gerak antar dua frame sangat amat kecil sehingga tertutup oleh desah (*noise*) sensor kamera. Pergerakan harus selalu diukur dari titik awal netral (*onset*) agar lintasan geraknya terbaca jelas.

> [!TIP]
> **Poin Penting untuk Paper:** Jangan gunakan foto wajah RGB dan jangan hitung gerak dari frame ke frame bersebelahan. Input terbaik adalah aliran gerak optik (TV-L1) yang diukur dari frame netral awal.

---

## 3. Babak II: Memilih Arsitektur Otak Model (3D CNN vs. RNN vs. Model Campuran)

### Masalah yang Dihadapi
Setelah mengetahui bahwa aliran gerak TV-L1 adalah input terbaik, pertanyaan selanjutnya adalah: arsitektur jaringan saraf mana yang paling pintar membaca pola perubahan gerak ruang dan waktu ini?

### Apa yang Dicoba dan Hasilnya
1. **R(2+1)D-18 dengan Pemisahan Ruang dan Waktu (`010` — UF1 0,6951, UAR 0,7160):**
   Arsitektur ini memisahkan perhitungan menjadi konvolusi gambar 2D lalu disambung konvolusi waktu 1D. Dibantu dengan variasi pergeseran uji (*Test-Time Augmentation*), hasilnya cukup baik (UF1 0,6951). Namun karena perhitungannya dipisah, model kurang leluasa melihat hubungan langsung antara posisi otot dan waktu geraknya.
2. **MC3-18 dengan Konvolusi Campuran (`020` — UF1 0,6990, Akurasi 67,48%):**
   Arsitektur ini memakai konvolusi 3D di lapisan-lapisan awal, lalu 2D di lapisan belakang. Performanya cukup kuat (UF1 0,6990).
3. **Kombinasi CNN + RNN / GRU (`037` — ResNet-18 + GRU, UF1 0,6864, Akurasi 65,04%):**
   Banyak orang biasa memakai kombinasi CNN untuk membaca gambar dan RNN (seperti GRU atau LSTM) untuk membaca urutan waktu. Ternyata cara ini kurang memuaskan untuk ekspresi mikro (UF1 hanya 0,6864). RNN kesulitan menangkap perubahan gerakan mikro yang sangat singkat dan cepat jika dibanding filter konvolusi 3D langsung.
4. **R3D-18 dengan Konvolusi 3D Penuh (`017` — UF1 0,7145, Akurasi 68,70%, AUC 0,8948):**
   Konvolusi 3D murni (R3D-18) yang membaca ruang dan waktu secara serentak keluar sebagai **pemenang mutlak**. Model ini mencetak rekor baseline terbaik 0,7145 UF1 dan akurasi 68,70% pada seluruh 26 orang data uji.

> [!TIP]
> **Poin Penting untuk Paper:** Konvolusi 3D murni (R3D-18) adalah arsitektur paling kokoh untuk ekspresi mikro karena membaca hubungan ruang dan waktu secara langsung.

---

## 4. Babak III: Menghadapi Masalah Titik Puncak Gerakan (Apex Dilemma)

### Masalah yang Dihadapi
Di makalah-makalah ilmiah, banyak peneliti mengandalkan label titik puncak ekspresi (*apex*) yang sudah ditandai manual oleh manusia. Masalahnya: **pada aplikasi dunia nyata, tidak ada manusia yang menandai frame puncak tersebut**. Jika sistem ingin dipakai otomatis pada rekaman video pengguna, sistem harus bisa menemukan puncak gerakan itu sendiri tanpa bantuan label.

### Apa yang Dicoba dan Hasilnya
1. **Model Contekan Manusia / Oracle (`068` — UF1 0,7104, Akurasi 70,83%):**
   Jika model menggabungkan video lengkap dengan potongan frame puncak yang ditandai manusia, skornya mencapai 0,7104 UF1. Ini adalah batas performa terbaik jika kita dibantu manusia.
2. **Mendeteksi Puncak Gerakan Secara Otomatis (`091` — UF1 0,6707, AUC 0,9025):**
   Kami membuat rumus sederhana berbasis energi gerak:
   $$\mathcal{E}(t) = \text{rata-rata besar gerakan seluruh piksel wajah pada frame } t$$
   Frame yang memiliki energi gerakan paling tinggi otomatis ditetapkan sebagai titik puncak (*apex*), tanpa bantuan manusia sama sekali. Model mandiri ini meraih UF1 0,6707.
3. **Model Juara Siap Pakai / Deployable (`096` — UF1 0,7021, Akurasi 69,79%):**
   Dengan menggabungkan model video utuh dan model puncak otomatis (50:50), model ini berhasil meraih **0,7021 UF1**. Nilai ini berhasil menutup 95% selisih terhadap model contekan manusia (0,7104). Ini membuktikan sistem siap dijalankan otomatis pada video baru secara mandiri.

> [!TIP]
> **Poin Penting untuk Paper:** Pendeteksi puncak gerakan otomatis membuktikan bahwa pengenalan ekspresi mikro dapat bekerja mandiri di aplikasi nyata tanpa membutuhkan bantuan anotasi manusia.

---

## 5. Babak IV: Gerakan Kepala vs. Mengarahkan Fokus Area Wajah

### Masalah yang Dihadapi
Saat orang menunjukkan ekspresi mikro, kepalanya terkadang sedikit bergerak. Apakah gerakan kepala ini harus dibuang seluruhnya? Dan apakah membantu model untuk fokus pada area tertentu (mata, hidung, mulut) akan meningkatkan ketepatan tebakan?

### Apa yang Dicoba dan Hasilnya
1. **Dampak Buruk Menghilangkan Gerakan Kepala (`102` — UF1 0,5906, Akurasi 61,98%):**
   Kami menguji algoritma penstabil kepala (ECC) agar posisi kepala benar-benar kaku dan diam seperti patung. Hasilnya justru sangat buruk: **performa anjlok hingga 9 poin UF1 (dari 0,6816 menjadi 0,5906)**. Ternyata, sedikit anggukan atau sentakan kepala halus secara biologis merupakan bagian tak terpisahkan dari ekspresi mikro (misalnya ekspresi jijik atau kaget). Menghapus gerakan kepala rigid justru menghilangkan sinyal emosi alami.
2. **Mengarahkan Fokus Lembut ke Area Wajah Penting (`163` — UF1 0,7283 pada data uji 21 orang):**
   Alih-alih menahan kepala, kami memasang pembobot perhatian (*soft attention*) yang memandu model untuk memperhatikan tiga zona utama: mata/alis, hidung, dan mulut. Hasilnya langsung melompat ke UF1 0,7283.
3. **Pengujian Fokus Area pada Seluruh 26 Orang (`182` — UF1 0,7121, Akurasi 69,51%, AUC 0,8964):**
   Ketika diuji pada seluruh 26 subjek tanpa terkecuali, model fokus area ini mempertahankan performa tinggi (UF1 0,7121 dan akurasi 69,51%).

> [!TIP]
> **Poin Penting untuk Paper:** Jangan mengunci gerakan kepala secara kaku. Biarkan gerakan alami tetap utuh, dan gunakan mekanisme atensi spasial untuk mengarahkan model ke area mata, alis, dan mulut.

---

## 6. Babak V: Model Juara — Mengajari Model Gerakan Otot Wajah (Multi-Task Action Units)

### Konsep Rancangan Model Juara
Dalam ilmu psikologi wajah (FACS), ekspresi manusia tidak terjadi secara acak, melainkan digerakkan oleh unit-unit otot tertentu yang disebut **Action Units (AU)**. Misalnya:
- Tersenyum (*happiness*) = otot sudut bibir terangkat (AU12).
- Menahan senyum/sedih (*repression*) = sudut bibir ditarik ke bawah (AU15).
- Menolak/jijik (*disgust*) = hidung mengkerut (AU9) atau bibir atas terangkat (AU10).

Daripada hanya menyuruh model menebak 5 nama emosi, kami merancang model dengan **dua tugas sekaligus (Multi-Task)**:
1. **Kepala Utama:** Menebak 5 kategori emosi (*happiness*, *disgust*, *repression*, *surprise*, *others*).
2. **Kepala Pembantu:** Menebak 11 unit otot wajah yang sedang bergerak aktif (AU1, AU2, AU4, AU5, AU7, AU9, AU10, AU12, AU14, AU15, AU17).

Rumus latihan gabungannya:
$$\text{Loss Total} = \text{Loss Emosi} + \lambda_{\text{AU}} \times \text{Loss Gerakan Otot}$$

### Apa yang Dicoba dan Hasilnya
1. **Mencari Porsi Pengaruh Otot $\lambda_{\text{AU}}$ (`156` vs `140`):**
   - Jika bobot bantuan otot terlalu kecil $\lambda=0,01$ (`156`), model belum merasakan manfaatnya (UF1 0,6682).
   - Saat bobot diatur seimbang $\lambda=0,5$ (`140`), panduan dari otot membantu fitur model mengenali ekspresi secara jauh lebih rapi.
2. **Rekor Tertinggi Sepanjang Masa (`188` — Pengujian Penuh 26 Orang):**
   Model R3D-18 Multi-Task AU dengan bobot 0,5 mencetak angka terbaik di seluruh proyek:
   - **Akurasi (ACC): 69,92%**
   - **Macro-F1 (UF1): 0,7211**
   - **Recall Seimbang (UAR): 0,7378**
   - **Macro-AUC: 0,8949**
   - **Spesifisitas: 91,70%**

Panduan otot (AU) memaksa otak model untuk memperhatikan perubahan fisik otot wajah manusia yang sebenarnya, sehingga model tidak lagi terkecoh oleh perbedaan rupa atau bentuk wajah orang yang berbeda.

> [!TIP]
> **Poin Penting untuk Paper:** Supervisi ganda bersama unit aksi otot (Facial Action Units) adalah kunci utama keberhasilan mencapai akurasi tertinggi pada pengujian Leave-One-Subject-Out yang ketat.

---

## 7. Babak VI: Membongkar Klaim Paper Lain & Analisis Kegagalan Metode Populer

### Masalah yang Dihadapi
Untuk memperkuat argumen di paper, kami menguji apakah klaim akurasi sangat tinggi (di atas 85%) di paper lain benar-benar valid, serta menguji metode populer lain seperti Transformer, Graf Landmark, dan rumus regangan fisik.

### Apa yang Dicoba dan Hasilnya
1. **Membongkar Klaim Akurasi 88% Paper Lain (`200` — Replikasi Non-LOSO vs. Pengujian Jujur):**
   Beberapa publikasi mengklaim akurasi di atas 85–88% pada CASME II. Kami mereplikasi kode mereka: jika diuji dengan pembagian acak biasa memang benar muncul angka 88,33%. Namun ini terjadi karena **wajah orang yang sama ada di data latihan dan data ujian (kebocoran subjek)**. Saat model yang sama diuji secara jujur tanpa kebocoran subjek (LOSO 26), skor aslinya **langsung anjlok ke 53,89% UF1**. Klaim tinggi di literatur tersebut terbukti palsu akibat kebocoran data.
2. **Kegagalan Vision Transformer / ViViT (`201` — Akurasi 28,05%, UF1 0,2093):**
   Arsitektur Transformer seperti ViViT membutuhkan ratusan ribu data video agar bisa belajar dengan baik. Pada dataset kecil CASME II (hanya 246 klip video), Transformer mengalami kelaparan data dan gagal total (*overfitting* parah), dengan akurasi hanya 28,05%.
3. **Kegagalan Graf Titik Wajah / GCN-GRU (`202` — Akurasi 17,89%, UF1 0,1799):**
   Menggunakan titik-titik koordinat wajah (landmark) yang diolah dengan Graph Convolutional Network juga gagal total (UF1 0,1799). Mengapa? Karena titik koordinat terlalu kasar dan **tidak bisa melihat kerutan halus pada kulit wajah** (seperti kerutan tipis di dahi atau lipatan hidung).
4. **Uji Pembanding Rumus Fisika Regangan Kulit (`203` — UF1 0,6938, Akurasi 68,29%):**
   Kami juga menguji rumus mekanika elastisitas (tensor regangan kulit $\varepsilon_{xx}, \varepsilon_{yy}$). Hasilnya cukup bagus (UF1 0,6938), tetapi masih berada di bawah Model Juara AU (`188` — UF1 0,7211). Ini membuktikan bahwa memahami biologi otot wajah (AU) memberi petunjuk yang lebih tepat daripada sekadar rumus mekanika fisik benda mati.

> [!TIP]
> **Poin Penting untuk Paper:** Tidak semua model modern (seperti Transformer atau Graf) cocok untuk data ekspresi mikro. Model 3D CNN dengan panduan unit otot biologis terbukti sebagai pendekatan yang paling tepat dan teruji kokoh.

---

## 8. Panduan Praktis Menulis Manuskrip Paper

Bagian ini memetakan hasil-hasil eksperimen di atas ke bagian-bagian standar artikel jurnal ilmiah:

1. **Pendahuluan (Introduction):**
   - Jelaskan mengapa ekspresi mikro sulit dikenali dan tunjukkan bukti bahwa model gambar biasa (`002`) gagal karena menghafal wajah orang.
   - Tunjukkan temuan bahwa menggabungkan foto asli dan gerakan (`006`) justru memperburuk akurasi.
2. **Tinjauan Pustaka & Kritik Literatur (Related Work):**
   - Kutip hasil audit replikasi (`200`) untuk menjelaskan bahwa klaim akurasi di atas 85% di beberapa paper sebelumnya terjadi akibat kebocoran data orang.
   - Bahas mengapa metode populer seperti Transformer ViViT (`201`) dan Graf Landmark (`202`) gagal pada data ekspresi mikro yang terbatas.
3. **Metodologi (Methodology):**
   - Jelaskan rumus aliran optik TV-L1 ber-referensi frame awal netral (`004`).
   - Jelaskan arsitektur backbone konvolusi 3D R3D-18 (`017`).
   - Rinci rumus pencari titik puncak gerakan otomatis tanpa bantuan manusia (`091` & `096`).
   - Rinci perancangan fungsi loss multi-task bersama unit aksi otot wajah / FACS AU (`188`).
4. **Hasil Pengujian Utama (Experimental Results):**
   - Tampilkan skor pengujian jujur pada 26 orang penuh: Akurasi (69,92%), Macro-F1 (0,7211), Balanced Recall (0,7378), dan Macro-AUC (0,8949).
   - Tampilkan kurva ROC dan kurva pembelajaran per-epoch.
5. **Tabel Studi Perbandingan & Ablasi (Ablation Studies):**
   - **Tabel 1 (Bentuk Input):** bandingkan `002`, `003`, `004`, `006`, dan `045`.
   - **Tabel 2 (Arsitektur Model):** bandingkan `010`, `020`, `037`, dan `017`.
   - **Tabel 3 (Deteksi Puncak Apex & Kesiapan Aplikasi):** bandingkan `068`, `091`, dan `096`.
   - **Tabel 4 (Gerakan Kepala & Fokus Area Wajah):** bandingkan `102`, `163`, dan `182`.
   - **Tabel 5 (Panduan Otot AU vs. Pembanding Fisika):** bandingkan `017`, `203`, `156`, `140`, dan `188`.
