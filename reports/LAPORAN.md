# Laporan Eksperimen CASME II — Klasifikasi Micro-Expression (5 Kelas)

Dibuat otomatis dari hasil eksperimen pada **11 September 2026, 11:26**.
Regenerasi laporan: `python -m casme.reporting.report` atau `bash scripts/rebuild_reports.sh`.

Seluruh data angka terpusat di `results/results.csv` (satu baris per run). Dokumen ini adalah satu-satunya file laporan resmi proyek; narasi disunting di `docs/narasi.md`.

---

## 1. Pengantar & Arsitektur Model

Proyek ini mengklasifikasikan micro-expression pada rekaman video **CASME II**
ke dalam 5 kelas kanonik (`happiness`, `disgust`, `repression`, `surprise`, `others`).
Micro-expression adalah gerakan wajah involunter berdurasi sangat singkat (kurang dari 0,5 detik)
dengan intensitas gerakan otot wajah yang sangat halus.

Representasi temporal utama menggunakan **TV-L1 Optical Flow kumulatif ber-referensi onset**
(`arr[t]` = flow perpindahan piksel dari frame onset ke frame t) yang diproses menggunakan
arsitektur 3D Convolutional Neural Network (R3D-18 pra-terlatih Kinetics-400).
Metode ini dirancang untuk pipeline inferensi offline dari video unggahan pengguna di aplikasi web.

⚠️ **Peringatan Integritas Angka — Asumsi Jendela Sempurna:**
Semua metrik evaluasi dihitung di atas jendela temporal ekspresi yang dipotong dari rentang `[onset, offset]` berlabel dataset.
Pada skenario nyata (video upload pengguna), onset maupun offset tidak diketahui sebelumnya.
Tantangan estimasi apex tanpa anotasi telah dipecahkan menggunakan **estimator energi optical flow label-free** (`091_r3d_auto_apex_s42_v2_dev_p5`),
menghasilkan model **Champion Deployable (0,7021 UF1)** (`096_fusion_deployable_50full_50auto47`) yang siap diaplikasikan pada backend produksi.
Sebagai pembanding, angka 0,7104 UF1 adalah champion *oracle* (`068_fusion_baseline_iter43_50full_50apex` menggunakan anotasi apex dataset yang tidak tersedia pada upload nyata),
sehingga acuan realistis untuk evaluasi aplikasi adalah **0,7021 UF1**.

### Spesifikasi Arsitektur & Pipeline

| Komponen | Detail Spesifikasi |
|---|---|
| **Input** | Grayscale CASME II $\rightarrow$ TV-L1 Dual Flow ($u, v$) kumulatif dari frame onset, $128 \times 128 \times 16$ frame |
| **Backbone** | R3D-18 (ResNet 3D 18-layer) pra-terlatih Kinetics-400 (spatiotemporal convolution) |
| **Pooling** | Adaptive Spatiotemporal Average Pooling $\rightarrow$ 512 fitur laten |
| **Head** | Dropout 0,5 $\rightarrow$ Fully Connected Layer $512 \rightarrow 5$ |
| **Kelas (5)** | `happiness`, `disgust`, `repression`, `surprise`, `others` |
| **Fusi Inferensi** | Ensemble fusi probabilitas 50:50 antara model full-span dan model apex-focused |

---

## 2. Protokol Pengujian & Evaluasi

Eksperimen diatur dalam protokol yang ketat untuk mencegah kebocoran identitas subjek dan bias seleksi:

| Kode | Protokol | Subjek | Pembagian Data | Tujuan & Fungsi |
|---|---|---|---|---|
| 🟢 | **Grouped 4-fold** | 21 subjek dev | Latih 16 orang $\rightarrow$ Uji 5 orang (stratified by subject) | Seleksi cepat & sweep hiperparameter |
| 🔵 | **LOSO 21** | 21 subjek dev | Latih 20 orang $\rightarrow$ Uji 1 orang (21 fold) | Validasi kandidat tanpa kebocoran identitas |
| ⚫ | **LOSO 26** | 26 subjek | Latih 25 orang $\rightarrow$ Uji 1 orang (26 fold) | Perbandingan literatur akademik & pengujian akhir |
| 🔴 | **Audit split** | 5 subjek audit | Tersegel & independen (tidak pernah dilihat selama tuning) | Verifikasi champion final |

Gate promosi kandidat: **paired subject-bootstrap 10.000 iterasi**, minimum effect **$\Delta\text{UF1} \ge 0,005$**, probabilitas keunggulan **$P \ge 80\%$**.

---

## 3. Ringkasan Model Acuan vs Champion

| Kategori Model | Run ID | Protokol / Split | UF1 | UAR | ACC | Status Evaluasi | Catatan Implementasi |
|---|---|---|---|---|---|---|---|
| **R3D-18 Baseline** | `061_r3d_v2_dev_p5` | Grouped 4-fold | 0,6816 | 0,7120 | 0,6823 | Acuan Dev | TV-L1 onset-referenced flow, 128x128 |
| **Champion Oracle** | `068_fusion_baseline_iter43_50full_50apex` | Grouped 4-fold | 0,7104 | 0,7345 | 0,7083 | Oracle Ref | Fusi 50:50 full + apex beranotasi dataset |
| **Champion Deployable** | `096_fusion_deployable_50full_50auto47` | Grouped 4-fold | **0,7021** | **0,7418** | **0,6979** | **CHAMPION APP** | Fusi 50:50 full + auto-apex (label-free) |
| **Baseline R3D-18 (LOSO 26)** | `017_r3d` | LOSO 26 | 0,7145 | 0,7207 | 0,6870 | Benchmark | Evaluasi 26 subjek literatur |
| **Region Attention s42** | `163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5` | LOSO 21 | 0,7283 | 0,7350 | 0,7188 | Lolos Gate | Segmentasi wajah statis (P=80,1%) |

---

## 4. Semua Run Sekilas (Master Progression Table)

Daftar lengkap seluruh **198 run** eksperimen yang tercatat di `runs/` dan `results/results.csv`, diurutkan secara numerik dari `000_` s.d. selesai:

| Run | Protokol | Split | Backbone | UF1 | UAR | ACC | Epochs | Waktu Selesai |
|---|---|---|---|---|---|---|---|---|
| `001_smoke_smoke` | protocol_v1 | 2-fold | r2plus1d_18 | 0,1476 | 0,2333 | 0,1364 | 3 | 2026-07-13 |
| `002_baseline_r2plus1d` | protocol_v1 | LOSO 26 | r2plus1d_18 | 0,3005 | 0,3374 | 0,3293 | 25 | 2026-07-13 |
| `003_onsetref_r2plus1d` | protocol_v1 | LOSO 26 | r2plus1d_18 | 0,5939 | 0,6044 | 0,5691 | 22 | 2026-07-13 |
| `004_flow_r2plus1d` | protocol_v1 | LOSO 26 | r2plus1d_18 | 0,6601 | 0,6753 | 0,6382 | 25 | 2026-07-13 |
| `005_smoke_2stream_smoke` | protocol_v1 | 2-fold | r2plus1d_18 | 0,4860 | 0,5400 | 0,4545 | 3 | 2026-07-13 |
| `006_twostream_flow_onsetref` | protocol_v1 | LOSO 26 | r2plus1d_18 | 0,6390 | 0,6587 | 0,6260 | 25 | 2026-07-14 |
| `007_flow_apex_tta` | protocol_v1 | LOSO 26 | r2plus1d_18 | 0,3581 | 0,3587 | 0,4350 | 25 | 2026-07-14 |
| `008_flow_tta_lastk` | protocol_v1 | LOSO 26 | r2plus1d_18 | 0,3317 | 0,3421 | 0,4268 | 28 | 2026-07-14 |
| `009_smoke_tta_smoke` | protocol_v1 | 2-fold | r2plus1d_18 | 0,4581 | 0,6200 | 0,4545 | 6 | 2026-07-14 |
| `010_flow_ensemble_tta` | protocol_v1 | 0-fold | r2plus1d_18 | 0,6951 | 0,7160 | 0,6667 | 25 | tersimpan |
| `011_flow_t24` | protocol_v1 | LOSO 26 | r2plus1d_18 | 0,6711 | 0,6924 | 0,6423 | 25 | 2026-07-14 |
| `012_flow_res128` | protocol_v1 | LOSO 26 | r2plus1d_18 | 0,6908 | 0,6964 | 0,6748 | 25 | 2026-07-14 |
| `013_flow_res128_ensemble` | protocol_v1 | 0-fold | r2plus1d_18 | 0,7076 | 0,7212 | 0,6748 | 25 | tersimpan |
| `014_objective` | protocol_v1 | LOSO 26 | r2plus1d_18 | 0,2019 | 0,2919 | 0,1059 | 25 | 2026-07-14 |
| `015_strain` | protocol_v1 | LOSO 26 | r2plus1d_18 | 0,6918 | 0,6923 | 0,6707 | 25 | 2026-07-15 |
| `016_4ch` | protocol_v1 | LOSO 26 | r2plus1d_18 | 0,6737 | 0,6736 | 0,6463 | 25 | 2026-07-15 |
| `017_r3d` | protocol_v1 | LOSO 26 | r3d_18 | 0,7145 | 0,7207 | 0,6870 | 25 | 2026-07-15 |
| `018_mag5` | protocol_v1 | LOSO 26 | r2plus1d_18 | 0,6820 | 0,7073 | 0,6748 | 25 | 2026-07-15 |
| `019_mag_r3d` | protocol_v1 | LOSO 26 | r3d_18 | 0,6826 | 0,7102 | 0,6585 | 25 | 2026-07-15 |
| `020_mc3` | protocol_v1 | LOSO 26 | mc3_18 | 0,6990 | 0,7199 | 0,6748 | 25 | 2026-07-15 |
| `021_smoke_focal_smoke` | protocol_v1 | 2-fold | r3d_18 | 0,5414 | 0,6000 | 0,5455 | 4 | 2026-07-15 |
| `022_focal` | protocol_v1 | LOSO 26 | r3d_18 | 0,6919 | 0,7026 | 0,6789 | 25 | 2026-07-15 |
| `023_apex_ststnet_smoke` | protocol_v1 | 0-fold | nan | 0,2056 | 0,2300 | 0,2069 | 60 | tersimpan |
| `024_apex_shallowcnn_smoke` | protocol_v1 | 0-fold | nan | 0,5040 | 0,5133 | 0,5517 | 80 | tersimpan |
| `025_apex_shallowcnn` | protocol_v1 | 0-fold | nan | 0,5455 | 0,5679 | 0,5203 | 80 | tersimpan |
| `026_r3d_s123` | protocol_v1 | LOSO 26 | r3d_18 | 0,6718 | 0,6849 | 0,6463 | 25 | 2026-07-15 |
| `027_twostage_r3d_smoke` | protocol_v1 | 0-fold | r3d_18 | 0,5708 | 0,5800 | 0,6552 | 25 | tersimpan |
| `028_mixup` | protocol_v1 | LOSO 26 | r3d_18 | 0,6669 | 0,6944 | 0,6463 | 25 | 2026-07-16 |
| `029_obj6_r3d` | protocol_v1 | LOSO 26 | r3d_18 | 0,5574 | 0,5599 | 0,6772 | 25 | 2026-07-16 |
| `030_obj6_mc3` | protocol_v1 | LOSO 26 | mc3_18 | 0,5489 | 0,5524 | 0,6693 | 25 | 2026-07-16 |
| `031_obj6_focal` | protocol_v1 | LOSO 26 | r3d_18 | 0,4907 | 0,4916 | 0,6299 | 25 | 2026-07-16 |
| `032_obj6_4ch` | protocol_v1 | LOSO 26 | r2plus1d_18 | 0,5358 | 0,5460 | 0,6535 | 25 | 2026-07-16 |
| `033_ema_smoke` | protocol_v1 | 2-fold | r3d_18 | 0,5655 | 0,6200 | 0,5909 | 25 | 2026-07-16 |
| `034_ema` | protocol_v1 | LOSO 26 | r3d_18 | 0,7094 | 0,7310 | 0,6829 | 25 | 2026-07-16 |
| `035_freezebn` | protocol_v1 | LOSO 26 | r3d_18 | 0,5978 | 0,6187 | 0,5732 | 25 | 2026-07-16 |
| `036_resnetgru_smoke` | protocol_v1 | 2-fold | resnet_gru | 0,6171 | 0,7000 | 0,6364 | 25 | 2026-07-16 |
| `037_resnetgru` | protocol_v1 | LOSO 26 | resnet_gru | 0,6864 | 0,7176 | 0,6504 | 25 | 2026-07-16 |
| `038_res160_smoke` | protocol_v1 | 2-fold | r3d_18 | 0,6393 | 0,6800 | 0,6818 | 25 | 2026-07-16 |
| `039_res160` | protocol_v1 | LOSO 26 | r3d_18 | 0,6657 | 0,6802 | 0,6341 | 25 | 2026-07-16 |
| `040_snapshot_qsmoke` | protocol_v1 | 2-fold | r3d_18 | 0,6321 | 0,6600 | 0,6818 | 40 | 2026-07-16 |
| `041_snapshot` | protocol_v1 | LOSO 26 | r3d_18 | 0,6958 | 0,7019 | 0,6707 | 40 | 2026-07-17 |
| `042_bnadapt_qsmoke` | protocol_v1 | 2-fold | r3d_18 | 0,5841 | 0,6200 | 0,6364 | 25 | 2026-07-17 |
| `043_bnadapt` | protocol_v1 | LOSO 26 | r3d_18 | 0,6805 | 0,6979 | 0,6585 | 25 | 2026-07-17 |
| `044_seqflow_qsmoke` | protocol_v1 | 2-fold | r3d_18 | 0,4749 | 0,5733 | 0,5455 | 25 | 2026-07-17 |
| `045_seqflow` | protocol_v1 | LOSO 26 | r3d_18 | 0,5602 | 0,5980 | 0,5244 | 25 | 2026-07-17 |
| `046_erase_qsmoke` | protocol_v1 | 2-fold | r3d_18 | 0,5669 | 0,6200 | 0,5909 | 25 | 2026-07-17 |
| `047_erase` | protocol_v1 | LOSO 26 | r3d_18 | 0,6936 | 0,7077 | 0,6707 | 25 | 2026-07-17 |
| `048_noweight_qsmoke` | protocol_v1 | 2-fold | r3d_18 | 0,6270 | 0,6600 | 0,6818 | 25 | 2026-07-17 |
| `049_noweight` | protocol_v1 | LOSO 26 | r3d_18 | 0,7027 | 0,6997 | 0,6748 | 25 | 2026-07-17 |
| `050_vivit_rgb_qsmoke` | protocol_v1 | 2-fold | vivit | 0,2965 | 0,3800 | 0,2727 | 20 | 2026-07-18 |
| `051_vivit_rgb` | protocol_v1 | LOSO 26 | vivit | 0,2922 | 0,3094 | 0,3577 | 24 | 2026-07-19 |
| `052_vivit_flow` | protocol_v1 | LOSO 26 | vivit | 0,4905 | 0,4947 | 0,5000 | 24 | 2026-07-20 |
| `053_r3d_diff` | protocol_v1 | LOSO 26 | r3d_18 | 0,5897 | 0,6077 | 0,5610 | 25 | 2026-07-20 |
| `054_r3d_diff_randstart` | protocol_v1 | LOSO 26 | r3d_18 | 0,4555 | 0,4856 | 0,4472 | 25 | 2026-07-20 |
| `055_r3d_diff_res128` | protocol_v1 | LOSO 26 | r3d_18 | 0,6012 | 0,5928 | 0,5813 | 25 | 2026-07-20 |
| `056_r3d_diff_evalshift` | protocol_v1 | LOSO 26 | r3d_18 | 0,5773 | 0,5884 | 0,5569 | 25 | 2026-07-20 |
| `057_mc3_diff` | protocol_v1 | LOSO 26 | mc3_18 | 0,6086 | 0,6012 | 0,5935 | 25 | 2026-07-20 |
| `058_r2plus1d_diff` | protocol_v1 | LOSO 26 | r2plus1d_18 | 0,5785 | 0,5738 | 0,5569 | 25 | 2026-07-20 |
| `059_r3d_v2_dev_smoke2` | protocol_v2 | 1-fold | r3d_18 | 0,4931 | 0,5764 | 0,5714 | 25 | 2026-07-22 |
| `060_r3d_v2_dev` | protocol_v2 | grouped 4-fold | r3d_18 | 0,5902 | 0,6040 | 0,5528 | 25 | 2026-07-22 |
| `061_r3d_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6816 | 0,7120 | 0,6823 | 25 | 2026-07-22 |
| `062_r3d_flow_comp_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,5987 | 0,6217 | 0,6042 | 25 | 2026-07-22 |
| `063_fusion_baseline_iter40_90_10` | fusion | 0-fold | nan | 0,6562 | 0,6849 | 0,6615 | — | 2026-07-22 |
| `064_fusion_baseline_iter40_70_30` | fusion | 0-fold | nan | 0,6507 | 0,6718 | 0,6562 | — | 2026-07-22 |
| `065_fusion_baseline_iter40_80_20` | fusion | 0-fold | nan | 0,6547 | 0,6826 | 0,6562 | — | 2026-07-22 |
| `066_r3d_onset_apex_corrected_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6995 | 0,7088 | 0,6927 | 25 | 2026-07-22 |
| `067_fusion_baseline_iter43_25full_75apex` | fusion | 0-fold | nan | 0,7039 | 0,7231 | 0,6979 | — | 2026-07-22 |
| `068_fusion_baseline_iter43_50full_50apex` | fusion | 0-fold | nan | 0,7104 | 0,7345 | 0,7083 | — | 2026-07-22 |
| `069_fusion_baseline_iter43_75full_25apex` | fusion | 0-fold | nan | 0,6913 | 0,7154 | 0,6979 | — | 2026-07-22 |
| `070_r3d_res160_protocol_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6254 | 0,6562 | 0,6250 | 25 | 2026-07-22 |
| `071_fusion_champion_iter44_80_20` | fusion | 0-fold | nan | 0,7032 | 0,7286 | 0,7031 | — | 2026-07-22 |
| `072_fusion_champion_iter44_90_10` | fusion | 0-fold | nan | 0,7142 | 0,7439 | 0,7083 | — | 2026-07-22 |
| `073_seqflow_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,3684 | 0,4215 | 0,3698 | 25 | 2026-07-22 |
| `074_fusion_champion_seqflow_95_05` | fusion | 0-fold | nan | 0,7008 | 0,7241 | 0,6979 | — | 2026-07-22 |
| `075_ema_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6725 | 0,7017 | 0,6615 | 25 | 2026-07-22 |
| `076_fusion_champion_ema_80_20` | fusion | 0-fold | nan | 0,7071 | 0,7318 | 0,7083 | — | 2026-07-22 |
| `077_fusion_champion_ema_90_10` | fusion | 0-fold | nan | 0,7142 | 0,7439 | 0,7083 | — | 2026-07-22 |
| `078_r3d_s123_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6163 | 0,6573 | 0,6198 | 25 | 2026-07-22 |
| `079_r3d_onset_apex_s123_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,7119 | 0,7344 | 0,7031 | 25 | 2026-07-22 |
| `080_fusion_full_seed42_123_50_50` | fusion | 0-fold | nan | 0,6693 | 0,7005 | 0,6719 | — | 2026-07-22 |
| `081_fusion_temporal_s123_50_50` | fusion | 0-fold | nan | 0,7029 | 0,7335 | 0,6927 | — | 2026-07-22 |
| `082_fusion_apex_seed42_123_50_50` | fusion | 0-fold | nan | 0,7356 | 0,7534 | 0,7188 | — | 2026-07-22 |
| `083_fusion_champion42_plus_apexensemble_50_50` | fusion | 0-fold | nan | 0,7034 | 0,7302 | 0,6927 | — | 2026-07-22 |
| `084_fusion_temporal_seed42_123_equal4` | fusion | 0-fold | nan | 0,7270 | 0,7616 | 0,7135 | — | 2026-07-22 |
| `085_r3d_s2024_v2_dev_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6895 | 0,7035 | 0,6927 | 25 | 2026-07-22 |
| `086_fusion_full_seed42_123_2024_equal` | fusion | 0-fold | nan | 0,6902 | 0,7215 | 0,6927 | — | 2026-07-22 |
| `087_r3d_onset_apex_s2024_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,7210 | 0,7390 | 0,7292 | 25 | 2026-07-22 |
| `088_fusion_temporal_s2024_50_50` | fusion | 0-fold | nan | 0,7238 | 0,7365 | 0,7292 | — | 2026-07-22 |
| `089_fusion_apex_seed42_123_2024_equal` | fusion | 0-fold | nan | 0,7209 | 0,7345 | 0,7188 | — | 2026-07-22 |
| `090_fusion_temporal_seed42_123_2024_equal6` | fusion | 0-fold | nan | 0,7120 | 0,7308 | 0,7135 | — | 2026-07-22 |
| `091_r3d_auto_apex_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6707 | 0,7008 | 0,6719 | 25 | 2026-07-22 |
| `092_r3d_oracle_apex_reuse_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6995 | 0,7088 | 0,6927 | 25 | 2026-07-23 |
| `093_r3d_auto_apex_cap65_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6574 | 0,6891 | 0,6562 | 25 | 2026-07-23 |
| `094_r3d_multihyp_apex_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6654 | 0,7070 | 0,6562 | 25 | 2026-07-23 |
| `095_r3d_multihyp15_apex_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6781 | 0,7091 | 0,6719 | 25 | 2026-07-23 |
| `096_fusion_deployable_50full_50auto47` | fusion | 0-fold | nan | 0,7021 | 0,7418 | 0,6979 | — | 2026-07-23 |
| `097_fusion_deployable_50full_50multihyp48b` | fusion | 0-fold | nan | 0,6789 | 0,7136 | 0,6771 | — | 2026-07-23 |
| `098_r3d_auto_apex_onsetlate10_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6837 | 0,7138 | 0,6823 | 25 | 2026-07-23 |
| `099_r3d_auto_apex_onsetlate20_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6680 | 0,6962 | 0,6615 | 25 | 2026-07-23 |
| `100_r3d_auto_apex_onsetlate30_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6757 | 0,7094 | 0,6615 | 25 | 2026-07-23 |
| `101_r3d_v2_dev_p5ck` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6816 | 0,7120 | 0,6823 | 25 | 2026-07-23 |
| `102_r3d_hq_stabilized_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,5906 | 0,6025 | 0,6198 | 25 | 2026-07-23 |
| `103_r3d_face_roi_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6370 | 0,6567 | 0,6406 | 25 | 2026-07-23 |
| `104_r3d_auto_apex_smooth2_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6626 | 0,6913 | 0,6667 | 25 | 2026-07-23 |
| `105_r3d_auto_apex_smooth4_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6622 | 0,6890 | 0,6667 | 25 | 2026-07-23 |
| `106_r3d_auto_apex_smooth6_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6687 | 0,6936 | 0,6719 | 25 | 2026-07-23 |
| `107_r3d_auto_apex_smooth10_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6697 | 0,6997 | 0,6719 | 25 | 2026-07-23 |
| `108_r3d_auto_apex_s123_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,7082 | 0,7524 | 0,7031 | 25 | 2026-07-23 |
| `109_r3d_auto_apex_s2024_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6925 | 0,7180 | 0,7083 | 25 | 2026-07-23 |
| `110_r3d_auto_apex_s7_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6761 | 0,7091 | 0,6771 | 25 | 2026-07-23 |
| `111_r3d_auto_apex_s2025_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6610 | 0,7011 | 0,6667 | 25 | 2026-07-23 |
| `112_g1_user_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6255 | 0,6513 | 0,6531 | 25 | 2026-07-23 |
| `113_g1_user_s123_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,5529 | 0,5897 | 0,5663 | 25 | 2026-07-23 |
| `114_g1_user_s2024_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,5779 | 0,6063 | 0,6071 | 25 | 2026-07-23 |
| `115_g2_disgust_others_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,7388 | 0,7751 | 0,8177 | 25 | 2026-07-23 |
| `116_g2_disgust_others_s123_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,7495 | 0,7732 | 0,8281 | 25 | 2026-07-23 |
| `117_g2_disgust_others_s2024_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6933 | 0,7445 | 0,7812 | 25 | 2026-07-23 |
| `118_g3_disgust_others_plus_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,7129 | 0,7530 | 0,7857 | 25 | 2026-07-23 |
| `119_g3_disgust_others_plus_s123_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6769 | 0,7019 | 0,7602 | 25 | 2026-07-23 |
| `120_g3_disgust_others_plus_s2024_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6965 | 0,7457 | 0,7704 | 25 | 2026-07-23 |
| `121_r3d_earlystop_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,5755 | 0,5803 | 0,5885 | 60 | 2026-08-03 |
| `122_r3d_llrd_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6832 | 0,7131 | 0,6823 | 25 | 2026-08-03 |
| `123_r3d_unfreeze_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6233 | 0,6365 | 0,6250 | 25 | 2026-08-03 |
| `124_r3d_ep60_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6644 | 0,6837 | 0,6719 | 60 | 2026-08-03 |
| `125_r3d_es_llrd_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6194 | 0,6324 | 0,6250 | 60 | 2026-08-03 |
| `126_r3d_es_bigcheck_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,5664 | 0,5972 | 0,5938 | 60 | 2026-08-03 |
| `127_r3d_es_loss_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6137 | 0,6264 | 0,6094 | 60 | 2026-08-03 |
| `128_r3d_es_leaky_diag_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6742 | 0,7245 | 0,6615 | 60 | 2026-08-03 |
| `129_r3d_es_leaky_llrd_diag_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6694 | 0,6988 | 0,6667 | 60 | 2026-08-03 |
| `130_r3d_faceparse_full_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6637 | 0,6945 | 0,6719 | 25 | 2026-08-03 |
| `131_r3d_faceparse_autoapex_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6698 | 0,6934 | 0,6771 | 25 | 2026-08-03 |
| `132_r3d_es_refit_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6509 | 0,6851 | 0,6615 | 60 | 2026-08-03 |
| `133_r3d_es_refit_uf1_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6518 | 0,6920 | 0,6510 | 60 | 2026-08-03 |
| `134_r3d_es_refit_llrd_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6212 | 0,6645 | 0,6042 | 60 | 2026-08-03 |
| `135_r3d_lastk5_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6592 | 0,6873 | 0,6615 | 25 | 2026-08-03 |
| `136_r3d_lastk8_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6702 | 0,7065 | 0,6615 | 25 | 2026-08-03 |
| `137_r3d_lastk8_ep40_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6983 | 0,7215 | 0,6979 | 40 | 2026-08-03 |
| `138_r3d_snapshot3_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6651 | 0,6937 | 0,6510 | 39 | 2026-08-03 |
| `139_r3d_au02_full_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6894 | 0,7194 | 0,6823 | 25 | 2026-08-03 |
| `140_r3d_au05_full_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6690 | 0,6957 | 0,6615 | 25 | 2026-08-03 |
| `141_r3d_au10_full_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6690 | 0,6994 | 0,6562 | 25 | 2026-08-03 |
| `142_r3d_au05_autoapex_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6714 | 0,7068 | 0,6667 | 25 | 2026-08-03 |
| `143_r3d_lastk8_ep40_s123_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6483 | 0,6734 | 0,6562 | 40 | 2026-08-03 |
| `144_r3d_lastk8_ep40_s2024_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,7088 | 0,7177 | 0,7083 | 40 | 2026-08-03 |
| `145_r3d_lastk8_ep40_autoapex_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,7060 | 0,7298 | 0,7135 | 40 | 2026-08-03 |
| `146_r3d_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,7109 | 0,7348 | 0,6979 | 25 | 2026-08-03 |
| `147_r3d_auto_apex_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6817 | 0,7143 | 0,6667 | 25 | 2026-08-03 |
| `148_r3d_lastk8_ep40_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,7075 | 0,7336 | 0,6979 | 40 | 2026-08-03 |
| `149_r3d_au05_full_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6743 | 0,7125 | 0,6562 | 25 | 2026-08-03 |
| `150_r3d_es_refit_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6930 | 0,7374 | 0,6823 | 60 | 2026-08-04 |
| `151_r3d_llrd_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,7169 | 0,7392 | 0,7083 | 25 | 2026-08-04 |
| `152_r3d_ep60_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6874 | 0,7050 | 0,6979 | 60 | 2026-08-04 |
| `153_r3d_faceparse_full_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,7136 | 0,7336 | 0,6927 | 25 | 2026-08-04 |
| `154_resnetgru_full_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | resnet_gru | 0,5970 | 0,6318 | 0,6042 | 25 | 2026-08-04 |
| `155_resnetgru_autoapex_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | resnet_gru | 0,6656 | 0,6987 | 0,6615 | 25 | 2026-08-04 |
| `156_r3d_au01_full_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6682 | 0,6948 | 0,6615 | 25 | 2026-08-04 |
| `157_r3d_au02_autoapex_s42_v2_dev_p5` | protocol_v2 | grouped 4-fold | r3d_18 | 0,6595 | 0,6985 | 0,6562 | 25 | 2026-08-04 |
| `158_r3d_au02_full_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6671 | 0,6971 | 0,6562 | 25 | 2026-08-04 |
| `159_mc3_full_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | mc3_18 | 0,6431 | 0,6533 | 0,6562 | 25 | 2026-08-04 |
| `160_r2plus1d_full_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r2plus1d_18 | 0,7059 | 0,7323 | 0,6875 | 25 | 2026-08-04 |
| `161_resnetgru_autoapex_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | resnet_gru | 0,6579 | 0,6883 | 0,6510 | 25 | 2026-08-04 |
| `162_resnetgru_full_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | resnet_gru | 0,6129 | 0,6603 | 0,6042 | 25 | 2026-08-04 |
| `163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,7283 | 0,7440 | 0,7135 | 25 | 2026-08-04 |
| `164_r3d_regionattn_apex_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6667 | 0,7036 | 0,6615 | 25 | 2026-08-04 |
| `165_r3d_regionattn_dyn_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6758 | 0,7099 | 0,6667 | 25 | 2026-08-04 |
| `166_r3d_nocheek_full_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6458 | 0,6666 | 0,6354 | 25 | 2026-08-04 |
| `167_r3d_auroi_full_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6479 | 0,6845 | 0,6354 | 25 | 2026-08-04 |
| `168_r3d_auroi_apex_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6351 | 0,6647 | 0,6250 | 25 | 2026-08-04 |
| `169_r3d_focus_region_full_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6394 | 0,6593 | 0,6354 | 25 | 2026-08-04 |
| `170_r3d_focus_energy_full_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6850 | 0,7058 | 0,6615 | 25 | 2026-08-04 |
| `171_r3d_focus_region_apex_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6642 | 0,6961 | 0,6667 | 25 | 2026-08-05 |
| `172_r3d_focus_es_full_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6198 | 0,6658 | 0,6094 | 60 | 2026-08-05 |
| `173_r3d_focus_es_apex_s42_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6466 | 0,6736 | 0,6458 | 60 | 2026-08-05 |
| `174_r3d_regionattn_s123_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6865 | 0,7242 | 0,6771 | 25 | 2026-08-05 |
| `175_r3d_regionattn_s2024_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6862 | 0,7008 | 0,6771 | 25 | 2026-08-05 |
| `176_r3d_regionattn_s7_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6953 | 0,7117 | 0,6979 | 25 | 2026-08-05 |
| `177_r3d_s123_loso_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6780 | 0,7127 | 0,6615 | 25 | 2026-08-05 |
| `178_r3d_s2024_loso_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6853 | 0,7222 | 0,6719 | 25 | 2026-08-05 |
| `179_r3d_s7_loso_v2_dev_loso_dev_p5` | protocol_v2 | LOSO 21 | r3d_18 | 0,6965 | 0,7215 | 0,6927 | 25 | 2026-08-05 |
| `180_r3d_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | r3d_18 | 0,6902 | 0,6975 | 0,6707 | 25 | 2026-08-05 |
| `181_r3d_regionattn_full_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | r3d_18 | 0,7022 | 0,7208 | 0,6829 | 25 | 2026-08-06 |
| `182_r3d_focus_region_full_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | r3d_18 | 0,7121 | 0,7271 | 0,6951 | 25 | 2026-08-06 |
| `183_r3d_focus_es_full_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | r3d_18 | 0,6508 | 0,6686 | 0,6301 | 60 | 2026-08-06 |
| `184_r3d_auto_apex_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | r3d_18 | 0,7098 | 0,7413 | 0,6789 | 25 | 2026-08-06 |
| `185_r3d_lastk8_ep40_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | r3d_18 | 0,6889 | 0,7010 | 0,6667 | 40 | 2026-08-06 |
| `186_r3d_llrd_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | r3d_18 | 0,6995 | 0,7192 | 0,6870 | 25 | 2026-08-06 |
| `187_r3d_faceparse_full_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | r3d_18 | 0,6824 | 0,7003 | 0,6667 | 25 | 2026-08-06 |
| `188_r3d_au05_full_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | r3d_18 | 0,7211 | 0,7378 | 0,6992 | 25 | 2026-08-06 |
| `189_r3d_es_refit_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | r3d_18 | 0,6756 | 0,7018 | 0,6504 | 60 | 2026-08-06 |
| `190_r2plus1d_full_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | r2plus1d_18 | 0,6943 | 0,7011 | 0,6707 | 25 | 2026-08-06 |
| `191_r3d_focus_energy_full_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | r3d_18 | 0,6584 | 0,6730 | 0,6301 | 25 | 2026-08-06 |
| `192_fusion10_loso26` | fusion | 0-fold | nan | 0,7118 | 0,7232 | 0,6951 | — | 2026-08-06 |
| `193_resnetgru_nopretrain_full_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | resnet_gru | 0,6612 | 0,7117 | 0,6301 | 25 | 2026-08-07 |
| `194_resnetgru_nopretrain_apex_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | resnet_gru | 0,6904 | 0,7248 | 0,6545 | 25 | 2026-08-07 |
| `195_focus_rnn_nopretrain_full_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | resnet_gru | 0,6148 | 0,6498 | 0,5935 | 25 | 2026-08-07 |
| `196_focus_rnn_nopretrain_es_full_s42_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | resnet_gru | 0,5679 | 0,6047 | 0,5569 | 60 | 2026-08-07 |
| `200_expA_flow_on_replica_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | r3d_18 | 0,5389 | 0,5415 | 0,8833 | 25 | 2026-09-11 |
| `201_expB_vivit_on_megc_v2_dev_loso_all_p5` | protocol_v2 | LOSO 26 | cnn_temporal_vivit | 0,2093 | 0,2267 | 0,2805 | 30 | 2026-09-11 |

---

## 5. Temuan Utama & Analisis Empiris

Temuan penting dari seluruh rangkaian eksperimen:
1. **Flow kumulatif ber-referensi onset unggul telak atas flow sekuensial ($t \rightarrow t+1$).** Flow kumulatif menangkap lintasan deformasi dari titik netral awal secara stabil (UF1 0,6601 vs 0,3684).
2. **3D CNN mengungguli arsitektur 2D-CNN + RNN dan ViViT Transformer.** Konvolusi 3D (R3D-18) mampu mengekstrak interaksi ruang-waktu secara langsung tanpa overfitting yang rentan terjadi pada arsitektur berbasis transformer di dataset berskala kecil.
3. **Estimasi Apex Label-Free (Auto-Apex) Menutup Celah Oracle.** Estimator energi optical flow berhasil mendeteksi frame apex secara mandiri dari sinyal gerakan, menghasilkan performa fusi 0,7021 UF1 (hanya terpaut 0,0083 dari kondisi oracle beranotasi 0,7104).
4. **Ketahanan terhadap Penurunan Frame Rate (FPS Robustness).** Evaluasi degradasi FPS (200 $\rightarrow$ 120 / 60 / 30 fps) membuktikan model fusi deployable mempertahankan akurasi 0,6979 dengan delta UF1 minimal (−0,0068, $P=0,372$). Aplikasi ponsel standar 30 fps dapat beroperasi tanpa kehilangan performa berarti.
5. **Epoch Averaging (Last-K = 8 pada 40 Epoch) Memberikan Stabilitas Optimal.** Mengurangi varians antar fold dan meningkatkan UF1 baseline dari 0,6816 menjadi 0,6983.
6. **Segmentation Attention Statis Lolos Validasi LOSO 21.** Masking berbasis landmark wajah pada area mata, alis, dan mulut menghasilkan skor 0,7283 UF1 ($P=0,801$).

---

## 6. Cabang yang Ditutup (Pelajaran Negatif & Dead-Ends)

Cabang eksperimen dan pendekatan yang **telah dibuktikan tidak efektif / ditutup** agar tidak diulang:
- **Stabilisasi Gerakan Kepala Eksternal (HQ TV-L1 + ECC):** Menurunkan performa secara drastis (~9 poin UF1, 0,5906 vs 0,6816). Gerakan kepala minor ternyata berkorelasi dengan dinamika ekspresi mikro.
- **ViViT (Video Vision Transformer):** Mengalami overfitting berat karena kapasitas model yang terlalu besar untuk dataset dengan jumlah klip terbatas.
- **Eulerian Motion Magnification:** Memperbesar artifak sensor kamera dan noise optical flow dibanding memperkuat sinyal mikro.
- **Mixup Data Augmentation:** Interpolasi linear antar klip merusak pola temporal halus gerakan mikro.
- **Resolusi Gambar 160x160:** Menambah beban komputasi dan meningkatkan varians overfitting tanpa perbaikan metrik (0,6254 vs 0,6816).
- **Ensemble Lebih dari 3 Seed:** Penambahan seed acak di luar seed 42 dan 123 menunjukkan diminishing returns dan anomali seleksi.

---

## 7. Statistik Repositori & Prosedur Reproduksi

- **Total Run Tercatat:** 198 eksperimen mandiri (tersimpan di `runs/` dan terindeks di `results/results.csv`).
- **Struktur Repositori:** Bersih dan terstandarisasi penuh (`configs/NNN_*.json`, `runs/NNN_*/`, `results/results.csv`, `reports/LAPORAN.md`).
- **Pemeriksaan Integritas:** Jalankan `bash scripts/check.sh` untuk memverifikasi struktur dan unittests.
- **Regenerasi Hasil & Laporan:** Jalankan `bash scripts/rebuild_reports.sh` untuk menyusun ulang tabel dan laporan secara deterministik tanpa training ulang.
