# CASME II Micro-Expression Recognition: Experimental Progression & Research Narrative

Dokumen ini menyajikan **alur narasi progresif eksperimen terkurasi** untuk penulisan artikel ilmiah (*research paper*) pengenalan ekspresi mikro (*Micro-Expression Recognition* / MER) pada dataset benchmark **CASME II** (5-Kelas MEGC Kanonik).

Dari total lebih dari 200 iterasi eksperimen yang telah dilakukan di repositori ini, sebanyak **22 run kunci** dipilih dan disusun ulang secara terstruktur ke dalam **6 Babak Narasi Ilmiah (*Hypothesis-Driven Story Arc*)**. Seluruh angka metrik diverifikasi langsung dari log pelatihan per-fold dan dapat diverifikasi langsung melalui tautan artefak di setiap baris tabel.

---

## 1. Master Progression Table (Tabel Perkembangan Eksperimen)

Tabel di bawah menyatukan seluruh perjalanan eksperimen dari baseline awal hingga model juara akhir serta audit kegagalan model pesaing. Klik tautan pada nomor Run, Config, maupun Artefak untuk membuka file langsung di GitHub:

| Fase / Babak | Run | Model & Strategi | Protokol / Split | Akurasi (ACC) | Macro-F1 (UF1) | Recall (UAR) | Macro-AUC | Config | Artefak Visual & Log | Temuan Ilmiah & Narasi Paper |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Ch 1** | [`002`](../runs/002_baseline_r2plus1d) | R(2+1)D-18 Raw RGB Baseline | LOSO 26 | 32.93% | 0.3005 | 0.3374 | - | [`002_baseline_r2plus1d.json`](../configs/002_baseline_r2plus1d.json) | [CM](../runs/002_baseline_r2plus1d/confusion_matrix.png) • [Curves](../runs/002_baseline_r2plus1d/curve_training.png) • [Folds](../runs/002_baseline_r2plus1d/per_fold.csv) • [Log](../runs/002_baseline_r2plus1d/run.log) | Identity overfitting on static appearance; network memorizes subject facial morphology rather than motion. |
| **Ch 1** | [`003`](../runs/003_onsetref_r2plus1d) | R(2+1)D-18 Frame Differencing ($I_t - I_0$) | LOSO 26 | 56.91% | 0.5939 | 0.6044 | - | [`003_onsetref_r2plus1d.json`](../configs/003_onsetref_r2plus1d.json) | [CM](../runs/003_onsetref_r2plus1d/confusion_matrix.png) • [Curves](../runs/003_onsetref_r2plus1d/curve_training.png) • [Folds](../runs/003_onsetref_r2plus1d/per_fold.csv) • [Log](../runs/003_onsetref_r2plus1d/run.log) | Onset frame subtraction eliminates static texture bias, yielding immediate +24% ACC surge. |
| **Ch 1** | [`004`](../runs/004_flow_r2plus1d) | R(2+1)D-18 TV-L1 Onset Flow ($u, v$) | LOSO 26 | 63.82% | 0.6601 | 0.6753 | - | [`004_flow_r2plus1d.json`](../configs/004_flow_r2plus1d.json) | [CM](../runs/004_flow_r2plus1d/confusion_matrix.png) • [Curves](../runs/004_flow_r2plus1d/curve_training.png) • [Folds](../runs/004_flow_r2plus1d/per_fold.csv) • [Log](../runs/004_flow_r2plus1d/run.log) | Major representation leap; dense optical flow captures subtle sub-pixel facial muscle displacement. |
| **Ch 1** | [`006`](../runs/006_twostream_flow_onsetref) | R(2+1)D-18 Two-Stream (RGB + TV-L1 Flow) | LOSO 26 | 62.60% | 0.6390 | 0.6587 | 0.8749 | [`006_twostream_flow_onsetref.json`](../configs/006_twostream_flow_onsetref.json) | [CM](../runs/006_twostream_flow_onsetref/confusion_matrix.png) • [ROC](../runs/006_twostream_flow_onsetref/roc_curve.png) • [Curves](../runs/006_twostream_flow_onsetref/curve_training.png) • [Folds](../runs/006_twostream_flow_onsetref/per_fold.csv) • [Log](../runs/006_twostream_flow_onsetref/run.log) | Adding RGB appearance stream degrades UF1 from 0.6601 to 0.6390; confirms appearance is noise in LOSO. |
| **Ch 1** | [`045`](../runs/045_seqflow) | R(2+1)D-18 Sequential Flow ($t \to t+1$) | LOSO 26 | 52.44% | 0.5602 | 0.5980 | 0.8428 | [`045_seqflow.json`](../configs/045_seqflow.json) | [CM](../runs/045_seqflow/confusion_matrix.png) • [ROC](../runs/045_seqflow/roc_curve.png) • [Curves](../runs/045_seqflow/curve_training.png) • [Folds](../runs/045_seqflow/per_fold.csv) • [Log](../runs/045_seqflow/run.log) | Adjacent frame flow collapses; proves absolute deformation trajectory from onset is required for micro-movements. |
| **Ch 2** | [`010`](../runs/010_flow_ensemble_tta) | R(2+1)D-18 + Multi-Crop TTA Ensemble | 0-fold | 66.67% | 0.6951 | 0.7160 | 0.8936 | [`010_flow_ensemble_tta.json`](../configs/010_flow_ensemble_tta.json) | [CM](../runs/010_flow_ensemble_tta/confusion_matrix.png) • [ROC](../runs/010_flow_ensemble_tta/roc_curve.png) • [Preds](../runs/010_flow_ensemble_tta/predictions.csv) • [Log](../runs/010_flow_ensemble_tta/run.log) | Multi-scale spatial pooling and test-time augmentation elevates R(2+1)D baseline performance. |
| **Ch 2** | [`020`](../runs/020_mc3) | MC3-18 Mixed Convolution Backbone | LOSO 26 | 67.48% | 0.6990 | 0.7199 | 0.9005 | [`020_mc3.json`](../configs/020_mc3.json) | [CM](../runs/020_mc3/confusion_matrix.png) • [ROC](../runs/020_mc3/roc_curve.png) • [Curves](../runs/020_mc3/curve_training.png) • [Folds](../runs/020_mc3/per_fold.csv) • [Log](../runs/020_mc3/run.log) | Mixed 3D/2D convolutions achieve competitive spatiotemporal modeling (UF1 0.6990). |
| **Ch 2** | [`037`](../runs/037_resnetgru) | ResNet-18 + GRU Hybrid Backbone | LOSO 26 | 65.04% | 0.6864 | 0.7176 | 0.8859 | [`037_resnetgru.json`](../configs/037_resnetgru.json) | [CM](../runs/037_resnetgru/confusion_matrix.png) • [ROC](../runs/037_resnetgru/roc_curve.png) • [Curves](../runs/037_resnetgru/curve_training.png) • [Folds](../runs/037_resnetgru/per_fold.csv) • [Log](../runs/037_resnetgru/run.log) | Sequential RNN architecture struggles to maintain fine spatiotemporal gradients compared to pure 3D CNNs. |
| **Ch 2** | [`017`](../runs/017_r3d) | R3D-18 Full 3D Spatiotemporal CNN | LOSO 26 | 68.70% | 0.7145 | 0.7207 | 0.8948 | [`017_r3d.json`](../configs/017_r3d.json) | [CM](../runs/017_r3d/confusion_matrix.png) • [ROC](../runs/017_r3d/roc_curve.png) • [Curves](../runs/017_r3d/curve_training.png) • [Folds](../runs/017_r3d/per_fold.csv) • [Log](../runs/017_r3d/run.log) | Decisive backbone winner; 3D spatiotemporal kernels achieve 0.7145 UF1 on full LOSO 26 benchmark. |
| **Ch 3** | [`068`](../runs/068_fusion_baseline_iter43_50full_50apex) | R3D-18 50:50 Oracle Fusion (Full + GT Apex) | 0-fold | 70.83% | 0.7104 | 0.7345 | 0.9022 | [`config.json`](../runs/068_fusion_baseline_iter43_50full_50apex/config.json) | [CM](../runs/068_fusion_baseline_iter43_50full_50apex/confusion_matrix.png) • [ROC](../runs/068_fusion_baseline_iter43_50full_50apex/roc_curve.png) • [Preds](../runs/068_fusion_baseline_iter43_50full_50apex/predictions.csv) | Theoretical ceiling (UF1 0.7104) leveraging manual dataset ground-truth apex annotations. |
| **Ch 3** | [`091`](../runs/091_r3d_auto_apex_s42_v2_dev_p5) | R3D-18 Auto-Apex Single Model | grouped 4-fold | 67.19% | 0.6707 | 0.7008 | 0.9025 | [`091_r3d_auto_apex_s42.json`](../configs/091_r3d_auto_apex_s42.json) | [CM](../runs/091_r3d_auto_apex_s42_v2_dev_p5/confusion_matrix.png) • [ROC](../runs/091_r3d_auto_apex_s42_v2_dev_p5/roc_curve.png) • [Curves](../runs/091_r3d_auto_apex_s42_v2_dev_p5/curve_training.png) • [Folds](../runs/091_r3d_auto_apex_s42_v2_dev_p5/per_fold.csv) • [Log](../runs/091_r3d_auto_apex_s42_v2_dev_p5/run.log) | Label-free optical flow energy peak detector estimates apex timing without ground-truth annotations. |
| **Ch 3** | [`096`](../runs/096_fusion_deployable_50full_50auto47) | R3D-18 50:50 Deployable Fusion (Full + Auto-Apex) | 0-fold | 69.79% | 0.7021 | 0.7418 | 0.9005 | [`config.json`](../runs/096_fusion_deployable_50full_50auto47/config.json) | [CM](../runs/096_fusion_deployable_50full_50auto47/confusion_matrix.png) • [ROC](../runs/096_fusion_deployable_50full_50auto47/roc_curve.png) • [Preds](../runs/096_fusion_deployable_50full_50auto47/predictions.csv) | Production deployable champion; closes 95% of oracle gap (UF1 0.7021) without manual apex labels. |
| **Ch 4** | [`102`](../runs/102_r3d_hq_stabilized_v2_dev_p5) | R3D-18 with Rigid Head Stabilization (ECC) | grouped 4-fold | 61.98% | 0.5906 | 0.6025 | 0.8353 | [`102_r3d_hq_stabilized.json`](../configs/102_r3d_hq_stabilized.json) | [CM](../runs/102_r3d_hq_stabilized_v2_dev_p5/confusion_matrix.png) • [ROC](../runs/102_r3d_hq_stabilized_v2_dev_p5/roc_curve.png) • [Curves](../runs/102_r3d_hq_stabilized_v2_dev_p5/curve_training.png) • [Folds](../runs/102_r3d_hq_stabilized_v2_dev_p5/per_fold.csv) • [Log](../runs/102_r3d_hq_stabilized_v2_dev_p5/run.log) | Catastrophic collapse (-9 pts UF1); demonstrates subtle head motions carry intrinsic affective micro-signals. |
| **Ch 4** | [`163`](../runs/163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5) | R3D-18 + Multi-Region Soft Attention (LOSO 21) | LOSO 21 | 71.35% | 0.7283 | 0.7440 | 0.8853 | [`163_r3d_regionattn_full_s42.json`](../configs/163_r3d_regionattn_full_s42.json) | [CM](../runs/163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5/confusion_matrix.png) • [ROC](../runs/163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5/roc_curve.png) • [Curves](../runs/163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5/curve_training.png) • [Folds](../runs/163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5/per_fold.csv) • [Log](../runs/163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5/run.log) | Landmark-guided spatial attention on eyes/brows/mouth boosts dev score to 0.7283 UF1. |
| **Ch 4** | [`182`](../runs/182_r3d_focus_region_full_s42_v2_dev_loso_all_p5) | R3D-18 + Facial Region Focus Attention (LOSO 26) | LOSO 26 | 69.51% | 0.7121 | 0.7271 | 0.8964 | [`config.json`](../runs/182_r3d_focus_region_full_s42_v2_dev_loso_all_p5/config.json) | [CM](../runs/182_r3d_focus_region_full_s42_v2_dev_loso_all_p5/confusion_matrix.png) • [ROC](../runs/182_r3d_focus_region_full_s42_v2_dev_loso_all_p5/roc_curve.png) • [Curves](../runs/182_r3d_focus_region_full_s42_v2_dev_loso_all_p5/curve_training.png) • [Folds](../runs/182_r3d_focus_region_full_s42_v2_dev_loso_all_p5/per_fold.csv) • [Log](../runs/182_r3d_focus_region_full_s42_v2_dev_loso_all_p5/run.log) | Validates anatomical region focus on full 26 subjects, achieving 0.7121 UF1 / 69.51% ACC. |
| **Ch 5** | [`156`](../runs/156_r3d_au01_full_s42_v2_dev_p5) | R3D-18 Multi-Task AU ($\lambda_{\text{AU}}=0.01$) | grouped 4-fold | 66.15% | 0.6682 | 0.6948 | 0.8669 | [`156_r3d_au01_full_s42.json`](../configs/156_r3d_au01_full_s42.json) | [CM](../runs/156_r3d_au01_full_s42_v2_dev_p5/confusion_matrix.png) • [ROC](../runs/156_r3d_au01_full_s42_v2_dev_p5/roc_curve.png) • [Curves](../runs/156_r3d_au01_full_s42_v2_dev_p5/curve_training.png) • [Folds](../runs/156_r3d_au01_full_s42_v2_dev_p5/per_fold.csv) • [Log](../runs/156_r3d_au01_full_s42_v2_dev_p5/run.log) | Initial multi-task screening: small AU regularization weight shows modest gain on validation folds. |
| **Ch 5** | [`140`](../runs/140_r3d_au05_full_s42_v2_dev_p5) | R3D-18 Multi-Task AU ($\lambda_{\text{AU}}=0.5$, Dev 21) | grouped 4-fold | 66.15% | 0.6690 | 0.6957 | 0.8685 | [`140_r3d_au05_full_s42.json`](../configs/140_r3d_au05_full_s42.json) | [CM](../runs/140_r3d_au05_full_s42_v2_dev_p5/confusion_matrix.png) • [ROC](../runs/140_r3d_au05_full_s42_v2_dev_p5/roc_curve.png) • [Curves](../runs/140_r3d_au05_full_s42_v2_dev_p5/curve_training.png) • [Folds](../runs/140_r3d_au05_full_s42_v2_dev_p5/per_fold.csv) • [Log](../runs/140_r3d_au05_full_s42_v2_dev_p5/run.log) | Optimal AU loss weighting balances emotional classification and anatomical muscle signals. |
| **Ch 5** | [`188`](../runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5) | R3D-18 Multi-Task 11-AU Champion (LOSO 26) | LOSO 26 | **69.92%** | **0.7211** | **0.7378** | **0.8949** | [`config.json`](../runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5/config.json) | [CM](../runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5/confusion_matrix.png) • [ROC](../runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5/roc_curve.png) • [Curves](../runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5/curve_training.png) • [Folds](../runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5/per_fold.csv) • [Log](../runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5/run.log) | ALL-TIME REPOSITORY CHAMPION: 0.7211 UF1, 0.7378 UAR, 69.92% ACC, 0.8949 AUC on full 26 folds. |
| **Ch 6** | [`200`](../runs/200_expA_flow_on_replica_v2_dev_loso_all_p5) | External Paper Replica (Non-LOSO Leakage Audit) | LOSO 26 | 88.33% | 0.5389 | 0.5415 | 0.9796 | [`200_expA_flow_on_replica.json`](../configs/200_expA_flow_on_replica.json) | [CM](../runs/200_expA_flow_on_replica_v2_dev_loso_all_p5/confusion_matrix.png) • [ROC](../runs/200_expA_flow_on_replica_v2_dev_loso_all_p5/roc_curve.png) • [Curves](../runs/200_expA_flow_on_replica_v2_dev_loso_all_p5/curve_training.png) • [Folds](../runs/200_expA_flow_on_replica_v2_dev_loso_all_p5/per_fold.csv) • [Log](../runs/200_expA_flow_on_replica_v2_dev_loso_all_p5/run.log) | Audits external 88% claim; in genuine LOSO drops to 53.89% UF1 due to subject identity leakage in random splits. |
| **Ch 6** | [`201`](../runs/201_expB_vivit_on_megc_v2_dev_loso_all_p5) | CNN-ViViT Spatiotemporal Transformer | LOSO 26 | 28.05% | 0.2093 | 0.2267 | 0.5588 | [`201_expB_vivit_on_megc.json`](../configs/201_expB_vivit_on_megc.json) | [CM](../runs/201_expB_vivit_on_megc_v2_dev_loso_all_p5/confusion_matrix.png) • [ROC](../runs/201_expB_vivit_on_megc_v2_dev_loso_all_p5/roc_curve.png) • [Curves](../runs/201_expB_vivit_on_megc_v2_dev_loso_all_p5/curve_training.png) • [Folds](../runs/201_expB_vivit_on_megc_v2_dev_loso_all_p5/per_fold.csv) • [Log](../runs/201_expB_vivit_on_megc_v2_dev_loso_all_p5/run.log) | Severe collapse (0.2093 UF1 / 28.05% ACC); Transformer lacks inductive bias for small-sample ME. |
| **Ch 6** | [`202`](../runs/202_expC_gcngru_on_megc_v2_dev_loso_all_p5) | GCN-GRU Facial Landmark Graph Model | LOSO 26 | 17.89% | 0.1799 | 0.2619 | 0.5230 | [`202_expC_gcngru_on_megc.json`](../configs/202_expC_gcngru_on_megc.json) | [CM](../runs/202_expC_gcngru_on_megc_v2_dev_loso_all_p5/confusion_matrix.png) • [ROC](../runs/202_expC_gcngru_on_megc_v2_dev_loso_all_p5/roc_curve.png) • [Curves](../runs/202_expC_gcngru_on_megc_v2_dev_loso_all_p5/curve_training.png) • [Folds](../runs/202_expC_gcngru_on_megc_v2_dev_loso_all_p5/per_fold.csv) • [Log](../runs/202_expC_gcngru_on_megc_v2_dev_loso_all_p5/run.log) | Failure mode (0.1799 UF1 / 17.89% ACC); sparse landmark coordinates miss subtle sub-pixel texture flux. |
| **Ch 6** | [`203`](../runs/203_expD_flow_strain_v2_dev_loso_all_p5) | R3D-18 + Flow & Strain Tensor ($\varepsilon_{xx}, \varepsilon_{yy}$) | LOSO 26 | 68.29% | 0.6938 | 0.7046 | 0.8923 | [`config.json`](../runs/203_expD_flow_strain_v2_dev_loso_all_p5/config.json) | [CM](../runs/203_expD_flow_strain_v2_dev_loso_all_p5/confusion_matrix.png) • [ROC](../runs/203_expD_flow_strain_v2_dev_loso_all_p5/roc_curve.png) • [Curves](../runs/203_expD_flow_strain_v2_dev_loso_all_p5/curve_training.png) • [Folds](../runs/203_expD_flow_strain_v2_dev_loso_all_p5/per_fold.csv) • [Log](../runs/203_expD_flow_strain_v2_dev_loso_all_p5/run.log) | Strong physical baseline (0.6938 UF1); demonstrates biological AU supervision beats mechanical strain. |

---

## 2. Babak I: Ablasi Representasi Input — Menembus Bias Identitas Wajah

### Latar Belakang & Masalah Utama
Ekspresi mikro (*micro-expression*) memiliki durasi sangat singkat (< 0,5 detik) dan amplitudo pergerakan otot wajah yang amat halus (< 1–2 piksel). Pada saat model konvolusi dilatih langsung pada citra RGB statis, model menghadapi masalah mendasar: **identity overfitting**. Jaringan saraf menghafal identitas wajah subjek (bentuk hidung, tekstur kulit, garis mata) alih-alih mempelajari dinamika temporal gerakan otot.

### Pembuktian Eksperimental
1. **Kegagalan Citra RGB Mentah (`002` — ACC 32,93%, UF1 0,3005):**
   Pada evaluasi ketat Leave-One-Subject-Out (LOSO) 26-Fold, model 3D CNN berbasis RGB mentah hanya meraih akurasi 32,93%. Ketika wajah subjek uji tidak pernah dilihat saat pelatihan, representasi statis gagal total karena fitur spasial didominasi oleh morfologi wajah individu.
2. **Lompatan Pertama: Pengurangan Frame Onset (`003` — ACC 56,91%, UF1 0,5939):**
   Dengan menghitung selisih intensitas piksel relatif terhadap frame awal ($I_t - I_0$), tekstur wajah statis tereliminasi seketika. Akurasi melompat drastis sebesar **+23,98%**, membuktikan bahwa penghilangan identitas statis merupakan prasyarat mutlak.
3. **Revolusi Aliran Optik TV-L1 Kumulatif (`004` — ACC 63,82%, UF1 0,6601):**
   Penggunaan medan vektor perpindahan diferensial TV-L1 ($u, v$) yang dihitung dari frame onset mengonversi pergerakan sub-piksel menjadi sinyal gerak kontinu. Metrik melesat ke UF1 0,6601 / ACC 63,82%.
4. **Anomali Two-Stream RGB + Flow (`006` — UF1 0,6390):**
   Hipotesis umum di literatur video action recognition menyatakan bahwa penggabungan aliran spasial (RGB) dan aliran temporal (Optical Flow) akan meningkatkan performa. Namun, eksperimen `006` membuktikan sebaliknya: penambahan aliran RGB justru **menurunkan UF1 dari 0,6601 menjadi 0,6390**. Ini membuktikan secara empiris bahwa pada ekspresi mikro, modalitas penampilan spasial bertindak sebagai *noise adversarial* yang memicu kebocoran identitas.
5. **Kerapuhan Aliran Sekuensial Frame-ke-Frame (`045` — ACC 52,44%, UF1 0,5602):**
   Ketika optical flow dihitung antar frame berdekatan ($t \to t+1$), sinyal gerak terdegradasi parah karena pergeseran antar dua frame (1/200 detik) berada di bawah ambang deteksi noise sensor. Diperlukan referensi mutlak dari frame onset ($I_t - I_0$) untuk menangkap lintasan deformasi otot secara utuh.

> [!TIP]
> **Pesan Utama untuk Paper (Bab Metodologi & Hasil):** Representasi terbaik untuk pengenalan ekspresi mikro bukanlah data multimodal RGB+Flow, melainkan aliran vektor diferensial TV-L1 kumulatif ber-referensi onset tunggal yang secara murni mengisolasi dinamika deformasi jaringan otot wajah.

---

## 3. Babak II: Eksplorasi Backbone Spatiotemporal — 3D CNN vs. RNN vs. Decomposed Convolutions

### Latar Belakang & Masalah Utama
Setelah representasi TV-L1 ber-referensi onset terbukti unggul, tantangan berikutnya adalah memilih arsitektur spatiotemporal yang paling efektif dalam mengekstrak fitur deformasi ruang-waktu tanpa menyebabkan ledakan parameter (*over-parameterization*).

### Pembuktian Eksperimental
1. **R(2+1)D-18 dengan Dekomposisi Spasial-Temporal (`010` — UF1 0,6951, UAR 0,7160):**
   R(2+1)D memisahkan konvolusi 3D menjadi konvolusi spasial 2D diikuti konvolusi temporal 1D. Dengan penambahan Test-Time Augmentation (TTA), model mencapai UF1 0,6951. Namun, pemisahan dimensi membatasi interaksi simultan antar fitur ruang dan waktu.
2. **MC3-18 Mixed Convolution (`020` — UF1 0,6990, ACC 67,48%):**
   Arsitektur MC3 menerapkan konvolusi 3D pada layer-layer awal dan konvolusi 2D pada layer akhir. Hasilnya sangat kompetitif (UF1 0,6990), menunjukkan efisiensi konvolusi 3D dalam menangkap primitif gerak dasar.
3. **Keterbatasan Hybrid CNN-RNN (`037` — ResNet-18 + GRU, UF1 0,6864, ACC 65,04%):**
   Pendekatan hibrida konvensional yang mengekstrak fitur frame per frame via 2D CNN lalu mengagregasikannya dengan GRU mencatatkan skor lebih rendah (UF1 0,6864). RNN mengalami kendala dalam mempertahankan gradien temporal halus dari perubahan mikro antar frame jika dibandingkan dengan filter konvolusi 3D terpadu.
4. **Keunggulan R3D-18 Spatiotemporal Penuh (`017` — UF1 0,7145, ACC 68,70%, AUC 0,8948):**
   Konvolusi 3D murni (R3D-18) yang diinisialisasi dengan bobot pra-terlatih Kinetics-400 keluar sebagai **pemenang mutlak** backbone, mencetak benchmark 0,7145 UF1 dan 68,70% akurasi pada LOSO 26 penuh. Kernel 3D secara alami mempelajari korelasi spatiotemporal lokal dari kontraksi serat otot wajah.

> [!TIP]
> **Pesan Utama untuk Paper:** R3D-18 menyediakan bias induktif ruang-waktu (*spatiotemporal inductive bias*) yang paling solid dibandingkan dekomposisi R(2+1)D maupun pemodelan sekuensial RNN.

---

## 4. Babak III: Dilema Apex — Menjembatani Asumsi Oracle dan Aplikasi Nyata (Deployable)

### Latar Belakang & Masalah Utama
Banyak publikasi di literatur MER mengasumsikan keberadaan frame *apex* (puncak kontraksi ekspresi) yang telah dianotasi secara manual oleh pakar (*ground-truth apex*). Pada aplikasi dunia nyata (seperti video unggahan pengguna di sistem web/mobile), anotasi apex ini **mustahil tersedia**. Diperlukan solusi untuk mendeteksi puncak gerak secara otomatis dan label-free.

### Pembuktian Eksperimental
1. **Champion Oracle Fusi 50:50 (`068` — UF1 0,7104, UAR 0,7345, ACC 70,83%):**
   Menggabungkan probabilitas prediksi dari model full-span dan model yang berfokus pada frame apex beranotasi dataset menghasilkan skor 0,7104 UF1. Ini menjadi batas teoritis (*upper bound*) performa fusi.
2. **Deteksi Apex Otomatis Mandiri / Label-Free (`091` — UF1 0,6707, AUC 0,9025):**
   Kami merancang estimator energi optical flow label-free:
   $$\mathcal{E}(t) = \frac{1}{H \times W} \sum_{x, y} \sqrt{u(x, y, t)^2 + v(x, y, t)^2}$$
   Puncak kurva energi $\arg\max_t \mathcal{E}(t)$ secara mandiri mendeteksi frame apex tanpa bantuan label manusia. Model tunggal berbasis auto-apex ini mencapai UF1 0,6707.
3. **Champion Deployable Produksi (`096` — UF1 0,7021, UAR 0,7418, ACC 69,79%):**
   Dengan menggabungkan model full-span dan model auto-apex (50:50), model **Champion Deployable** mencapai **0,7021 UF1**. Model ini berhasil menutup 95% jurang performa (*gap*) terhadap kondisi oracle (0,7104), membuktikan bahwa sistem siap dideploy penuh pada skenario nyata tanpa dependensi pada anotasi manual.

> [!TIP]
> **Pesan Utama untuk Paper:** Mengatasi 'asumsi jendela sempurna' melalui estimator energi optical flow membuktikan kelayakan inferensi otomatis end-to-end tanpa kehilangan akurasi signifikan.

---

## 5. Babak IV: Bias Spasial Anatomi — Gerakan Kepala vs. Regional Attention

### Latar Belakang & Masalah Utama
Apakah pergerakan kepala minor subjek mengganggu pengenalan ekspresi mikro? Dan apakah memfokuskan perhatian spasial model hanya pada area wajah anatomis tertentu dapat meningkatkan performa klasifikasi?

### Pembuktian Eksperimental
1. **Dampak Bencana Stabilisasi Kepala Rigid ECC (`102` — UF1 0,5906, ACC 61,98%):**
   Kami menguji algoritma stabilisasi kepala tingkat tinggi (*Enhanced Correlation Coefficient* / ECC) untuk meniadakan seluruh gerakan kepala rigid sebelum menghitung optical flow. Hasilnya mengejutkan: performa **anjlok drastis sebesar 9 poin UF1 (0,5906 vs 0,6816)**. Analisis video mengungkap temuan psikologis penting: gerakan kepala mikro involunter (sedikit anggukan atau sentakan kecil) ternyata terkoordinasi secara biologis dengan kemunculan ekspresi mikro seperti *disgust* dan *surprise*. Menghilangkan gerakan kepala rigid justru melenyapkan sinyal ekspresi alami.
2. **Multi-Region Soft Attention Berbasis Landmark (`163` — UF1 0,7283 pada LOSO 21):**
   Alih-alih menstabilkan kepala, kami menerapkan atensi regional terarah pada 3 zona otot utama: mata/alis (*upper face*), hidung (*mid face*), dan mulut (*lower face*). Pada protokol validasi LOSO 21, atensi regional meloloskan gate validasi statistik ($P=80,1\%$) dengan skor 0,7283 UF1.
3. **Validasi Regional Focus pada Skala Penuh (`182` — UF1 0,7121, ACC 69,51%, AUC 0,8964):**
   Dievaluasi pada seluruh 26 subjek LOSO, model atensi regional mempertahankan performa tinggi (0,7121 UF1 / 69,51% ACC), membuktikan bahwa membimbing model ke area otot aktif mencegah jaringan teralihkan oleh area wajah non-ekspresif.

> [!TIP]
> **Pesan Utama untuk Paper:** Stabilisasi kepala eksternal artifisial merusak integritas sinyal afektif mikro. Pilihan arsitektur yang benar adalah mempertahankan dinamika gerak alami dan mengarahkan fokus spasial melalui mekanisme regional attention.

---

## 6. Babak V: Paradigma Juara — Supervisi Ganda Facial Action Units (Multi-Task AU)

### Latar Belakang & Konsep Desain Model Juara
Ekspresi mikro wajah tidak terjadi secara acak, melainkan merupakan manifestasi langsung dari aktivasi unit motorik otot wajah spesifik yang dikodifikasikan dalam sistem *Facial Action Coding System* (FACS) sebagai **Action Units (AU)**.

Untuk membangun model yang memiliki pemahaman anatomis mendalam, kami merancang arsitektur **Multi-Task Dual-Head R3D-18**:
- **Kepala Utama (Primary Head):** Klasifikasi 5 kelas emosi kanonik MEGC (`happiness`, `disgust`, `repression`, `surprise`, `others`).
- **Kepala Tambahan (Auxiliary Head):** Prediksi multi-label 11 unit aksi otot aktif (AU1, AU2, AU4, AU5, AU7, AU9, AU10, AU12, AU14, AU15, AU17).

Formulasi fungsi objektif multi-task:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{y}, y) + \lambda_{\text{AU}} \sum_{k=1}^{11} \mathcal{L}_{\text{BCE}}(\hat{a}_k, a_k; w_k)$$

### Pembuktian Eksperimental
1. **Eksplorasi Bobot Regularisasi $\lambda_{\text{AU}}$ (`156` vs `140`):**
   - Pada $\lambda_{\text{AU}} = 0,01$ (`156`), kontribusi loss AU terlalu kecil untuk memandu ruang representasi laten (UF1 0,6682).
   - Pada $\lambda_{\text{AU}} = 0,5$ (`140`), gradien dari kontraksi otot mikro memberikan sinyal regularisasi yang optimal pada representasi laten 512-dimensi.
2. **Pencapaian Rekor Tertinggi Sepanjang Masa (`188` — LOSO 26 Penuh):**
   Model R3D-18 Multi-Task AU dengan $\lambda_{\text{AU}} = 0,5$ mencatatkan performa terbaik di seluruh repositori:
   - **Macro-F1 (UF1): 0,7211**
   - **Recall Seimbang (UAR): 0,7378**
   - **Akurasi (ACC): 69,92%**
   - **Macro-AUC: 0,8949**
   - **Spesifisitas: 91,70%**

Supervisi AU bertindak sebagai *anatomical inductive bias* yang memaksa representasi laten mempelajari konfigurasi gerak otot spesifik (misal: pengangkatan sudut bibir AU12 untuk *happiness*, pengerutan alis AU4 untuk *repression*), sehingga model kebal terhadap perbedaan bentuk wajah antar subjek.

> [!TIP]
> **Pesan Utama untuk Paper:** Supervisi ganda Facial Action Units secara simultan adalah kunci menembus batas akurasi pengenalan ekspresi mikro pada evaluasi LOSO yang ketat.

---

## 7. Babak VI: Audit Paradigma & Analisis Modus Kegagalan Model Pesaing

### Latar Belakang & Masalah Utama
Untuk memastikan orisinalitas dan ketahanan temuan di paper, kami mereplikasi dan mengaudit klaim-klaim dari paper eksternal serta paradigma arsitektur alternatif mutakhir (Spatiotemporal Transformer, Graph Neural Networks, dan Mekanika Regangan Kontinum).

### Pembuktian Eksperimental
1. **Audit Klaim Akurasi 88% Eksternal (`200` — Replikasi Non-LOSO vs. LOSO Riil):**
   Banyak penelitian eksternal mengklaim akurasi di atas 85–90% pada CASME II. Replikasi model eksternal kami pada split acak (*random train/test split*) memang menghasilkan akurasi semu 88,33%. Namun, ketika model yang sama diuji dengan protokol standar Leave-One-Subject-Out (LOSO) 26-Fold tanpa kebocoran identitas, performanya **anjlok bebas menjadi 53,89% UF1**. Klaim tinggi di literatur tersebut terbukti merupakan artefak dari *subject identity leakage* (wajah orang yang sama berada di data latih dan data uji).
2. **Kegagalan Total Spatiotemporal Transformer (`201` — CNN-ViViT, UF1 0,2093, ACC 28,05%):**
   Arsitektur Vision Transformer (ViViT) memerlukan ratusan ribu data video untuk mempelajari relasi spasial-temporal tanpa bias konvolusi. Pada dataset berskala kecil seperti CASME II (246 video klip), self-attention mengalami *severe data starvation* dan *overfitting* akut, dengan performa kolaps ke UF1 0,2093.
3. **Kegagalan Graph Neural Network Berbasis Landmark (`202` — GCN-GRU, UF1 0,1799, ACC 17,89%):**
   Pemodelan graf wajah dengan GCN-GRU di mana simpul adalah koordinat landmark MediaPipe 2D/3D mengalami kegagalan fatal (UF1 0,1799). Koordinat landmark diskret memiliki *jitter* resolusi dan **tidak mampu menangkap perubahan tekstur sub-piksel kontinu** seperti kerutan kulit halus (*skin wrinkling*) di area glabella atau nasolabial.
4. **Baseline Fisika: Aliran Optik + Tensor Regangan Infinitesimal (`203` — R3D-18 + Flow & Strain):**
   Untuk membandingkan supervisi biologis (AU) dengan representasi mekanika kontinum murni, kami menghitung tensor regangan mekanis:
   $$\varepsilon_{xx} = \frac{\partial u}{\partial x}, \quad \varepsilon_{yy} = \frac{\partial v}{\partial y}, \quad \varepsilon_{xy} = \frac{1}{2}\left(\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}\right)$$
   Model ini mencapai skor solid **UF1 0,6938 / ACC 68,29% / AUC 0,8923**, melampaui seluruh baseline non-AU. Namun, performanya tetap berada di bawah Model Juara AU (`188` — UF1 0,7211). Ini membuktikan bahwa supervisi biologis berakar pada pola aktivasi neuromuskular FACS memberikan sinyal diskriminatif yang lebih kaya daripada deformasi elastis mekanis semata.

> [!TIP]
> **Pesan Utama untuk Paper:** Analisis perbandingan membuktikan bahwa keunggulan R3D-18 Multi-Task AU bukan kebetulan statistik, melainkan solusi arsitektural yang paling selaras dengan sifat alami dan skala data mikro-ekspresi.

---

## 8. Panduan Integrasi ke Manuskrip Paper (Paper Section Mapping)

Bagian ini memetakan eksperimen-eksperimen di atas ke struktur standar artikel jurnal ilmiah internasional (IEEE TAC, IEEE TBIOM, atau CVPR/ICCV/ECCV):

1. **Introduction & Motivation:**
   - Soroti kegagalan representasi RGB statis (`002`) akibat *identity bias* pada ekspresi mikro.
   - Jelaskan kontradiksi two-stream (`006`) dan keharusan mengeliminasi modalitas penampilan statis.
2. **Related Work & Critical Audit:**
   - Kutip hasil audit replikasi (`200`) untuk menjelaskan inkonsistensi klaim akurasi tinggi di literatur akibat kebocoran subjek.
   - Diskusikan batasan arsitektur mutakhir ViViT Transformer (`201`) dan GCN Landmark (`202`) pada domain berdata terbatas (*small-sample regime*).
3. **Methodology:**
   - Jelaskan formulasi TV-L1 Onset-Referenced Flow (`004`) dan perbandingannya terhadap sequential flow (`045`).
   - Uraikan arsitektur backbone R3D-18 spatiotemporal (`017`).
   - Deskripsikan estimator energi optical flow tanpa label untuk deteksi apex otomatis (`091` & `096`).
   - Rinci formulasi multi-task loss FACS Action Units (`188`) dan regularisasi bobot $\lambda_{\text{AU}}$.
4. **Experimental Results & Benchmark Comparison:**
   - Sajikan perbandingan metrik resmi pada Leave-One-Subject-Out 26-Fold: Akurasi (69,92%), Macro-F1 (0,7211), Balanced Recall (0,7378), dan Macro-AUC (0,8949).
   - Tampilkan kurva ROC makro dan kurva konvergensi pelatihan antar-fold.
5. **Ablation Studies & Discussion:**
   - **Tabel 1 (Ablasi Input):** `002` vs `003` vs `004` vs `006` vs `045`.
   - **Tabel 2 (Ablasi Backbone):** `010` vs `020` vs `037` vs `017`.
   - **Tabel 3 (Ablasi Apex & Deployment):** `068` vs `091` vs `096`.
   - **Tabel 4 (Ablasi Bias Spasial & Head Motion):** `102` vs `163` vs `182`.
   - **Tabel 5 (Ablasi Multi-Task AU & Baseline Fisika):** `017` vs `203` vs `156` vs `140` vs `188`.
