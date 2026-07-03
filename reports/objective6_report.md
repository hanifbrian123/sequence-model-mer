# CASME II Micro-Expression Recognition — Objective Classes (6-class) Final Report

**Method family:** sequence models on optical-flow clips (same pipeline as the Estimated-Emotion task).
**Protocol:** LOSO, 26 folds, subject-independent, pooled UF1 / UAR / ACC.
**Classes (CASME2-ObjectiveClasses, AU-defined):** 6-class — original objective classes **{1, 2, 3, 4, 5, 7}**, **254 samples**.

> **Class 6 (1 sample) was dropped.** In LOSO a single-sample class is untrainable/unevaluable and, via inverse-frequency class weighting, blows the loss weight up to ~36× and wrecks training. The full 7-class run confirmed this: **UF1 0.20 / ACC 0.11**. Dropping class 6 (a standard, defensible choice) plus a **class-weight cap (10×)** fixed it.

Class distribution: obj1=25, obj2=15, obj3=**99**, obj4=26, obj5=20, obj7=69 (heavily imbalanced; obj3 ≈ 39%).

---

## 1. Final results (honest, pooled LOSO)

| Model | UF1 | UAR | ACC |
|---|---|---|---|
| Single **r3d_18** (flow, res128, TTA×5) | 0.5574 | 0.5599 | 0.6772 |
| **Ensemble** (r3d + mc3 + r2plus1d-4ch + r3d-focal), fuse-all | **0.5880** | **0.5934** | **0.7008** |

Per-class F1 (single r3d): obj1 0.45, obj2 0.45, obj3 **0.88**, obj4 0.47, obj5 0.44, obj7 0.65.

**Why UF1 ≪ ACC:** severe class imbalance. The big class obj3 (99) is easy (F1 0.88) and drives ACC up, while the small classes (obj2=15, obj5=20) are hard (F1 ~0.45) and drag the macro-UF1 down. ACC ~0.70 is respectable; UF1 ~0.59 reflects the small-class difficulty.

---

## 2. Method (same winning recipe as emotion)

- Input: **onset-referenced TV-L1 optical flow** (res 128), not raw RGB.
- Backbone: **r3d_18** (Kinetics-pretrained) — best of r3d / mc3 / r2plus1d.
- TTA×5, class weighting (capped 10×), label smoothing.
- **Ensemble = fuse all 4 diverse models.** As on emotion, individually-weaker models (focal, 4ch) still improve the fusion because their errors are decorrelated. (focal solo UF1 0.49, but it lifts the ensemble.)

---

## 3. Deployment / inference

Same interface as emotion; class names come from `deploy.json` (6 objective classes).

```bash
# single (recommended, realtime-friendly)
python src/infer.py --frames <folder_of_clip_frames> --models models/objective6_single

# ensemble (max accuracy, offline; 4 models)
python src/infer_ensemble.py --frames <folder_of_clip_frames> --models_root models/objective6_ensemble
```

Input contract identical to the emotion task (face-cropped aligned frames of one clip, first frame = onset). Realistic accuracy = the honest LOSO number (~0.70 ACC / 0.59 UF1) on unseen subjects.

---

## 4. Emotion vs Objective (summary)

| Task | classes | samples | best UF1 | best ACC |
|---|---|---|---|---|
| Estimated Emotion | 5 | 246 | 0.7177 | 0.6911 |
| Objective | 6 | 254 | 0.5880 | 0.7008 |

Objective has **higher ACC** (the dominant obj3 class is easy) but **lower UF1** (more classes + worse imbalance). Both are honest LOSO results with no protocol manipulation.

---

## 5. Reproducibility

- Objective index/manifests: `cache/index_objective.csv`, `cache/flow144/manifest_obj6.csv` (254 samples, 6-class).
- Configs: `configs/obj6_r3d.json` (+ `obj6_mc3/focal/4ch`). Train-on-all: `src/train_final.py`.
- LOSO eval: `src/run_experiment.py`; per-run logs in `experiments/obj6_*`; ledger `results_ledger.csv`.
