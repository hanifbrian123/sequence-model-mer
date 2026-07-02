# CASME II Micro-Expression Recognition — Estimated Emotion (5-class) Final Report

**Method family:** sequence (spatiotemporal) models on optical-flow clips.
**Protocol:** LOSO (Leave-One-Subject-Out), 26 folds, subject-independent; predictions pooled across all folds; metrics = **UF1** (macro-F1), **UAR** (macro-recall / balanced acc), **ACC**.
**Classes (Estimated Emotion, CASME2-coding):** happiness(32), disgust(63), repression(27), surprise(25), others(99) — **246 samples**. (sadness=7, fear=2 dropped — standard.)

---

## 1. Final results (honest, pooled LOSO)

| Model | UF1 | UAR | ACC |
|---|---|---|---|
| Single **r3d_18** (flow, res128, TTA×5) — *recommended for realtime* | 0.7145 | 0.7207 | 0.6870 |
| **Ensemble** (r3d + mc3 + r2plus1d-4ch + r3d-focal), late-fusion | **0.7177** | **0.7239** | **0.6911** |

Per-class F1 (ensemble): happiness 0.66, disgust 0.63, repression 0.69, surprise **0.92**, others 0.64.

Confusion matrix (ensemble, rows = true):
```
          happ  disg  repr  surp  othr
happiness   25     0     0     1     6
disgust      2    39     1     0    21
repression   4     0    16     0     7
surprise     0     0     0    24     1
others       8    17     7     1    66
```

**Bottleneck:** the "others" class. 88% of errors involve it, dominated by `disgust↔others` (38 cases). "others" is a semantic grab-bag (40% of data) and is the structural ceiling.

---

## 2. What worked (evidence-based)

The winning recipe, built by systematic ablation (see §3):
1. **Optical flow, onset-referenced (TV-L1)** as input — not raw RGB. This was the single biggest jump (RGB ACC 0.33 → flow 0.64). Flow discards static appearance/identity that a 3D-CNN otherwise overfits.
2. **3D-CNN backbone `r3d_18`** (Kinetics-pretrained) > mc3_18 > r2plus1d_18.
3. **Resolution 128** > 112.
4. **Test-time augmentation (TTA×5)** and **class weighting + label smoothing**.
5. **Late-fusion** of a few diverse models for a small extra lift.

## 3. What did NOT work (dropped after testing)

| Lever | Result | Verdict |
|---|---|---|
| Raw RGB sequence | ACC 0.33 | overfits identity |
| onset_ref residual | ACC 0.57 | good, but flow better |
| Two-stream (flow + appearance) | 0.639 | appearance stream drags ↓ |
| Onset→apex sampling | 0.36 | truncation destroys info |
| T=24 frames | 0.671 | T=16 better |
| Optical strain (3ch / 4ch) | 0.69 / 0.67 | neutral / worse |
| Motion magnification (α=5) | 0.68 | distorts, hurts surprise |
| Extra r3d seeds | s123=0.672 | high seed variance; hurts fusion |
| Two-stage (others-vs-expr → 4-way) | 0.655 < 0.724 (matched folds) | error propagation |
| Mixup | 0.667 | hurts fusion |
| Apex-flow + shallow net | 0.545 | far below sequence (also unfit for realtime) |

A serious **bug was found and fixed**: TTA initially collapsed results (UF1 0.33) because the eval loader was shuffled, misaligning per-sample probabilities across passes. Fixed by decoupling dataset augmentation from loader shuffle. Every new code path is now smoke-tested on 2 folds first.

---

## 4. Honest assessment vs the 0.80 target

The target was ACC ≥ 0.80. After 19+ configurations plus fusion, two-stage, and magnification, the honest ceiling for this approach on the **strict 5-class (incl. "others") LOSO** protocol is **~0.72 UF1 / ~0.69 ACC**. Reasons:
- **"others" class** (40% of data, semantically mixed) is a structural limiter — 88% of errors touch it.
- **246 samples** is very small; large 3D-CNNs overfit (train-acc ~0.9 vs test ~0.69), and seed variance is high (0.672–0.7145).
- ~15% of samples are misclassified by *every* model tried — a near-irreducible core for this model family.

Published results near 0.80 on CASME II generally use different protocols/sample-sets or heavy per-method engineering that does not reproduce as a quick lever here. We did **not** change the protocol to inflate the number (no 3-class variant), per the integrity requirement. **0.72 UF1 is a competitive, honest sequence-model result.**

---

## 5. Deployment / inference

**Recommended for a realtime video app: the SINGLE model.** A sliding window of face-cropped frames → prediction. (The single r3d is ~as accurate as the ensemble, far cheaper, and needs no apex detection.)

```bash
# single (recommended, realtime-friendly)
python src/infer.py --frames <folder_of_clip_frames> --models models/emotion_single

# ensemble (max accuracy, offline; 4 models, dual preprocessing)
python src/infer_ensemble.py --frames <folder_of_clip_frames> --models_root models/emotion_ensemble
```

**Input contract the app must satisfy:**
1. A folder of **face-cropped, aligned frames** of ONE micro-expression clip (onset→offset), filenames sorting chronologically.
2. Faces cropped/aligned like CASME II `Cropped` (the app needs a face detector + aligner upstream).
3. The **first frame is treated as onset (neutral)**; optical flow is computed relative to it internally.

**Output:** predicted class + full probability distribution.

**Realistic accuracy:** the honest LOSO number (~0.69 ACC / 0.72 UF1) reflects unseen subjects — that is what to expect in production. (A quick sanity demo on a training clip predicts confidently but is optimistic.)

---

## 6. Reproducibility

- Preprocessing: `src/flow_preprocess.py` (TV-L1 onset-ref flow → `cache/flow144`, base 144).
- Best single config: `configs/iter_14_r3d.json`. Train-on-all: `src/train_final.py`.
- LOSO eval: `src/run_experiment.py`; multi-seed: `src/run_ensemble.py`; late-fusion: `src/fuse.py`.
- Full run history: `results_ledger.csv`; per-run logs in `experiments/<name>/`.

---

## 7. Extended push v2 (exhaustive ceiling test)

After the initial plateau, a second campaign of ~11 further levers (~12 h GPU) was run to try to beat UF1 0.72 / reach ACC ≥ 0.70. **None beat the `r3d@128` champion**, and the robust best is essentially unchanged.

| Lever (v2) | UF1 | ACC | Verdict |
|---|---|---|---|
| **EMA weight-averaging** (`ema`) | 0.7094 | 0.6829 | ➖ ≈ baseline solo; **+0.004 ACC in fusion** (kept) |
| freeze BatchNorm | 0.5978 | 0.5732 | ❌ Kinetics-RGB BN stats mismatch flow |
| resnet_gru (2D-CNN+BiGRU) | 0.6864 | 0.6504 | ❌ drags fusion (breaks>fixes) |
| res160 (higher resolution) | 0.6657 | 0.6341 | ❌ overfits; 128 is the sweet spot |
| snapshot ensemble (cyclic LR) | 0.6958 | 0.6707 | ❌ cycles undertrain; drags fusion |
| test-time BN adaptation | 0.6805 | 0.6585 | ❌ noisy on small LOSO folds |
| sequential (consecutive) flow | 0.5602 | 0.5244 | ❌ ≪ onset-referenced flow |
| random-erase augmentation | 0.6936 | 0.6707 | ❌ modest, below champion |
| no class-weighting (ACC-oriented) | 0.7027 | 0.6748 | ❌ raises "others" recall but drags ensemble |
| multi-seed averaging | 0.6819 | 0.6545 | ❌ mixes lucky+unlucky seed → regress to mean |
| nested-LOSO decision-tuning | 0.66 | 0.63 | ❌ overfits the 26-subject selection set |

**Best robust deployable (fixed-rule):** `deployed-4 + EMA` = **UF1 0.7182 / UAR 0.7229 / ACC 0.6951** (a marginal EMA gain over the shipped 0.7177/0.6911). A *curated-pool* nested-LOSO selective fusion touched **UF1 ~0.7302 / ACC ~0.6992**, but it is pool-sensitive (degrades to 0.686 once weak members enter the pool), so it is reported as an upper estimate, **not** the deployable headline.

**Conclusion (honest):** ~0.72 UF1 / ~0.695 ACC is a genuine structural ceiling for strict 5-class LOSO (incl. "others") on 246 samples — confirmed now from ~6 independent angles: every new base model ≤ champion and drags fusion; every post-hoc trick (seed-averaging, decision-tuning, fusion re-selection) overfits the tiny 26-subject set; and two SOTA neighbours score lower on the same protocol (HTNet 0.556; a Micron-BERT frozen-feature LOSO probe 0.36). **ACC ≥ 0.70 was approached (0.6992) but not robustly cleared.** A real jump would require more data (composite SAMM/SMIC pretraining — not available) or apex-based methods (which break the realtime/no-apex requirement). No numbers were inflated and the protocol was never changed.
