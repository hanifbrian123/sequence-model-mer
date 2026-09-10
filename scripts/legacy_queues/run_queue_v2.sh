#!/bin/bash
# Protocol-v2 autonomous GPU queue (2026-07-23).
#
# Ordered by information value, not by hope. Every run saves fold checkpoints so
# that any later evaluation-side question (fps, window error, apex hypothesis)
# can be answered by re-evaluation instead of retraining.
#
#   1 iter_14  baseline WITH checkpoints  -> unlocks fusion-level robustness tests,
#                                            and must reproduce UF1 0.6816 exactly;
#                                            if it does not, training is nondeterministic
#                                            and every paired comparison is suspect.
#   2 iter_41  HQ TV-L1 + ECC stabilized  -> the cache finished 246/246 and was never
#                                            used; the audit called motion stabilization
#                                            the single biggest remaining lever.
#   3 iter_49  soft face-ellipse ROI      -> independent axis, config was already written.
#   4 iter_50  auto-apex seed 123         -> label-free leaf at a second seed.
#   5 iter_51  auto-apex seed 2024        -> label-free leaf at a third seed;
#                                            together these test whether the deployable
#                                            champion survives away from seed 42.
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V2.log"
mkdir -p experiments
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

run_one(){  # config_stem  tag
  local stem="$1"; local tag="$2"
  local cfg="configs/${stem}.json"
  local out="experiments/protocol_v2/${stem}_v2_dev_${tag}"
  if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "SKIP ${stem} (already complete)"
    return
  fi
  log "START ${stem} (tag=${tag})"
  $CONDA src/run_protocol_v2.py --config "$cfg" --role dev --tag "$tag" \
      --save_fold_checkpoints > "experiments/${stem}_${tag}_out.log" 2>&1
  if grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "DONE  ${stem}: $(grep 'FINAL dev' "${out}/run.log" | tail -1)"
  else
    log "!! FAILED ${stem} -- see experiments/${stem}_${tag}_out.log"
  fi
}

log "=== QUEUE V2 START ==="
run_one iter_14_r3d                p5ck
run_one iter_41_r3d_hq_stabilized  p5
run_one iter_49_r3d_face_roi_s42   p5
run_one iter_50_r3d_auto_apex_s123 p5
run_one iter_51_r3d_auto_apex_s2024 p5
log "=== QUEUE V2 COMPLETE ==="
