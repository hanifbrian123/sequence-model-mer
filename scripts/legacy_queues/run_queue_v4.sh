#!/bin/bash
# Label-grouping comparison, 3 groupings x 3 seeds (2026-07-23).
#
# The recipe is held fixed at the iter_14 full-span baseline so that what varies
# is the taxonomy and nothing else. Three seeds per grouping because a single
# seed moves UF1 by up to 0.073 on this data -- point scores are not decidable.
#
#   g1_user               happiness / disgust+surprise / repression+fear+sadness / others
#   g2_disgust_others     happiness / disgust+others / repression / surprise
#   g3_disgust_others_plus  as g2, with fear+sadness folded into repression
#
# Each grouping uses a DERIVED protocol that copies the 5-class subject
# partition verbatim, so audit subjects [4, 6, 8, 17, 24] stay sealed in all of
# them. Cross-taxonomy numbers are NOT comparable to the 5-class champion.
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V4.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE V4 START ==="
for grouping in g1_user g2_disgust_others g3_disgust_others_plus; do
  for seed in 42 123 2024; do
    stem="iter_56_${grouping}_s${seed}"
    out="experiments/protocol_v2/${stem}_v2_dev_p5"
    if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
      log "SKIP ${stem}"
      continue
    fi
    log "START ${stem}"
    $CONDA src/run_protocol_v2.py \
        --config "configs/${stem}.json" \
        --role dev \
        --protocol "protocols/accuracy_v5_${grouping}.json" \
        --tag p5 --save_fold_checkpoints \
        > "experiments/${stem}_p5_out.log" 2>&1
    if grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
      log "DONE  ${stem}: $(grep 'FINAL dev' "${out}/run.log" | tail -1)"
    else
      log "!! FAILED ${stem} -- see experiments/${stem}_p5_out.log"
    fi
  done
done

log "MENYUSUN RINGKASAN..."
$CONDA src/summarize_groupings.py > experiments/GROUPING_SUMMARY.txt 2>&1
log "=== QUEUE V4 COMPLETE ==="
