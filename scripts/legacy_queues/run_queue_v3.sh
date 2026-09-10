#!/bin/bash
# Two more seeds for the label-free apex leaf. The 3-seed ensemble scored 0.7241
# but gated at P=0.771 against a 0.80 threshold; adding seeds tightens the
# estimate, which is the principled way to resolve a near-miss. Adjusting fusion
# weights until it passes would not be.
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V3.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }
log "=== QUEUE V3 START ==="
for stem in iter_54_r3d_auto_apex_s7 iter_55_r3d_auto_apex_s2025; do
  out="experiments/protocol_v2/${stem}_v2_dev_p5"
  if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "SKIP ${stem}"; continue
  fi
  log "START ${stem}"
  $CONDA src/run_protocol_v2.py --config "configs/${stem}.json" --role dev --tag p5 \
      --save_fold_checkpoints > "experiments/${stem}_p5_out.log" 2>&1
  log "DONE ${stem}: $(grep 'FINAL dev' "${out}/run.log" | tail -1)"
done
log "=== QUEUE V3 COMPLETE ==="
