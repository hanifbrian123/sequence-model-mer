#!/bin/bash
# Second-stage queue: after the main queue completes, run the ACC-targeted
# no-class-weighting variant (raises 'others' accuracy) + final consolidation.
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE.log"
log(){ echo "[$(date +%H:%M:%S)] [q2] $*" | tee -a "$QLOG"; }

log "waiting for main queue to complete..."
while ! grep -q "QUEUE COMPLETE" experiments/QUEUE.log 2>/dev/null; do sleep 60; done
log "main queue done -> starting stage 2"

base="iter_29_noweight"
log "SMOKE $base"
$CONDA src/run_experiment.py --config "configs/${base}.json" --max_folds 2 --tag qsmoke > "experiments/${base}_qsmoke_out.log" 2>&1
if grep -q "^elapsed" "experiments/${base}_qsmoke/run.log" 2>/dev/null; then
  log "FULL $base"
  $CONDA src/run_experiment.py --config "configs/${base}.json" > "experiments/${base}_full_out.log" 2>&1
  log "DONE $base: $(grep -E 'UF1|UAR|ACC =' experiments/${base}/run.log | tail -3 | tr '\n' ' ')"
else
  log "!! SMOKE FAILED $base -> skip"
fi

log "FINAL CONSOLIDATION (stage 2)..."
$CONDA src/consolidate_final.py > experiments/CONSOLIDATION_out.log 2>&1
log "=== QUEUE2 COMPLETE ==="
