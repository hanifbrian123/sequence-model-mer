#!/bin/bash
# Autonomous 12h experiment queue. Keeps GPU busy: waits for res160, then runs
# each queued config (smoke-gate new code paths), finally auto-consolidates.
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE START ==="

# 1) wait for res160 (currently on GPU) to finish
log "waiting for res160 to finish..."
while ! grep -q "^elapsed" experiments/iter_24_res160/run.log 2>/dev/null; do sleep 60; done
log "res160 done: $(grep -E 'UF1|ACC =' experiments/iter_24_res160/run.log | tail -2 | tr '\n' ' ')"

run_one(){   # cfg_basename  needs_smoke
  local base="$1"; local smoke="$2"; local cfg="configs/${base}.json"
  if [ "$smoke" = "1" ]; then
    log "SMOKE $base"
    $CONDA src/run_experiment.py --config "$cfg" --max_folds 2 --tag qsmoke > "experiments/${base}_qsmoke_out.log" 2>&1
    if ! grep -q "^elapsed" "experiments/${base}_qsmoke/run.log" 2>/dev/null; then
      log "!! SMOKE FAILED $base -> SKIP full run (see ${base}_qsmoke_out.log)"; return
    fi
    log "smoke ok $base"
  fi
  log "FULL $base"
  $CONDA src/run_experiment.py --config "$cfg" > "experiments/${base}_full_out.log" 2>&1
  log "DONE $base: $(grep -E 'UF1|UAR|ACC =' experiments/${base}/run.log | tail -3 | tr '\n' ' ')"
}

# 2) queue, highest-EV first
run_one iter_27_snapshot 1
run_one iter_25_bnadapt  1

# seq-flow cache must be built before iter_28
log "waiting for seq-flow cache..."
while ! grep -q "^DONE" experiments/flow144_seq_preprocess.log 2>/dev/null; do sleep 60; done
log "seqflow cache ready"
run_one iter_28_seqflow  1
run_one iter_26_erase    1

# 3) consolidate everything
log "CONSOLIDATING..."
$CONDA src/consolidate_final.py > experiments/CONSOLIDATION_out.log 2>&1
log "=== QUEUE COMPLETE ==="
