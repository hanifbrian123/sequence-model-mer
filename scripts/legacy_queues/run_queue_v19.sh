#!/bin/bash
# Acuan baseline di seed yang SAMA dengan kandidat segmentation attention.
#
# Kandidat punya 4 seed di LOSO (0,7283 / 0,6865 / 0,6862 / 0,6953, rata-rata
# 0,6991) sementara acuannya cuma ada di seed 42 (0,7109). Membandingkan
# rata-rata 4 seed dengan satu seed tidak sah: variasi seed di project ini
# mencapai 0,073 pada baseline yang sama.
#
# Tanpa acuan berpasangan, kita tidak bisa tahu apakah seed 42 kebetulan bagus
# untuk KANDIDAT, atau memang seed 42 bagus untuk SEMUA model.
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V19.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE V19 START: acuan baseline seed 123/2024/7 di LOSO ==="
for stem in iter_14_r3d_s123_loso iter_14_r3d_s2024_loso iter_14_r3d_s7_loso; do
  out="experiments/protocol_v2/${stem}_v2_dev_loso_dev_p5"
  if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "SKIP ${stem}"; continue
  fi
  log "START ${stem} (loso_dev)"
  $CONDA src/run_protocol_v2.py --config "configs/${stem}.json" \
      --role dev --split loso_dev --protocol "protocols/accuracy_v5.json" \
      --tag p5 > "experiments/${stem}_loso_dev_p5_out.log" 2>&1
  if grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "DONE  ${stem}: $(grep 'FINAL dev' "${out}/run.log" | tail -1)"
  else
    log "!! FAILED ${stem}"
  fi
done
log "=== QUEUE V19 COMPLETE ==="
