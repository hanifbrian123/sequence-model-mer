#!/bin/bash
# FASE 2 -- segmentation nyata (face parsing MediaPipe), bukan mask elips.
#
# Diukur lebih dulu, bukan diasumsikan:
#   masker elips lama (iter_49)  membuang 28,5% energi gerakan  -> UF1 0,6370
#   face parsing MediaPipe       membuang  4,7% energi gerakan  -> diuji di sini
#   83% dari yang dibuang parsing ada di pita bawah (dagu/leher), bukan rambut.
#   Dahi -- pita terpadat di seluruh frame -- justru DIPERTAHANKAN oleh parsing
#   dan dulu ikut terpotong oleh elips.
#
#   67  face parsing pada leaf full-span   (pembanding iter_14  = 0,6816)
#   68  face parsing pada leaf auto-apex   (pembanding iter_47  = 0,6707)
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V7.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE V7 START (fase 2: face parsing) ==="
for stem in \
    iter_67_r3d_faceparse_full_s42 \
    iter_68_r3d_faceparse_autoapex_s42
do
  out="experiments/protocol_v2/${stem}_v2_dev_p5"
  if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "SKIP ${stem}"
    continue
  fi
  log "START ${stem}"
  $CONDA src/run_protocol_v2.py \
      --config "configs/${stem}.json" \
      --role dev \
      --protocol "protocols/accuracy_v5.json" \
      --tag p5 --save_fold_checkpoints \
      > "experiments/${stem}_p5_out.log" 2>&1
  if grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "DONE  ${stem}: $(grep 'FINAL dev' "${out}/run.log" | tail -1)"
  else
    log "!! FAILED ${stem} -- lihat experiments/${stem}_p5_out.log"
  fi
done
log "=== QUEUE V7 COMPLETE ==="
