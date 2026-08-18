#!/bin/bash
# FASE 1 -- cara melatih (early stopping + adaptive fine-tuning).
#
# Semua varian memakai resep baseline iter_14 (full-span flow, r3d_18, seed 42)
# dan HANYA mengubah cara melatihnya, supaya yang dibandingkan benar-benar satu
# variabel. Pembanding: iter_14_r3d_v2_dev_p5 = UF1 0,6816.
#
#   58  early stopping saja (data cek diambil dari subject latih, bukan ujian)
#   59  adaptive fine-tuning: lapisan bawah pelan, head cepat (LLRD)
#   60  gradual unfreeze: 3 epoch pertama hanya head
#   61  kontrol -- 60 epoch tanpa early stopping (apakah 25 memang kurang?)
#   62  gabungan 58 + 59
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V5.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE V5 START (fase 1: cara melatih) ==="
for stem in \
    iter_58_r3d_earlystop_s42 \
    iter_59_r3d_llrd_s42 \
    iter_60_r3d_unfreeze_s42 \
    iter_61_r3d_ep60_s42 \
    iter_62_r3d_es_llrd_s42
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
log "=== QUEUE V5 COMPLETE ==="
