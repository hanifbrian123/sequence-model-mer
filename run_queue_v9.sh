#!/bin/bash
# FASE 1 putaran keempat -- mendorong sumbu yang TERBUKTI bekerja.
#
# Temuan V5/V6: early stopping gagal di semua bentuknya, bahkan versi yang
# mengintip fold ujian (0,6742) masih di bawah baseline (0,6816). Yang membuat
# baseline unggul adalah eval_last_k=3 -- merata-ratakan prediksi 3 epoch
# terakhir. Merata-ratakan beberapa epoch lebih stabil daripada bertaruh pada
# satu epoch, dan di data 192 sampel kestabilan itu yang menentukan.
#
# Maka: perbesar jendela rata-rata, dan uji snapshot ensembling.
#   72  rata-rata 5 epoch terakhir
#   73  rata-rata 8 epoch terakhir
#   74  rata-rata 8 epoch terakhir, latih 40 epoch
#   75  snapshot ensembling 3 siklus cosine warm restart
# Pembanding: iter_14 = 0,6816 (rata-rata 3 epoch)
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V9.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE V9 START (perbesar jendela rata-rata epoch) ==="
for stem in \
    iter_72_r3d_lastk5_s42 \
    iter_73_r3d_lastk8_s42 \
    iter_74_r3d_lastk8_ep40_s42 \
    iter_75_r3d_snapshot3_s42
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
log "=== QUEUE V9 COMPLETE ==="
