#!/bin/bash
# FASE 1 putaran ketiga -- early stopping gaya "refit".
#
# Bukti dari V5/V6: menahan subject untuk data cek selalu merugikan.
#   es_val_fraction 0,20 -> data cek 17 sampel  (terlalu berisik, UF1 0,5755)
#   es_val_fraction 0,35 -> data latih 89 sampel (terlalu sedikit)
# Refit memisahkan dua pekerjaan itu: probe hanya untuk MENCARI JUMLAH EPOCH,
# lalu model dilatih ulang dari nol memakai SELURUH subject latih. Tidak ada
# data latih yang hilang permanen, dan fold ujian tetap tidak pernah dilihat.
# Biayanya waktu latih 2x.
#
#   69  refit, penilai = loss
#   70  refit, penilai = UF1
#   71  refit + adaptive fine-tuning
# Pembanding jujur: iter_14 = 0,6816
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V8.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE V8 START (early stopping refit) ==="
for stem in \
    iter_69_r3d_es_refit_s42 \
    iter_70_r3d_es_refit_uf1_s42 \
    iter_71_r3d_es_refit_llrd_s42
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
log "=== QUEUE V8 COMPLETE ==="
