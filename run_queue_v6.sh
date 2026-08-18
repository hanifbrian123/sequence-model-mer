#!/bin/bash
# FASE 1 lanjutan -- memperbaiki early stopping, lalu MENGUKUR versi yang bocor.
#
#   63  data cek diperbesar (17 sampel terlalu berisik untuk memilih epoch)
#   64  penilai diganti ke loss (lebih halus daripada UF1 di data kecil)
#   65  DIAGNOSTIK: early stopping yang mengintip fold ujian
#   66  DIAGNOSTIK: sama seperti 65 + adaptive fine-tuning
#
# 65 dan 66 BUKAN hasil yang sah. Keduanya ada semata-mata untuk mengukur
# berapa besar skor naik kalau epoch berhenti dipilih dengan melihat kunci
# jawaban. Jangan pernah dipromosikan, difusikan, atau dibandingkan sebagai
# hasil nyata. Pembanding jujurnya adalah iter_58/62.
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V6.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE V6 START ==="
for stem in \
    iter_63_r3d_es_bigcheck_s42 \
    iter_64_r3d_es_loss_s42 \
    iter_65_r3d_es_leaky_diag_s42 \
    iter_66_r3d_es_leaky_llrd_diag_s42
do
  out="experiments/protocol_v2/${stem}_v2_dev_p5"
  if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "SKIP ${stem}"
    continue
  fi
  case "$stem" in *_diag_*) log "CATATAN: ${stem} adalah DIAGNOSTIK BOCOR, bukan hasil sah." ;; esac
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
log "=== QUEUE V6 COMPLETE ==="
