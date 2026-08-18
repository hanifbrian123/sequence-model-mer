#!/bin/bash
# FASE 4 -- Action Unit multi-task.
#
# Alasannya berasal dari struktur error, bukan dari tebakan arsitektur:
# 76% error champion melibatkan kelas `others`, dan `others` didefinisikan
# secara negatif ("bukan empat yang lain") sehingga tidak punya ciri emosi yang
# konsisten untuk dipelajari. Yang dia punya adalah ciri AU. Melatih backbone
# yang sama untuk menebak 11 AU sekaligus memberi target yang padat dan
# berdasar fisik -- persis yang kurang di data 192 sampel.
#
# Kepala AU hanya perancah saat latihan; saat inference cuma kepala emosi yang
# dibaca, jadi video upload tidak perlu anotasi AU apa pun.
#
#   76  bobot loss AU 0,2   (leaf full-span, pembanding iter_14 = 0,6816)
#   77  bobot loss AU 0,5
#   78  bobot loss AU 1,0
#   79  bobot 0,5 pada leaf auto-apex (pembanding iter_47 = 0,6707)
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V10.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE V10 START (fase 4: AU multi-task) ==="
for stem in \
    iter_76_r3d_au02_full_s42 \
    iter_77_r3d_au05_full_s42 \
    iter_78_r3d_au10_full_s42 \
    iter_79_r3d_au05_autoapex_s42
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
log "=== QUEUE V10 COMPLETE ==="
