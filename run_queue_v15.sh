#!/bin/bash
# FASE 2.4 -- FOKUS KE AREA AU SAJA (buang pipi, opsional buang dahi).
#
# Ini yang user tanyakan dan memang belum pernah dijalankan: versi di mana
# hanya mata/alis/hidung/mulut yang dipakai, bukan seluruh wajah.
#
# Diukur lebih dulu, bukan diasumsikan:
#   buang pipi saja      -> sisa 67,0% luas, 71,5% energi gerakan
#   buang pipi + dahi    -> sisa 51,1% luas, 51,6% energi gerakan
# Sebagai pembanding, face parsing yang sudah gagal cuma membuang 4,7% energi.
# Jadi ini pemotongan 6x-10x lebih agresif.
#
# Peringatan dari anotasi dataset: AU6 (pipi terangkat, 13 sampel) dan AU14
# (lesung, 27 sampel) berada DI PIPI. Membuang pipi berarti membuang isyarat
# utama 40 dari 246 sampel. Itu sebabnya ini diuji, bukan diasumsikan benar.
#
#   92  buang pipi saja, full-span      (acuan LOSO 0,7109)
#   93  buang pipi + dahi, full-span    (acuan LOSO 0,7109)
#   94  buang pipi + dahi, auto-apex    (acuan LOSO 0,6817)
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V15.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE V15 START: Fase 2.4 fokus area AU (LOSO-21) ==="
for stem in \
    iter_92_r3d_nocheek_full_s42 \
    iter_93_r3d_auroi_full_s42 \
    iter_94_r3d_auroi_apex_s42
do
  out="experiments/protocol_v2/${stem}_v2_dev_loso_dev_p5"
  if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "SKIP ${stem}"
    continue
  fi
  log "START ${stem} (loso_dev)"
  $CONDA src/run_protocol_v2.py \
      --config "configs/${stem}.json" \
      --role dev --split loso_dev \
      --protocol "protocols/accuracy_v5.json" \
      --tag p5 \
      > "experiments/${stem}_loso_dev_p5_out.log" 2>&1
  if grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "DONE  ${stem}: $(grep 'FINAL dev' "${out}/run.log" | tail -1)"
  else
    log "!! FAILED ${stem} -- lihat experiments/${stem}_loso_dev_p5_out.log"
  fi
done
log "=== QUEUE V15 COMPLETE ==="
