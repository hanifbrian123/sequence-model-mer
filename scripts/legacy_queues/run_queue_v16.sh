#!/bin/bash
# FASE 2.5 -- SALURAN FOKUS: input segmentasi area yang harus difokuskan.
#
# Ini yang user minta sejak awal dan sempat saya salah terjemahkan menjadi
# "buang pipi". Bedanya mendasar:
#
#   membuang (Fase 2.2/2.4)  input dipotong, informasi hilang permanen
#   memfokuskan (Fase 2.5)   input LENGKAP + satu saluran petunjuk "lihat sini"
#
# Dari mana area fokusnya? Dihitung dari gerakan di video itu sendiri, bukan
# peta anatomi tetap. Alasannya: micro-expression terlokalisasi dan letaknya
# BERBEDA tiap sampel -- jijik bergerak di hidung dan bibir atas, kaget di alis.
# Satu peta tetap tidak bisa mewakili keduanya. Perhitungannya tidak memakai
# label sama sekali, jadi berlaku sama persis untuk video upload.
#
# Sudah diverifikasi: 3 saluran asli TIDAK berubah sedikit pun (identik bit per
# bit), saluran ke-4 ditambahkan, dan peta fokusnya terbukti menunjuk ke area
# yang memang bergerak.
#
#   95  fokus halus mengikuti energi gerakan, full-span
#   96  fokus TERSEGMENTASI per region wajah, full-span
#   97  fokus tersegmentasi, auto-apex
# Acuan LOSO: full-span 0,7109  auto-apex 0,6817
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V16.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE V16 START: Fase 2.5 saluran fokus (LOSO-21) ==="
for stem in \
    iter_96_r3d_focus_region_full_s42 \
    iter_95_r3d_focus_energy_full_s42 \
    iter_97_r3d_focus_region_apex_s42
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
log "=== QUEUE V16 COMPLETE ==="
