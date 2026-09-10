#!/bin/bash
# FASE 2.6 -- input segmentasi (area difokuskan) DIGABUNG dengan early stopping.
#
# Diminta user untuk masuk daftar LOSO 26. Diuji dulu di LOSO 21 supaya
# kandidatnya sudah punya angka sebelum segel audit dipecah.
#
# Ekspektasi jujur: kemungkinan besar kalah. Early stopping sudah gagal 9 kali
# di project ini; terbaiknya 0,6930 lawan acuan 0,7109 di protokol yang sama,
# dan bahkan versi yang mengintip fold ujian masih kalah dari baseline.
#
# Tetap dijalankan karena kombinasi dua perubahan bisa berperilaku berbeda dari
# masing-masingnya -- persis kasus iter_74, di mana "epoch 40" dan "rata-rata 8
# epoch" sama-sama merugikan sendirian tapi menguntungkan saat digabung.
#
# Varian early stopping yang dipakai adalah yang TERBAIK dari 9: mode refit
# (probe untuk cari jumlah epoch, lalu latih ulang dari nol dengan seluruh
# subject latih sehingga tidak ada data latih yang hilang) dengan penilai loss.
#
#   98  saluran fokus tersegmentasi + early stopping refit, full-span
#   99  saluran fokus tersegmentasi + early stopping refit, auto-apex
# Acuan LOSO 21: full-span 0,7109  auto-apex 0,6817
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V17.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE V17 START: Fase 2.6 fokus + early stopping (LOSO-21) ==="
for stem in \
    iter_98_r3d_focus_es_full_s42 \
    iter_99_r3d_focus_es_apex_s42
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
log "=== QUEUE V17 COMPLETE ==="
