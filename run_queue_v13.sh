#!/bin/bash
# FASE 5.1 -- perbanyak KERAGAMAN untuk penggabungan, langsung di LOSO.
#
# Dari 25 percobaan, hanya satu yang lolos gate: menggabungkan 7 model berbeda
# di LOSO (0,7237 lawan 0,7109, P=0,858). Uji ketahanan menunjukkan hampir
# semua subset 4/5/6 anggota juga menang, jadi yang bekerja adalah keragaman
# itu sendiri, bukan satu kombinasi ajaib.
#
# Maka langkah paling masuk akal bukan mencari satu model juara, melainkan
# menambah ANGGOTA yang cara salahnya berbeda. Skor solo sengaja diabaikan
# sebagai kriteria: leaf AU (0,6743) dan auto-apex (0,6817) sama-sama di bawah
# baseline 0,7109 dan tetap menaikkan gabungan.
#
# Semua langsung di LOSO, tidak disaring dulu di 4-fold, karena kemarin terbukti
# saringan 4-fold bisa membalik hasil ke DUA arah (iter_74 menang palsu di
# 4-fold lalu mati di LOSO; face parsing kalah di 4-fold lalu netral di LOSO).
#
#   87  mc3_18        -- 3D-CNN keluarga lain
#   88  r2plus1d_18   -- 3D-CNN yang memisah ruang dan waktu
#   86  resnet_gru auto-apex -- CNN per frame + RNN (arsitektur dosen user)
#   85  resnet_gru full-span -- solo 0,5970 di 4-fold, dimasukkan justru karena
#       paling berbeda; gate yang akan memutuskan, bukan skor solonya.
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V13.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE V13 START: keragaman arsitektur di LOSO-21 ==="
for stem in \
    iter_87_mc3_full_s42 \
    iter_88_r2plus1d_full_s42 \
    iter_86_resnetgru_autoapex_s42 \
    iter_85_resnetgru_full_s42
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
log "=== QUEUE V13 COMPLETE ==="
