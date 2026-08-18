#!/bin/bash
# FASE 2.3 -- SEGMENTATION ATTENTION. Bagian saran dosen yang belum dikerjakan.
#
# Bedanya dengan Fase 2.2 (face parsing) yang sudah gagal:
#   Fase 2.2  KITA yang memutuskan bagian mana dibuang -> informasi hilang
#   Fase 2.3  MODEL yang belajar bagian mana lebih penting -> tidak ada yang hilang
#
# Enam masker region (alis, mata, hidung, mulut, pipi, dahi) dibuat dari landmark
# MediaPipe dan dinormalisasi sehingga jumlahnya tepat 1 di setiap piksel --
# sudah diverifikasi 100,0%. Gerbangnya sum_k w_k * mask_k dengan semua w_k = 1
# saat awal, jadi model dimulai dari posisi identik dengan baseline dan hanya
# bergerak kalau datanya membayar. Sudah diuji: selisih keluaran saat awal
# 1,9e-07 terhadap model tanpa attention.
#
# Ini penting karena SEMUA percobaan spasial yang menghapus sinyal kalah di
# project ini: elips 0,6370, kompensasi translasi 0,5987, ECC 0,5906, face
# parsing netral. Attention tidak menghapus apa pun.
#
# Langsung di LOSO-21, tidak disaring di 4-fold, karena saringan 4-fold sudah
# terbukti membalik hasil ke dua arah.
#
#   89  region attention statis, full-span   (acuan LOSO 0,7109)
#   90  region attention statis, auto-apex   (acuan LOSO 0,6817)
#   91  region attention dinamis, full-span  (bobot diprediksi dari klip)
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V14.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE V14 START: Fase 2.3 segmentation attention (LOSO-21) ==="
for stem in \
    iter_89_r3d_regionattn_full_s42 \
    iter_90_r3d_regionattn_apex_s42 \
    iter_91_r3d_regionattn_dyn_s42
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
log "=== QUEUE V14 COMPLETE ==="
