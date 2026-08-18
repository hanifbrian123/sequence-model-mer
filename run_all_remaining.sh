#!/bin/bash
# Rantai tunggal untuk seluruh antrean 🔵 LOSO 21 yang tersisa.
#
# Sebelumnya tiap antrean dirantai lewat tugas latar terpisah, dan waktu sesi
# berhenti, kelima rantai itu ikut mati sementara antrean yang sedang jalan
# selamat. Satu proses berantai tidak punya titik putus itu: kalau prosesnya
# hidup, seluruh sisa antrean pasti jalan.
#
#   V14  Fase 2.3  segmentation attention        (89, 90, 91)
#   V15  Fase 2.4  fokus area AU (pembanding)    (92, 93, 94)
#   V16  Fase 2.5  saluran fokus                 (95, 96, 97)
#   V17  Fase 2.6  saluran fokus + early stopping (98, 99)
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
QLOG="experiments/QUEUE_ALL.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== RANTAI SISA ANTREAN START ==="

# Tunggu V13 kalau masih jalan. Selesai ditandai baris COMPLETE di lognya.
if [ -f experiments/QUEUE_V13.log ] && \
   ! grep -q "QUEUE V13 COMPLETE" experiments/QUEUE_V13.log; then
  log "menunggu V13 (arsitektur) selesai..."
  while ! grep -q "QUEUE V13 COMPLETE" experiments/QUEUE_V13.log 2>/dev/null; do
    sleep 60
  done
  log "V13 selesai"
fi

for queue in v14 v15 v16 v17; do
  upper=$(echo "$queue" | tr 'a-z' 'A-Z')
  if grep -q "QUEUE ${upper} COMPLETE" "experiments/QUEUE_${upper}.log" 2>/dev/null; then
    log "SKIP ${upper} (sudah selesai)"
    continue
  fi
  log ">>> mulai ${upper}"
  bash "run_queue_${queue}.sh"
  log "<<< selesai ${upper}"
done

log "menyusun ulang laporan..."
conda run -n facesleuth python src/make_experiment_report.py >> "$QLOG" 2>&1
log "=== RANTAI SISA ANTREAN COMPLETE ==="
