#!/bin/bash
# FASE 3.1 (CNN+RNN) + sisa Fase 4 (AU multi-task).
#
# CNN+RNN adalah arsitektur yang dosen user sebut, dan di protocol v2 belum
# pernah dijalankan sama sekali -- satu-satunya angkanya berasal dari LOSO lama
# (iter_23, UF1 0,6864 lawan iter_14 0,7145 di protokol yang sama). Perlu diuji
# di protokol sekarang, dan juga sebagai anggota fusi: pelajaran 48b di project
# ini adalah skor solo tidak memprediksi nilai dalam fusi, dan CNN+RNN memproses
# waktu dengan cara yang berbeda dari 3D-CNN sehingga errornya berpeluang tidak
# berkorelasi.
#
# AU multi-task: bobot 0,2 memberi +0,0077 (P=0,627, gate gagal), 0,5 dan 1,0
# sama-sama merugikan. Trennya menurun terhadap bobot, jadi 0,1 diuji sekali
# untuk memastikan ini bukan sekadar "makin mendekati tanpa-AU makin bagus".
#
#   83  AU bobot 0,1 (full-span)
#   84  AU bobot 0,2 (auto-apex)
#   85  CNN+RNN full-span
#   86  CNN+RNN auto-apex
#   lalu LOSO 21 subject untuk AU 0,2 -- versi terbaiknya, bukan 0,5 yang
#   terlanjur masuk antrean sebelumnya.
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V12.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

run_one(){   # $1 = stem, $2 = split
  local stem="$1" split="$2" suffix="" extra=""
  if [ "$split" != "grouped" ]; then suffix="_${split}"; extra="--split ${split}"; fi
  local out="experiments/protocol_v2/${stem}_v2_dev${suffix}_p5"
  if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "SKIP ${stem} (${split})"
    return
  fi
  log "START ${stem} (${split})"
  $CONDA src/run_protocol_v2.py \
      --config "configs/${stem}.json" \
      --role dev ${extra} \
      --protocol "protocols/accuracy_v5.json" \
      --tag p5 \
      > "experiments/${stem}${suffix}_p5_out.log" 2>&1
  if grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "DONE  ${stem} (${split}): $(grep 'FINAL dev' "${out}/run.log" | tail -1)"
  else
    log "!! FAILED ${stem} (${split}) -- lihat experiments/${stem}${suffix}_p5_out.log"
  fi
}

log "=== QUEUE V12 BAGIAN A: Fase 3.1 CNN+RNN dan sisa Fase 4 ==="
for stem in \
    iter_85_resnetgru_full_s42 \
    iter_86_resnetgru_autoapex_s42 \
    iter_83_r3d_au01_full_s42 \
    iter_84_r3d_au02_autoapex_s42
do
  run_one "$stem" grouped
done

log "=== QUEUE V12 BAGIAN B: LOSO 21 untuk AU versi terbaik ==="
run_one iter_76_r3d_au02_full_s42 loso_dev
log "=== QUEUE V12 COMPLETE ==="
