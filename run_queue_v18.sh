#!/bin/bash
# Replikasi seed untuk iter_89 -- segmentation attention, kandidat gate-pass
# tunggal pertama di sesi ini (0,7283 lawan 0,7109, dUF1 +0,0174, P=0,801).
#
# P=0,801 hanya 0,001 di atas ambang, dan kandidat ini muncul setelah belasan
# perbandingan lain di data yang sama, jadi sebagian dari skornya bisa saja
# efek memilih dari banyak pilihan. Riwayat project ini tegas soal ini: iter_74
# lolos replikasi 3/3 seed di 4-fold dan tetap mati di LOSO.
#
# Tiga seed tambahan, semuanya di LOSO 21. Kalau efeknya muncul lagi, ini
# kandidat champion. Kalau tidak, ini keberuntungan pemilihan.
#
# iter_102 juga menyimpan checkpoint supaya bobot region yang DIPELAJARI model
# bisa dibaca -- itu jawaban langsung atas "bagian wajah mana yang penting",
# datang dari model, bukan dari tebakan.
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V18.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

log "=== QUEUE V18 START: replikasi seed segmentation attention ==="
for stem in iter_100_r3d_regionattn_s123 iter_101_r3d_regionattn_s2024 \
            iter_102_r3d_regionattn_s7
do
  out="experiments/protocol_v2/${stem}_v2_dev_loso_dev_p5"
  if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "SKIP ${stem}"; continue
  fi
  log "START ${stem} (loso_dev)"
  $CONDA src/run_protocol_v2.py --config "configs/${stem}.json" \
      --role dev --split loso_dev --protocol "protocols/accuracy_v5.json" \
      --tag p5 --save_fold_checkpoints \
      > "experiments/${stem}_loso_dev_p5_out.log" 2>&1
  if grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "DONE  ${stem}: $(grep 'FINAL dev' "${out}/run.log" | tail -1)"
  else
    log "!! FAILED ${stem}"
  fi
done
log "menyusun ulang laporan..."
$CONDA src/make_experiment_report.py >> "$QLOG" 2>&1
log "=== QUEUE V18 COMPLETE ==="
