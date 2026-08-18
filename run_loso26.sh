#!/bin/bash
# ⚫ LOSO 26 -- SEGEL AUDIT DIPECAH (dikonfirmasi user 2026-08-05).
#
# Menjalankan kelima kandidat pra-registrasi, lalu RNN tanpa pretrain.
# LOSO 26 memakai kelima subject audit [4,6,8,17,24]; setelah ini segel tidak
# bisa dikembalikan. Angka LOSO 26 sebanding langsung dengan literatur & dosen.
#
# Kandidat 2 (gabungan 10 model) butuh kesepuluh leaf-nya di LOSO 26 dulu, jadi
# leaf dijalankan lebih dulu; fusinya dihitung di akhir dari probabilitas.
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_LOSO26.log"
P2="experiments/protocol_v2"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

run26(){   # $1 = stem config
  local stem="$1"
  local out="${P2}/${stem}_v2_dev_loso_all_p5"
  if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "SKIP ${stem}"; return
  fi
  log "START ${stem} (loso_all, 26 fold)"
  $CONDA src/run_protocol_v2.py --config "configs/${stem}.json" \
      --role dev --split loso_all --break_audit_seal \
      --protocol "protocols/accuracy_v5.json" --tag p5 --save_fold_checkpoints \
      > "experiments/${stem}_loso_all_p5_out.log" 2>&1
  if grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "DONE  ${stem}: $(grep 'FINAL dev' "${out}/run.log" | tail -1)"
  else
    log "!! FAILED ${stem} -- lihat experiments/${stem}_loso_all_p5_out.log"
  fi
}

runL(){    # $1 = stem, jalankan di LOSO 21 (untuk RNN)
  local stem="$1"
  local out="${P2}/${stem}_v2_dev_loso_dev_p5"
  if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "SKIP ${stem}"; return
  fi
  log "START ${stem} (loso_dev)"
  $CONDA src/run_protocol_v2.py --config "configs/${stem}.json" \
      --role dev --split loso_dev \
      --protocol "protocols/accuracy_v5.json" --tag p5 \
      > "experiments/${stem}_loso_dev_p5_out.log" 2>&1
  if grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "DONE  ${stem}: $(grep 'FINAL dev' "${out}/run.log" | tail -1)"
  else
    log "!! FAILED ${stem}"
  fi
}

log "=== LOSO 26 START -- SEGEL AUDIT DIPECAH ==="

# Kandidat 1,3,4,5 (model tunggal) dulu supaya hasilnya cepat terlihat.
log ">>> kandidat tunggal (1,3,4,5)"
run26 iter_14_r3d                      # kandidat 1 (dan leaf 'full')
run26 iter_89_r3d_regionattn_full_s42  # kandidat 5 (dan leaf 'regattn')
run26 iter_96_r3d_focus_region_full_s42   # kandidat 3 (ide user)
run26 iter_98_r3d_focus_es_full_s42       # kandidat 4 (ide user)

# Sisa leaf untuk kandidat 2 (gabungan 10 model).
log ">>> sisa leaf untuk gabungan"
for stem in iter_47_r3d_auto_apex_s42 iter_74_r3d_lastk8_ep40_s42 \
            iter_59_r3d_llrd_s42 iter_67_r3d_faceparse_full_s42 \
            iter_77_r3d_au05_full_s42 iter_69_r3d_es_refit_s42 \
            iter_88_r2plus1d_full_s42 iter_95_r3d_focus_energy_full_s42; do
  run26 "$stem"
done

# Kandidat 2: fusi 10 model di LOSO 26.
log ">>> kandidat 2: gabungan 10 model (LOSO 26)"
MEMBERS=""
for stem in iter_14_r3d iter_47_r3d_auto_apex_s42 iter_74_r3d_lastk8_ep40_s42 \
            iter_59_r3d_llrd_s42 iter_67_r3d_faceparse_full_s42 \
            iter_77_r3d_au05_full_s42 iter_69_r3d_es_refit_s42 \
            iter_89_r3d_regionattn_full_s42 iter_88_r2plus1d_full_s42 \
            iter_95_r3d_focus_energy_full_s42; do
  MEMBERS="$MEMBERS ${P2}/${stem}_v2_dev_loso_all_p5"
done
WEIGHTS=$(printf '1 %.0s' {1..10})
$CONDA src/fuse_protocol.py --members $MEMBERS --weights $WEIGHTS \
    --out "${P2}/fusion10_loso26" --name gabungan10_loso26 \
    >> "$QLOG" 2>&1 && log "fusi 10 model LOSO 26 selesai"
$CONDA src/compare_protocol.py \
    --baseline "${P2}/iter_14_r3d_v2_dev_loso_all_p5" \
    --candidate "${P2}/fusion10_loso26" \
    --out "${P2}/comparisons/fusion10_loso26_vs_baseline.json" \
    >> "$QLOG" 2>&1 && log "gate fusi LOSO 26 selesai"

# Lanjut RNN tanpa pretrain -- langsung di LOSO 26 (permintaan user).
log ">>> RNN tanpa pretrain + kombinasi (LOSO 26)"
for stem in iter_103_resnetgru_nopretrain_full_s42 \
            iter_104_resnetgru_nopretrain_apex_s42 \
            iter_105_focus_rnn_nopretrain_full_s42 \
            iter_106_focus_rnn_nopretrain_es_full_s42; do
  run26 "$stem"
done

log "menyusun ulang laporan..."
$CONDA src/make_experiment_report.py >> "$QLOG" 2>&1
log "=== LOSO 26 + RNN COMPLETE ==="
