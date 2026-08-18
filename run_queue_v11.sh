#!/bin/bash
# BAGIAN A -- replikasi kandidat pertama yang mendekati lolos gate.
# BAGIAN B -- Fase 5.0: uji ulang di LOSO 21 subject.
#
# BAGIAN A. iter_74 (epoch 40 + rata-rata 8 epoch terakhir) mencetak 0,6983
# lawan baseline 0,6816: dUF1 +0,0166 dengan P=0,798, meleset dari ambang 0,80
# sebesar 0,002. Riwayat project ini jelas soal skor titik -- ensemble 3-seed
# pernah mencetak 0,7241 lalu runtuh ke 0,6936 saat diuji 5 seed. Jadi yang
# menentukan bukan angka 0,6983, melainkan apakah efeknya muncul lagi di seed
# lain. Catatan: dua perubahannya sendiri-sendiri justru MERUGIKAN (epoch 60
# saja 0,6644; rata-rata 8 saja 0,6702), jadi ini murni efek interaksi dan
# karena itu makin perlu direplikasi.
#
# BAGIAN B. Saringan 4-fold hanya melatih dengan ~144 sampel; LOSO 21 subject
# melatih dengan ~183. Ide yang gagal karena kelaparan data diuji ulang di sini.
# loso_dev dipakai, bukan loso_all, karena loso_all memakai 5 subject audit --
# yang meskipun sudah dipakai 37x di percobaan LAMA, belum pernah menyentuh satu
# pun keputusan yang dibuat sejak protokol v2 berlaku.
cd "d:/AI-Projects/casmeII-new-from-sequence-model" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_V11.log"
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

log "=== QUEUE V11 BAGIAN A: replikasi iter_74 (3 percobaan, ~70 menit) ==="
for stem in \
    iter_80_r3d_lastk8_ep40_s123 \
    iter_81_r3d_lastk8_ep40_s2024 \
    iter_82_r3d_lastk8_ep40_autoapex_s42
do
  run_one "$stem" grouped
done

log "=== QUEUE V11 BAGIAN B: LOSO 21 subject (8 percobaan, ~9 jam) ==="
log "audit subject [4,6,8,17,24] TIDAK dipakai di bagian ini"
for stem in \
    iter_14_r3d \
    iter_47_r3d_auto_apex_s42 \
    iter_74_r3d_lastk8_ep40_s42 \
    iter_77_r3d_au05_full_s42 \
    iter_69_r3d_es_refit_s42 \
    iter_59_r3d_llrd_s42 \
    iter_61_r3d_ep60_s42 \
    iter_67_r3d_faceparse_full_s42
do
  run_one "$stem" loso_dev
done
log "=== QUEUE V11 COMPLETE ==="
