#!/bin/bash
# Jalankan satu atau beberapa run eksperimen CASME II, lalu susun ulang laporan.
#
#   bash scripts/run_queue.sh 014              # cocokkan awalan nomor
#   bash scripts/run_queue.sh 047_r3d_auto_apex_s42 # nama lengkap
#   bash scripts/run_queue.sh all              # semua config di configs/
#
# Run yang summary.json-nya sudah "complete": true akan DILEWATI (aman diulang).
# Paksa ulang dengan FORCE=1.
#
# Log stdout ditaruh di logs/ (sekali pakai, gitignored).
# Log permanen per-run tersimpan di runs/<nama>/run.log.
set -u
cd "$(dirname "$0")/.." || exit 1

if [ -f "/c/Users/OWNER/miniconda3/envs/facesleuth/python.exe" ]; then
  DEFAULT_PY="/c/Users/OWNER/miniconda3/envs/facesleuth/python.exe"
elif [ -f "C:/Users/OWNER/miniconda3/envs/facesleuth/python.exe" ]; then
  DEFAULT_PY="C:/Users/OWNER/miniconda3/envs/facesleuth/python.exe"
else
  DEFAULT_PY="python"
fi

PY="$DEFAULT_PY"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs logs/runs
QLOG="logs/queue_$(date +%Y%m%d_%H%M%S).log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

if [ "$#" -eq 0 ]; then
  echo "Gunakan: bash scripts/run_queue.sh <awalan|nama|all> ..." >&2
  exit 2
fi

stems=()
for arg in "$@"; do
  if [ "$arg" = "all" ]; then
    for f in configs/*.json; do stems+=("$(basename "$f" .json)"); done
  else
    found=0
    for f in configs/"$arg"*.json; do
      [ -e "$f" ] || continue
      stems+=("$(basename "$f" .json)"); found=1
    done
    [ "$found" -eq 0 ] && log "!! Tidak ada config cocok: $arg"
  fi
done
[ "${#stems[@]}" -eq 0 ] && { log "Tidak ada yang dijalankan"; exit 1; }

log "=== ANTREAN CASME II: ${stems[*]} ==="

for stem in "${stems[@]}"; do
  out="runs/${stem}_v2_dev_loso_all_p5"
  
  if [ -z "${FORCE:-}" ] && [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "LEWATI ${stem} (sudah selesai; gunakan FORCE=1 untuk menjalankan ulang)"
    continue
  fi

  log "MULAI  ${stem}"
  $PY -m casme.training.run_protocol_v2 --config "configs/${stem}.json" --role dev --split loso_all --break_audit_seal --tag p5 \
      > "logs/runs/${stem}_stdout.log" 2>&1 || true

  if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "SELESAI ${stem}: $(grep 'FINAL dev' "${out}/run.log" 2>/dev/null | tail -1)"
  else
    log "!! GAGAL ${stem} -- lihat logs/runs/${stem}_stdout.log"
  fi
done

log "Menyusun ulang tabel hasil & laporan..."
$PY -m casme.reporting.collect >> "$QLOG" 2>&1
$PY -m casme.reporting.report >> "$QLOG" 2>&1

log "Memeriksa integritas struktur..."
if $PY -m unittest discover -s tests >> "$QLOG" 2>&1; then
  log "=== ANTREAN SELESAI -> results/results.csv + reports/LAPORAN.md ==="
else
  log "!! PEMERIKSAAN GAGAL -- jalankan 'bash scripts/check.sh' untuk detail"
  exit 1
fi
