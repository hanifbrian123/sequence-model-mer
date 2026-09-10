#!/bin/bash
# Susun ulang SEMUA keluaran turunan tanpa training apa pun:
#   - results/results.csv  dari runs/*/summary.json
#   - reports/LAPORAN.md   dari results.csv + docs/narasi.md
#
# Aman diulang kapan saja; tidak menyentuh bobot atau probabilitas.
set -eu
cd "$(dirname "$0")/.." || exit 1

if [ -f "/mnt/c/Users/OWNER/miniconda3/envs/facesleuth/python.exe" ]; then
  DEFAULT_PY="/mnt/c/Users/OWNER/miniconda3/envs/facesleuth/python.exe"
elif [ -f "/c/Users/OWNER/miniconda3/envs/facesleuth/python.exe" ]; then
  DEFAULT_PY="/c/Users/OWNER/miniconda3/envs/facesleuth/python.exe"
elif [ -f "C:/Users/OWNER/miniconda3/envs/facesleuth/python.exe" ]; then
  DEFAULT_PY="C:/Users/OWNER/miniconda3/envs/facesleuth/python.exe"
else
  DEFAULT_PY="python"
fi

PY="${CASME_PYTHON:-$DEFAULT_PY}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

$PY -m casme.reporting.collect
$PY -m casme.reporting.report
$PY -m unittest discover -s tests
