#!/bin/bash
# Pemeriksaan struktur + integritas repositori CASME II.
# Jalankan SEBELUM menyerahkan pekerjaan.
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

$PY -m unittest discover -s tests -v
