#!/bin/bash
# Section S.2.2 -- unrestricted subsampling in the reparameterized space.
# Runs the requested sample sizes ONE AT A TIME (each uses all cores via
# joblib internally), writing summary_n<n>.csv / meta_n<n>.json / png as it
# goes -- so if you kill it partway, re-run with the same sizes and it will
# skip whatever already has a meta_n<n>.json (--skip-existing).
#
#   bash hpc/run_s22.sh                       # n = 250 500 1000, M=200
#   bash hpc/run_s22.sh "250 500 1000" 200 8   # sizes, M, cores
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
cd "$(dirname "$0")/.."

SIZES="${1:-250 500 1000}"
M="${2:-200}"
NCORES="${3:-$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu )}"
OUTDIR=resultats_reparam

echo "sizes=$SIZES  M=$M  cores=$NCORES  started=$(date)"
python3 hpc/preflight_check.py || { echo "PREFLIGHT FAILED"; exit 1; }

# shellcheck disable=SC2086
python3 export_s22_reparam.py \
    --sample-sizes $SIZES \
    --M "$M" \
    --n-cores "$NCORES" \
    --skip-existing \
    --output-dir "$OUTDIR"

echo "finished=$(date)"
