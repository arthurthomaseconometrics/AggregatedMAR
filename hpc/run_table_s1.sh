#!/bin/bash
# Table S.1 -- Monte Carlo estimation accuracy (3 panels x 3 sample sizes,
# 1000 replications, 10 multi-starts). ~3 min on 15 cores.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
cd "$(dirname "$0")/.."
NCORES="${1:-$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu )}"
echo "cores=$NCORES  started=$(date)"
python3 hpc/preflight_check.py || { echo "PREFLIGHT FAILED"; exit 1; }
python3 table_s1_mc_accuracy.py --n-jobs "$NCORES" --output-dir results_table_s1
echo "finished=$(date)"
