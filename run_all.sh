#!/bin/bash
# Replication driver.
#
#   bash run_all.sh            # "fast": Figure 1, OVX application (Tables S.14-S.15,
#                              #  Figures 2, S.6-S.8), and every Monte Carlo table
#                              #  rebuilt from the stored simulation results.  ~5 min.
#   bash run_all.sh full       # also reruns every Monte Carlo from scratch
#                              #  (S.1-S.13).  Several hundred core-hours: see README.
#
# PYTHON=/path/to/python bash run_all.sh   to choose the interpreter.
# NCORES=32 bash run_all.sh full           to set the number of workers.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
NCORES="${NCORES:-$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu )}"
MODE="${1:-fast}"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

step() { echo; echo "=== $* ($(date +%H:%M:%S))"; }

step "Preflight checks"
"$PY" hpc/preflight_check.py

step "Figure 1 (main paper)"
"$PY" figure1_stable_aggregate.py

if [ "$MODE" = "full" ]; then
    step "Table S.1 -- Monte Carlo, Case 1"
    "$PY" table_s1_mc_accuracy.py --n-jobs "$NCORES" --output-dir results_table_s1
    step "Table S.2 -- Monte Carlo, Case 2 (MAR(1,1))"
    "$PY" table_s2_mc_accuracy.py --n-jobs "$NCORES" --output-dir results_table_s2
    step "Tables S.3-S.9, Figures S.1-S.5 -- subsampling"
    "$PY" export_s22_reparam.py --sample-sizes 500 1000 10000 100000 500000 --M 200 \
        --n-cores "$NCORES" --skip-existing --output-dir resultats_reparam
    step "Tables S.11-S.13 -- test of H0: min psi_j = 0"
    (cd boundary_test && "$PY" boundary_test_mc.py --reps 2000 --workers "$NCORES" \
        && "$PY" subsampling_check.py --reps 500 --workers "$NCORES")
fi

step "Tables S.3-S.9 LaTeX (from resultats_reparam/summary_n*.csv)"
"$PY" export_s22_reparam.py --sample-sizes 500 1000 10000 100000 500000 --M 200 \
    --tex-only --output-dir resultats_reparam
step "Tables S.11-S.13 LaTeX (from boundary_test/results/)"
(cd boundary_test && "$PY" boundary_test_mc.py --table-only && "$PY" supplement_tables.py)

step "OVX application: estimation of the three models (~1 min)"
(cd ovx_estimation && "$PY" run_ovx_estimation.py)
step "OVX application: Table S.14"
"$PY" table_ovx_estimation.py
step "OVX application: Figure S.6 (filtering, ~1.5 min)"
"$PY" figure_ovx_deconvolution.py
step "OVX application: Table S.15"
"$PY" table_ovx_forecast.py
step "OVX application: Figures S.7-S.8"
"$PY" figure_ovx_forecast_component.py
step "OVX application: Figure 2 (main paper)"
"$PY" figure_ovx_combined_forecast.py

echo; echo "Done ($MODE)."
