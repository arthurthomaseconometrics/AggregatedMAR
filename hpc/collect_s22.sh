#!/bin/bash
# Assemble results_reparam/s22_reparam.tex from the per-size results already
# on disk.  Safe to run at any point: it reports which sizes are still missing
# and builds the .tex from the ones that are there.  Seconds, no cluster needed.
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
python3 export_s22_reparam.py \
    --sample-sizes 500 1000 10000 100000 500000 \
    --M 200 --tex-only --output-dir results_reparam
