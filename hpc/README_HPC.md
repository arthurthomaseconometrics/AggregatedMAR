# Running the supplement's Section S.2.2 on a cluster

Audit of 2026-09-03.  See the header of `export_s22_reparam.py` for what was
fixed and why; the short version is that `pi_1` is identified and the apparent
non-identification was scipy's L-BFGS-B stopping on its default **absolute**
`gtol = 1e-5` while the Assumption-C2 weighted criterion is of order `1e-6`.

## What to copy

The whole repository directory. The files that matter:

    cf_weights.py              shared quadrature + optimizer tolerances (MDE_GTOL)
    stablemixture.py           DGP (burn-in side + independent seed streams fixed)
    subsampling_reparam.py     engine (single MDE functional, C1 ordering, thinning)
    export_s22_reparam.py      S.2.2 driver (restartable, --tex-only)
    table_s1_mc_accuracy.py    Table S.1 driver
    latex_table_export.py      booktabs formatting
    hpc/                       these scripts

Dependencies: `numpy scipy pandas matplotlib joblib`.  Nothing else, no
compilation, no GPU.

## Step 0 — preflight (seconds)

    python3 hpc/preflight_check.py

Exit code 0 means the four audit fixes are active in *this* environment.  If it
fails, do not launch anything else; send me the output.

## Step 1 — Table S.1 (~3 min on 15 cores)

    sbatch hpc/run_table_s1.sbatch      # or: bash hpc/run_table_s1.sbatch

Output: `results_table_s1/` (`mc_summary.csv`, `mc_raw_*.csv`,
`table_s1_mc_accuracy.tex`, `run_config.json`).

## Step 2 — Section S.2.2 (the expensive one)

    mkdir -p logs
    sbatch hpc/run_s22.sbatch           # array of 6 tasks, one per sample size

One task per `n` in `250 500 1000 10000 50000 100000`, `M = 200` replications
each.  Tasks are independent and write disjoint files, so a failed or
wall-clocked task can be resubmitted on its own:

    sbatch --array=5 hpc/run_s22.sbatch     # just n = 100000

`--skip-existing` is on, so a resubmitted task that already produced
`meta_n<n>.json` exits immediately.

### Rough cost

Measured locally: ~30 ms per block estimate at `b = 39`, ~35 ms at `b = 62`,
scaling roughly linearly in `b`.  The number of block estimates per replication
is capped at `OV_BLOCKS_TARGET + N_b` (about 2 000, see
`subsampling_reparam.OV_BLOCKS_TARGET`).

| n       | b     | blocks/rep | est. core-hours |
|---------|-------|-----------|-----------------|
| 250     | 39    | 218       | 0.8             |
| 500     | 62    | 447       | 1.7             |
| 1 000   | 99    | 912       | 4               |
| 10 000  | 464   | 1 929     | 21              |
| 50 000  | 1 357 | 1 982     | 55              |
| 100 000 | 2 154 | 2 044     | 91              |

Total ≈ 175 core-hours, i.e. under 2 h wall-clock for the largest task on 64
cores.  **Treat these as order-of-magnitude only** — they extrapolate from two
small sizes on a laptop, and the per-solve cost depends on how many L-BFGS-B
iterations the tighter tolerances need at large `b`.  Start with
`--array=0-3` if you want to see real timings before committing the big two.

If it turns out too expensive, the cheapest lever is the number of overlapping
blocks used for the confidence intervals: pass `--ci-step` explicitly (larger =
fewer blocks = cheaper, at the cost of a noisier CI quantile).  It does not
affect `Avg Est.`, `Std Est.` or the normality diagnostics, which use the
non-overlapping blocks.

## Step 3 — assemble the LaTeX

    bash hpc/collect_s22.sh

Reads whatever `summary_n<n>.csv` / `meta_n<n>.json` pairs exist and writes
`resultats_reparam/s22_reparam.tex` (design table + summary table + one detail
table and one figure per size).  It tells you which sizes are still missing.

## What to send back

    results_table_s1/mc_summary.csv
    results_table_s1/table_s1_mc_accuracy.tex
    resultats_reparam/summary_n*.csv
    resultats_reparam/meta_n*.json
    resultats_reparam/run_config_n*.json
    resultats_reparam/s22_reparam.tex
    logs/*.out

The CSVs are small (a few kB each).  The PNGs are ~1 MB each — send them only
if you want me to look at the diagnostics.

## What to look at first

In `resultats_reparam/summary_n*.csv`, across sample sizes:

- `Avg Est.` for `pi_1` should sit near 0.4375 and stop drifting toward 0.5.
- `Ratio` = `Dev.Std / (sqrt(n) * Std Est.)` should approach **1**.  It was
  0.54 at n = 250 and 0.51 at n = 500 locally — that is the small-`b` bias of
  subsampling, and it is the thing to watch: if it is still far from 1 at
  n = 100 000, the problem is the block size, not the estimator.
- `qCov90` / `nCov90` should approach 0.90, and `qCov95` / `nCov95` 0.95.
- `Dev.Mean` should stay bounded rather than growing with n.
