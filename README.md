# Replication files — Estimating and forecasting stable aggregates of moving averages

This folder reproduces every table and figure of the paper (`main_v26.tex`) and of
its Online Supplement (`online_supplement4.tex`), including the Monte Carlo study
of the test of $H_0:\min_j\psi_j = 0$ (Supplement, Section S.4).

Everything is in Python. The Monte Carlo results that are too expensive to rerun
on a laptop are included (`results_*/`, `resultats_reparam/`,
`boundary_test/results/`), so the fast mode rebuilds Tables S.3–S.15 from them in
a few minutes (Tables S.1–S.2 have no rebuild-only option: their stored `.tex` is
the output of the full run). The scripts that produced those results are in the
package too, with the exact settings used.

## 1. Quick start

```bash
pip install -r requirements.txt
bash run_all.sh          # fast mode, about 6 minutes
bash run_all.sh full     # also reruns every Monte Carlo (cluster scale, see Section 4)
```

`PYTHON=/path/to/python` and `NCORES=<n>` can be set before the command.
Each script also runs on its own (Section 3), writes its outputs to its own folder,
and records its settings, package versions and timestamp in a `run_config.json`.

## 2. Requirements

Python ≥ 3.10 with `numpy scipy pandas matplotlib joblib tqdm` (see
`requirements.txt`). No compilation, no GPU, no data download. Tested with
Python 3.13–3.14, numpy 2.5, scipy 1.18, pandas 3.0–3.1 on macOS (arm64), and,
for the largest simulations, Linux x86-64 (30 cores).

`python hpc/preflight_check.py` checks the environment in a few seconds (optimizer
tolerances, simulation of the data-generating process, ordering of the labels,
multi-start). It must print `ALL CHECKS PASSED` before any long run.

## 3. Map from the paper to the code

Numbering is that of `main_v26.pdf` and `online_supplement4.pdf`.

### Main paper

| Output | Script | Output folder | Time |
|---|---|---|---|
| Figure 1 (`fig:contagg`, `discretestableagg.pdf`) | `figure1_stable_aggregate.py` | `results_figure1/` | seconds |
| Figure 2 (`fig:combined_forecasts`, `InSCombinedForecast.png`) | `figure_ovx_combined_forecast.py` | `results_ovx_combined_forecast/` | seconds |

The Monte Carlo and OVX results discussed in Sections 4–5 of the paper are those of
the Supplement tables below.

### Online Supplement

| Output | Script | Output folder | Stored results computed on | Time (full run) |
|---|---|---|---|---|
| Table S.1 (`tab:mc_accuracy_full`), Monte Carlo, Case 1 | `table_s1_mc_accuracy.py` | `results_table_s1/` | laptop | ~3 min (laptop or 15 cores) |
| Table S.2 (`tab:mc_accuracy_mar11`), Monte Carlo, Case 2 | `table_s2_mc_accuracy.py` | `results_table_s2/` | laptop | not timed; longer than S.1 |
| Table S.3 (`tab:mc_design`) | `export_s22_reparam.py` | `resultats_reparam/s22_reparam.tex` | — | from the runs below |
| Table S.4 (`tab:subsampling_summary`) | `export_s22_reparam.py` | idem | — | from the runs below |
| Tables S.5–S.9 + Figures S.1–S.5 (n = 500, 1,000, 10,000, 100,000, 500,000) | `export_s22_reparam.py` | `resultats_reparam/` (`summary_n*.csv`, `diagnostics_n*.png`) | n ≤ 10,000: laptop; n = 100,000 and 500,000: cluster (30 cores) | 14 min (n = 1,000), 27 min (10,000) on a laptop; 33 min (100,000), 41 min (500,000) on 30 cores |
| Table S.10 (`tab:bt_design`) | design table, typed in the supplement; values are the constants at the top of `boundary_test/boundary_test_mc.py` | — | — | — |
| Table S.11 (`tab:bt_mc_supp`) | `boundary_test/boundary_test_mc.py` | `boundary_test/results/tab_boundary_mc.tex`, `summary.csv`, `replications.csv` | laptop | ~75 min on 12 cores |
| Tables S.12 (`tab:bt_mc_cv`), S.13 (`tab:bt_mc_diag`) | `boundary_test/supplement_tables.py` | `boundary_test/results/tab_supp_cv.tex`, `tab_supp_diag.tex` | built from `replications.csv` | seconds |
| Subsampling version of the test (Section S.4.2, "Alternative critical values") | `boundary_test/subsampling_check.py` | `boundary_test/results/subsampling_check.csv` | laptop | ~8 min on 12 cores |
| Table S.14 (`tab:ovx_estimation`) | `ovx_estimation/run_ovx_estimation.py` (estimation) then `table_ovx_estimation.py` | `ovx_estimation/estimation.csv`, `results_ovx_estimation/` | — | ~1 min |
| Figure S.6 (`fig:ovx_deconv`, `OVXDeconv48.pdf`) | `figure_ovx_deconvolution.py` | `results_ovx_deconvolution/` | — | ~1.5 min |
| Table S.15 (`tab:bubble_forecast`) | `table_ovx_forecast.py` | `results_ovx_forecast/` | — | seconds |
| Figures S.7–S.8 (`fig:bubble_forecast_component1/2`) | `figure_ovx_forecast_component.py` | `results_ovx_forecast_component{1,2}/` | — | seconds |

Section S.1 (proofs) has no numerical content.

## 4. Running the Monte Carlo simulations

All simulations are seeded (`--seed 42` by default). Seeds are derived from the
replication, sample size and design, so any subset can be regenerated on its own
and the results do not depend on the number of workers.

**Tables S.1 and S.2** (1,000 replications, T = 250, 500, 1,000, three
distributional settings, 10 starting values):

```bash
python table_s1_mc_accuracy.py --n-jobs 16      # or: bash hpc/run_table_s1.sh 16
python table_s2_mc_accuracy.py --n-jobs 16
python table_s1_mc_accuracy.py --quick          # smoke test, seconds
```

Each writes the raw estimates (`mc_raw_<panel>.csv`), the summary
(`mc_summary.csv`) and the LaTeX table.

**Tables S.3–S.9 and Figures S.1–S.5** (subsampling, M = 200 replications per
sample size, blocks of size b = ⌊n^{2/3}⌋):

```bash
bash hpc/run_s22.sh "500 1000 10000 100000 500000" 200 30   # sizes, M, cores
bash hpc/collect_s22.sh                                     # rebuild s22_reparam.tex
python export_s22_reparam.py --quick --output-dir /tmp/s22  # smoke test, ~30 s
```

Sample sizes run one after the other; each one uses every core through `joblib`
and writes `summary_n<n>.csv`, `meta_n<n>.json` and `diagnostics_n<n>.png`.
With `--skip-existing`, a size already done is skipped, so an interrupted run can be
restarted with the same command. `--tex-only` rebuilds `s22_reparam.tex` (Tables
S.3–S.9, Figure environments S.1–S.5) from the stored CSV files. The two largest
sizes were run on the INSEE computing cluster.

**Tables S.11–S.13 and the subsampling check** (test of $H_0:\min_j\psi_j=0$,
2,000 replications × 40 designs, n from 500 to 100,000):

```bash
cd boundary_test
python boundary_test_mc.py --reps 2000 --workers 12    # Table S.11 + replications.csv
python supplement_tables.py                            # Tables S.12-S.13
python subsampling_check.py --reps 500 --workers 12    # Section S.4.2
python boundary_test_mc.py --table-only                # rebuild Table S.11 from summary.csv
python boundary_test_mc.py --reps 20 --sizes 500       # smoke test, seconds
```

`boundary_test_mc.py` contains its own estimator (same criterion, 15 × 15
quadrature grid on [−1.5, 1.5]², κ = 1, and L-BFGS-B tolerances as the main
code), the HAC sandwich variance, and the restricted parametric bootstrap
(warp-speed method of Giacomini, Politis and White, 2013). It does not import the
rest of the package. `--append` adds sample sizes to an existing `replications.csv`.

## 5. OVX application

Data: `data/OVX_History.csv`, the CBOE Crude Oil ETF Volatility Index (FRED, series
OVXCLS), daily, through 23/05/2025. `ovx_common.py` takes the window
23/05/2015–23/05/2025, averages by week, and removes a linear trend: **T = 522**
weekly observations (2015-05-31 to 2025-05-25). The estimation, the filtering and the
forecasts all use this one series.

**Estimation** (`ovx_estimation/`). `run_ovx_estimation.py` estimates the three
specifications of Table S.14 (GaS, SaS, S1S) with the original project estimator,
copied unchanged from the author's `code Python` folder (`stablemixture.py`,
`stable_aggregates.py`), called as in the interactive application
`stablemixture_app.py` (`step1=False`): HAC "optimal" weights (Bartlett kernel) are
computed **once, at the initial guess**, on a 15 × 15 grid, then the weighted criterion
is minimised by L-BFGS-B; Hessian standard errors. Data preparation follows
`stablemixture_app.py` (weekly mean, linear detrend, centred, rescaled to max |Y| = 10).
GaS starts from the initial guess archived in `Latex/figures/OVXResults.xlsx`
("init from reg"); SaS and S1S start from the sequential (NLS) initial guess of the
current `stable_aggregates.py`. Components are ordered ψ₁ < ψ₂.

```bash
cd ovx_estimation && python run_ovx_estimation.py        # ~1 min
```

It writes `ovx_estimation/estimation.csv` (estimate, standard error, t-statistic), which
`ovx_common.py` reads: it feeds Table S.14 (`table_ovx_estimation.py`), the filtering of
Figure S.6, the forecasts of Table S.15 and Figures 2, S.7–S.8. GaS estimates:

| ψ₁ | ψ₂ | α | β | σ | π₁ |
|---|---|---|---|---|---|
| 0.771 | 0.850 | 1.474 | −0.131 | 2.102 | 0.262 |

Notes on this estimation:

* **The weights are computed at the starting point, so the result depends on it.** The
  estimate is a single pass, not a fixed point of the weighted criterion: re-estimating
  from the solution with weights recomputed there drifts to the bounds of the parameter
  space (ψ = 0.01, 0.99). The GaS starting point is the one archived with the earlier
  version of the paper, which is why the estimates are close to it (ψ = 0.799, 0.847,
  α = 1.469, β = −0.128, π₁ = 0.279); the remaining differences come from the sample and
  from the numerical details of the optimizer (e.g. the version of scipy).
* The criterion is flat, in particular in π₁ (standard error 0.39): moving π₁ from 0.2 to
  0.7 changes the criterion by about 1%. A two-step variant (`step1=True`: uniform
  weights, then HAC weights) gives π₁ = 0.48 with the same ψ; a small multistart that
  keeps the lowest criterion ends on the bounds or on degenerate solutions
  (ψ₁ ≈ ψ₂). The estimator is not the Gaussian weight of Assumption C2.
* SaS: ψ₁ = 0.012 is close to its lower bound, so its standard error (9.2) is not
  informative. In S1S the standard errors of π are not available (shown as "--").

**Filtering** (Figure S.6): MCMC filter of de Truchis, Fries and Thomas, 1,000 particles,
window 5, seed 11, GaS estimates.

**Forecast** (Table S.15, Figures 2, S.7–S.8): in-sample pattern-matching forecast
at the 2020-01-05 cut-off (m = 20, horizon 20, positive bubble), GaS estimates. The
trajectory is anchored on |X_t0| measured on the series (11.2711), and k₀ is the value
returned by `kfinder`. `ovx_common.foreprob` / `kfinder` reproduce `StableMAR.foreprob` /
`kfinder` of the application (`code Python/stablemar.py`) exactly: crash and survival
probabilities and forecast trajectories agree to the last digit (checked with the
estimates above), and so do k₀ (10 and 5). Results at the 99% threshold: component 1 (ψ₁ = 0.771, k₀ = 10) crosses the threshold at
h* = 13, component 2 (ψ₂ = 0.850, k₀ = 5) at h* = 20 (beyond the 19 weeks of Table S.15); at
90% and 95%, h* = 7, 8 and 10, 13. The combined path of Figure 2 peaks at about 254
(observed peak ≈ 195). Since k₀ is not identified by the matching criterion (see the
docstring of `ovx_common.py`), it only changes which early probabilities are reported.

## 6. Notes on reproducibility

* **Figure 1** is a single simulated path, for illustration. The script used for
  the published PDF was not archived; `figure1_stable_aggregate.py` has the same
  design (ψ = (0.9, 0.7), π = (0.4, 0.6), T = 500, two purely anticipative
  α-stable AR(1) components) and layout, but the path itself depends on the draw
  (`--seed`, `--alpha`, `--beta`).
* **Tables S.1–S.13, S.15** and **Figures 2, S.1–S.8** are reproduced by the stored
  results and the scripts above. Rebuilding the LaTeX gives the published numbers;
  rerunning `table_s1_mc_accuracy.py` from scratch gives Table S.1 to the last digit.
* **Table S.14** and the OVX illustration are produced from `ovx_estimation/estimation.csv`
  (Section 5). The estimates are not those of the earlier version of the paper.
* Small differences in the last digit can arise with other versions of
  numpy/scipy (stable random variates, optimizer), mainly in the Monte Carlo
  tables.

## 7. Contents

```
replication_files/
├── README.md, requirements.txt, run_all.sh
├── data/OVX_History.csv      daily OVX through 23/05/2025 (T = 522 weekly)
├── cf_weights.py             quadrature of the criterion (weight of Assumption C2), optimizer tolerances, seeds
├── stablemixture.py          simulation of stable aggregates, estimator, MCMC filtering
├── subsampling_reparam.py    subsampling engine (Section S.3)
├── latex_table_export.py     LaTeX formatting for Section S.3
├── ovx_common.py             OVX data, estimates (read from ovx_estimation/estimation.csv), pattern matching (kfinder, foreprob, crash_horizon)
├── figure1_stable_aggregate.py
├── table_s1_mc_accuracy.py, table_s2_mc_accuracy.py, export_s22_reparam.py
├── table_ovx_estimation.py, figure_ovx_deconvolution.py, table_ovx_forecast.py,
│   figure_ovx_forecast_component.py, figure_ovx_combined_forecast.py
├── ovx_estimation/           run_ovx_estimation.py, original stablemixture.py / stable_aggregates.py, estimation.csv
├── boundary_test/            test of H0: min psi_j = 0 (Section S.4)
│   ├── boundary_test_mc.py, supplement_tables.py, subsampling_check.py
│   └── results/
├── hpc/                      preflight_check.py, run_table_s1.sh, run_s22.sh, collect_s22.sh
└── results_*/, resultats_reparam/   stored outputs
```
