# Replication package

de Truchis, G., Fries, S., and Thomas, A. (2026). *Estimation and prediction of bubbles
with α-stable aggregates of moving averages*, and its Online Supplement.

This package reproduces every table and figure of the paper and of the Online
Supplement. It is written in Python. The results of the long simulations are included,
so all tables can be rebuilt from them in a few minutes; the scripts that produce them
are included with the settings used.

## 1. Requirements

Python ≥ 3.10 with `numpy scipy pandas matplotlib joblib tqdm` (`requirements.txt`).
No compilation, no GPU, no data download.

```bash
pip install -r requirements.txt
python hpc/preflight_check.py      # checks the environment in a few seconds
```

## 2. Quick start

```bash
bash run_all.sh          # ~5 min: Figure 1, OVX illustration, Tables S.3–S.13 from stored results
bash run_all.sh full     # also reruns all Monte Carlo simulations (see run times below)
```

`PYTHON=/path/to/python` and `NCORES=<n>` can be set before the command. Each script
also runs on its own, writes its outputs to its own folder and records its settings,
package versions and timestamp in a `run_config.json`.

## 3. Tables and figures

Run times measured on a 15-core laptop (macOS, arm64) unless stated otherwise.

### Paper

| Output | Script | Output folder | Run time |
|---|---|---|---|
| Figure 1 | `figure1_stable_aggregate.py` | `results_figure1/` | seconds |
| Figure 2 | `figure_ovx_combined_forecast.py` | `results_ovx_combined_forecast/` | seconds |

### Online Supplement

| Output | Script | Output folder | Run time |
|---|---|---|---|
| Table S.1 | `table_s1_mc_accuracy.py` | `results_table_s1/` | 3 min |
| Table S.2 | `table_s2_mc_accuracy.py` | `results_table_s2/` | 3 min |
| Tables S.3–S.9, Figures S.1–S.5 | `export_s22_reparam.py` | `results_reparam/` | per sample size: 14 min (n = 1,000), 27 min (n = 10,000); 33 min (n = 100,000) and 41 min (n = 500,000) on a 30-core server |
| Table S.10 | design of the simulation (constants of `boundary_test/boundary_test_mc.py`) | — | — |
| Table S.11 | `boundary_test/boundary_test_mc.py` | `boundary_test/results/` | 75 min on 12 cores |
| Tables S.12–S.13 | `boundary_test/supplement_tables.py` | `boundary_test/results/` | seconds |
| Section S.4.4, subsampling version of the test | `boundary_test/subsampling_check.py` | `boundary_test/results/` | 8 min on 12 cores |
| Table S.14 | `ovx_estimation/run_ovx_estimation.py`, then `table_ovx_estimation.py` | `ovx_estimation/`, `results_ovx_estimation/` | 1 min |
| Figure S.6 | `figure_ovx_deconvolution.py` | `results_ovx_deconvolution/` | 1.5 min |
| Table S.15 | `table_ovx_forecast.py` | `results_ovx_forecast/` | seconds |
| Figures S.7–S.8 | `figure_ovx_forecast_component.py` | `results_ovx_forecast_component{1,2}/` | seconds |

Section S.1 of the Online Supplement (proofs) has no numerical content.

## 4. How each part works

### Monte Carlo accuracy (Tables S.1–S.2)

Aggregates of two anticipative α-stable AR(1) components (Table S.1) or two mixed
causal–noncausal MAR(1,1) components (Table S.2); T = 250, 500, 1,000; 1,000
replications; three distributional settings (Cauchy, SαS, GαS). Estimator: minimum
distance on the joint empirical characteristic function with the weight of
Assumption C2 (κ = 1, 15 × 15 grid on [−1.5, 1.5]², `cf_weights.py`), 10 starting
values. Seeds are derived from the replication, sample size and design, so results
do not depend on the number of workers.

```bash
python table_s1_mc_accuracy.py --n-jobs 15
python table_s2_mc_accuracy.py --n-jobs 15
python table_s1_mc_accuracy.py --quick          # smoke test
```

Outputs: LaTeX table, `mc_summary.csv`, `mc_raw_<panel>.csv`.

### Subsampling (Tables S.3–S.9, Figures S.1–S.5)

M = 200 replications per sample size n ∈ {500, 1,000, 10,000, 100,000, 500,000}; the
model is estimated on the full sample and on subsamples of size b = ⌊n^{2/3}⌋ with the
same estimator; diagnostics are reported in the reparameterized space
(ψ₁, ψ₂, σ, π₁, α). Engine: `subsampling_reparam.py`.

```bash
bash hpc/run_s22.sh "500 1000 10000 100000 500000" 200 30   # sizes, M, cores
bash hpc/collect_s22.sh                                     # rebuild s22_reparam.tex
```

Sample sizes run one after the other and each writes `summary_n<n>.csv`,
`meta_n<n>.json` and `diagnostics_n<n>.png`. `--skip-existing` restarts an interrupted
run; `--tex-only` rebuilds `s22_reparam.tex` (Tables S.3–S.9 and the figure
environments of Figures S.1–S.5) from the stored CSV files.

### Test of H₀: min ψⱼ = 0 (Tables S.10–S.13, Section S.4)

2,000 replications × 40 designs, n from 500 to 100,000; asymptotic, warp-speed
bootstrap and two-sided versions of the test. `boundary_test_mc.py` is self-contained
(same criterion and quadrature as the rest of the package).

```bash
cd boundary_test
python boundary_test_mc.py --reps 2000 --workers 12    # Table S.11, replications.csv
python supplement_tables.py                            # Tables S.12–S.13
python subsampling_check.py --reps 500 --workers 12    # Section S.4.4
python boundary_test_mc.py --table-only                # rebuild Table S.11 from summary.csv
```

### OVX illustration (Section 5, Figure 2; Section S.5, Tables S.14–S.15, Figures S.6–S.8)

Data: `data/OVX_History.csv`, CBOE Crude Oil ETF Volatility Index (FRED, OVXCLS),
daily. Window 23/05/2015–23/05/2025, weekly mean, linear detrend: T = 522
(`ovx_common.py`).

1. **Estimation** — `ovx_estimation/run_ovx_estimation.py` estimates the GαS, SαS and
   S1S specifications with `ovx_estimation/stablemixture.py`: minimum distance
   estimation on a 15 × 15 grid, computed by L-BFGS-B from an initial guess; Hessian
   standard errors. The series is centred and
   rescaled to max |Y| = 10 for the estimation. It writes
   `ovx_estimation/estimation.csv`, which all the other OVX scripts read.
2. **Table S.14** — `table_ovx_estimation.py` formats `estimation.csv`.
3. **Filtering (Figure S.6)** — dual MCMC filter of de Truchis, Thomas and Vaudrée
   (2026), 1,000 particles, window 5, seed 11, GαS estimates.
4. **Forecast (Table S.15, Figures 2, S.7–S.8)** — pattern-matching forecast at the
   2020-01-05 cut-off (m = 20, horizon 20, positive bubble), GαS estimates; risk
   thresholds 90%, 95%, 99%. Figure 2 combines the two components as their upper
   envelope (`--combine max`).

```bash
cd ovx_estimation && python run_ovx_estimation.py && cd ..
python table_ovx_estimation.py
python figure_ovx_deconvolution.py
python table_ovx_forecast.py
python figure_ovx_forecast_component.py
python figure_ovx_combined_forecast.py
```

### Figure 1

`figure1_stable_aggregate.py` simulates two anticipative α-stable AR(1) components
(ψ = (0.9, 0.7), π = (0.4, 0.6), α = 1.5, T = 500, seed 42) and plots the weights, the
aggregate and the two weighted components.

## 5. Stored results

| Folder | Content |
|---|---|
| `results_table_s1/`, `results_table_s2/` | full runs of Tables S.1–S.2 |
| `results_reparam/` | `summary_n*.csv`, `meta_n*.json`, `diagnostics_n*.png` for the five sample sizes (n = 100,000 and 500,000 computed on a 30-core server) |
| `boundary_test/results/` | `replications.csv`, `summary.csv`, `subsampling_check.csv` and the LaTeX tables |
| `ovx_estimation/`, `results_ovx_*/`, `results_figure1/` | OVX and Figure 1 outputs |

## 6. Contents

```
replication_files/
├── README.md, requirements.txt, run_all.sh
├── data/OVX_History.csv
├── cf_weights.py             quadrature of the criterion, estimation bounds, optimizer tolerances, seeds
├── stablemixture.py          simulation of stable aggregates and estimator (Monte Carlo)
├── subsampling_reparam.py    subsampling engine (Section S.3)
├── latex_table_export.py     LaTeX formatting for Section S.3
├── ovx_common.py             OVX data, estimates, pattern matching (kfinder, foreprob, crash_horizon)
├── figure1_stable_aggregate.py
├── table_s1_mc_accuracy.py, table_s2_mc_accuracy.py, export_s22_reparam.py
├── table_ovx_estimation.py, figure_ovx_deconvolution.py, table_ovx_forecast.py,
│   figure_ovx_forecast_component.py, figure_ovx_combined_forecast.py
├── ovx_estimation/           run_ovx_estimation.py, stablemixture.py, stable_aggregates.py, estimation.csv
├── boundary_test/            boundary_test_mc.py, supplement_tables.py, subsampling_check.py, results/
├── hpc/                      preflight_check.py, run_table_s1.sh, run_s22.sh, collect_s22.sh
└── results_*/                stored outputs
```
