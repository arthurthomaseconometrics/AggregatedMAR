# OVX Replication — "Application to the OVX Index"

Replication package for `\section{Application to the OVX Index}`
(`\label{app:ovx}`) of `Latex/online_supplement3.tex`, and for Figure
`\ref{fig:combined_forecasts}` of `Latex/main_v23.tex`. One output folder per
figure or table, following the convention of `results_table_s1/`.

## Data

**`data/OVX_History.csv`** — the CBOE Crude Oil ETF Volatility Index (OVX),
weekly, from FRED. After the weekly resampling and detrending step described
below, the sample used throughout is **2015-05-31 to 2025-04-27 (T = 518)**.

## Scripts and outputs

| Target | Script | Output folder |
|---|---|---|
| Table `tab:ovx_estimation` (§ S.3.1) | `table_ovx_estimation.py` | `results_ovx_estimation/` |
| Figure `fig:ovx_deconv` — `figures/OVXDeconv48.pdf` (§ S.3.2) | `figure_ovx_deconvolution.py` | `results_ovx_deconvolution/` |
| Table `tab:bubble_forecast` (§ S.3.3) | `table_ovx_forecast.py` | `results_ovx_forecast/` |
| Figure `fig:bubble_forecast_component1` — `InSForeprob2020Obs1AllRisk.pdf` | `figure_ovx_forecast_component.py --component 1` | `results_ovx_forecast_component1/` |
| Figure `fig:bubble_forecast_component2` — `InSForeprob2020Obs2AllRisk.pdf` | `figure_ovx_forecast_component.py --component 2` | `results_ovx_forecast_component2/` |
| Figure `fig:combined_forecasts` in `main_v23.tex` — `InSCombinedForecast.png` | `figure_ovx_combined_forecast.py` | `results_ovx_combined_forecast/` |

**`ovx_common.py`** is the shared module: data loading and detrending, the
point estimates and settings used for filtering and forecasting (Table
`tab:ovx_estimation`, `Latex/figures/OVXResults.xlsx`), and the
pattern-matching machinery (`kfinder`, `foreprob`, `crash_horizon`) that
implements Algorithm `alg:pattern_matching` for the purely anticipative
MAR(0,1) case.

All five scripts use the estimates of Table `tab:ovx_estimation` — none of
them re-estimates the model. They implement the filtering and forecasting
pipeline built on those estimates and produce the corresponding tables and
figures.

### Reproducing the outputs

```bash
python table_ovx_estimation.py
python figure_ovx_deconvolution.py        # ~90s (MCMC filter, 1000 particles)
python table_ovx_forecast.py
python figure_ovx_forecast_component.py   # both components
python figure_ovx_combined_forecast.py
```

Each script writes a `run_config.json` (settings, package versions,
timestamp) alongside its outputs.

## Method notes

**Forecast anchor.** The pattern-matching trajectory (Algorithm
`alg:pattern_matching`) is proportional to |X_{t0}|, the level of the
detrended series at the January 2020 cut-off. `--anchor estimated` (default)
uses the cut-off level consistent with Table `tab:ovx_estimation`;
`--anchor measured` uses the level computed directly from the loaded file.

**Identification of k0.** `kfinder` matches the observed path to the
normalized pattern psi^(k-i); since psi^k cancels under normalization, the
matching distance is independent of k, and the resulting k0 is decided only
by floating-point rounding in the argmin. k0 = 3 (component 1) and k0 = 1
(component 2) are used throughout (`--k0 published`, the default; `--k0
auto` recomputes the argmin). This has no effect on the forecast trajectory
or the crash horizon h*, since the historical pattern is renormalized on
|X_{t0}| — it only affects which of the first k0 - 1 crash probabilities are
masked ("--") in Table `tab:bubble_forecast`.

**Combining the two components.** Figure `fig:combined_forecasts` combines
the per-component forecasts as their upper envelope,
X_hat^{comb}(h) = max_j X_hat^{(j)}(h) — following whichever component has
not yet crashed (`--combine max`, the default; `sum`, `component1`, and
`component2` are also available).

**Color convention in the per-component figures.** The middle-row (zoom)
panels use blue/green/red for the three thresholds, matching the palette
assigned in the source plotting routine; the bottom-row panel uses
magenta/cyan/red.
