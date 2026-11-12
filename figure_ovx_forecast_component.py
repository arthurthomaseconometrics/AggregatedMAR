#!/usr/bin/env python3
r"""
Figures \ref{fig:bubble_forecast_component1} and
\ref{fig:bubble_forecast_component2} -- per-component graphical diagnostics,
Section \ref{app:ovx_figures} of online_supplement3.tex (figures
`figures/InSForeprob2020Obs1AllRisk.pdf` and `...Obs2AllRisk.pdf`).

Layout reproduced from `stablemixture_app.create_forecast_plots`, a 3 x 3
grid:

  * top row:    crash-probability profiles for thresholds 0.90, 0.95, 0.99.
    Probability of crashing at the date (red, round markers), of surviving
    beyond it (green), cumulative crash probability (blue), threshold
    (dashed red).
  * middle row: zoomed forecast trajectories, one panel per threshold.
    Colored squares are predicted values; dashed red vertical is the cut-off.
  * bottom row: full 2015-2025 view, historical theoretical pattern in
    green, forecasts at the three thresholds, shaded forecast zone.

Outputs (results_ovx_forecast_component1/ and ...component2/)
--------------------------------------------------------------------------
    figure_ovx_forecast_component<j>.pdf / .png
    probabilities.csv     probabilities by horizon, one block per threshold
    trajectories.csv      past pattern + predicted trajectory, per threshold
    run_config.json

Anchor (|X_t0|) and k0 conventions follow table_ovx_forecast.py.

Usage
--------------------------------------------------------------------------
    python figure_ovx_forecast_component.py              # both components
    python figure_ovx_forecast_component.py --component 1
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ovx_common as oc

# Color scheme matching `create_forecast_plots`: the `colors` list there is
# initialized to ['r', 'purple', 'm'] and then reassigned to ['b', 'g', 'r']
# inside the probability-panel loop; the zoom row, drawn afterwards, inherits
# the reassigned palette.
ZOOM_COLORS = ["b", "g", "r"]
ZOOM_LINESTYLES = ["-", "-.", ":"]
FULLVIEW_COLORS = ["m", "c", "r"]


def compute(x: np.ndarray, t: int, psi: float, alpha: float, k0: int,
            m: int, h: int, vartheta: int, anchor) -> Dict[float, Dict]:
    """One pass of Algorithm \\ref{alg:pattern_matching} per threshold."""
    out = {}
    for thr in oc.THRESHOLDS:
        df_prob, traj = oc.foreprob(x, k0=k0, h=h, maxp=thr, t=t, psi=psi,
                                    alpha=alpha, vartheta=vartheta, m=m,
                                    anchor=anchor)
        out[thr] = {"df_prob": df_prob, "traj": traj,
                    "h_star": oc.crash_horizon(psi, alpha, thr)}
    return out


def make_figure(df: pd.DataFrame, x: np.ndarray, t: int, j: int, psi: float,
                k0: int, m: int, h: int, res: Dict[float, Dict],
                dates: pd.DatetimeIndex) -> plt.Figure:
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 3)

    thresholds = list(oc.THRESHOLDS)

    # ---- Row 1: probability profiles ---------------------------------------
    for i, thr in enumerate(thresholds):
        ax = fig.add_subplot(gs[0, i])
        p = res[thr]["df_prob"]
        ax.plot(p.index, p["Crash Before h"], "b-", linewidth=2,
                label="Crash at h")
        ax.plot(p.index, p["Survive at h"], "g-", linewidth=2,
                label="Survive beyond h")
        ax.plot(p.index, p["Crash at h"], "r-", marker="o", linewidth=2,
                label="Crash by h")
        ax.axhline(y=thr, color="r", alpha=0.5, linestyle="--",
                   label=f"Threshold={thr}")
        ax.set_xlabel("Horizon h", fontsize=12)
        if i == 0:
            ax.set_ylabel("Probability", fontsize=12)
        ax.set_title(f"Threshold = {thr}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    # ---- Row 2: zoom on the bubble ------------------------------------------
    focus_start = max(t - 3, 0)
    focus_end = min(t + h + 3, len(x))
    # Reference pattern for the first threshold (0.90), drawn as green
    # markers on all three panels (hence the green circles at zero after
    # its crash).
    ref_traj = res[thresholds[0]]["traj"]
    for i, thr in enumerate(thresholds):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(df.index[focus_start:focus_end], x[focus_start:focus_end],
                "b-", linewidth=3, marker="o", markersize=8, label="Real series")

        for idx in range(focus_start, min(focus_end, t + h + 1)):
            k = idx - (t - m)
            if 0 <= k < len(ref_traj):
                ax.plot(dates[k], ref_traj[k], "g", marker="o", markersize=8,
                        alpha=0.8)

        traj = res[thr]["traj"]
        seg_dates, seg_vals = [], []
        for idx in range(focus_start, min(focus_end, t + h + 1)):
            k = idx - (t - m)
            if 0 <= k < len(traj):
                seg_dates.append(dates[k])
                seg_vals.append(traj[k])
        if seg_dates:
            ax.plot(seg_dates, seg_vals, ZOOM_COLORS[i],
                    linestyle=ZOOM_LINESTYLES[i], linewidth=2,
                    marker="s", markersize=8,
                    label=f"Forecast (threshold={thr})")

        ax.axvline(df.index[t], color="r", linestyle="--", linewidth=2,
                   label="End of in-sample")
        ax.set_title(f"Zoom - Threshold {thr}", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
        if i == 0:
            ax.set_ylabel("Value", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # ---- Row 3: full sample --------------------------------------------------
    ax = fig.add_subplot(gs[2, :])
    ax.plot(df.index, x, color="b", linestyle="-", linewidth=1, alpha=0.7,
            label="Detrended series")

    ref = res[thresholds[0]]["traj"]
    ax.plot(dates[:m + 1], ref[:m + 1], color="g", linestyle="-", linewidth=3,
            label=f"Historical theoretical pattern (k0={k0})")

    for i, thr in enumerate(thresholds):
        traj = res[thr]["traj"]
        ax.plot(dates[m:], traj[m:], color=FULLVIEW_COLORS[i], linestyle=":",
                linewidth=3, alpha=0.9, label=f"Forecast (threshold={thr})")

    ax.axvspan(dates[m], dates[-1], alpha=0.2, color="gray",
               label="Forecast zone")
    ax.axvline(dates[0], color="k", linestyle=":", linewidth=2,
               label="Pattern start")
    ax.axvline(dates[m], color="r", linestyle="-", linewidth=2,
               label="End of in-sample")

    ax.set_title("Detrended series with theoretical pattern and scenarios",
                 fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def run_one(df: pd.DataFrame, x: np.ndarray, t: int, j: int, params: Dict,
            k0: int, args, anchor) -> str:
    psi = params[f"rho_{j}"]
    alpha = params["alpha"]
    outdir = oc.ensure_outdir(f"results_ovx_forecast_component{j}")

    res = compute(x, t, psi, alpha, k0, args.m, args.horizon, args.vartheta,
                  anchor)
    dates = oc.forecast_dates(df.index, t, args.m, args.horizon)

    print(f"\n  Component {j}: psi={psi:.4f}, k0={k0}")
    for thr in oc.THRESHOLDS:
        print(f"    threshold {thr:.2f} -> h* = {res[thr]['h_star']}")

    # --- probabilities.csv ---
    prob_rows = []
    for thr in oc.THRESHOLDS:
        p = res[thr]["df_prob"].reset_index()
        p.insert(0, "threshold", thr)
        p.insert(0, "component", j)
        prob_rows.append(p)
    pd.concat(prob_rows, ignore_index=True).to_csv(
        os.path.join(outdir, "probabilities.csv"), index=False)

    # --- trajectories.csv ---
    traj_rows = []
    for thr in oc.THRESHOLDS:
        traj = res[thr]["traj"]
        for k, val in enumerate(traj):
            traj_rows.append({
                "component": j, "threshold": thr,
                "date": dates[k], "h": k - args.m,
                "segment": "pattern" if k <= args.m else "forecast",
                "value": val,
                "observed": x[t - args.m + k] if (t - args.m + k) < len(x) else np.nan,
            })
    pd.DataFrame(traj_rows).to_csv(
        os.path.join(outdir, "trajectories.csv"), index=False)

    fig = make_figure(df, x, t, j, psi, k0, args.m, args.horizon, res, dates)
    paths = oc.save_figure(fig, outdir, f"figure_ovx_forecast_component{j}")
    plt.close(fig)

    oc.write_run_config(outdir, {
        "target": f"online_supplement3.tex :: fig:bubble_forecast_component{j} "
                  f"(figures/InSForeprob2020Obs{j}AllRisk.pdf)",
        "data_file": os.path.relpath(args.data, oc.HERE),
        "T": int(len(df)),
        "sample_note": oc.sample_note(df),
        "component": j, "psi": float(psi), "alpha": float(alpha),
        "k0": k0, "k0_mode": "kfinder",
        "cutoff": f"{df.index[t]:%Y-%m-%d}", "cutoff_index": t,
        "anchor_mode": "measured",
        "anchor_value": float(abs(x[t])),
        "m": args.m, "horizon": args.horizon, "vartheta": args.vartheta,
        "thresholds": list(oc.THRESHOLDS),
        "crash_horizons": {str(thr): res[thr]["h_star"] for thr in oc.THRESHOLDS},
    })
    return os.path.dirname(paths[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=oc.DEFAULT_DATA)
    ap.add_argument("--component", type=int, choices=(1, 2), default=None,
                    help="process a single component (default: both)")
    ap.add_argument("--cutoff", default=oc.FORECAST_CUTOFF)
    ap.add_argument("--m", type=int, default=oc.FORECAST_M)
    ap.add_argument("--horizon", type=int, default=oc.FORECAST_H)
    ap.add_argument("--vartheta", type=int, default=oc.FORECAST_VARTHETA,
                    choices=(-1, 1))
    args = ap.parse_args()

    df = oc.load_ovx(args.data)
    x = df["detrended"].to_numpy()
    params = oc.estimated_params(oc.MAIN_SPEC)

    cutoff = pd.Timestamp(args.cutoff)
    t = int(df.index.get_loc(cutoff)) if cutoff in df.index \
        else int(df.index.get_indexer([cutoff], method="nearest")[0])
    anchor = None      # |X_t0| measured on the series, as in the application

    print("=" * 72)
    print("Figures \\ref{fig:bubble_forecast_component1/2} -- OVX diagnostics")
    print("=" * 72)
    print(f"  sample: {oc.sample_note(df)}")
    print(f"  cut-off t_0 = {df.index[t]:%Y-%m-%d} (index {t}), "
          f"anchor: |X_t0| = {abs(x[t]):.4f}")

    components = [args.component] if args.component else [1, 2]
    for j in components:
        k0, _, _ = oc.kfinder(x, t, params[f"rho_{j}"], m=args.m,
                              kmax=oc.FORECAST_KMAX, vartheta=args.vartheta)
        out = run_one(df, x, t, j, params, k0, args, anchor)
        print(f"    -> {os.path.basename(out)}/")


if __name__ == "__main__":
    main()
