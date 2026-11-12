#!/usr/bin/env python3
r"""
Figure \ref{fig:combined_forecasts} of main_v23.tex -- "Combined in-sample
forecast at the 99% risk threshold for the 2020 OVX bubble"
(figure `figures/InSCombinedForecast.png`).

Single panel, full sample: the detrended series in blue, the matched
historical theoretical pattern in green from the dashed black vertical
(t_0 - m), and the combined forecast in red beyond the dashed red vertical
(January 2020 cut-off).

Combining the two components
--------------------------------------------------------------------------
The combined trajectory is the upper envelope of the two component
forecasts,

    X_hat_t^{comb}(h) = max_j X_hat_t^{(j)}(h),

which amounts to following whichever component has not yet crashed. This is
the default (`--combine max`); `sum`, `component1`, and `component2` are
also available. The historical pattern is combined the same way; since both
patterns equal |X_{t0}| at the cut-off and decay as psi_j^(m-i) moving left,
the envelope there is governed by the more persistent component.

Outputs (results_ovx_combined_forecast/)
--------------------------------------------------------------------------
    figure_ovx_combined_forecast.pdf / .png
    combined_forecast.csv   combined pattern and forecast, alongside the two
                            per-component trajectories
    run_config.json

Anchor (|X_t0|) and k0 conventions follow table_ovx_forecast.py.

Usage
--------------------------------------------------------------------------
    python figure_ovx_combined_forecast.py
    python figure_ovx_combined_forecast.py --combine sum
    python figure_ovx_combined_forecast.py --threshold 0.95
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ovx_common as oc

OUTDIR_NAME = "results_ovx_combined_forecast"

# Date range shown in the figure.
XLIM = ("2015-01-01", "2022-01-01")


def combine(trajs: Dict[int, np.ndarray], how: str) -> np.ndarray:
    """Combine the two per-component trajectories into a single curve."""
    stack = np.vstack([trajs[1], trajs[2]])
    if how == "max":
        return np.max(stack, axis=0)
    if how == "sum":
        return np.sum(stack, axis=0)
    if how == "component1":
        return trajs[1].copy()
    if how == "component2":
        return trajs[2].copy()
    raise ValueError(how)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=oc.DEFAULT_DATA)
    ap.add_argument("--cutoff", default=oc.FORECAST_CUTOFF)
    ap.add_argument("--m", type=int, default=oc.FORECAST_M)
    ap.add_argument("--horizon", type=int, default=oc.FORECAST_H)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--vartheta", type=int, default=oc.FORECAST_VARTHETA,
                    choices=(-1, 1))
    ap.add_argument("--combine", choices=("max", "sum", "component1", "component2"),
                    default="max", help="how to combine the two components")
    ap.add_argument("--xlim", nargs=2, default=list(XLIM),
                    help="date-axis bounds (empty for the full sample)")
    args = ap.parse_args()

    outdir = oc.ensure_outdir(OUTDIR_NAME)
    df = oc.load_ovx(args.data)
    x = df["detrended"].to_numpy()
    params = oc.estimated_params(oc.MAIN_SPEC)

    cutoff = pd.Timestamp(args.cutoff)
    t = int(df.index.get_loc(cutoff)) if cutoff in df.index \
        else int(df.index.get_indexer([cutoff], method="nearest")[0])
    anchor = None      # |X_t0| measured on the series, as in the application

    print("=" * 72)
    print("Figure \\ref{fig:combined_forecasts} (main_v23.tex) -- "
          "OVX combined forecast")
    print("=" * 72)
    print(f"  sample: {oc.sample_note(df)}")
    print(f"  cut-off t_0 = {df.index[t]:%Y-%m-%d}, threshold {args.threshold:.0%}, "
          f"combination '{args.combine}'")

    trajs, k0s = {}, {}
    for j in (1, 2):
        psi = params[f"rho_{j}"]
        k0, _, _ = oc.kfinder(x, t, psi, m=args.m, kmax=oc.FORECAST_KMAX,
                              vartheta=args.vartheta)
        k0s[j] = k0
        _, traj = oc.foreprob(x, k0=k0, h=args.horizon, maxp=args.threshold,
                              t=t, psi=psi, alpha=params["alpha"],
                              vartheta=args.vartheta, m=args.m, anchor=anchor)
        trajs[j] = traj
        h_star = oc.crash_horizon(psi, params["alpha"], args.threshold)
        print(f"    component {j}: psi={psi:.4f}, k0={k0}, h*={h_star}, "
              f"peak={np.max(traj):.2f}")

    comb = combine(trajs, args.combine)
    dates = oc.forecast_dates(df.index, t, args.m, args.horizon)
    m = args.m
    print(f"  combined trajectory: peak={np.max(comb):.2f} on "
          f"{dates[int(np.argmax(comb))]:%Y-%m-%d}")

    # --- CSV ---
    out = pd.DataFrame({
        "date": dates,
        "h": np.arange(-m, args.horizon + 1),
        "segment": ["pattern"] * (m + 1) + ["forecast"] * args.horizon,
        "combined": comb,
        "component_1": trajs[1],
        "component_2": trajs[2],
        "observed": [x[t - m + k] if (t - m + k) < len(x) else np.nan
                     for k in range(len(dates))],
    })
    out.to_csv(os.path.join(outdir, "combined_forecast.csv"), index=False)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.plot(df.index, x, color="#6a7fbf", linewidth=1.0,
            label="Detrended series")
    ax.plot(dates[:m + 1], comb[:m + 1], color="darkgreen", linewidth=3,
            label="Historical theoretical pattern")
    ax.plot(dates[m:], comb[m:], color="crimson", linewidth=3,
            label=f"Forecast (threshold={args.threshold})")
    ax.axvline(dates[0], color="k", linestyle=":", linewidth=2,
               label="Pattern start")
    ax.axvline(dates[m], color="r", linestyle=":", linewidth=2,
               label="End of in-sample")

    if args.xlim:
        ax.set_xlim(pd.Timestamp(args.xlim[0]), pd.Timestamp(args.xlim[1]))
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Value", fontsize=14)
    ax.legend(fontsize=12, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    paths = oc.save_figure(fig, outdir, "figure_ovx_combined_forecast")
    plt.close(fig)

    oc.write_run_config(outdir, {
        "target": "main_v23.tex :: fig:combined_forecasts "
                  "(figures/InSCombinedForecast.png)",
        "data_file": os.path.relpath(args.data, oc.HERE),
        "T": int(len(df)),
        "sample_note": oc.sample_note(df),
        "cutoff": f"{df.index[t]:%Y-%m-%d}", "cutoff_index": t,
        "anchor_mode": "measured",
        "anchor_value": float(abs(x[t])),
        "m": args.m, "horizon": args.horizon, "threshold": args.threshold,
        "vartheta": args.vartheta,
        "k0_mode": "kfinder", "k0_used": k0s,
        "combine": args.combine,
        "xlim": list(args.xlim) if args.xlim else None,
        "peak_combined": float(np.max(comb)),
    })

    print(f"\nWritten to {os.path.basename(outdir)}/")
    for p in paths:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
