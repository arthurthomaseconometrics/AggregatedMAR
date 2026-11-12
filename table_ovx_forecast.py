#!/usr/bin/env python3
r"""
Table \ref{tab:bubble_forecast} -- "In-sample bubble forecast: predicted
values and crash/survival probabilities", Section \ref{app:ovx_forecast} of
online_supplement3.tex.

Applies Algorithm \ref{alg:pattern_matching} to both components of the
G-alpha-S model at the January 2020 cut-off, and tabulates, for each weekly
horizon h, the predicted value, the crash probability 1 - |psi|^(alpha h),
and the survival probability |psi|^(alpha h).

Outputs (results_ovx_forecast/)
--------------------------------------------------------------------------
    table_ovx_forecast.tex   booktabs table, supplement format
    forecast_table.csv       same content, machine-readable
    crash_horizons.csv       h* by (component, threshold)
    pattern_matching.csv     matching distance by k, for every component

Usage
--------------------------------------------------------------------------
    python table_ovx_forecast.py
    python table_ovx_forecast.py --threshold 0.95
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

import ovx_common as oc

OUTDIR_NAME = "results_ovx_forecast"


def run_component(x: np.ndarray, t: int, j: int, params: Dict[str, float],
                  k0: int, h: int, threshold: float, m: int, vartheta: int,
                  anchor) -> Dict:
    """Algorithm \\ref{alg:pattern_matching} for component j."""
    psi = params[f"rho_{j}"]
    alpha = params["alpha"]
    df_prob, traj = oc.foreprob(x, k0=k0, h=h, maxp=threshold, t=t, psi=psi,
                                alpha=alpha, vartheta=vartheta, m=m,
                                anchor=anchor)
    return {"j": j, "psi": psi, "alpha": alpha, "k0": k0,
            "df_prob": df_prob, "traj": traj}


def assemble(res1: Dict, res2: Dict, dates: pd.DatetimeIndex, m: int,
             h_show: int) -> pd.DataFrame:
    """Table \\ref{tab:bubble_forecast}: one row per horizon h = 0..h_show."""
    rows = []
    for h in range(0, h_show + 1):
        row = {"date": dates[m + h], "h": h}
        for res, tag in ((res1, "c1"), (res2, "c2")):
            row[f"forecast_{tag}"] = res["traj"][m + h]
            if h == 0:
                row[f"crash_{tag}"] = np.nan
                row[f"survive_{tag}"] = np.nan
            else:
                p = res["df_prob"].loc[float(h)]
                row[f"crash_{tag}"] = p["Crash at h"]
                row[f"survive_{tag}"] = p["Survive at h"]
        rows.append(row)
    return pd.DataFrame(rows)


def to_tex(tab: pd.DataFrame, res1: Dict, res2: Dict, threshold: float,
           m: int, h_star: Dict[int, int]) -> str:
    """Table matching the layout of Table \\ref{tab:bubble_forecast}."""
    pct = f"{threshold:.0%}".replace("%", r"\%")
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{In-sample bubble forecast: predicted values and "
        r"crash/survival probabilities (cut-off: January 2020, risk "
        rf"threshold: {pct})}}".replace("}}", "}"),
        r"\label{tab:bubble_forecast}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrcccccc}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{Component 1 ($k_0=" + str(res1["k0"])
        + r"$, $\hat{\psi}_1=" + f"{res1['psi']:.4f}" + r"$)}",
        r"  & \multicolumn{3}{c}{Component 2 ($k_0=" + str(res2["k0"])
        + r"$, $\hat{\psi}_2=" + f"{res2['psi']:.4f}" + r"$)} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Date & $h$ & Forecast & Crash by $h$ & Survive beyond $h$",
        r"          & Forecast & Crash by $h$ & Survive beyond $h$ \\",
        r"\midrule",
    ]
    for _, r in tab.iterrows():
        h = int(r["h"])
        cells = [f"{r['date']:%Y-%m-%d}", f"{h:2d}"]
        for tag, res in (("c1", res1), ("c2", res2)):
            fc = r[f"forecast_{tag}"]
            hs = h_star.get(res["j"])
            crashed = (hs is not None) and (h >= hs)
            cells.append((f"{fc:.4f}" + r"$^*$") if crashed else f"{fc:.4f}")
            cells.append(oc.tex_num(r[f"crash_{tag}"]))
            cells.append(oc.tex_num(r[f"survive_{tag}"]))
        lines.append(" & ".join(cells) + r" \\")

    h_last = int(tab["h"].max())
    star = "; ".join(
        (f"Component {j}: $h={hs}$" + (", beyond the last horizon shown" if hs > h_last else ""))
        if hs is not None else f"Component {j}: no crash within the horizon"
        for j, hs in sorted(h_star.items()))
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\begin{flushleft}",
        r"\footnotesize\textit{Notes:} Crash by $h$ denotes "
        r"$1-|\psi_{j_0}|^{\alpha h}$; Survive beyond $h$ denotes "
        r"$|\psi_{j_0}|^{\alpha h}$. $^*$Crash predicted before this horizon "
        rf"at the {pct} threshold ({star}). Forecast values after the "
        rf"predicted crash are set to zero. Pattern matching window: $m={m}$.",
        r"\end{flushleft}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


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
    args = ap.parse_args()

    outdir = oc.ensure_outdir(OUTDIR_NAME)
    df = oc.load_ovx(args.data)
    x = df["detrended"].to_numpy()
    params = oc.estimated_params(oc.MAIN_SPEC)

    cutoff = pd.Timestamp(args.cutoff)
    if cutoff in df.index:
        t = int(df.index.get_loc(cutoff))
    else:
        t = int(df.index.get_indexer([cutoff], method="nearest")[0])
        print(f"  cut-off {args.cutoff} not in sample, using nearest date: "
              f"{df.index[t]:%Y-%m-%d}")

    print("=" * 72)
    print("Table \\ref{tab:bubble_forecast} -- OVX 2020 in-sample forecast")
    print("=" * 72)
    print(f"  sample: {oc.sample_note(df)}")
    print(f"  cut-off t_0 = {df.index[t]:%Y-%m-%d} (index {t})")

    anchor = None      # |X_t0| measured on the series, as in the application
    print(f"  anchor: |X_t0| = {abs(x[t]):.4f}")

    # --- Pattern matching ---
    pm_rows: List[Dict] = []
    k0_used: Dict[int, int] = {}
    for j in (1, 2):
        psi = params[f"rho_{j}"]
        k0_auto, quality, dists = oc.kfinder(x, t, psi, m=args.m,
                                             kmax=oc.FORECAST_KMAX,
                                             vartheta=args.vartheta)
        k0_used[j] = k0_auto
        print(f"  component {j}: psi={psi:.4f}, k0={k0_used[j]}, "
              f"matching quality {quality:.4f}")
        for k in range(1, oc.FORECAST_KMAX + 1):
            pm_rows.append({"component": j, "k": k, "distance": dists[k],
                            "quality": 1 - dists[k] / (2 * args.m)})
    pd.DataFrame(pm_rows).to_csv(
        os.path.join(outdir, "pattern_matching.csv"), index=False)

    # --- Trajectories and probabilities ---
    res = {j: run_component(x, t, j, params, k0_used[j], args.horizon,
                            args.threshold, args.m, args.vartheta, anchor)
           for j in (1, 2)}
    dates = oc.forecast_dates(df.index, t, args.m, args.horizon)
    h_star = {j: oc.crash_horizon(res[j]["psi"], params["alpha"], args.threshold)
              for j in (1, 2)}
    print(f"  crash horizons at the {args.threshold:.0%} threshold: "
          f"component 1 h*={h_star[1]}, component 2 h*={h_star[2]}")

    tab = assemble(res[1], res[2], dates, args.m, args.horizon - 1)
    tab.to_csv(os.path.join(outdir, "forecast_table.csv"), index=False)

    # --- Crash horizons across thresholds ---
    ch_rows = []
    for j in (1, 2):
        for thr in oc.THRESHOLDS:
            ch_rows.append({"component": j, "threshold": thr,
                            "h_star": oc.crash_horizon(res[j]["psi"], params["alpha"], thr)})
    pd.DataFrame(ch_rows).to_csv(os.path.join(outdir, "crash_horizons.csv"), index=False)

    # --- LaTeX ---
    tex = to_tex(tab, res[1], res[2], args.threshold, args.m, h_star)
    tex_path = os.path.join(outdir, "table_ovx_forecast.tex")
    with open(tex_path, "w") as fh:
        fh.write(tex)

    oc.write_run_config(outdir, {
        "target": "online_supplement3.tex :: tab:bubble_forecast",
        "data_file": os.path.relpath(args.data, oc.HERE),
        "T": int(len(df)),
        "sample_note": oc.sample_note(df),
        "cutoff": f"{df.index[t]:%Y-%m-%d}", "cutoff_index": t,
        "anchor_mode": "measured",
        "anchor_value": float(abs(x[t])),
        "m": args.m, "horizon": args.horizon, "threshold": args.threshold,
        "vartheta": args.vartheta,
        "k0_mode": "kfinder", "k0_used": k0_used,
        "params": {k: float(v) for k, v in params.items()},
        "crash_horizons": {f"component_{j}": h_star[j] for j in (1, 2)},
    })

    print(f"\nWritten to {OUTDIR_NAME}/")
    print(f"  {os.path.basename(tex_path)}")
    print("  forecast_table.csv, crash_horizons.csv, pattern_matching.csv")


if __name__ == "__main__":
    main()
