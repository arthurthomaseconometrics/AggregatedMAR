#!/usr/bin/env python3
r"""
Table S.14 of the Online Supplement of de Truchis, Fries and Thomas (2026): estimation results for the OVX
index under three specifications (GaS, SaS, S1S).

Formats the estimates of ovx_estimation/estimation.csv into the booktabs table used in the
supplement.

Outputs (results_ovx_estimation/)
--------------------------------------------------------------------------
    table_ovx_estimation.tex     booktabs table, supplement format
    estimation_table.csv          estimate / std. error / t-stat per specification
    ovx_series.csv                series used (level, trend, detrended)
    run_config.json               settings and environment

Usage
--------------------------------------------------------------------------
    python table_ovx_estimation.py
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

import ovx_common as oc

OUTDIR_NAME = "results_ovx_estimation"
SPECS = ("gas", "sas", "s1s")

REPORT_ROWS = ["rho_1", "rho_2", "alpha", "beta", "sigma", "pi_1", "pi_2"]

TEX_SYMBOL = {
    "rho_1": r"$\psi_1$", "rho_2": r"$\psi_2$", "alpha": r"$\alpha$",
    "beta": r"$\beta$", "sigma": r"$\sigma$",
    "pi_1": r"$\pi_1$", "pi_2": r"$\pi_2$",
}

TABLE_NOTE = (
    r"Minimum distance estimator with HAC weights computed at the initial guess, on the "
    r"weekly OVX, $T = 522$; the components are ordered $\psi_1 < \psi_2$. Standard errors "
    r"are the Hessian-based asymptotic standard errors; a dash marks a standard error that "
    r"is not available. The $t$-statistics test the null hypothesis that each parameter "
    r"equals zero. For the $\mathcal{S}1\mathcal{S}$ specification, $\alpha = 1$ is imposed. "
    r"In the $\mathcal{S}\alpha\mathcal{S}$ specification, $\psi_1$ is close to the lower "
    r"bound of the parameter space and its standard error is not informative.")


def build_tables() -> Dict[str, pd.DataFrame]:
    """Point estimates for each specification, in Table S.14 row order."""
    out = {}
    for spec in SPECS:
        rows = []
        for name in REPORT_ROWS:
            cell = oc.ESTIMATION[spec].get(name)
            if cell is None:
                rows.append({"parameter": name, "estimate": np.nan,
                             "std_error": np.nan, "t_stat": np.nan})
            else:
                se_, t_ = cell[1], cell[2]
                if not se_ > 1e-6:      # parameter on a bound: standard error not informative
                    se_, t_ = np.nan, np.nan
                rows.append({"parameter": name, "estimate": cell[0],
                             "std_error": se_, "t_stat": t_})
        out[spec] = pd.DataFrame(rows)
    return out


def to_tex(tables: Dict[str, pd.DataFrame], note: str) -> str:
    """Booktabs table matching Table S.14."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Estimation results for the OVX index under three specifications}",
        r"\label{tab:ovx_estimation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{" + oc.SPEC_LABELS["gas"] + "}",
        r"& \multicolumn{3}{c}{" + oc.SPEC_LABELS["sas"] + "}",
        r"& \multicolumn{3}{c}{" + oc.SPEC_LABELS["s1s"] + r"} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"$\widehat{\theta}$ & Estimate & Std.\ & $t$-stat",
        r"                   & Estimate & Std.\ & $t$-stat",
        r"                   & Estimate & Std.\ & $t$-stat \\",
        r"\midrule",
    ]
    for name in REPORT_ROWS:
        cells = [TEX_SYMBOL[name]]
        for spec in SPECS:
            row = tables[spec].set_index("parameter").loc[name]
            cells += [oc.tex_num(row["estimate"], 4),
                      oc.tex_num(row["std_error"], 4),
                      oc.tex_num(row["t_stat"], 3)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\begin{flushleft}",
        r"\footnotesize\textit{Notes:} " + note,
        r"\end{flushleft}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    outdir = oc.ensure_outdir(OUTDIR_NAME)
    df = oc.load_ovx()
    note_sample = oc.sample_note(df)

    print("=" * 72)
    print("Table S.14 -- OVX estimation")
    print("=" * 72)
    print(f"  sample: {note_sample}")
    print()

    df.to_csv(os.path.join(outdir, "ovx_series.csv"))

    tables = build_tables()

    long_rows = []
    for spec in SPECS:
        t = tables[spec].copy()
        t.insert(0, "spec", spec)
        long_rows.append(t)
    combined = pd.concat(long_rows, ignore_index=True)
    csv_path = os.path.join(outdir, "estimation_table.csv")
    combined.to_csv(csv_path, index=False)

    tex_path = os.path.join(outdir, "table_ovx_estimation.tex")
    with open(tex_path, "w") as fh:
        fh.write(to_tex(tables, TABLE_NOTE))

    oc.write_run_config(outdir, {
        "target": "Online Supplement, Table S.14",
        "data_file": os.path.relpath(oc.DEFAULT_DATA, oc.HERE),
        "sample_start": oc.SAMPLE_START, "sample_end": oc.SAMPLE_END,
        "resample_rule": oc.RESAMPLE_RULE, "resample_how": oc.RESAMPLE_HOW,
        "detrend_order": oc.DETREND_ORDER,
        "T": int(len(df)),
        "sample_note": note_sample,
        "specs": list(SPECS),
    })

    print(f"Written to {OUTDIR_NAME}/")
    print(f"  {os.path.basename(tex_path)}")
    print(f"  {os.path.basename(csv_path)}")


if __name__ == "__main__":
    main()
