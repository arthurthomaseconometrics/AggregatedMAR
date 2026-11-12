"""Extra tables for Section S.4 of the Online Supplement, from results/replications.csv.

Writes results/tab_supp_cv.tex (bootstrap critical values, power of the asymptotic test)
and results/tab_supp_diag.tex (behaviour of the estimator and numerical diagnostics).
No simulation is run.

Usage: python supplement_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PSI = [0.0, 0.05, 0.1, 0.2, 0.3]
Z95 = 1.6448536269514722


def _n(n):
    return f"{int(n):,}".replace(",", "{,}")


def cv_table(df):
    sizes = sorted(df.n.unique())
    alts = [p for p in PSI if p > 0]
    lines = [r"\begin{tabular}{rccccc}", r"\toprule",
             r"\multicolumn{6}{l}{\textit{Panel A. 95\% quantile of the bootstrap statistic $t_n^{*}$}} \\",
             r"\midrule",
             r"$n$ & " + " & ".join(f"$\\psi_1={p:g}$" for p in PSI) + r" \\", r"\midrule"]
    for n in sizes:
        cells = [f"{df[(df.n == n) & (df.psi_small == p)].t_star.quantile(0.95):.2f}" for p in PSI]
        lines.append(_n(n) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule",
              r"\multicolumn{6}{l}{\textit{Panel B. Rejection rates of the asymptotic test at 5\% under $H_1$ (not size-adjusted)}} \\",
              r"\midrule",
              r"$n$ & " + " & ".join(f"$\\psi_1={p:g}$" for p in alts) + r" & \\", r"\midrule"]
    for n in sizes:
        cells = [f"{np.mean(df[(df.n == n) & (df.psi_small == p)].t_unres > Z95):.3f}" for p in alts]
        lines.append(_n(n) + " & " + " & ".join(cells) + r" & \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def diag_table(df):
    lines = [r"\begin{tabular}{rcccccc}", r"\toprule",
             r"& \multicolumn{4}{c}{Under $H_0$: $\psi_1 = 0$} & \multicolumn{2}{c}{All five designs} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
             r"$n$ & Mean $\hat\psi_1$ & $q_{0.95}(\sqrt{n}\hat\psi_1)$ & Median $\hat{V}_{\psi_1\psi_1}(\tilde\theta_n)^{1/2}$ "
             r"& $q_{0.95}(t_n)$ & Min.\ conv.\ & Max.\ $V<0$ \\", r"\midrule"]
    for n, g in df.groupby("n"):
        h = g[g.psi_small == 0.0]
        conv = g.groupby("psi_small").converged.mean().min()
        neg = g.groupby("psi_small").t_unres.apply(lambda s: s.isna().mean()).max()
        lines.append(f"{_n(n)} & {h.psi1_hat.mean():.3f} & {h.S.quantile(0.95):.2f} & "
                     f"{h.sd_h0.median():.2f} & {h.t_unres.quantile(0.95):.2f} & "
                     f"{conv:.3f} & {neg:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    df = pd.read_csv(HERE / "results" / "replications.csv")
    (HERE / "results" / "tab_supp_cv.tex").write_text(cv_table(df))
    (HERE / "results" / "tab_supp_diag.tex").write_text(diag_table(df))


if __name__ == "__main__":
    main()
