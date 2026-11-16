r"""LaTeX formatting of the subsampling tables of the Online Supplement of de Truchis,
Fries and Thomas (2026), Section S.3 (Tables S.3-S.9, figure environments of
Figures S.1-S.5).  No computation: it formats the DataFrames produced by
subsampling_reparam.py; called by export_s22_reparam.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Parameter symbols used in the tables.
DETAIL_SYMBOL = {
    "rho_1": r"$\psi_1$", "rho_2": r"$\psi_2$",
    "sigma": r"$\sigma$", "pi_1": r"$\pi_1$", "alpha": r"$\alpha$",
}
SUMMARY_SYMBOL = {
    "rho_1": r"$\psi_1$", "rho_2": r"$\psi_2$",
    "sigma": r"$\sigma$", "pi_1": r"$\pi_1$", "alpha": r"$\alpha$",
}

_N_TO_B = {}  # filled by set_block_sizes() or passed explicitly


def fmt_n(n: int) -> str:
    """1000 -> '1,000' (LaTeX '{,}' convention of the document)."""
    return f"{n:,}".replace(",", "{,}")


def _f(x: float, dp: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = ("" if x >= 0 else r"$-$") + format(abs(x), f".{dp}f")
    return s


def _f_signed_math(x: float, dp: int = 2) -> str:
    """For Dev.Mean/Skew/Kurt: always within $...$, with an explicit sign."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"${x:.{dp}f}$"


# =============================================================================
# Detailed table (one per n) -- reproduces tab:subsampling_n*, tab:restr_n*,
# tab:restr_true_n*
# =============================================================================

def format_detail_table(df: pd.DataFrame, n: int, b: int, M: int,
                         caption: str, label: str, notes: str) -> str:
    """
    df : DataFrame produced by Engine.table(agg) -- columns Parameter, True,
         Avg Est., Std Est., Dev.Mean, Dev.Std, Ratio, Skew, Kurt, SW Rej.,
         JB Rej., qCov90, nCov90, qCov95, nCov95 (native order of the engine).
         If `Skew.Trim`/`Outlier%` are present (subsampling_reparam.robust_shape_stats),
         two robust columns are inserted after Kurt: trimmed skewness and the share of
         extreme standardized points (median/MAD).

    Layout of the detail tables: the Ratio column is dropped; coverage columns are
    grouped as Quantile CI (qCov90, qCov95) then Normal CI (nCov90, nCov95).
    """
    has_robust = "Skew.Trim" in df.columns and "Outlier%" in df.columns
    ncols = 16 if has_robust else 14
    colspec = "l" + "c" * ncols
    qci_start, qci_end = (12, 13) if has_robust else (10, 11)
    nci_start, nci_end = (14, 15) if has_robust else (12, 13)
    blanks = "& " * (ncols - 4)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        rf"{blanks}\multicolumn{{2}}{{c}}{{Quantile CI}} & \multicolumn{{2}}{{c}}{{Normal CI}} \\",
        rf"\cmidrule(lr){{{qci_start}-{qci_end}}} \cmidrule(lr){{{nci_start}-{nci_end}}}",
        r"Parameter & True & Avg Est.\ & Std Est.\ & Dev.\ Mean & Dev.\ Std & "
        r"Skew & Kurt & "
        + (r"Skew$_{\text{trim}}$ & Out.\% & " if has_robust else "")
        + r"SW Rej.\ & JB Rej.\ & qCov90 & qCov95 & nCov90 & nCov95 \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        sym = DETAIL_SYMBOL.get(row["Parameter"], row["Parameter"])
        cells = [
            sym,
            format(row["True"], ".3f"),
            _f(row["Avg Est."]), _f(row["Std Est."]),
            _f_signed_math(row["Dev.Mean"]), format(row["Dev.Std"], ".3f"),
            _f_signed_math(row["Skew"]), _f_signed_math(row["Kurt"]),
        ]
        if has_robust:
            cells += [_f_signed_math(row["Skew.Trim"]),
                      format(row["Outlier%"] * 100, ".1f") + r"\%"]
        cells += [
            format(row["SW Rej."], ".3f"), format(row["JB Rej."], ".3f"),
            format(row["qCov90"], ".3f"), format(row["qCov95"], ".3f"),
            format(row["nCov90"], ".3f"), format(row["nCov95"], ".3f"),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        r"\begin{flushleft}",
        rf"\footnotesize\textit{{Notes:}} {notes}",
        r"\end{flushleft}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# =============================================================================
# Tableau resume multirow (tab:subsampling_summary, tab:restr_true_summary)
# =============================================================================

def format_summary_table(dfs_by_n: Dict[int, pd.DataFrame], param_order: Sequence[str],
                          caption: str, label: str, notes: str) -> str:
    """
    dfs_by_n : {n: DataFrame}, one DataFrame per sample size (same columns as
               format_detail_table).  Columns shown: True, Avg Est.,
               Std Est., Dev.Mean, Skew, Kurt, qCov90, qCov95, nCov90, nCov95
               (Dev.Std and SW/JB Rej. omitted, as in the paper).
    """
    lines = [
        r"\begin{table}[!h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * 11 + "}",  # n, Param, True, AvgEst, StdEst,
                                                  # DevMean, Skew, Kurt, qCov90/95, nCov90/95
        r"\toprule",
        r"& & & & & & & \multicolumn{2}{c}{Quantile CI} & \multicolumn{2}{c}{Normal CI} \\",
        r"\cmidrule(lr){8-9} \cmidrule(lr){10-11}",
        r"$n$ & Param.\ & True & Avg.\ Est.\ & Std.\ Est.\ & Dev.\ Mean & Skew & "
        r"Kurt & qCov90 & qCov95 & nCov90 & nCov95 \\",
        r"\midrule",
    ]
    ns = sorted(dfs_by_n)
    for i, n in enumerate(ns):
        df = dfs_by_n[n]
        lines.append(rf"\multirow{{{len(param_order)}}}{{*}}{{${fmt_n(n)}$}}")
        for p in param_order:
            row = df[df["Parameter"] == p]
            if row.empty:
                continue
            row = row.iloc[0]
            sym = SUMMARY_SYMBOL.get(p, p)
            cells = [
                "", sym,
                format(row["True"], ".4f" if p == "pi_1" else ".2f"),
                format(row["Avg Est."], ".3f"), format(row["Std Est."], ".3f"),
                _f_signed_math(row["Dev.Mean"]),
                _f_signed_math(row["Skew"]), _f_signed_math(row["Kurt"]),
                format(row["qCov90"], ".3f"), format(row["qCov95"], ".3f"),
                format(row["nCov90"], ".3f"), format(row["nCov95"], ".3f"),
            ]
            lines.append("& " + " & ".join(cells[1:]) + r" \\")
        if i < len(ns) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        r"\begin{flushleft}",
        rf"\footnotesize\textit{{Notes:}} {notes}",
        r"\end{flushleft}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# =============================================================================
# Diagnostic figure (same structure for the three families reparam/restr/
# restr_true -- only the image path and the caption change)
# =============================================================================

def format_figure(image_path: str, caption: str, label: str) -> str:
    return "\n".join([
        r"\begin{figure}[htbp]",
        r"\centering",
        rf"\includegraphics[width=\textwidth]{{{image_path}}}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{figure}",
    ])


# =============================================================================
# Design table (tab:mc_design) -- documents the full design, including the
# sample sizes that are not detailed in their own table.
# =============================================================================

def format_design_table(sample_sizes: Sequence[int], b_exponent: float, M: int) -> str:
    lines = [
        r"\begin{table}[!h]",
        r"\centering",
        r"\caption{Monte Carlo Design Parameters}",
        r"\label{tab:mc_design}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Sample Size $n$ & Subsample Size $b$ & Non-overlapping Blocks $N_b$ & MC Replications $M$ \\",
        r"\midrule",
    ]
    for n in sample_sizes:
        b = int(np.floor(n ** b_exponent))
        Nb = n // b
        lines.append(f"{fmt_n(n)} & {fmt_n(b)} & {Nb} & {M} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize\textit{Notes:} The subsample size is set to "
        r"$b = \lfloor n^{2/3} \rfloor$ following \cite{PR99}.  "
        r"$N_b = \lfloor n/b \rfloor$ denotes the number of non-overlapping "
        r"blocks per replication.",
        r"\end{flushleft}",
        r"\end{table}",
    ]
    return "\n".join(lines)
