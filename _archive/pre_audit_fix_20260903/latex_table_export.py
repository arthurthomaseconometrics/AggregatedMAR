r"""
Formateurs LaTeX partages pour les quatre exports de la Section S.2 du
supplement (online_supplement3.tex) : reparam non restreint, test de
critere, restreint pi_1=1/2, restreint pi_1=0.4375 (valeur vraie).

Ce module ne fait AUCUN calcul MDE/subsampling -- il ne fait que mettre en
forme les DataFrames deja produits par les moteurs de subsampling_reparam.py,
subsampling_restricted.py, subsampling_restricted_truepi.py et
criterion_difference_test.py, en respectant EXACTEMENT le format des
tableaux actuellement dans online_supplement3.tex (memes en-tetes, meme
ordre de colonnes Quantile CI / Normal CI, memes notes).

Chaque script d'export (export_s2*_*.py) appelle ces fonctions et ecrit un
seul fichier .tex par section, pret a etre colle dans online_supplement3.tex.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Symboles des parametres tels qu'utilises dans online_supplement3.tex.
# Les tableaux DETAILLES (un par n) utilisent rho_1/rho_2 ; les tableaux
# RESUME (en tete de section S.2) utilisent psi_1/psi_2 -- c'est deja
# l'incoherence de notation presente dans le document actuel, on la
# reproduit telle quelle plutot que de la "corriger" sans qu'on le demande.
DETAIL_SYMBOL = {
    "rho_1": r"$\rho_1$", "rho_2": r"$\rho_2$",
    "sigma": r"$\sigma$", "pi_1": r"$\pi_1$", "alpha": r"$\alpha$",
}
SUMMARY_SYMBOL = {
    "rho_1": r"$\psi_1$", "rho_2": r"$\psi_2$",
    "sigma": r"$\sigma$", "pi_1": r"$\pi_1$", "alpha": r"$\alpha$",
}

_N_TO_B = {}  # renseigne par set_block_sizes() ou passe explicitement


def fmt_n(n: int) -> str:
    """1000 -> '1,000' (convention LaTeX '{,}' du document)."""
    return f"{n:,}".replace(",", "{,}")


def _f(x: float, dp: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = ("" if x >= 0 else r"$-$") + format(abs(x), f".{dp}f")
    return s


def _f_signed_math(x: float, dp: int = 2) -> str:
    """Pour Dev.Mean/Skew/Kurt : toujours entre $...$, avec signe explicite."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"${x:.{dp}f}$"


# =============================================================================
# Tableau detaille (un par n) -- reproduit tab:subsampling_n*, tab:restr_n*,
# tab:restr_true_n*
# =============================================================================

def format_detail_table(df: pd.DataFrame, n: int, b: int, M: int,
                         caption: str, label: str, notes: str) -> str:
    """
    df : DataFrame produit par Engine.table(agg) -- colonnes Parameter, True,
         Avg Est., Std Est., Dev.Mean, Dev.Std, Ratio, Skew, Kurt, SW Rej.,
         JB Rej., qCov90, nCov90, qCov95, nCov95 (ordre natif de l'engine).

    Reproduit exactement la mise en page de online_supplement3.tex : colonne
    Ratio supprimee (absente du document publie), colonnes de couverture
    regroupees Quantile CI (qCov90, qCov95) puis Normal CI (nCov90, nCov95).
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccccccccc}",
        r"\toprule",
        r"& & & & & & & & & \multicolumn{2}{c}{Quantile CI} & \multicolumn{2}{c}{Normal CI} \\",
        r"\cmidrule(lr){10-11} \cmidrule(lr){12-13}",
        r"Parameter & True & Avg Est.\ & Std Est.\ & Dev.\ Mean & Dev.\ Std & "
        r"Skew & Kurt & SW Rej.\ & JB Rej.\ & qCov90 & qCov95 & nCov90 & nCov95 \\",
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
    dfs_by_n : {n: DataFrame}, un DataFrame par taille (memes colonnes que
               format_detail_table). Colonnes montrees : True, Avg Est.,
               Std Est., Dev.Mean, Skew, Kurt, qCov90, qCov95, nCov90, nCov95
               (Dev.Std, SW/JB Rej. omis, comme dans le document publie).
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
# Figure diagnostique (meme structure pour les trois familles reparam/restr/
# restr_true -- seul le chemin de l'image et la legende changent)
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
# Table de conception (tab:mc_design) -- documente le design complet a 6
# tailles meme quand seules 3 sont detaillees en profondeur.
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
