#!/usr/bin/env python3
r"""
Export S.2.3 -- Criterion difference test for $H_0\colon \pi_1 = 1/2$
(online_supplement3.tex, \label{app:criterion_test}).

Reuses run_mc_criterion_test() from criterion_difference_test.py
(Assumption C2 weight + guaranteed nesting of the restricted model in the
free model, see cf_weights.py and the 2026-09 patch: Delta_n is now >= 0 by
construction, fixing the negative medians of the archived table) -- this
script does NOT reimplement any estimation logic, it only runs that
function and formats its returned dict into the single Table
(tab:criterion_test) exactly as it appears in online_supplement3.tex.

Unlike the three subsampling sections, this exercise has no per-n figures
and (already, in the document) shows n = 1,000 / 10,000 / 100,000 as three
rows of ONE table rather than three separate tables.

Usage
-----
    python export_s23_criterion_test.py
    python export_s23_criterion_test.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy

from cf_weights import KAPPA_DEFAULT
from criterion_difference_test import run_mc_criterion_test

OUTPUT_DIR_DEFAULT = "resultats_restr"  # criterion test has no figures; shares
                                        # the numeric appendix with S.2.4 in the
                                        # archive layout (Latex/resultats_restr/)


def _fmt_n(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def format_criterion_table(results: dict, sample_sizes, M: int) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Criterion difference test for $H_0\colon \pi_1 = 1/2$ "
        r"(true $\pi_1 = 0.4375$, $M = " + str(M) + r"$)}",
        r"\label{tab:criterion_test}",
        r"\begin{tabular}{l c c c c c c c}",
        r"\toprule",
        r"& & & \multicolumn{3}{c}{$\Delta_n$} & \multicolumn{2}{c}{Rejection rate} \\",
        r"\cmidrule(lr){4-6} \cmidrule(lr){7-8}",
        r"$n$ & $b$ & $M$ & Mean & Median & Std & $\chi^2(1)$ at 5\% & Subsampling at 5\% \\",
        r"\midrule",
    ]
    for n in sample_sizes:
        a = results[n]
        med = a["delta_median"]
        med_s = f"{med:.3f}" if med >= 0 else f"$-{abs(med):.3f}$"
        lines.append(
            f"{_fmt_n(n)} & {_fmt_n(a['b'])} & {a['M']} & {a['delta_mean']:.3f} & "
            f"{med_s} & {a['delta_std']:.3f} & {a['rej_chi2_05']:.3f} & "
            f"{a['rej_sub_05']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}",
        r"\footnotesize\textit{Notes:} "
        r"$\Delta_n = n[Q_n(\hat{\theta}_n^{\mathrm{restr}}) - Q_n(\hat{\theta}_n^{\mathrm{free}})]$ "
        r"is the criterion difference statistic defined in \eqref{eq:criterion_diff}.  "
        r"The $\chi^2(1)$ column reports the fraction of replications for which the "
        r"asymptotic $p$-value falls below 5\%.  The subsampling column reports the "
        r"fraction for which $\hat{p}_{\mathrm{sub}} < 0.05$.  The subsample size is "
        r"$b = \lfloor n^{2/3} \rfloor$.  By construction (nested optimization, see "
        r"criterion\_difference\_test.py), $\Delta_n \geq 0$ for every replication.",
        r"\end{flushleft}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--b-exponent", type=float, default=2 / 3)
    ap.add_argument("--kappa", type=float, default=KAPPA_DEFAULT)
    ap.add_argument("--n-cores", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--true-params", type=float, nargs=6,
                     default=[0.8, 0.3, 0.7, 0.9, 1.5, 0.0],
                     metavar=("RHO1", "RHO2", "VARS1", "VARS2", "ALPHA", "BETA"))
    ap.add_argument("--output-dir", default="resultats_criterion_test")
    ap.add_argument("--quick", action="store_true",
                     help="smoke test: sample_sizes=[500], M=6")
    args = ap.parse_args()

    sample_sizes = args.sample_sizes
    M = args.M
    if args.quick:
        sample_sizes, M = [500], 6

    os.makedirs(args.output_dir, exist_ok=True)
    true_params = tuple(args.true_params)

    config = dict(
        section="S.2.3 (app:criterion_test)", sample_sizes=sample_sizes, M=M,
        b_exponent=args.b_exponent, kappa=args.kappa, seed=args.seed,
        true_params=true_params, n_cores=args.n_cores,
        python_version=sys.version, platform=platform.platform(),
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        pandas_version=pd.__version__, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    t_start = time.time()
    results = run_mc_criterion_test(
        sample_sizes=sample_sizes, M=M, b_exponent=args.b_exponent,
        n_cores=args.n_cores, seed=args.seed, output_dir=args.output_dir,
        true_params=true_params, alpha=true_params[4], beta=true_params[5],
        fix_alpha=False, fix_beta=True, J=2, kappa=args.kappa,
    )

    bad = {n: a["frac_delta_n_negative"] for n, a in results.items()
           if a["frac_delta_n_negative"] > 0}
    if bad:
        print(f"\n!! ATTENTION : Delta_n < 0 subsiste pour {bad} -- "
              "l'optimum libre n'est pas atteint, augmenter maxiter/n_starts.")
    else:
        print("\nOK : Delta_n >= 0 pour toutes les replications et tailles.")

    tex = format_criterion_table(results, sample_sizes, M)
    tex_path = os.path.join(args.output_dir, "s23_criterion_test.tex")
    with open(tex_path, "w") as f:
        f.write(tex + "\n")

    print(f"\nLaTeX written to {tex_path}")
    print(f"Done in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
