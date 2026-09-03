#!/usr/bin/env python3
r"""
Export S.2.5 -- Restricted subsampling under $\pi_1 = 0.4375$ (true value)
(online_supplement3.tex, \label{app:subsampling_restr_true}).  This is the
"flagship" exercise cited by name in main_v23.tex's conclusion for Section
4.2 ("confirming that the correctly restricted estimator converges to its
asymptotic Gaussian limit").

Reuses the engine of subsampling_restricted_truepi.py (Assumption C2
weight, see cf_weights.py and the 2026-09 patch) -- this script does NOT
reimplement any estimation logic.  It runs that engine for the
representative sample sizes and formats the output exactly as it appears
in online_supplement3.tex: one Table (tab:restr_true_n<n>) and one Figure
(fig:diagnostics_restr_true_n<n>) per sample size, plus the top-of-S.2
summary table (tab:restr_true_summary).  Figures are saved to
resultats_restr_true_param/diagnostics_n<n>.png, matching the existing
\includegraphics paths.

Usage
-----
    python export_s25_restricted_true.py
    python export_s25_restricted_true.py --quick
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

from cf_weights import KAPPA_DEFAULT, check_grid
from latex_table_export import fmt_n, format_detail_table, format_figure, format_summary_table
from subsampling_restricted_truepi import _EngineRestricted

OUTPUT_DIR_DEFAULT = "resultats_restr_true_param"
PARAM_ORDER = ["rho_1", "rho_2", "sigma", "alpha"]


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
    ap.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    ap.add_argument("--quick", action="store_true",
                     help="smoke test: sample_sizes=[500], M=6")
    args = ap.parse_args()

    sample_sizes = args.sample_sizes
    M = args.M
    if args.quick:
        sample_sizes, M = [500], 6

    os.makedirs(args.output_dir, exist_ok=True)
    true_params = tuple(args.true_params)
    pi1_true = true_params[2] / (true_params[2] + true_params[3])

    eng = _EngineRestricted(J=2, true_params=true_params, alpha=true_params[4],
                            beta=true_params[5], fix_alpha=False, fix_beta=True,
                            kappa=args.kappa)
    print("Critere MDE (Assumption C2, main_v23.tex):")
    check_grid(eng.u_grid, eng.v_grid, args.kappa)
    print(f"pi_1 (true) = {pi1_true:.6f}")

    config = dict(
        section="S.2.5 (app:subsampling_restr_true, pi_1=true)",
        sample_sizes=sample_sizes, M=M, b_exponent=args.b_exponent, kappa=args.kappa,
        seed=args.seed, true_params=true_params, pi1_true=pi1_true, n_cores=args.n_cores,
        python_version=sys.version, platform=platform.platform(),
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        pandas_version=pd.__version__, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    tex_blocks = []
    dfs_by_n = {}
    t_start = time.time()
    for n in sample_sizes:
        b = int(np.floor(n ** args.b_exponent))
        print(f"\n{'=' * 65}\nn={n}, b={b} [RESTRICTED: pi_1={pi1_true:.4f}]")

        agg = eng.mc(n, M, b, seed=args.seed, n_cores=args.n_cores)
        df = eng.table(agg)
        dfs_by_n[n] = df
        print(df.to_string(index=False))

        df.to_csv(os.path.join(args.output_dir, f"summary_n{n}.csv"), index=False)
        png_path = os.path.join(args.output_dir, f"diagnostics_n{n}.png")
        eng.plot_diag(agg, png_path)

        caption = f"Restricted subsampling for $n = {fmt_n(n)}$ ($b = {fmt_n(b)}$, $M = {M}$, $\\pi_1 = 0.4375$)"
        tex_blocks.append(format_detail_table(
            df, n, b, M, caption=caption, label=f"tab:restr_true_n{n}",
            notes=(r"All results are in the reparameterized space $(\rho_1, \rho_2, "
                  r"\sigma, \alpha)$ under the true restriction $\pi_1 = 0.4375$.  "
                  r"SW and JB rejection rates are at the 5\% level.  qCov and nCov "
                  "denote quantile-based and normal-approximation CI coverage.  "
                  "Confidence intervals are constructed from thinned overlapping "
                  "blocks; normality diagnostics use non-overlapping blocks."
                  if n == sample_sizes[0] else
                  rf"See notes to Table~\ref{{tab:restr_true_n{sample_sizes[0]}}}.")))
        tex_blocks.append(format_figure(
            image_path=f"{args.output_dir}/diagnostics_n{n}.png",
            caption=(f"Diagnostic plots for restricted subsampling ($\\pi_1 = 0.4375$) "
                    f"at $n = {fmt_n(n)}$.  " +
                    ("For each parameter (rows): histogram with standard normal "
                     "overlay (left), Q-Q plot (center), and kernel density estimate "
                     r"vs.\ $\mathcal{N}(0,1)$ (right)."
                     if n == sample_sizes[0] else
                     rf"See caption to Figure~\ref{{fig:diagnostics_restr_true_n{sample_sizes[0]}}} for details.")),
            label=f"fig:diagnostics_restr_true_n{n}"))

    summary_tex = format_summary_table(
        dfs_by_n, PARAM_ORDER,
        caption=r"Summary of restricted subsampling results under $\pi_1 = 0.4375$ (true value)",
        label="tab:restr_true_summary",
        notes=(r"See notes to Table~\ref{tab:subsampling_summary}.  Estimation under "
              rf"the true restriction $\pi_1 = \varsigma_1/\sigma = {pi1_true:.4f}$.  "
              r"Detailed results for $n = " +
              ", ".join(f"{n:,}".replace(",", "{,}") for n in sample_sizes) +
              r"$ are in Section \ref{app:subsampling_restr_true} below."))

    full_tex = "\n\n\n".join([summary_tex] + tex_blocks)
    tex_path = os.path.join(args.output_dir, "s25_restricted_true.tex")
    with open(tex_path, "w") as f:
        f.write(full_tex + "\n")

    print(f"\nLaTeX written to {tex_path}")
    print(f"Figures written to {args.output_dir}/diagnostics_n<n>.png")
    print(f"Done in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
