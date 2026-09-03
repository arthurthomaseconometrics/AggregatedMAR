#!/usr/bin/env python3
r"""
Export S.2.2 -- Unrestricted subsampling, reparameterized space
(online_supplement3.tex, \label{app:subsampling_reparam}).

Reuses the engine of subsampling_reparam.py (Assumption C2 weight +
corrected centering, see cf_weights.py and the 2026-09 patch) -- this
script does NOT reimplement any estimation logic, it only runs that engine
for the representative sample sizes and formats the output exactly as it
appears in online_supplement3.tex:

  - one Table (tab:subsampling_n<n>) and one Figure (fig:diagnostics_n<n>)
    per sample size, concatenated into a single .tex file ready to paste
    into the "Unrestricted subsampling in the reparameterized space"
    subsection;
  - the top-of-S.2 summary table (tab:subsampling_summary);
  - figures saved to resultats_reparam/diagnostics_n<n>.png, matching the
    \includegraphics paths already in online_supplement3.tex;
  - raw per-replication CSVs, a summary CSV, and a run_config.json, in the
    same spirit as table_s1_mc_accuracy.py / MAR_1_1/table_s2_mc_accuracy.py.

Usage
-----
    python export_s22_reparam.py                       # n = 1000/10000/100000, M=200
    python export_s22_reparam.py --quick                # smoke test
    python export_s22_reparam.py --sample-sizes 1000 10000 100000 --M 200 --n-cores -1
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
from latex_table_export import fmt_n, format_design_table, format_detail_table, format_figure, format_summary_table
from subsampling_reparam import _EngineReparam

OUTPUT_DIR_DEFAULT = "resultats_reparam"
FULL_DESIGN_SIZES = [250, 500, 1000, 10000, 50000, 100000]
PARAM_ORDER = ["rho_1", "rho_2", "sigma", "pi_1", "alpha"]


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

    eng = _EngineReparam(J=2, true_params=true_params, alpha=true_params[4],
                         beta=true_params[5], fix_alpha=False, fix_beta=True,
                         kappa=args.kappa)
    print("Critere MDE (Assumption C2, main_v23.tex):")
    check_grid(eng.u_grid, eng.v_grid, args.kappa)

    config = dict(
        section="S.2.2 (app:subsampling_reparam)", sample_sizes=sample_sizes, M=M,
        b_exponent=args.b_exponent, kappa=args.kappa, seed=args.seed,
        true_params=true_params, n_cores=args.n_cores,
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
        Nb = n // b
        print(f"\n{'=' * 65}\nn={n}, b={b} (n^{args.b_exponent:.4f}) [REPARAM: sigma, pi_1]")

        agg = eng.mc(n, M, b, seed=args.seed, n_cores=args.n_cores)
        df = eng.table(agg)
        dfs_by_n[n] = df
        print(df.to_string(index=False))

        df.to_csv(os.path.join(args.output_dir, f"summary_n{n}.csv"), index=False)
        png_path = os.path.join(args.output_dir, f"diagnostics_n{n}.png")
        eng.plot_diag(agg, png_path)

        caption = (f"Subsampling results (reparameterized) for $n = {fmt_n(n)}$ "
                  f"($b = {fmt_n(b)}$, $N_b = {Nb}$, $M = {M}$)")
        tex_blocks.append(format_detail_table(
            df, n, b, M, caption=caption, label=f"tab:subsampling_n{n}",
            notes=("All results are in the reparameterized space "
                  r"$(\rho_1, \rho_2, \sigma, \pi_1, \alpha)$.  SW and JB rejection "
                  r"rates are at the 5\% level.  qCov and nCov denote quantile-based "
                  "and normal-approximation CI coverage.  Confidence intervals are "
                  "constructed from thinned overlapping blocks; normality diagnostics "
                  "use non-overlapping blocks."
                  if n == sample_sizes[0] else
                  rf"See notes to Table~\ref{{tab:subsampling_n{sample_sizes[0]}}}.")))
        tex_blocks.append(format_figure(
            image_path=f"{args.output_dir}/diagnostics_n{n}.png",
            caption=(f"Diagnostic plots for scaled subsample deviations "
                    f"(reparameterized) at $n = {fmt_n(n)}$. " +
                    ("For each parameter (rows), we display: (left) histogram with "
                     "standard normal overlay, (center) Q-Q plot against standard "
                     "normal quantiles, and (right) kernel density estimate compared "
                     "to the standard normal density."
                     if n == sample_sizes[0] else
                     rf"See caption to Figure~\ref{{fig:diagnostics_n{sample_sizes[0]}}} for details.")),
            label=f"fig:diagnostics_n{n}"))

    summary_tex = format_summary_table(
        dfs_by_n, PARAM_ORDER,
        caption="Summary of unrestricted subsampling results (reparameterized space)",
        label="tab:subsampling_summary",
        notes=(r"Avg.~Est.\ and Std.~Est.\ are computed over $M=" + str(M) + r"$ "
              r"replications.  Dev.~Mean, Skew, and Kurt refer to the mean, skewness, "
              r"and excess kurtosis of the non-overlapping block scaled deviations "
              r"$Z^{(i)}_b$.  qCov and nCov denote the average coverage of "
              r"quantile-based and normal-approximation confidence intervals."))

    design_tex = format_design_table(FULL_DESIGN_SIZES, args.b_exponent, 200)

    full_tex = "\n\n\n".join([design_tex, summary_tex] + tex_blocks)
    tex_path = os.path.join(args.output_dir, "s22_reparam.tex")
    with open(tex_path, "w") as f:
        f.write(full_tex + "\n")

    print(f"\nLaTeX written to {tex_path}")
    print(f"Figures written to {args.output_dir}/diagnostics_n<n>.png")
    print(f"Done in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
