#!/usr/bin/env python3
r"""
Export S.2.2 -- Unrestricted subsampling, reparameterized space
(online_supplement3.tex, \label{app:subsampling_reparam}).

Runs the engine of subsampling_reparam.py (Assumption C2 weight, single MDE
functional for the full sample and for every block) and emits, in one .tex
file, everything Section S.2 needs:

  - the Monte Carlo design table (tab:mc_design), built from the sample sizes
    ACTUALLY run rather than from a hard-coded list;
  - the top-of-S.2 summary table (tab:subsampling_summary);
  - one detail table (tab:subsampling_n<n>) and one figure
    (fig:diagnostics_n<n>) per sample size;
  - figures in <output-dir>/diagnostics_n<n>.png, matching the
    \includegraphics paths of online_supplement3.tex;
  - per-n summary CSVs and a run_config.json recording every setting.

This is intended to be the ONLY resampling exercise of the online supplement:
the criterion-difference test (old S.2.3), the pi_1 = 1/2 restricted study
(old S.2.4) and the pi_1 = 0.4375 restricted study (old S.2.5) were motivated
by an apparent non-identification of pi_1 that the 2026-09-03 audit traced to
the optimizer rather than to the criterion.  See the AUDIT block below.

AUDIT 2026-09-03 -- what changed and why
----------------------------------------
1. cf_weights.MDE_GTOL.  scipy's L-BFGS-B stops on ABSOLUTE criteria: `gtol`
   defaults to 1e-5 on max|projected gradient|, while the Assumption-C2
   weighted criterion is ~1e-6 at the optimum.  With the defaults the
   optimizer halted before moving along the weakly curved (pi_1) direction.
   Noiseless check -- replace phi_n by the exact phi(.;theta_0), i.e. n = inf:
       scipy defaults -> (psi1,psi2,sigma,pi1,alpha)
                       = (0.7699, 0.2358, 1.6119, 0.5090, 1.5017)
       with MDE_GTOL  -> (0.8000, 0.3000, 1.6000, 0.4375, 1.5000) = theta_0
   The stall is a FIXED displacement while the sampling s.d. shrinks like
   n^{-1/2}, so the bias in s.d. units grows like sqrt(n) -- which is exactly
   the Dev.Mean drift and coverage collapse previously read as evidence that
   pi_1 was unidentified.  pi_1 IS identified; the paper already proves it
   (main_v23.tex, eq. (2.11)-(2.13)).
2. subsampling_reparam.mde.  theta_n and every theta_b are now the SAME
   functional (same starts, bounds, maxiter, tolerances), as Politis-Romano
   requires.  The previous warm start of each block at theta_n deflated the
   s.d. of the pi_1 block deviations by 13%.
3. subsampling_reparam.canonical_order.  Assumption C1's ordering restriction
   is now enforced (3% of blocks came back label-switched at n = 1000).
4. stablemixture.generate.  The burn-in was discarded on the wrong side of a
   BACKWARD recursion, so the contaminated tail was kept and the clean head
   thrown away; the last observation was exactly 0 in 100% of replications.
   Independent per-component seed streams also replace `seed + j`, which made
   component 2 of replication m share its innovations with component 1 of
   replication m+1.
5. Dev.Mean / Dev.Std / Skew / Kurt are computed on the POOLED non-overlapping
   deviations, as Section S.2.1 already describes, instead of being averages
   of per-replication statistics computed on N_b = 6 to 10 points.

Usage
-----
    python export_s22_reparam.py                        # n = 250/500/1000, M=200
    python export_s22_reparam.py --quick                 # smoke test
    python export_s22_reparam.py --sample-sizes 250 500 1000 --M 200 --n-cores -1
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

from cf_weights import KAPPA_DEFAULT, MDE_FTOL, MDE_GTOL, MDE_MAXITER, check_grid
from latex_table_export import fmt_n, format_design_table, format_detail_table, format_figure, format_summary_table
from subsampling_reparam import N_STARTS_DEFAULT, OV_BLOCKS_TARGET, _EngineReparam

OUTPUT_DIR_DEFAULT = "resultats_reparam"
PARAM_ORDER = ["rho_1", "rho_2", "sigma", "pi_1", "alpha"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-sizes", type=int, nargs="+", default=[250, 500, 1000])
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--b-exponent", type=float, default=2 / 3)
    ap.add_argument("--kappa", type=float, default=KAPPA_DEFAULT)
    ap.add_argument("--n-cores", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--true-params", type=float, nargs=6,
                     default=[0.8, 0.3, 0.7, 0.9, 1.5, 0.0],
                     metavar=("RHO1", "RHO2", "VARS1", "VARS2", "ALPHA", "BETA"))
    ap.add_argument("--n-starts", type=int, default=N_STARTS_DEFAULT,
                     help="deterministic mirrored MDE starts (see subsampling_reparam.MDE_STARTS_J2)")
    ap.add_argument("--ci-step", type=int, default=None,
                     help="thinning step for the overlapping blocks; default targets "
                          f"~{OV_BLOCKS_TARGET} blocks per replication")
    ap.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    ap.add_argument("--tex-only", action="store_true",
                     help="skip computation; assemble s22_reparam.tex from the "
                          "summary_n<n>.csv / meta_n<n>.json already on disk")
    ap.add_argument("--skip-existing", action="store_true",
                     help="skip any sample size that already has meta_n<n>.json "
                          "(resume after a wall-clock kill)")
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
                         kappa=args.kappa, n_starts=args.n_starts)
    print("Critere MDE (Assumption C2, main_v23.tex):")
    check_grid(eng.u_grid, eng.v_grid, args.kappa)

    config = dict(
        section="S.2.2 (app:subsampling_reparam)", sample_sizes=sample_sizes, M=M,
        b_exponent=args.b_exponent, kappa=args.kappa, seed=args.seed,
        true_params=true_params, n_cores=args.n_cores,
        n_starts=args.n_starts, ci_step=args.ci_step,
        ov_blocks_target=OV_BLOCKS_TARGET,
        optimizer="L-BFGS-B", mde_gtol=MDE_GTOL, mde_ftol=MDE_FTOL,
        mde_maxiter=MDE_MAXITER,
        audit_fixes=[
            "explicit L-BFGS-B gtol/ftol (cf_weights.MDE_GTOL)",
            "single MDE functional for full sample and blocks (Politis-Romano)",
            "Assumption C1 ordering enforced (subsampling_reparam.canonical_order)",
            "burn-in kept on the correct side of the backward recursion",
            "independent per-component seed streams",
            "pooled non-overlapping deviations for Dev.Mean/Std/Skew/Kurt",
        ],
        python_version=sys.version, platform=platform.platform(),
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        pandas_version=pd.__version__, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    # Per-job name so concurrent array tasks do not clobber each other.
    cfg_name = "run_config_n" + "-".join(str(x) for x in sample_sizes) + ".json"
    if not args.tex_only:
        with open(os.path.join(args.output_dir, cfg_name), "w") as f:
            json.dump(config, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Compute phase.  Each sample size is self-contained: it writes
    # summary_n<n>.csv, meta_n<n>.json and diagnostics_n<n>.png.  A run can
    # therefore be split across jobs (one n per job) and resumed after a
    # wall-clock kill; `--tex-only` then assembles the .tex from whatever is
    # on disk.  This is what makes the large sizes tractable on a cluster.
    # ------------------------------------------------------------------
    t_start = time.time()
    if not args.tex_only:
        for n in sample_sizes:
            b = int(np.floor(n ** args.b_exponent))
            Nb = n // b
            if args.skip_existing and os.path.exists(
                    os.path.join(args.output_dir, "meta_n%d.json" % n)):
                print("\n[skip] n=%d: meta_n%d.json already present" % (n, n))
                continue
            print("\n%s\nn=%d, b=%d (n^%.4f) [REPARAM: sigma, pi_1]"
                  % ("=" * 65, n, b, args.b_exponent))

            agg = eng.mc(n, M, b, seed=args.seed, ci_step=args.ci_step,
                         n_cores=args.n_cores)
            df = eng.table(agg)
            print(df.to_string(index=False))

            df.to_csv(os.path.join(args.output_dir, "summary_n%d.csv" % n), index=False)
            eng.plot_diag(agg, os.path.join(args.output_dir, "diagnostics_n%d.png" % n))

            step = agg.get("ci_step") or 1
            meta = dict(
                n=n, b=b, Nb=Nb, M=M, ci_step=step,
                n_ov=len(range(0, n - b + 1, step)),
                n_pooled=(int(min(v.get("n_pooled", 0)
                                  for v in agg["moments_pooled"].values()))
                          if agg.get("moments_pooled") else 0),
                n_starts=args.n_starts, kappa=args.kappa, seed=args.seed,
                b_exponent=args.b_exponent,
                elapsed_min=round((time.time() - t_start) / 60, 2),
            )
            with open(os.path.join(args.output_dir, "meta_n%d.json" % n), "w") as f:
                json.dump(meta, f, indent=2)
            print("  -> summary_n%d.csv, meta_n%d.json, diagnostics_n%d.png" % (n, n, n))

    # ------------------------------------------------------------------
    # Collect phase: assemble the .tex from what is on disk.
    # ------------------------------------------------------------------
    dfs_by_n, metas, missing = {}, {}, []
    for n in sample_sizes:
        fcsv = os.path.join(args.output_dir, "summary_n%d.csv" % n)
        fmeta = os.path.join(args.output_dir, "meta_n%d.json" % n)
        if os.path.exists(fcsv) and os.path.exists(fmeta):
            dfs_by_n[n] = pd.read_csv(fcsv)
            metas[n] = json.load(open(fmeta))
        else:
            missing.append(n)
    if missing:
        print("\n[warn] no results yet for n = %s; the .tex will cover %s only.  "
              "Re-run those sizes, then --tex-only to assemble everything."
              % (missing, sorted(dfs_by_n)))
    if not dfs_by_n:
        print("Nothing to assemble.")
        return

    done = sorted(dfs_by_n)
    first = done[0]
    tex_blocks = []
    for n in done:
        mt = metas[n]
        ov_desc = ("all $n - b + 1$ of them, no thinning" if mt["ci_step"] == 1
                   else "thinned with step %d out of $n - b + 1 = %s$"
                        % (mt["ci_step"], fmt_n(n - mt["b"] + 1)))
        caption = ("Subsampling results (reparameterized) for $n = %s$ "
                   "($b = %s$, $N_b = %d$, $M = %d$)"
                   % (fmt_n(n), fmt_n(mt["b"]), mt["Nb"], mt["M"]))
        if n == first:
            notes = (
                "All results are in the reparameterized space "
                r"$(\rho_1, \rho_2, \sigma, \pi_1, \alpha)$.  The full-sample "
                r"estimator $\hat{\vartheta}_n$ and every block estimator "
                r"$\hat{\vartheta}^{(i)}_b$ are the same functional: identical "
                "criterion, grid, starting values, bounds and optimizer tolerances, "
                r"as required by the subsampling theory of \cite{PR94}.  "
                r"Dev.~Mean, Dev.~Std, Skew and Kurt are computed on the "
                r"non-overlapping block deviations $Z^{(i)}_b$ pooled across all "
                "$M = " + str(mt["M"]) + "$ replications (" + fmt_n(mt["n_pooled"]) +
                " values per parameter).  "
                r"SW and JB rejection rates are per-replication rejection "
                r"frequencies at the 5\% level.  qCov and nCov denote "
                "quantile-based and normal-approximation CI coverage; confidence "
                "intervals use the " + fmt_n(mt["n_ov"]) + " overlapping blocks (" +
                ov_desc + "), while the normality diagnostics use the "
                "non-overlapping blocks.")
        else:
            notes = r"See notes to Table~\ref{tab:subsampling_n%d}." % first
        tex_blocks.append(format_detail_table(
            dfs_by_n[n], n, mt["b"], mt["M"], caption=caption,
            label="tab:subsampling_n%d" % n, notes=notes))

        if n == first:
            fcap = ("For each parameter (rows), we display: (left) histogram with "
                    "standard normal overlay, (center) Q-Q plot against standard "
                    "normal quantiles, and (right) kernel density estimate compared "
                    "to the standard normal density.")
        else:
            fcap = r"See caption to Figure~\ref{fig:diagnostics_n%d} for details." % first
        tex_blocks.append(format_figure(
            image_path="%s/diagnostics_n%d.png" % (args.output_dir, n),
            caption=("Diagnostic plots for scaled subsample deviations "
                     "(reparameterized) at $n = %s$. " % fmt_n(n)) + fcap,
            label="fig:diagnostics_n%d" % n))

    summary_tex = format_summary_table(
        dfs_by_n, PARAM_ORDER,
        caption="Summary of unrestricted subsampling results (reparameterized space)",
        label="tab:subsampling_summary",
        notes=(r"Avg.~Est.\ and Std.~Est.\ are the mean and standard deviation of "
               r"$\hat{\vartheta}_n$ over $M=" + str(M) + r"$ replications.  "
               r"Dev.~Mean, Skew, and Kurt are the mean, skewness and excess kurtosis "
               r"of the non-overlapping block scaled deviations $Z^{(i)}_b$, pooled "
               r"across replications.  qCov and nCov denote the average coverage of "
               r"quantile-based and normal-approximation confidence intervals.  The "
               r"full-sample and block estimators are the same functional, evaluated "
               r"with identical starting values and optimizer tolerances."))

    design_tex = format_design_table(done, args.b_exponent, M)

    full_tex = "\n\n\n".join([design_tex, summary_tex] + tex_blocks)
    tex_path = os.path.join(args.output_dir, "s22_reparam.tex")
    with open(tex_path, "w") as f:
        f.write(full_tex + "\n")

    print("\nLaTeX written to %s  (sample sizes: %s)" % (tex_path, done))
    print("Figures written to %s/diagnostics_n<n>.png" % args.output_dir)
    print("Done in %.1f min." % ((time.time() - t_start) / 60))


if __name__ == "__main__":
    main()
