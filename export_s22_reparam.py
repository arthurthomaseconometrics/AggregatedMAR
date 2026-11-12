#!/usr/bin/env python3
r"""Tables S.3-S.9 and Figures S.1-S.5 of the Online Supplement of de Truchis, Fries and
Thomas (2026), Section S.3: subsampling verification of asymptotic normality.

Simulates M aggregates of two anticipative alpha-stable AR(1) components
(psi = (0.8, 0.3), sigma = 1.6, pi_1 = 0.4375, alpha = 1.5, beta = 0) for each
sample size n, estimates the model on the full sample and on subsamples of size
b = floor(n^{2/3}) with the same minimum distance estimator (Assumption C2 weight,
kappa = 1), and reports, in the reparameterized space (psi_1, psi_2, sigma, pi_1,
alpha), the normality diagnostics of the non-overlapping block deviations and the
coverage of quantile-based and normal confidence intervals (overlapping blocks,
thinned to about 2,000 per replication).  Engine: subsampling_reparam.py.

Writes, in <output-dir>:
    summary_n<n>.csv, meta_n<n>.json, diagnostics_n<n>.png   (one per sample size)
    s22_reparam.tex   Table S.3 (design), Table S.4 (summary), Tables S.5-S.9 and
                      the figure environments of Figures S.1-S.5
    run_config_n<...>.json

Usage
    python export_s22_reparam.py --sample-sizes 500 1000 10000 100000 500000 --M 200
    python export_s22_reparam.py ... --skip-existing     # restart an interrupted run
    python export_s22_reparam.py ... --tex-only          # rebuild the .tex from the CSVs
    python export_s22_reparam.py --quick                 # smoke test, ~30 s
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
from subsampling_reparam import (N_STARTS_DEFAULT, OUTLIER_Z_THRESHOLD, OV_BLOCKS_TARGET,
                                 ROBUST_TRIM_FRAC, _EngineReparam)

OUTPUT_DIR_DEFAULT = "resultats_reparam"
ROBUST_TRIM_FRAC_PCT = int(round(ROBUST_TRIM_FRAC * 100))
OUTLIER_Z_THRESHOLD_STR = ("%g" % OUTLIER_Z_THRESHOLD)
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
    print("Critere MDE (Assumption C2):")
    check_grid(eng.u_grid, eng.v_grid, args.kappa)

    config = dict(
        section="Online Supplement, Section S.3.2", sample_sizes=sample_sizes, M=M,
        b_exponent=args.b_exponent, kappa=args.kappa, seed=args.seed,
        true_params=true_params, n_cores=args.n_cores,
        n_starts=args.n_starts, ci_step=args.ci_step,
        ov_blocks_target=OV_BLOCKS_TARGET,
        optimizer="L-BFGS-B", mde_gtol=MDE_GTOL, mde_ftol=MDE_FTOL,
        mde_maxiter=MDE_MAXITER,
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
        has_robust = "Skew.Trim" in dfs_by_n[n].columns and "Outlier%" in dfs_by_n[n].columns
        if n == first:
            notes = (
                "All results are in the reparameterized space "
                r"$(\psi_1, \psi_2, \sigma, \pi_1, \alpha)$.  The full-sample "
                r"estimator $\hat{\vartheta}_n$ and every block estimator "
                r"$\hat{\vartheta}^{(i)}_b$ are the same functional: identical "
                "criterion, grid, starting values, bounds and optimizer tolerances, "
                r"as required by the subsampling theory of \cite{PR94}.  "
                r"Dev.~Mean, Dev.~Std, Skew and Kurt are computed on the "
                r"non-overlapping block deviations $Z^{(i)}_b$ pooled across all "
                "$M = " + str(mt["M"]) + "$ replications (" + fmt_n(mt["n_pooled"]) +
                " values per parameter).  "
                r"At large $n$, a small share of blocks reach the boundary of the "
                r"compact parameter space $\Theta$ (a genuine finite-$b$ feature of "
                r"subsampling, not a search failure: confirmed by dense multi-start "
                r"re-optimization) and, because $Z^{(i)}_b$ scales with $\sqrt{b}$, "
                r"a fixed absolute distance to the boundary maps into an "
                r"increasingly extreme standardized value as $b$ grows, even as the "
                r"share of affected blocks falls -- this inflates raw Skew and Kurt "
                r"without bearing on the asymptotic normality of $\hat\vartheta_n$ "
                r"itself, established on the full sample by Proposition~2.1.  ")
            if has_robust:
                notes += (
                    r"Skew$_{\text{trim}}$ is the skewness on the sample trimmed "
                    rf"symmetrically by {ROBUST_TRIM_FRAC_PCT}\% on each side (valid "
                    r"under trimming, verified on clean Gaussian data); Out.\% is the "
                    r"share of pooled deviations, standardized by median/MAD (robust to "
                    r"the same contamination), exceeding "
                    rf"{OUTLIER_Z_THRESHOLD_STR}$\,\sigma$ -- reported in place of a "
                    r"trimmed Kurt, which is mechanically biased toward negative values "
                    r"by the trimming itself (verified: $\approx -1.0$ on trimmed clean "
                    r"$\mathcal{N}(0,1)$ data at this trim fraction, not $\approx 0$).  ")
            notes += (
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
