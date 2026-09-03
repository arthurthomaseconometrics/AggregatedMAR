#!/usr/bin/env python3
r"""
Reproduction of Table S.1 ("Monte Carlo estimation accuracy") of the Online
Supplement, Section S.1 (\label{app:mc_accuracy}), consistent with the
minimum-distance estimator (MDE) as specified in main_v23.tex.

Data-generating process (Section 4.1 / eq. (ar1sim) of main_v23.tex)
----------------------------------------------------------------------
    Y_t   = X_{1,t} + X_{2,t},          X_{j,t} = varsigma_j * W_{j,t}
    W_{j,t} = psi_j * W_{j,t+1} + eps_{j,t},   eps_{j,t} ~iid  S(alpha, beta, 1, 0)

for j = 1, 2, with psi_j in (0, 1) (purely anticipative / noncausal AR(1)
components).  Following the paper, sigma = 1.6, pi_1 = 7/16, pi_2 = 9/16, so
that the combined scale parameters are varsigma_1 = sigma*pi_1 = 0.7 and
varsigma_2 = sigma*pi_2 = 0.9, with psi_1 = 0.8, psi_2 = 0.3.  Three
distributional settings are estimated:

    Panel A  Cauchy         S1S,  alpha = 1   (fixed), beta = 0   (fixed)
    Panel B  symmetric      SaS,  alpha = 1.5 (estimated), beta = 0   (fixed)
    Panel C  general        GaS,  alpha = 1.5 (estimated), beta = 0.3 (estimated)

Weighting: Assumption C2 of main_v23.tex / criterion (2.14)
----------------------------------------------------------------------
The archived table (Latex/resultats*, and the footnote of the currently
compiled online_supplement2.tex) was produced with UNIFORM weights on a
bounded grid, i.e. the estimator minimized `mean(|ecf - tcf|^2)`.  That is
not the criterion covered by Proposition 2.1 of main_v23.tex: Assumption C2
imposes the exponential weight

    w(u, v) = (kappa / pi) * exp(-kappa (u^2 + v^2)),   kappa > 0,

and it is this weight that appears in the proofs (Lemma 2.1, Lemma 4.1,
Proposition 2.1/4.1).  This script discretizes the MDE criterion

    Q_n(theta) = \iint |phi_n(u,v) - phi(u,v;theta)|^2 w(u,v) du dv

with the shared quadrature module `cf_weights.py` already used to regenerate
the subsampling sections of the supplement (S.2.2-S.2.6, see
`run_supplement_rerun.py`), so that Table S.1 is now estimated under the
SAME criterion whose asymptotics are established in the paper.  Pass
`--uniform-weights` to fall back to the legacy uniform-weight criterion
(kappa=None in `cf_weights.quadrature_weights`) for comparison with the
archived numbers.

Reproducibility
----------------------------------------------------------------------
  * All randomness is seeded deterministically from `--seed` (default 42):
    replication r, sample size T, panel p all enter the per-replication
    seed, so any subset of the run can be regenerated in isolation.
  * `run_config.json` records every setting (DGP, grid, kappa, seeds,
    n_starts, package versions) alongside the results.
  * Raw per-replication estimates (`mc_raw_<panel>.csv`) and the summary
    statistics (`mc_summary.csv`) are both saved, in addition to the
    booktabs LaTeX table (`table_s1_mc_accuracy.tex`) formatted exactly as
    Table S.1 of online_supplement2.tex.

Usage
----------------------------------------------------------------------
    python table_s1_mc_accuracy.py                   # full paper design (slow)
    python table_s1_mc_accuracy.py --quick            # fast smoke test
    python table_s1_mc_accuracy.py --n-jobs 8 --seed 42
    python table_s1_mc_accuracy.py --uniform-weights  # legacy criterion
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from scipy import optimize, stats

from cf_weights import (
    ALPHA_BOUNDS,
    KAPPA_DEFAULT,
    N_GRID_DEFAULT,
    U_MAX_DEFAULT,
    alpha_admissible,
    default_grid,
    is_alpha_one,
    quadrature_weights,
)

stats.levy_stable.parameterization = "S1"

TRUE_PSI = (0.8, 0.3)
TRUE_VARSIGMA = (0.7, 0.9)          # sigma=1.6, pi=(7/16, 9/16) => varsigma = sigma*pi
J = 2

PSI_BOUNDS = (0.02, 0.97)
VARSIGMA_BOUNDS = (0.02, 5.0)
BETA_BOUNDS = (-0.97, 0.97)

SAMPLE_SIZES = (250, 500, 1000)
N_REPLICATIONS = 1000
BURN_IN = 300                       # >> log(1e-12)/log(0.97) ~ 750 not needed: true psi <= 0.8


# =============================================================================
# Panel specification
# =============================================================================

@dataclass(frozen=True)
class PanelSpec:
    key: str
    label: str
    alpha_true: float
    beta_true: float
    estimate_alpha: bool
    estimate_beta: bool

    @property
    def param_names(self) -> List[str]:
        names = ["psi1", "varsigma1", "psi2", "varsigma2"]
        if self.estimate_alpha:
            names.append("alpha")
        if self.estimate_beta:
            names.append("beta")
        return names


PANELS: Tuple[PanelSpec, ...] = (
    PanelSpec("cauchy", "Cauchy ($\\mathcal{S}1\\mathcal{S}$, $\\alpha = 1$, $\\beta = 0$)",
              alpha_true=1.0, beta_true=0.0, estimate_alpha=False, estimate_beta=False),
    PanelSpec("sas", "Symmetric $\\alpha$-stable ($\\mathcal{S}\\alpha\\mathcal{S}$, $\\alpha = 1.5$, $\\beta = 0$)",
              alpha_true=1.5, beta_true=0.0, estimate_alpha=True, estimate_beta=False),
    PanelSpec("gas", "General $\\alpha$-stable ($\\mathcal{G}\\alpha\\mathcal{S}$, $\\alpha = 1.5$, $\\beta = 0.3$)",
              alpha_true=1.5, beta_true=0.3, estimate_alpha=True, estimate_beta=True),
)


# =============================================================================
# Data-generating process
# =============================================================================

def simulate_component(T: int, psi: float, alpha: float, beta: float,
                        burn_in: int, seed: int) -> np.ndarray:
    """
    Unit-scale (varsigma=1) anticipative alpha-stable AR(1):
    W_t = psi*W_{t+1} + eps_t, solved backward from an artificial boundary
    W[T+burn_in-1] = 0.  The boundary error decays like psi^k moving away
    from it; we KEEP the T points farthest from the boundary (the first T
    of the T+burn_in solved values) and discard the burn_in points nearest
    the artificial boundary.
    """
    n = T + burn_in
    eps = stats.levy_stable.rvs(alpha=alpha, beta=beta, size=n, random_state=seed)
    W = np.empty(n)
    W[-1] = 0.0
    for t in range(n - 2, -1, -1):
        W[t] = psi * W[t + 1] + eps[t]
    return W[:T]


def simulate_aggregate(T: int, psi: Sequence[float], varsigma: Sequence[float],
                        alpha: float, beta: float, burn_in: int,
                        seed: int) -> np.ndarray:
    Y = np.zeros(T)
    for j in range(J):
        Wj = simulate_component(T, psi[j], alpha, beta, burn_in, seed=seed + j * 7919)
        Y += varsigma[j] * Wj
    return Y


# =============================================================================
# Characteristic functions (joint CF of the pair (Y_t, Y_{t+1}))
# =============================================================================

def _component_log_cf(U: np.ndarray, V: np.ndarray, psi: float, varsigma: float,
                       alpha: float, beta: float) -> np.ndarray:
    """
    log E[exp(i(u*W_t + v*W_{t+1}))] for a SCALED component varsigma*W, with
    W the unit-scale anticipative AR(1).  Derivation: W_t = sum_{k>=0}
    psi^k eps_{t+k}, so u*W_t + v*W_{t+1} = u*eps_t + sum_{k>=1} (u*psi+v)
    psi^{k-1} eps_{t+k}; summing the log-CF of each iid term and using
    sum_{k>=1} psi^{alpha(k-1)} = 1/(1-psi^alpha) gives the closed form
    below.  Verified numerically against the empirical CF of simulated
    paths (see tests at the bottom of this file / --self-test).
    """
    C = U * psi + V
    if is_alpha_one(alpha):
        if beta != 0.0:
            raise NotImplementedError(
                "alpha=1 with beta!=0 requires the log|.|-drift term of "
                "cf_weights.logcf_alpha1_component; not needed for Table S.1 "
                "(Panel A has beta=0 fixed)."
            )
        return -varsigma * (np.abs(U) + np.abs(C) / (1.0 - psi))
    tan_t = np.tan(np.pi * alpha / 2.0)
    term_u = np.abs(U) ** alpha * (1.0 - 1j * beta * np.sign(U) * tan_t)
    term_c = np.abs(C) ** alpha * (1.0 - 1j * beta * np.sign(C) * tan_t)
    return -(varsigma ** alpha) * (term_u + term_c / (1.0 - psi ** alpha))


def tcf_grid(u_grid: np.ndarray, v_grid: np.ndarray, psi: Sequence[float],
             varsigma: Sequence[float], alpha: float, beta: float) -> np.ndarray:
    """Vectorized theoretical CF of (Y_t, Y_{t+1}) over the full (u, v) grid."""
    U, V = np.meshgrid(u_grid, v_grid, indexing="ij")
    log_phi = np.zeros(U.shape, dtype=complex)
    for j in range(J):
        log_phi += _component_log_cf(U, V, psi[j], varsigma[j], alpha, beta)
    return np.exp(log_phi)


def ecf_grid(Y: np.ndarray, u_grid: np.ndarray, v_grid: np.ndarray) -> np.ndarray:
    """Empirical CF of the pair (Y_t, Y_{t+1}) over the full (u, v) grid."""
    Yt, Ytp1 = Y[:-1], Y[1:]
    phase = u_grid[:, None, None] * Yt[None, None, :] + v_grid[None, :, None] * Ytp1[None, None, :]
    return np.mean(np.exp(1j * phase), axis=2)


# =============================================================================
# Parameter (un)packing and MDE objective
# =============================================================================

def unpack(theta: np.ndarray, spec: PanelSpec) -> Dict[str, float]:
    p: Dict[str, float] = {}
    p["psi1"], p["varsigma1"], p["psi2"], p["varsigma2"] = theta[:4]
    idx = 4
    if spec.estimate_alpha:
        p["alpha"] = theta[idx]; idx += 1
    else:
        p["alpha"] = spec.alpha_true
    if spec.estimate_beta:
        p["beta"] = theta[idx]
    else:
        p["beta"] = spec.beta_true
    return p


def bounds_for(spec: PanelSpec) -> List[Tuple[float, float]]:
    b = [PSI_BOUNDS, VARSIGMA_BOUNDS, PSI_BOUNDS, VARSIGMA_BOUNDS]
    if spec.estimate_alpha:
        b.append(ALPHA_BOUNDS)
    if spec.estimate_beta:
        b.append(BETA_BOUNDS)
    return b


def mde_objective(theta: np.ndarray, spec: PanelSpec, ecf: np.ndarray,
                   u_grid: np.ndarray, v_grid: np.ndarray, W: np.ndarray) -> float:
    p = unpack(theta, spec)
    if spec.estimate_alpha and not alpha_admissible(p["alpha"], estimated=True):
        return 1e10
    if spec.estimate_beta and not (-0.99 < p["beta"] < 0.99):
        return 1e10
    tcf = tcf_grid(u_grid, v_grid, (p["psi1"], p["psi2"]),
                    (p["varsigma1"], p["varsigma2"]), p["alpha"], p["beta"])
    return float(np.sum(W * np.abs(ecf - tcf) ** 2))


def sort_by_psi_desc(p: Dict[str, float]) -> Dict[str, float]:
    """Break the label-switching symmetry: component 1 is the one with the
    larger psi, matching the DGP convention psi_1 = 0.8 > psi_2 = 0.3."""
    q = dict(p)
    if q["psi1"] < q["psi2"]:
        q["psi1"], q["psi2"] = q["psi2"], q["psi1"]
        q["varsigma1"], q["varsigma2"] = q["varsigma2"], q["varsigma1"]
    return q


def _starting_points(spec: PanelSpec, n_starts: int, rng: np.random.Generator,
                      bounds: List[Tuple[float, float]]) -> List[np.ndarray]:
    base = [0.7, 0.7, 0.4, 0.9]
    swapped = [0.4, 0.9, 0.7, 0.7]
    tail = []
    if spec.estimate_alpha:
        tail.append(1.4)
    if spec.estimate_beta:
        tail.append(0.1)
    starts = [np.array(base + tail), np.array(swapped + tail)]
    for _ in range(max(0, n_starts - 2)):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))
    return starts


def estimate_one(Y: np.ndarray, spec: PanelSpec, u_grid: np.ndarray, v_grid: np.ndarray,
                  W: np.ndarray, n_starts: int, seed: int) -> Dict[str, object]:
    ecf = ecf_grid(Y, u_grid, v_grid)
    bounds = bounds_for(spec)
    rng = np.random.default_rng(seed)
    best_fun, best_x, best_success = np.inf, None, False
    for x0 in _starting_points(spec, n_starts, rng, bounds):
        res = optimize.minimize(
            mde_objective, x0, args=(spec, ecf, u_grid, v_grid, W),
            method="L-BFGS-B", bounds=bounds, options={"maxiter": 300},
        )
        if res.fun < best_fun:
            best_fun, best_x, best_success = res.fun, res.x, bool(res.success)

    p_hat = unpack(best_x, spec)
    p_hat = sort_by_psi_desc(p_hat)
    p_hat["objective_value"] = best_fun
    p_hat["success"] = best_success
    return p_hat


# =============================================================================
# One Monte Carlo replication
# =============================================================================

def single_replication(rep_idx: int, T: int, spec: PanelSpec, seed: int,
                        u_grid: np.ndarray, v_grid: np.ndarray, W: np.ndarray,
                        n_starts: int, burn_in: int) -> Dict[str, object]:
    out: Dict[str, object] = {"rep": rep_idx, "T": T}
    try:
        Y = simulate_aggregate(T, TRUE_PSI, TRUE_VARSIGMA, spec.alpha_true,
                                spec.beta_true, burn_in, seed=seed)
        est = estimate_one(Y, spec, u_grid, v_grid, W, n_starts, seed=seed + 1)
        out.update({f"{k}_est": v for k, v in est.items()
                     if k not in ("objective_value", "success")})
        out["objective_value"] = est["objective_value"]
        out["success"] = est["success"]
    except Exception as exc:  # keep the MC loop alive; report failures explicitly
        out["success"] = False
        out["error"] = repr(exc)
    return out


# =============================================================================
# Panel-level Monte Carlo study
# =============================================================================

def run_panel(spec: PanelSpec, sample_sizes: Sequence[int], n_replications: int,
              u_grid: np.ndarray, v_grid: np.ndarray, W: np.ndarray,
              n_starts: int, burn_in: int, seed_base: int, n_jobs: int,
              verbose: bool) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for T in sample_sizes:
        t0 = time.time()
        seeds = [seed_base + hash((spec.key, T, r)) % 1_000_000_000
                 for r in range(n_replications)]
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(single_replication)(r, T, spec, seeds[r], u_grid, v_grid, W,
                                         n_starts, burn_in)
            for r in range(n_replications)
        )
        rows.extend(results)
        if verbose:
            n_ok = sum(1 for r in results if r.get("success"))
            print(f"    [{spec.key}] T={T:>5d}: {n_ok}/{n_replications} converged "
                  f"({time.time() - t0:.1f}s)")
    return pd.DataFrame(rows)


def true_value(spec: PanelSpec, param: str) -> float:
    return {"psi1": TRUE_PSI[0], "psi2": TRUE_PSI[1],
            "varsigma1": TRUE_VARSIGMA[0], "varsigma2": TRUE_VARSIGMA[1],
            "alpha": spec.alpha_true, "beta": spec.beta_true}[param]


def compute_summary(df: pd.DataFrame, spec: PanelSpec) -> pd.DataFrame:
    rows = []
    for T in sorted(df["T"].unique()):
        sub = df[(df["T"] == T) & (df["success"] == True)]  # noqa: E712
        for param in spec.param_names:
            col = f"{param}_est"
            vals = sub[col].dropna().to_numpy()
            if vals.size == 0:
                continue
            tv = true_value(spec, param)
            rows.append({
                "panel": spec.key, "Parameter": param, "True": tv, "T": T,
                "Bias": float(np.mean(vals - tv)),
                "RMSE": float(np.sqrt(np.mean((vals - tv) ** 2))),
                "MRE": float(np.mean(np.abs(vals - tv) / abs(tv))),
                "N_valid": int(vals.size),
            })
    return pd.DataFrame(rows)


# =============================================================================
# LaTeX table (format matches Table S.1 of online_supplement2.tex)
# =============================================================================

_SYMBOL = {
    "psi1": r"$\psi_1$", "psi2": r"$\psi_2$",
    "varsigma1": r"$\varsigma_{1}$", "varsigma2": r"$\varsigma_{2}$",
    "alpha": r"$\alpha$", "beta": r"$\beta$",
}
_PANEL_LETTER = {"cauchy": "A", "sas": "B", "gas": "C"}


def _fmt(x: float, dp: int = 3) -> str:
    s = ("" if x >= 0 else r"$-$") + format(abs(x), f".{dp}f")
    return s


def format_table_s1(summaries: Dict[str, pd.DataFrame], sample_sizes: Sequence[int],
                     n_replications: int, kappa: Optional[float]) -> str:
    lines = [
        r"\begin{table}[!h]",
        r"\centering",
        r"\caption{Monte Carlo estimation accuracy for $\alpha$-stable AR(1) "
        f"aggregates ({n_replications:,}".replace(",", "{,}") + " replications)}",
        r"\label{tab:mc_accuracy_full}",
        r"\begin{tabular}{ccccccccccc}",
        r"\toprule",
        r"& & " + " & ".join(
            r"\multicolumn{3}{c}{$T = " + str(T) + r"$}" for T in sample_sizes
        ) + r" \\",
    ]
    c = 3
    cmid = []
    for _ in sample_sizes:
        cmid.append(r"\cmidrule(lr){" + f"{c}-{c + 2}" + "}")
        c += 3
    lines.append(" ".join(cmid))
    lines.append(r"$\theta$ & True & " + " & ".join(["Bias", "RMSE", "MRE"] * len(sample_sizes)) + r" \\")
    lines.append(r"\midrule")

    for spec in PANELS:
        summ = summaries[spec.key]
        letter = _PANEL_LETTER[spec.key]
        lines.append(
            r"\multicolumn{11}{l}{\textit{Panel " + letter + ": " + spec.label + r"}} \\[3pt]"
        )
        for i, param in enumerate(spec.param_names):
            pdata = summ[summ["Parameter"] == param]
            if pdata.empty:
                continue
            true_val = pdata["True"].iloc[0]
            row = [_SYMBOL.get(param, param), format(true_val, ".3f")]
            for T in sample_sizes:
                td = pdata[pdata["T"] == T]
                if td.empty:
                    row += ["--", "--", "--"]
                else:
                    row += [_fmt(td["Bias"].iloc[0]), _fmt(td["RMSE"].iloc[0]),
                            _fmt(td["MRE"].iloc[0])]
            sep = r" \\[6pt]" if (i == len(spec.param_names) - 1 and spec.key != PANELS[-1].key) else r" \\"
            lines.append(" & ".join(row) + sep)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{flushleft}")
    weight_txt = (
        r"uniform weights on a bounded grid (legacy criterion)" if kappa is None else
        rf"the Assumption~C2 weight $w(u,v)=(\kappa/\pi)\exp(-\kappa(u^2+v^2))$ with $\kappa={kappa:g}$"
    )
    lines.append(
        r"\footnotesize\textit{Notes:} True parameter values are $\psi_1 = 0.8$, "
        r"$\psi_2 = 0.3$, $\varsigma_1 = 0.7$, $\varsigma_2 = 0.9$.  MRE denotes the "
        r"mean relative error $\mathbb{E}[|\hat\theta - \theta_0|/|\theta_0|]$.  "
        f"Estimation minimizes the discretized criterion (2.14) of main\\_v23.tex using {weight_txt}."
    )
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =============================================================================
# Orchestration
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-sizes", type=int, nargs="+", default=list(SAMPLE_SIZES))
    ap.add_argument("--n-replications", type=int, default=N_REPLICATIONS)
    ap.add_argument("--n-starts", type=int, default=10, help="multi-start restarts per replication")
    ap.add_argument("--burn-in", type=int, default=BURN_IN)
    ap.add_argument("--kappa", type=float, default=KAPPA_DEFAULT,
                     help="Assumption C2 weight decay (default: cf_weights.KAPPA_DEFAULT)")
    ap.add_argument("--uniform-weights", action="store_true",
                     help="legacy mode: uniform weights (kappa=None), for comparison "
                          "with the archived table")
    ap.add_argument("--n-grid", type=int, default=N_GRID_DEFAULT)
    ap.add_argument("--u-max", type=float, default=U_MAX_DEFAULT)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="results_table_s1")
    ap.add_argument("--panels", nargs="+", choices=[p.key for p in PANELS],
                     default=[p.key for p in PANELS])
    ap.add_argument("--quick", action="store_true",
                     help="smoke test: sample_sizes=[100], n_replications=10, n_starts=2")
    args = ap.parse_args()

    sample_sizes = args.sample_sizes
    n_replications = args.n_replications
    n_starts = args.n_starts
    if args.quick:
        sample_sizes, n_replications, n_starts = [100], 10, 2

    kappa = None if args.uniform_weights else args.kappa
    u_grid, v_grid = default_grid(n_grid=args.n_grid, u_max=args.u_max)
    W = quadrature_weights(u_grid, v_grid, kappa=kappa)

    os.makedirs(args.output_dir, exist_ok=True)

    config = dict(
        sample_sizes=sample_sizes, n_replications=n_replications, n_starts=n_starts,
        burn_in=args.burn_in, kappa=kappa, n_grid=args.n_grid, u_max=args.u_max,
        seed=args.seed, panels=args.panels,
        true_psi=TRUE_PSI, true_varsigma=TRUE_VARSIGMA,
        psi_bounds=PSI_BOUNDS, varsigma_bounds=VARSIGMA_BOUNDS,
        alpha_bounds=ALPHA_BOUNDS, beta_bounds=BETA_BOUNDS,
        python_version=sys.version, platform=platform.platform(),
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        pandas_version=pd.__version__,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    print("=" * 78)
    print("  Table S.1 -- Monte Carlo estimation accuracy (main_v23 / online_supplement2)")
    print("=" * 78)
    print(f"  Weights      : {'uniform (legacy)' if kappa is None else f'Assumption C2, kappa={kappa}'}")
    print(f"  Grid         : {args.n_grid}x{args.n_grid} on [-{args.u_max}, {args.u_max}]^2")
    print(f"  Sample sizes : {sample_sizes}")
    print(f"  Replications : {n_replications}   |   multi-start: {n_starts}")
    print(f"  Panels       : {args.panels}")
    print(f"  Output       : {args.output_dir}/")

    summaries: Dict[str, pd.DataFrame] = {}
    t_start = time.time()
    for spec in PANELS:
        if spec.key not in args.panels:
            continue
        print(f"\n  Panel {_PANEL_LETTER[spec.key]} ({spec.key}) ...")
        df_raw = run_panel(spec, sample_sizes, n_replications, u_grid, v_grid, W,
                            n_starts, args.burn_in, args.seed, args.n_jobs, verbose=True)
        df_raw.to_csv(os.path.join(args.output_dir, f"mc_raw_{spec.key}.csv"), index=False)
        summaries[spec.key] = compute_summary(df_raw, spec)

    if len(summaries) == len(PANELS):
        combined = pd.concat(summaries.values(), ignore_index=True)
        combined.to_csv(os.path.join(args.output_dir, "mc_summary.csv"), index=False)
        tex = format_table_s1(summaries, sample_sizes, n_replications, kappa)
        tex_path = os.path.join(args.output_dir, "table_s1_mc_accuracy.tex")
        with open(tex_path, "w") as f:
            f.write(tex + "\n")
        print(f"\n  LaTeX table written to {tex_path}")
    else:
        print("\n  Skipped combined table/summary (only a subset of panels was run; "
              "use --panels cauchy sas gas together to regenerate the full table).")

    print(f"\n  Done in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
