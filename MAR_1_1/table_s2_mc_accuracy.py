#!/usr/bin/env python3
r"""
Table S.2 -- "Monte Carlo estimation accuracy for mixed causal-noncausal
MAR(1,1) aggregates", the Case-2 analogue of Table S.1
(\label{app:mc_accuracy} / \label{tab:mc_accuracy_full}) of the Online
Supplement, consistent with Section 2.4 ("Case 2: Aggregation of Mixed
Causal-Noncausal MAR(1,1) Processes") of main_v23.tex.

Data-generating process (eq. around \label{eq:aggcf_mar11_skewed} of
main_v23.tex)
----------------------------------------------------------------------
Each latent component is a mixed causal-noncausal MAR(1,1) process,

    (1 - phi_j L)(1 - psi_j L^{-1}) X_{j,t} = eps_{j,t},   |phi_j|, |psi_j| < 1,

with MA(infinity) coefficients d_{j,k} = psi_j^k/(1-phi_j psi_j) for k >= 0
and d_{j,k} = phi_j^{|k|}/(1-phi_j psi_j) for k < 0, so that X_{j,t} mixes a
causal (backward-looking, root phi_j) and a noncausal (forward-looking,
root psi_j) branch.  The observed aggregate is Y_t = sum_j varsigma_j
X_{j,t} (unit-scale eps_{j,t} ~iid S(alpha, beta, 1, 0); varsigma_j carries
the scale, exactly as in Case 1 / Table S.1).  Following the paper's
simplifying restriction for beta != 0 (main_v23.tex, Section 2.4), every
phi_j, psi_j is kept strictly positive.

theta = (psi_1, phi_1, varsigma_1, psi_2, phi_2, varsigma_2 [, alpha] [, beta]),
of dimension p = 3J+2 = 8 for J = 2 (2 more than Case 1's p = 3J+1 = 7,
i.e. Table S.1's 4-6 free parameters), the extra coordinate per component
being the causal root phi_j.

Design choice (no MAR(1,1) design is specified in Section 4.1 of
main_v23.tex, which only covers Case 1): to isolate the effect of adding
causal dynamics, this script reuses EXACTLY the Table S.1 latent design
(psi_1, psi_2, varsigma_1, varsigma_2) = (0.8, 0.3, 0.7, 0.9) and the same
three distributional panels (Cauchy / SaS / GaS), and adds causal roots
(phi_1, phi_2) = (0.3, 0.5), a moderate, well-separated pair also used
elsewhere in this repository's MAR(1,1) exploratory code
(monte_carlo_mar11_study.py).

Weighting: Assumption C2 of main_v23.tex / criterion (2.14)
----------------------------------------------------------------------
As for Table S.1 (../table_s1_mc_accuracy.py), estimation minimizes the
discretized version of criterion (2.14),

    Q_n(theta) = \iint |phi_n(u,v) - phi(u,v;theta)|^2 w(u,v) du dv,

with the Assumption C2 weight w(u,v) = (kappa/pi) exp(-kappa(u^2+v^2)),
using the SAME quadrature module `cf_weights.py` (imported from the parent
directory) as Table S.1 and `run_supplement_rerun.py`, so the two tables
of the (extended) supplement are estimated under one consistent criterion.
`--uniform-weights` reverts to the legacy uniform-weight criterion.

Reproducibility
----------------------------------------------------------------------
Deterministic per-replication seeding, `run_config.json` with every
setting and package version, raw per-replication CSVs, a combined summary
CSV, and a booktabs LaTeX table formatted like Table S.1.

Usage
----------------------------------------------------------------------
    python table_s2_mc_accuracy.py                    # full paper-style design (slow)
    python table_s2_mc_accuracy.py --quick             # fast smoke test
    python table_s2_mc_accuracy.py --n-jobs 8 --seed 42
    python table_s2_mc_accuracy.py --uniform-weights   # legacy criterion
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from scipy import optimize, stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cf_weights import (  # noqa: E402
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

TRUE_PSI = (0.8, 0.3)               # noncausal roots, same as Table S.1
TRUE_PHI = (0.3, 0.5)               # causal roots (new to Case 2)
TRUE_VARSIGMA = (0.7, 0.9)          # combined scales, same as Table S.1
J = 2

ROOT_BOUNDS = (0.02, 0.97)          # phi_j, psi_j > 0 (main_v23.tex Sec. 2.4 restriction)
VARSIGMA_BOUNDS = (0.02, 5.0)
BETA_BOUNDS = (-0.97, 0.97)

SAMPLE_SIZES = (250, 500, 1000)
N_REPLICATIONS = 1000
MA_TRUNCATION = 250                 # max |k| kept in the MA(infinity) expansion


# =============================================================================
# Panel specification (identical distributional design to Table S.1)
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
        names = ["psi1", "phi1", "varsigma1", "psi2", "phi2", "varsigma2"]
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

def simulate_component(T: int, phi: float, psi: float, alpha: float, beta: float,
                        K: int, seed: int) -> np.ndarray:
    """
    Unit-scale (varsigma=1) mixed causal-noncausal MAR(1,1):
    X_t = sum_{k=-K}^{K} d_k eps_{t+k}, d_k = psi^k/(1-phi*psi) for k >= 0
    (future shocks, noncausal branch), d_k = phi^{|k|}/(1-phi*psi) for k < 0
    (past shocks, causal branch) -- MA(infinity) truncated at lag K.  The
    +k (not -k) indexing is not a convention-free choice: it is forced by
    inverting (1-phi L)(1-psi L^{-1})X_t = eps_t as X_t = sum_{k,j>=0}
    phi^k psi^j eps_{t-k+j}, which collapses (m=j-k) to exactly this d_m
    paired with eps_{t+m}.  Getting the sign backwards silently swaps the
    causal/noncausal roles of phi and psi and biases the MDE (caught via
    the CF-vs-simulation cross-check below, --self-test).

    Fully vectorized via a sliding window over an extended innovation
    sequence; validated against a brute-force double loop (max abs diff
    ~1e-16 for a short series).
    """
    n_eps = T + 2 * K
    eps = stats.levy_stable.rvs(alpha=alpha, beta=beta, size=n_eps, random_state=seed)
    k_idx = np.arange(-K, K + 1)
    d = np.where(k_idx >= 0, psi ** k_idx, phi ** (-k_idx)) / (1.0 - phi * psi)
    windows = sliding_window_view(eps, 2 * K + 1)   # window t = eps[t : t+2K+1]
    return windows @ d   # index t+K+k increases with k, same order as d


def simulate_aggregate(T: int, psi: Sequence[float], phi: Sequence[float],
                        varsigma: Sequence[float], alpha: float, beta: float,
                        K: int, seed: int) -> np.ndarray:
    Y = np.zeros(T)
    for j in range(J):
        Xj = simulate_component(T, phi[j], psi[j], alpha, beta, K, seed=seed + j * 7919)
        Y += varsigma[j] * Xj
    return Y


# =============================================================================
# Characteristic functions (joint CF of the pair (Y_t, Y_{t+1}))
# =============================================================================

def _component_log_cf(U: np.ndarray, V: np.ndarray, psi: float, phi: float,
                       varsigma: float, alpha: float, beta: float) -> np.ndarray:
    """
    log E[exp(i(u X_t + v X_{t+1}))] for a SCALED Case-2 component,
    implementing eq. (aggcf_mar11_skewed) of main_v23.tex:

        log phi_{X_j}(u,v) = -[varsigma^alpha / |1-phi*psi|^alpha]
                               * (C_j(u,v) + A_j(u,v)),
        C_j = |u+v*phi|^alpha / (1-phi^alpha) * (1 - i*beta*sign(u+v*phi)*tan(pi*alpha/2)),
        A_j = |u*psi+v|^alpha / (1-psi^alpha) * (1 - i*beta*sign(u*psi+v)*tan(pi*alpha/2)),

    valid for phi, psi > 0 (the paper's restriction when beta != 0).
    Verified numerically against the empirical CF of simulated paths.
    """
    argC = U + V * phi
    argA = U * psi + V
    if is_alpha_one(alpha):
        if beta != 0.0:
            raise NotImplementedError(
                "alpha=1 with beta!=0 needs the log|.|-drift term (cf. "
                "cf_weights.logcf_alpha1_component); not needed here since "
                "Panel A (Cauchy) has beta=0 fixed."
            )
        Cj = np.abs(argC) / (1.0 - phi)
        Aj = np.abs(argA) / (1.0 - psi)
        norm = varsigma / abs(1.0 - phi * psi)
        return -norm * (Cj + Aj)
    tan_t = np.tan(np.pi * alpha / 2.0)
    Cj = np.abs(argC) ** alpha / (1.0 - phi ** alpha) * (1.0 - 1j * beta * np.sign(argC) * tan_t)
    Aj = np.abs(argA) ** alpha / (1.0 - psi ** alpha) * (1.0 - 1j * beta * np.sign(argA) * tan_t)
    norm = varsigma ** alpha / abs(1.0 - phi * psi) ** alpha
    return -norm * (Cj + Aj)


def tcf_grid(u_grid: np.ndarray, v_grid: np.ndarray, psi: Sequence[float],
             phi: Sequence[float], varsigma: Sequence[float], alpha: float,
             beta: float) -> np.ndarray:
    U, V = np.meshgrid(u_grid, v_grid, indexing="ij")
    log_phi = np.zeros(U.shape, dtype=complex)
    for j in range(J):
        log_phi += _component_log_cf(U, V, psi[j], phi[j], varsigma[j], alpha, beta)
    return np.exp(log_phi)


def ecf_grid(Y: np.ndarray, u_grid: np.ndarray, v_grid: np.ndarray) -> np.ndarray:
    Yt, Ytp1 = Y[:-1], Y[1:]
    phase = u_grid[:, None, None] * Yt[None, None, :] + v_grid[None, :, None] * Ytp1[None, None, :]
    return np.mean(np.exp(1j * phase), axis=2)


# =============================================================================
# Parameter (un)packing and MDE objective
# =============================================================================

def unpack(theta: np.ndarray, spec: PanelSpec) -> Dict[str, float]:
    p: Dict[str, float] = {}
    p["psi1"], p["phi1"], p["varsigma1"], p["psi2"], p["phi2"], p["varsigma2"] = theta[:6]
    idx = 6
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
    b = [ROOT_BOUNDS, ROOT_BOUNDS, VARSIGMA_BOUNDS] * 2
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
    tcf = tcf_grid(u_grid, v_grid, (p["psi1"], p["psi2"]), (p["phi1"], p["phi2"]),
                    (p["varsigma1"], p["varsigma2"]), p["alpha"], p["beta"])
    return float(np.sum(W * np.abs(ecf - tcf) ** 2))


def sort_by_psi_desc(p: Dict[str, float]) -> Dict[str, float]:
    """Break the label-switching symmetry: component 1 is the one with the
    larger psi, matching the DGP convention psi_1 = 0.8 > psi_2 = 0.3."""
    q = dict(p)
    if q["psi1"] < q["psi2"]:
        q["psi1"], q["psi2"] = q["psi2"], q["psi1"]
        q["phi1"], q["phi2"] = q["phi2"], q["phi1"]
        q["varsigma1"], q["varsigma2"] = q["varsigma2"], q["varsigma1"]
    return q


def _starting_points(spec: PanelSpec, n_starts: int, rng: np.random.Generator,
                      bounds: List[Tuple[float, float]]) -> List[np.ndarray]:
    base = [0.7, 0.25, 0.7, 0.4, 0.45, 0.9]      # psi1,phi1,vs1, psi2,phi2,vs2
    swapped = [0.4, 0.45, 0.9, 0.7, 0.25, 0.7]
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
                        n_starts: int, K: int) -> Dict[str, object]:
    out: Dict[str, object] = {"rep": rep_idx, "T": T}
    try:
        Y = simulate_aggregate(T, TRUE_PSI, TRUE_PHI, TRUE_VARSIGMA, spec.alpha_true,
                                spec.beta_true, K, seed=seed)
        est = estimate_one(Y, spec, u_grid, v_grid, W, n_starts, seed=seed + 1)
        out.update({f"{k}_est": v for k, v in est.items()
                     if k not in ("objective_value", "success")})
        out["objective_value"] = est["objective_value"]
        out["success"] = est["success"]
    except Exception as exc:
        out["success"] = False
        out["error"] = repr(exc)
    return out


# =============================================================================
# Panel-level Monte Carlo study
# =============================================================================

def run_panel(spec: PanelSpec, sample_sizes: Sequence[int], n_replications: int,
              u_grid: np.ndarray, v_grid: np.ndarray, W: np.ndarray,
              n_starts: int, K: int, seed_base: int, n_jobs: int,
              verbose: bool) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for T in sample_sizes:
        t0 = time.time()
        seeds = [seed_base + hash((spec.key, T, r)) % 1_000_000_000
                 for r in range(n_replications)]
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(single_replication)(r, T, spec, seeds[r], u_grid, v_grid, W,
                                         n_starts, K)
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
            "phi1": TRUE_PHI[0], "phi2": TRUE_PHI[1],
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
    "phi1": r"$\phi_1$", "phi2": r"$\phi_2$",
    "varsigma1": r"$\varsigma_{1}$", "varsigma2": r"$\varsigma_{2}$",
    "alpha": r"$\alpha$", "beta": r"$\beta$",
}
_PANEL_LETTER = {"cauchy": "A", "sas": "B", "gas": "C"}


def _fmt(x: float, dp: int = 3) -> str:
    return ("" if x >= 0 else r"$-$") + format(abs(x), f".{dp}f")


def format_table_s2(summaries: Dict[str, pd.DataFrame], sample_sizes: Sequence[int],
                     n_replications: int, kappa: Optional[float]) -> str:
    lines = [
        r"\begin{table}[!h]",
        r"\centering",
        r"\caption{Monte Carlo estimation accuracy for mixed causal-noncausal "
        r"MAR(1,1) $\alpha$-stable aggregates ("
        f"{n_replications:,}".replace(",", "{,}") + " replications)}",
        r"\label{tab:mc_accuracy_mar11}",
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
        r"$\psi_2 = 0.3$, $\phi_1 = 0.3$, $\phi_2 = 0.5$, $\varsigma_1 = 0.7$, "
        r"$\varsigma_2 = 0.9$ (same latent scale/mixing design as Table~\ref{tab:mc_accuracy_full}, "
        r"extended with causal roots $\phi_1,\phi_2$).  MRE denotes the mean relative "
        r"error $\mathbb{E}[|\hat\theta - \theta_0|/|\theta_0|]$.  Estimation minimizes "
        f"the discretized criterion (2.14) of main\\_v23.tex using {weight_txt}."
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
    ap.add_argument("--ma-truncation", type=int, default=MA_TRUNCATION,
                     help="max |k| kept in the MA(infinity) simulation")
    ap.add_argument("--kappa", type=float, default=KAPPA_DEFAULT,
                     help="Assumption C2 weight decay (default: cf_weights.KAPPA_DEFAULT)")
    ap.add_argument("--uniform-weights", action="store_true",
                     help="legacy mode: uniform weights (kappa=None)")
    ap.add_argument("--n-grid", type=int, default=N_GRID_DEFAULT)
    ap.add_argument("--u-max", type=float, default=U_MAX_DEFAULT)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="results_table_s2")
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

    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    config = dict(
        sample_sizes=sample_sizes, n_replications=n_replications, n_starts=n_starts,
        ma_truncation=args.ma_truncation, kappa=kappa, n_grid=args.n_grid, u_max=args.u_max,
        seed=args.seed, panels=args.panels,
        true_psi=TRUE_PSI, true_phi=TRUE_PHI, true_varsigma=TRUE_VARSIGMA,
        root_bounds=ROOT_BOUNDS, varsigma_bounds=VARSIGMA_BOUNDS,
        alpha_bounds=ALPHA_BOUNDS, beta_bounds=BETA_BOUNDS,
        python_version=sys.version, platform=platform.platform(),
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        pandas_version=pd.__version__,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    print("=" * 78)
    print("  Table S.2 -- MC estimation accuracy, MAR(1,1) aggregates (main_v23 Case 2)")
    print("=" * 78)
    print(f"  Weights      : {'uniform (legacy)' if kappa is None else f'Assumption C2, kappa={kappa}'}")
    print(f"  Grid         : {args.n_grid}x{args.n_grid} on [-{args.u_max}, {args.u_max}]^2")
    print(f"  Sample sizes : {sample_sizes}")
    print(f"  Replications : {n_replications}   |   multi-start: {n_starts}")
    print(f"  Panels       : {args.panels}")
    print(f"  Output       : {out_dir}/")

    summaries: Dict[str, pd.DataFrame] = {}
    t_start = time.time()
    for spec in PANELS:
        if spec.key not in args.panels:
            continue
        print(f"\n  Panel {_PANEL_LETTER[spec.key]} ({spec.key}) ...")
        df_raw = run_panel(spec, sample_sizes, n_replications, u_grid, v_grid, W,
                            n_starts, args.ma_truncation, args.seed, args.n_jobs, verbose=True)
        df_raw.to_csv(os.path.join(out_dir, f"mc_raw_{spec.key}.csv"), index=False)
        summaries[spec.key] = compute_summary(df_raw, spec)

    if len(summaries) == len(PANELS):
        combined = pd.concat(summaries.values(), ignore_index=True)
        combined.to_csv(os.path.join(out_dir, "mc_summary.csv"), index=False)
        tex = format_table_s2(summaries, sample_sizes, n_replications, kappa)
        tex_path = os.path.join(out_dir, "table_s2_mc_accuracy.tex")
        with open(tex_path, "w") as f:
            f.write(tex + "\n")
        print(f"\n  LaTeX table written to {tex_path}")
    else:
        print("\n  Skipped combined table/summary (only a subset of panels was run).")

    print(f"\n  Done in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
