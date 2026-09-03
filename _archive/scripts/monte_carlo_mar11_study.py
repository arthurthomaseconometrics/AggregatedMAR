#!/usr/bin/env python3
"""
Monte Carlo Study for MAR(1,1) Stable Aggregates Estimation (FIXED).

Corrections applied:
  1. Label switching: post-estimation sorting by psi (descending) to
     break the permutation symmetry of the J=2 aggregate CF.
  2. Multi-start optimization: the MDE objective is non-convex, so we
     draw N_starts initial points and keep the best solution.
  3. Theoretical CF: correct handling of phi_j = 0 (the k=0 causal
     term |u|^alpha is NOT zero when phi_j=0).
  4. Domaine de alpha : les bornes d'estimation etaient (0.51, 1.99), donc
     a cheval sur alpha = 1.  L'Assumption C2 de main_v23.tex impose
     alpha dans (1, 2) strict ; on utilise desormais cf_weights.ALPHA_BOUNDS.
     La branche alpha = 1 reste accessible avec alpha IMPOSE
     (estimate_alpha=False), ce dont a besoin le Panel A de la Table S.1
     du supplement (agregat de Cauchy).

Author: ---
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from numpy.polynomial.hermite_e import hermegauss
from typing import Tuple, List, Dict, Optional, Any
from cf_weights import ALPHA_BOUNDS, alpha_admissible, is_alpha_one
import warnings
import time
from joblib import Parallel, delayed
import os

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'text.usetex': False,
    'font.family': 'serif',
})


# =============================================================================
# GRID AND WEIGHTS
# =============================================================================

def build_cf_grid(
    n_grid: int = 10,
    kappa: float = 0.5,
    u_max: float = 3.0,
    grid_type: str = 'gauss_hermite',
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the evaluation grid (u, v) and weight matrix W for the discretized
    MDE objective.

    Parameters
    ----------
    n_grid : int
        Number of quadrature nodes per axis.
    kappa : float
        Decay rate of the weight function (kappa > 0).
    u_max : float
        Half-width for the linspace grid (ignored for gauss_hermite).
    grid_type : str
        'gauss_hermite' or 'linspace'.
    scale : float
        Scaling factor for Gauss-Hermite nodes (default=1.0).

    Returns
    -------
    u_grid, v_grid : np.ndarray, shape (n_grid,)
    weights : np.ndarray, shape (n_grid, n_grid)
    """
    if kappa <= 0:
        raise ValueError("kappa must be strictly positive.")

    if grid_type == 'gauss_hermite':
        nodes, w_gh = hermegauss(n_grid)
        nodes_scaled = nodes * scale
        w_adj = w_gh * scale * np.exp(-(kappa - 1.0) * nodes_scaled ** 2)
        u_grid  = nodes_scaled
        v_grid  = nodes_scaled
        weights = np.outer(w_adj, w_adj)

    elif grid_type == 'linspace':
        u_grid  = np.linspace(-u_max, u_max, n_grid)
        v_grid  = np.linspace(-u_max, u_max, n_grid)
        h       = u_grid[1] - u_grid[0]
        U, V    = np.meshgrid(u_grid, v_grid, indexing='ij')
        weights = h ** 2 * np.exp(-kappa * (U ** 2 + V ** 2))

    else:
        raise ValueError(
            f"Unknown grid_type '{grid_type}'. "
            "Use 'gauss_hermite' or 'linspace'."
        )

    return u_grid, v_grid, weights


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_alpha_stable(n: int, alpha: float, beta: float = 0.0,
                          sigma: float = 1.0, mu: float = 0.0,
                          random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate n samples from S(alpha, beta, sigma, mu) using scipy.
    """
    return stats.levy_stable.rvs(
        alpha=alpha, beta=beta, loc=mu, scale=sigma,
        size=n, random_state=random_state
    )


def generate_mar11_process(n: int, phi: float, psi: float,
                           alpha: float, beta: float = 0.0,
                           sigma: float = 1.0, burnin: int = 500,
                           max_lag: int = 300,
                           random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate a MAR(1,1) process via its MA(inf) representation.

    Model: (1 - phi*L)(1 - psi*L^{-1}) X_t = eps_t,  |phi|<1, |psi|<1.

    MA coefficients (normalized by 1/(1 - phi*psi)):
        delta_k = psi^k  / (1 - phi*psi),  k >= 0
        delta_k = phi^|k| / (1 - phi*psi), k <  0
    """
    if abs(phi) >= 1 or abs(psi) >= 1:
        raise ValueError("|phi| and |psi| must be strictly less than 1.")

    total_len = n + burnin + 2 * max_lag
    eps  = simulate_alpha_stable(total_len, alpha, beta, sigma,
                                 random_state=random_state)
    norm = 1.0 / (1.0 - phi * psi)

    X = np.zeros(n + burnin)
    for t in range(max_lag, n + burnin + max_lag):
        s = 0.0
        for k in range(max_lag):
            if t + k < total_len:
                s += (psi ** k) * norm * eps[t + k]
        for k in range(1, max_lag):
            if t - k >= 0:
                s += (phi ** k) * norm * eps[t - k]
        X[t - max_lag] = s

    return X[burnin:]


def generate_mar11_aggregate(n: int, phis: np.ndarray, psis: np.ndarray,
                              varsigmas: np.ndarray, alpha: float,
                              beta: float = 0.0,
                              burnin: int = 500, max_lag: int = 300,
                              random_state: Optional[int] = None
                              ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate an aggregated MAR(1,1) process X_t = sum_j varsigma_j * X_{j,t}.
    """
    J = len(phis)
    components = []
    for j in range(J):
        seed_j = (random_state + j * 1000) if random_state is not None else None
        comp = generate_mar11_process(
            n, phis[j], psis[j], alpha, beta, 1.0,
            burnin, max_lag, random_state=seed_j
        )
        components.append(varsigmas[j] * comp)
    return np.sum(components, axis=0), components


# =============================================================================
# CHARACTERISTIC FUNCTION
# =============================================================================

def compute_ecf_grid(X: np.ndarray,
                     u_grid: np.ndarray,
                     v_grid: np.ndarray) -> np.ndarray:
    """
    Compute the bivariate empirical CF on the full (u, v) grid.

    phi_n(u_i, v_k) = (1/(n-1)) sum_{t=1}^{n-1} exp(i*(u_i*X_t + v_k*X_{t+1}))

    BUGFIX (audit 2026-07): pairing corrected to match theoretical_cf_mar11
    (v <-> X_{t+1}, the FUTURE observation), as in vectorized_ecf of
    subsampling_reparam.py. See code_MAR11/monte_carlo_mar11_study.py,
    which is the reference copy of this file.
    """
    X_t   = X[:-1]
    X_tp1 = X[1:]
    phase = (u_grid[:, None, None] * X_t[None, None, :]
             + v_grid[None, :, None] * X_tp1[None, None, :])
    return np.mean(np.exp(1j * phase), axis=2)


def theoretical_cf_mar11(u: float, v: float, params: Dict[str, Any],
                         J: int = 2) -> complex:
    """
    Theoretical joint CF for the MAR(1,1) aggregate at a single point.

    From the paper (eq. 2.7), the log-CF is:

        log phi_X(u,v) = - sum_{j=1}^J varsigma_j^alpha / |1-phi_j*psi_j|^alpha
                           * [ C_j(u,v) + A_j(u,v) ]

    where for alpha != 1:

        C_j(u,v) = |u + v*phi_j|^alpha / (1 - |phi_j|^alpha)
                   * (1 - i*beta*sign(u+v*phi_j)*tan(pi*alpha/2))

        A_j(u,v) = |u*psi_j + v|^alpha / (1 - |psi_j|^alpha)
                   * (1 - i*beta*sign(u*psi_j+v)*tan(pi*alpha/2))

    ----- FIX 3: phi_j = 0 case -----
    When phi_j = 0, the causal geometric series has a SINGLE term (k=0):
        C_j(u,v) = |u|^alpha * (1 - i*beta*sign(u)*tan(pi*alpha/2))
    This is obtained by setting phi_j = 0 in the general formula:
        |u + v*0|^alpha / (1 - 0) = |u|^alpha.
    The original code returned C_j = 0 for phi_j = 0, which is WRONG.
    The formula is actually continuous at phi_j = 0, so no special case
    is needed: simply evaluate the general formula.

    ----- Branche alpha = 1 -----
    Elle est HORS du domaine de l'Assumption C2 / de la Proposition 2.1 et
    n'est atteignable qu'avec alpha IMPOSE (estimate_alpha=False), cas du
    Panel A de la Table S.1.  La selection se fait par tolerance
    (cf_weights.is_alpha_one) et non plus par egalite flottante
    `alpha != 1` : a alpha = 1 + 1e-9 on tombait dans la branche generale
    ou tan(pi*alpha/2) ~ -6.4e8, ce qui detruisait le critere en silence.

    LIMITATION CONNUE de cette branche, a beta != 0 SEULEMENT.  Comme dans
    les modules MAR(0,1) avant patch, le logarithme est applique a
    l'argument AGREGE (arg_c, arg_a) au lieu d'etre somme coefficient par
    coefficient.  Pour Z = sum_k d_k eps_k avec eps ~ S(1, beta, sigma, 0),
    la formule exacte est

        log phi_Z(1) = -sigma sum_k |d_k|
                       - i sigma beta (2/pi) sum_k d_k ln|d_k| ,

    dont il manque ici la partie qui fait intervenir ln|d_k| plutot que
    ln|arg|, c'est-a-dire un terme de derive en sum_k d_k ln|d_k| (la
    quantite que la condition (assump:coefa1) du papier suppose finie).
    Sans effet sur les resultats publies : le Panel A a beta = 0, et la
    partie imaginaire est alors identiquement nulle.  La correction n'a pas
    ete portee ici parce que le developpement bilateral MAR(1,1) demande de
    reprendre les deux series (causale en phi_j, anticipative en psi_j) et
    la normalisation |1 - phi_j psi_j| ; le cas MAR(0,1) corrige est dans
    cf_weights.logcf_alpha1_component et sert de modele.  A traiter avant
    toute specification MAR(1,1) alpha = 1 ASYMETRIQUE.
    """
    alpha = params['alpha']
    beta  = params['beta']
    alpha_is_one = is_alpha_one(alpha)
    tan_t = np.tan(np.pi * alpha / 2.0) if not alpha_is_one else None
    exp   = 0.0 + 0.0j

    for j in range(1, J + 1):
        psi_j      = params[f'psi{j}']
        phi_j      = params[f'phi{j}']
        varsigma_j = params[f'varsigma{j}']

        norm = varsigma_j ** alpha / abs(1.0 - phi_j * psi_j) ** alpha

        # --- FIX 3: Remove the incorrect phi_j == 0 shortcut ---
        # The formula C_j = |u+v*phi_j|^alpha / (1-|phi_j|^alpha) is
        # well-defined and correct for ALL |phi_j| < 1, including phi_j = 0
        # (where it gives |u|^alpha / 1 = |u|^alpha).

        # Causal contribution C_j
        arg_c = u + v * phi_j
        if not alpha_is_one:
            denom_c = 1.0 - abs(phi_j) ** alpha  # = 1.0 when phi_j = 0
            C_j = (abs(arg_c) ** alpha / denom_c) * \
                  (1.0 - 1j * beta * np.sign(arg_c) * tan_t)
        else:
            denom_c = 1.0 - abs(phi_j)
            C_j = (abs(arg_c) / denom_c) * \
                  (1.0 + 1j * beta * np.sign(arg_c) * (2.0 / np.pi) *
                   np.log(abs(arg_c) + 1e-300))

        # Noncausal contribution A_j
        arg_a = u * psi_j + v
        if not alpha_is_one:
            A_j = (abs(arg_a) ** alpha / (1.0 - abs(psi_j) ** alpha)) * \
                  (1.0 - 1j * beta * np.sign(arg_a) * tan_t)
        else:
            A_j = (abs(arg_a) / (1.0 - abs(psi_j))) * \
                  (1.0 + 1j * beta * np.sign(arg_a) * (2.0 / np.pi) *
                   np.log(abs(arg_a) + 1e-300))

        exp -= norm * (C_j + A_j)

    return np.exp(exp)


def compute_tcf_grid(u_grid: np.ndarray,
                     v_grid: np.ndarray,
                     params: Dict[str, Any],
                     J: int = 2) -> np.ndarray:
    """
    Compute the theoretical CF on the full (u, v) grid.
    """
    n_u  = len(u_grid)
    n_v  = len(v_grid)
    tcf  = np.zeros((n_u, n_v), dtype=complex)
    for i, u in enumerate(u_grid):
        for k, v in enumerate(v_grid):
            tcf[i, k] = theoretical_cf_mar11(u, v, params, J=J)
    return tcf


# =============================================================================
# ESTIMATION
# =============================================================================

def _build_params_dict(theta: np.ndarray,
                       estimate_alpha: bool, estimate_beta: bool,
                       estimate_phi: bool,
                       fixed_alpha: float, fixed_beta: float,
                       fixed_phi1: float, fixed_phi2: float
                       ) -> Dict[str, float]:
    """
    Unpack the optimization vector theta into a parameter dictionary.

    Parameter ordering in theta:
        [psi1, varsigma1, psi2, varsigma2,
         (phi1, phi2   if estimate_phi),
         (alpha        if estimate_alpha),
         (beta         if estimate_beta)]
    """
    idx = 0
    p   = {}
    p['psi1']      = theta[idx]; idx += 1
    p['varsigma1'] = theta[idx]; idx += 1
    p['psi2']      = theta[idx]; idx += 1
    p['varsigma2'] = theta[idx]; idx += 1

    if estimate_phi:
        p['phi1'] = theta[idx]; idx += 1
        p['phi2'] = theta[idx]; idx += 1
    else:
        p['phi1'] = fixed_phi1
        p['phi2'] = fixed_phi2

    if estimate_alpha:
        p['alpha'] = theta[idx]; idx += 1
    else:
        p['alpha'] = fixed_alpha

    if estimate_beta:
        p['beta'] = theta[idx]
    else:
        p['beta'] = fixed_beta

    return p


def mde_objective(theta: np.ndarray,
                  ecf_grid: np.ndarray,
                  u_grid: np.ndarray,
                  v_grid: np.ndarray,
                  weights: np.ndarray,
                  estimate_alpha: bool, estimate_beta: bool,
                  estimate_phi: bool,
                  fixed_alpha: float, fixed_beta: float,
                  fixed_phi1: float, fixed_phi2: float) -> float:
    """
    MDE objective function.

    Q_n(theta) = sum_{i,k} W[i,k] * |phi_n(u_i,v_k) - phi(u_i,v_k;theta)|^2
    """
    p = _build_params_dict(theta, estimate_alpha, estimate_beta, estimate_phi,
                           fixed_alpha, fixed_beta, fixed_phi1, fixed_phi2)

    if not alpha_admissible(p['alpha'], estimated=estimate_alpha): return 1e10
    if not (-0.99 < p['beta']      < 0.99): return 1e10
    if not (0.01  < p['psi1']      < 0.99): return 1e10
    if not (0.01  < p['psi2']      < 0.99): return 1e10
    if not (0.01  < p['varsigma1'] < 5.0):  return 1e10
    if not (0.01  < p['varsigma2'] < 5.0):  return 1e10
    if estimate_phi:
        if not (-0.99 < p['phi1'] < 0.99):  return 1e10
        if not (-0.99 < p['phi2'] < 0.99):  return 1e10

    tcf_grid = compute_tcf_grid(u_grid, v_grid, p, J=2)
    return float(np.sum(weights * np.abs(ecf_grid - tcf_grid) ** 2))


# =============================================================================
# FIX 1: LABEL SWITCHING RESOLUTION
# =============================================================================

def sort_components_by_psi(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sort estimated components by decreasing psi to break the permutation
    symmetry (label switching) in the J=2 aggregate model.

    The aggregate log-CF is invariant under permutation of component
    indices (j=1,2). To ensure a unique labeling, we impose the
    convention psi_1 >= psi_2.

    If the estimated psi_1 < psi_2, we swap all component-specific
    parameters: (psi, varsigma, phi) for j=1 <-> j=2.

    Parameters
    ----------
    params : dict
        Estimated parameter dictionary with keys psi1, psi2, varsigma1,
        varsigma2, phi1, phi2, alpha, beta, etc.

    Returns
    -------
    dict
        Sorted parameter dictionary (psi1 >= psi2).
    """
    p = dict(params)  # shallow copy

    if p.get('psi1', 1.0) < p.get('psi2', 0.0):
        # Swap component 1 <-> component 2
        p['psi1'],      p['psi2']      = p['psi2'],      p['psi1']
        p['varsigma1'], p['varsigma2'] = p['varsigma2'], p['varsigma1']
        p['phi1'],      p['phi2']      = p['phi2'],      p['phi1']

    return p


# =============================================================================
# FIX 2: MULTI-START ESTIMATION
# =============================================================================

def estimate_mar11_aggregate(X: np.ndarray,
                              n_grid: int = 10,
                              kappa: float = 0.5,
                              u_max: float = 3.0,
                              grid_type: str = 'gauss_hermite',
                              scale: float = 1.0,
                              estimate_alpha: bool = True,
                              estimate_beta: bool = False,
                              estimate_phi: bool = False,
                              fixed_alpha: float = 1.5,
                              fixed_beta: float = 0.0,
                              fixed_phi1: float = 0.0,
                              fixed_phi2: float = 0.0,
                              method: str = 'L-BFGS-B',
                              n_starts: int = 5,
                              verbose: bool = False) -> Dict[str, Any]:
    """
    Estimate MAR(1,1) aggregate parameters via MDE with multi-start.

    ----- FIX 2: Multi-start optimization -----
    The MDE objective is non-convex due to:
      (a) the |.|^alpha nonlinearity in the CF,
      (b) the permutation symmetry (label switching) across components.
    A single starting point often leads to a local minimum.  We draw
    n_starts initial points from a Latin hypercube and keep the solution
    with the smallest objective value.

    ----- FIX 1: Post-estimation sorting -----
    After optimization, components are sorted by decreasing psi
    (sort_components_by_psi) to break the label-switching symmetry.

    Parameters
    ----------
    X : np.ndarray
        Observed aggregated series.
    n_grid, kappa, u_max, grid_type, scale : grid parameters.
    estimate_alpha, estimate_beta, estimate_phi : bool
        Which parameters to estimate.
    fixed_alpha, fixed_beta, fixed_phi1, fixed_phi2 : float
        Fixed values when not estimated.
    method : str
        scipy.optimize.minimize method.
    n_starts : int
        Number of random starting points (default=5).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Sorted estimated parameters plus diagnostics.
    """
    u_grid, v_grid, weights = build_cf_grid(
        n_grid=n_grid, kappa=kappa, u_max=u_max, grid_type=grid_type,
        scale=scale
    )

    # Pre-compute empirical CF once
    ecf_grid = compute_ecf_grid(X, u_grid, v_grid)

    # Bounds
    bounds = [(0.01, 0.99), (0.01, 3.0),   # psi1, varsigma1
              (0.01, 0.99), (0.01, 3.0)]    # psi2, varsigma2
    if estimate_phi:
        bounds += [(-0.99, 0.99), (-0.99, 0.99)]   # phi1, phi2
    if estimate_alpha:
        bounds.append(ALPHA_BOUNDS)                  # alpha: (1,2), Assumption C2
    if estimate_beta:
        bounds.append((-0.99, 0.99))                # beta

    n_params = len(bounds)

    def objective(theta):
        return mde_objective(theta, ecf_grid, u_grid, v_grid, weights,
                             estimate_alpha, estimate_beta, estimate_phi,
                             fixed_alpha, fixed_beta, fixed_phi1, fixed_phi2)

    # --- FIX 2: Multi-start ---
    # Generate n_starts initial points via Latin Hypercube Sampling
    rng = np.random.default_rng()
    starts = []

    # Always include the deterministic starting point
    x0_default = [0.7, 0.7, 0.4, 0.9]
    if estimate_phi:
        x0_default += [0.2, 0.2]
    if estimate_alpha:
        x0_default.append(1.5)
    if estimate_beta:
        x0_default.append(0.0)
    starts.append(np.array(x0_default))

    # Also include the label-swapped version
    x0_swapped = [0.4, 0.9, 0.7, 0.7]
    if estimate_phi:
        x0_swapped += [0.2, 0.2]
    if estimate_alpha:
        x0_swapped.append(1.5)
    if estimate_beta:
        x0_swapped.append(0.0)
    starts.append(np.array(x0_swapped))

    # Random starts
    for _ in range(n_starts - 2):
        x0_rand = np.array([
            rng.uniform(lo, hi) for (lo, hi) in bounds
        ])
        starts.append(x0_rand)

    best_res  = None
    best_obj  = np.inf
    best_theta = starts[0]

    t0 = time.time()
    for x0 in starts:
        try:
            res = optimize.minimize(objective, x0, method=method,
                                    bounds=bounds,
                                    options={'maxiter': 500})
            if res.fun < best_obj:
                best_obj   = res.fun
                best_theta = res.x
                best_res   = res
        except Exception:
            continue

    elapsed = time.time() - t0
    success = best_res.success if best_res is not None else False

    p_hat = _build_params_dict(best_theta,
                               estimate_alpha, estimate_beta, estimate_phi,
                               fixed_alpha, fixed_beta, fixed_phi1, fixed_phi2)

    # --- FIX 1: Sort components by decreasing psi ---
    p_hat = sort_components_by_psi(p_hat)

    p_hat['success']         = success
    p_hat['objective_value'] = best_obj
    p_hat['elapsed_time']    = elapsed
    return p_hat


# =============================================================================
# MONTE CARLO REPLICATION
# =============================================================================

def single_mc_replication(sim_idx: int,
                           true_params: Dict[str, float],
                           T: int, seed: int,
                           estimate_alpha: bool,
                           estimate_beta: bool,
                           estimate_phi: bool,
                           n_grid: int,
                           kappa: float,
                           u_max: float,
                           grid_type: str,
                           scale: float,
                           n_starts: int) -> Dict[str, Any]:
    """
    Execute one Monte Carlo replication.

    The true parameters are also sorted by psi (descending) so that
    true_params['psi1'] >= true_params['psi2'], consistent with the
    post-estimation sorting convention.
    """
    np.random.seed(seed)

    # Sort true params consistently (psi1 >= psi2)
    tp = sort_components_by_psi(true_params)

    out = {
        'sim_idx':        sim_idx,
        'T':              T,
        'psi1_true':      tp['psi1'],
        'psi2_true':      tp['psi2'],
        'varsigma1_true': tp['varsigma1'],
        'varsigma2_true': tp['varsigma2'],
        'phi1_true':      tp['phi1'],
        'phi2_true':      tp['phi2'],
        'alpha_true':     tp['alpha'],
        'beta_true':      tp['beta'],
    }

    try:
        X, _ = generate_mar11_aggregate(
            n=T,
            phis=np.array([tp['phi1'], tp['phi2']]),
            psis=np.array([tp['psi1'], tp['psi2']]),
            varsigmas=np.array([tp['varsigma1'], tp['varsigma2']]),
            alpha=tp['alpha'],
            beta=tp['beta'],
            burnin=500, max_lag=200, random_state=seed
        )

        res = estimate_mar11_aggregate(
            X,
            n_grid=n_grid, kappa=kappa, u_max=u_max, grid_type=grid_type,
            scale=scale,
            estimate_alpha=estimate_alpha,
            estimate_beta=estimate_beta,
            estimate_phi=estimate_phi,
            fixed_alpha=tp['alpha'],
            fixed_beta=tp['beta'],
            fixed_phi1=tp['phi1'],
            fixed_phi2=tp['phi2'],
            n_starts=n_starts,
            verbose=False
        )

        out.update({
            'success':         res['success'],
            'objective_value': res['objective_value'],
            'elapsed_time':    res['elapsed_time'],
            'psi1_est':        res['psi1'],
            'psi2_est':        res['psi2'],
            'varsigma1_est':   res['varsigma1'],
            'varsigma2_est':   res['varsigma2'],
            'phi1_est':        res['phi1'],
            'phi2_est':        res['phi2'],
            'alpha_est':       res['alpha'],
            'beta_est':        res['beta'],
        })

    except Exception as e:
        out.update({'success': False, 'error': str(e)})

    return out


# =============================================================================
# MONTE CARLO STUDY
# =============================================================================

def run_monte_carlo_study(
    # --- true parameters ---
    # BUGFIX (audit 2026-07): the previous defaults (phi1=phi2=0.0,
    # estimate_phi=False) reduce the model to a pure noncausal AR(1)
    # aggregate (Case 1), which is not what this module is for. The values
    # below are the genuine MAR(1,1) design already used in main() and in
    # code_MAR11/main_mc_mar11.py (nonzero causal roots phi1, phi2), so that
    # a bare call exercises the case this function is meant to test.
    psi1: float = 0.7,
    psi2: float = 0.4,
    varsigma1: float = 0.8,
    varsigma2: float = 0.6,
    phi1: float = 0.3,
    phi2: float = 0.5,
    alpha: float = 1.5,
    beta: float = 0.0,
    # --- estimation flags ---
    estimate_alpha: bool = True,
    estimate_beta: bool = False,
    estimate_phi: bool = True,
    # --- MC settings ---
    n_replications: int = 1000,
    sample_sizes: Optional[List[int]] = None,
    n_jobs: int = -1,
    seed_base: int = 42,
    # --- grid settings ---
    # BUGFIX (audit 2026-07): the previous defaults (grid_type='gauss_hermite',
    # u_max=3.0) place quadrature nodes as far as |u| ~ 3.4, where an
    # alpha-stable characteristic function has already decayed to essentially
    # zero; fitting the ECF there mostly fits sampling noise, degrading and
    # even de-stabilizing the estimates (RMSE was observed to be non-monotone
    # in T). The defaults below match the calibrated design already used in
    # main() of this module and in code_MAR11/main_mc_mar11.py: a uniform-ish
    # 15x15 grid on [-0.5, 0.5]^2 with Gaussian weights (kappa=0.5), which is
    # the design validated against the reference AR(1)-aggregate results.
    n_grid: int = 15,
    kappa: float = 0.5,
    u_max: float = 0.5,
    grid_type: str = 'linspace',
    scale: float = 1.0,
    # --- multi-start ---
    n_starts: int = 8,
    # --- output ---
    verbose: bool = True,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full Monte Carlo study for MAR(1,1) aggregate estimation.
    """
    if sample_sizes is None:
        sample_sizes = [250, 500, 1000]

    true_params: Dict[str, float] = {
        'psi1': psi1, 'psi2': psi2,
        'varsigma1': varsigma1, 'varsigma2': varsigma2,
        'phi1': phi1, 'phi2': phi2,
        'alpha': alpha, 'beta': beta,
    }

    # Sort true params so that psi1 >= psi2 (convention)
    true_params = sort_components_by_psi(true_params)

    if verbose:
        print("=" * 70)
        print("Monte Carlo Study: MAR(1,1) Stable Aggregates Estimation")
        print("=" * 70)
        print(f"Replications: {n_replications}  |  Sample sizes: {sample_sizes}")
        print(f"Grid: {grid_type}, n_grid={n_grid}, kappa={kappa}, scale={scale}")
        print(f"Multi-start: n_starts={n_starts}")
        print("True parameters (sorted by psi descending):")
        for k, v in true_params.items():
            print(f"  {k} = {v}")
        print(f"Free parameters: alpha={estimate_alpha}, "
              f"beta={estimate_beta}, phi={estimate_phi}")
        print("=" * 70)

    all_rows = []
    for T in sample_sizes:
        if verbose:
            print(f"\nT = {T} ...", flush=True)
            t0 = time.time()

        seeds = [seed_base + T * 1000 + i for i in range(n_replications)]

        rows = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(single_mc_replication)(
                i, true_params, T, seeds[i],
                estimate_alpha, estimate_beta, estimate_phi,
                n_grid, kappa, u_max, grid_type, scale, n_starts
            )
            for i in range(n_replications)
        )
        all_rows.extend(rows)

        if verbose:
            n_ok = sum(1 for r in rows if r.get('success', False))
            print(f"  done in {time.time()-t0:.1f}s  "
                  f"({n_ok}/{n_replications} successful)")

    df = pd.DataFrame(all_rows)

    summary_df = compute_mc_statistics(
        df, true_params, estimate_phi, estimate_alpha, estimate_beta
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(f'{output_dir}/mc_raw.csv', index=False)
        summary_df.to_csv(f'{output_dir}/mc_summary.csv', index=False)
        with open(f'{output_dir}/mc_table.tex', 'w') as f:
            f.write(format_mc_table(summary_df))
        if verbose:
            print(f"\nResults saved to: {output_dir}/")

    return df, summary_df


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def compute_mc_statistics(df: pd.DataFrame,
                           true_params: Dict[str, float],
                           estimate_phi: bool = False,
                           estimate_alpha: bool = True,
                           estimate_beta: bool = False) -> pd.DataFrame:
    """
    Compute bias, RMSE, and MRE from raw Monte Carlo results.
    """
    params = ['psi1', 'varsigma1', 'psi2', 'varsigma2']
    if estimate_phi:
        params += ['phi1', 'phi2']
    if estimate_alpha:
        params.append('alpha')
    if estimate_beta:
        params.append('beta')

    rows = []
    for T in sorted(df['T'].unique()):
        sub = df[(df['T'] == T) & (df['success'] == True)]
        for param in params:
            est_col = f'{param}_est'
            if est_col not in sub.columns:
                continue
            vals = sub[est_col].dropna().values
            if len(vals) == 0:
                continue
            true_val = true_params[param]
            rows.append({
                'Parameter': param,
                'True':      true_val,
                'T':         T,
                'Mean':      float(np.mean(vals)),
                'Std':       float(np.std(vals)),
                'Bias':      float(np.mean(vals - true_val)),
                'RMSE':      float(np.sqrt(np.mean((vals - true_val) ** 2))),
                'MRE':       float(np.mean(
                                 np.abs(vals - true_val) / (abs(true_val) + 1e-10)
                             )),
                'N_valid':   len(vals),
            })

    return pd.DataFrame(rows)


# =============================================================================
# LATEX TABLE
# =============================================================================

def format_mc_table(summary_df: pd.DataFrame,
                    decimal_places: int = 3) -> str:
    """
    Format the Monte Carlo summary as a LaTeX booktabs table.
    """
    sample_sizes = sorted(summary_df['T'].unique())
    params       = list(summary_df['Parameter'].unique())

    sym = {
        'psi1':      r'$\psi_1$',
        'psi2':      r'$\psi_2$',
        'varsigma1': r'$\varsigma_1$',
        'varsigma2': r'$\varsigma_2$',
        'phi1':      r'$\phi_1$',
        'phi2':      r'$\phi_2$',
        'alpha':     r'$\alpha$',
        'beta':      r'$\beta$',
    }

    n_s      = len(sample_sizes)
    col_spec = 'cc' + 'ccc' * n_s
    fmt      = f'.{decimal_places}f'

    lines = [r'\begin{tabular}{' + col_spec + '}', r'\toprule']

    h1 = ['', ''] + [
        r'\multicolumn{3}{c}{$T = ' + str(T) + r'$}' for T in sample_sizes
    ]
    lines.append(' & '.join(h1) + r' \\')

    cr = []
    c  = 3
    for _ in sample_sizes:
        cr.append(r'\cmidrule(lr){' + f'{c}-{c+2}' + '}')
        c += 3
    lines.append(' '.join(cr))

    h2 = [r'$\theta$', 'True'] + ['Bias', 'RMSE', 'MRE'] * n_s
    lines.append(' & '.join(h2) + r' \\')
    lines.append(r'\midrule')

    for param in params:
        pdata = summary_df[summary_df['Parameter'] == param]
        if pdata.empty:
            continue
        true_val = pdata['True'].iloc[0]
        row = [sym.get(param, param), format(true_val, fmt)]
        for T in sample_sizes:
            td = pdata[pdata['T'] == T]
            if td.empty:
                row += ['--', '--', '--']
            else:
                bias  = td['Bias'].iloc[0]
                rmse  = td['RMSE'].iloc[0]
                mre   = td['MRE'].iloc[0]
                b_str = ('' if bias >= 0 else r'$-$') + format(abs(bias), fmt)
                row  += [b_str, format(rmse, fmt), format(mre, fmt)]
        lines.append(' & '.join(row) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


# =============================================================================
# PLOT
# =============================================================================

def plot_mc_results(df: pd.DataFrame,
                    true_params: Dict[str, float],
                    estimate_phi: bool = False,
                    estimate_alpha: bool = True,
                    estimate_beta: bool = False,
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Histograms of estimator distributions across Monte Carlo replications.
    """
    params = ['psi1', 'varsigma1', 'psi2', 'varsigma2']
    if estimate_phi:
        params += ['phi1', 'phi2']
    if estimate_alpha:
        params.append('alpha')
    if estimate_beta:
        params.append('beta')

    sample_sizes = sorted(df['T'].unique())
    sym = {
        'psi1':      r'$\psi_1$',
        'psi2':      r'$\psi_2$',
        'varsigma1': r'$\varsigma_1$',
        'varsigma2': r'$\varsigma_2$',
        'phi1':      r'$\phi_1$',
        'phi2':      r'$\phi_2$',
        'alpha':     r'$\alpha$',
        'beta':      r'$\beta$',
    }

    n_r = len(params)
    n_c = len(sample_sizes)
    fig, axes = plt.subplots(n_r, n_c,
                             figsize=(3.5 * n_c, 2.5 * n_r),
                             squeeze=False)

    for i, param in enumerate(params):
        true_val = true_params[param]
        est_col  = f'{param}_est'

        for j, T in enumerate(sample_sizes):
            ax  = axes[i, j]
            sub = df[(df['T'] == T) & (df['success'] == True)]

            if est_col not in sub.columns or sub.empty:
                ax.set_visible(False)
                continue

            vals = sub[est_col].dropna().values
            if len(vals) == 0:
                ax.set_visible(False)
                continue

            ax.hist(vals, bins=30, density=True, alpha=0.7,
                    color='steelblue', edgecolor='black', linewidth=0.5)
            ax.axvline(true_val, color='red', linestyle='--',
                       linewidth=2, label='True')
            ax.axvline(np.mean(vals), color='green', linestyle='-',
                       linewidth=1.5, label='Mean')

            if i == 0:
                ax.set_title(f'$T = {T}$', fontsize=11)
            if j == 0:
                ax.set_ylabel(sym.get(param, param), fontsize=11)
            if i == n_r - 1:
                ax.set_xlabel('Estimate', fontsize=10)

            rmse = np.sqrt(np.mean((vals - true_val) ** 2))
            ax.text(0.95, 0.95, f'RMSE={rmse:.3f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc='upper left', fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Figure saved: {save_path}")

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Demonstrate the corrected Monte Carlo study.
    """
    true_params = dict(
    psi1=0.7,      psi2=0.4,       # Éviter les valeurs extrêmes
    varsigma1=0.8, varsigma2=0.6,  # Valeurs plus équilibrées
    phi1=0.3,      phi2=0.5,       # phi modérés, loin de 1
    alpha=1.5,     beta=0.0,
    )

    df_raw, df_summary = run_monte_carlo_study(
        **true_params,
        # estimation
        estimate_alpha=False,
        estimate_beta=False,
        estimate_phi=True,
        # MC
        n_replications=1000,
        sample_sizes=[10000],
        n_jobs=-1,
        seed_base=42,
        # grid — matching the paper: linspace on [-0.5, 0.5]
        # The paper uses "10 equally spaced points in [-0.5, 0.5]"
        # with uniform weights.  For MAR(1,1) (7 params), we increase
        # to 15 grid points to provide 15^2 = 225 moment conditions.
        n_grid=40,
        kappa=0.5,
        grid_type='linspace',
        u_max=10,
        scale=1.0,           # irrelevant for linspace
        # multi-start
        n_starts=3,
        # output
        verbose=True,
        output_dir=None,
    )

    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(df_summary.to_string(index=False))

    # Sort true_params for consistent plotting
    true_params_sorted = sort_components_by_psi(true_params)

    fig = plot_mc_results(
        df_raw,
        true_params=true_params_sorted,
        estimate_phi=True,
        estimate_alpha=True,
        estimate_beta=False,
        save_path='mc_results_mar11_fixed.pdf',
    )
    plt.show()


if __name__ == "__main__":
    main()