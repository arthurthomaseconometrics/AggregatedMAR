"""
Subsampling with Reparameterization (sigma, pi_1) and Gaussian Weights

Weights: w(u,v) = (kappa/pi) exp(-kappa*(u^2 + v^2)) (de Truchis, Fries and Thomas,
2026, Assumption C2).  The objective is a Riemann sum of the criterion; the quadrature
is provided by cf_weights.quadrature_weights.  The full-sample estimate theta_n and
every subsample estimate theta_b are the minimisers of the same criterion, from the
same starting values, bounds and tolerances (Politis and Romano, 1994).

The MDE estimation remains in the original (rho_j, varsigma_j, alpha) space.
After each subsample estimation, results are mapped to the transformed space:
    (rho_1, rho_2, sigma, pi_1, alpha)
where sigma = varsigma_1 + varsigma_2 and pi_1 = varsigma_1 / sigma.

All diagnostics and confidence intervals are then computed on the
transformed parameters, which are expected to have smaller finite-sample bias.

Convention (u, v):
    u is associated with X_t  (Y[:-1])
    v is associated with X_{t+1} (Y[1:])
This matches the theoretical CF decomposition:
    u*X_{j,t} + v*X_{j,t+1} = (rho_j*u + v)*X_{j,t+1} + u*eps_{j,t}
giving |rho_j*u + v|^alpha and |u|^alpha terms in vectorized_tcf.

Usage:
    from subsampling_reparam import run_study_reparam

    run_study_reparam(
        sample_sizes=[250, 500, 1000, 10000, 50000, 100000],
        M=200, b_exponent=2/3, kappa=1.0, n_cores=-1
    )
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
from cf_weights import (KAPPA_DEFAULT, quadrature_weights, weighted_sse, check_grid,
                        ALPHA_BOUNDS, alpha_admissible, is_alpha_one,
                        logcf_alpha1_component, lbfgsb_options,
                        MDE_GTOL, MDE_FTOL, MDE_MAXITER)
from typing import Dict, List, Optional, Tuple
from joblib import Parallel, delayed
import warnings
import os
import time

warnings.filterwarnings('ignore')

# Target number of overlapping blocks per replication (see _EngineReparam.mc).
OV_BLOCKS_TARGET = 2000

# =============================================================================
# Robust statistics for Skew/Kurt
# =============================================================================
#
# At large n, a small share of blocks reaches the boundary of the parameter space,
# which produces extreme standardized deviations.  Trimmed skewness/kurtosis and the
# share of outlying deviations are reported next to the raw moments.
ROBUST_TRIM_FRAC = 0.10
OUTLIER_Z_THRESHOLD = 3.0


def robust_shape_stats(z, trim_frac=ROBUST_TRIM_FRAC, z_threshold=OUTLIER_Z_THRESHOLD):
    """Skewness tronquee (valide) + fraction de points extremes (remplace un
    kurtosis tronque, structurellement biase -- voir commentaire ci-dessus).

    La fraction extreme est calculee apres standardisation ROBUSTE (mediane /
    MAD), pas moyenne/ecart-type brut : la contamination elle-meme gonfle la
    moyenne et l'ecart-type, ce qui MASQUE les points extremes qu'on cherche
    justement a detecter (verifie : sur un melange 94% N(0,1) + 6% N(3.7,0.3),
    la standardisation par (moyenne, ecart-type) ne detecte que ~0.3% de
    points au-dela de 3 sigma au lieu des ~6% reels, la contamination ayant
    gonfle l'ecart-type brut jusqu'a ~1.45).  La mediane et le MAD ont un point
    de rupture de 50% et restent quasi inchanges par 6% de contamination.

    Returns (skew_trim, outlier_frac), NaN si trop peu de points valides.
    """
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 20:
        return np.nan, np.nan
    lo, hi = np.quantile(z, [trim_frac, 1 - trim_frac])
    zt = z[(z >= lo) & (z <= hi)]
    skew_trim = float(stats.skew(zt)) if zt.size >= 8 else np.nan
    med = np.median(z)
    mad = np.median(np.abs(z - med)) * 1.4826  # -> comparable a un ecart-type sous normalite
    zs = (z - med) / mad if mad > 0 else z - med
    outlier_frac = float(np.mean(np.abs(zs) > z_threshold))
    return skew_trim, outlier_frac


# =============================================================================
# Standalone functions (top-level for joblib pickling)
# =============================================================================

def vectorized_ecf(Y, u_grid, v_grid):
    """Compute empirical joint CF on (u,v) grid via broadcasting.

    Convention: u is associated with X_t = Y[:-1],
                v is associated with X_{t+1} = Y[1:].
    This is consistent with the theoretical CF where
    E[exp(i(u*X_t + v*X_{t+1}))] decomposes as
    (rho_j*u + v)*X_{j,t+1} + u*eps_{j,t}.
    """
    Y0, Y1 = Y[:-1], Y[1:]   # Y0 = X_t, Y1 = X_{t+1}
    phase = (u_grid[:, None, None] * Y0[None, None, :]
             + v_grid[None, :, None] * Y1[None, None, :])
    return np.mean(np.exp(1j * phase), axis=2)


def vectorized_tcf(params, u_grid, v_grid, J, alpha_fixed, beta_val):
    """Compute theoretical joint CF on (u,v) grid.

    Implements E[exp(i(u*X_t + v*X_{t+1}))] for the MAR(0,1) aggregate.
    The decomposition u*X_{j,t} + v*X_{j,t+1}
                    = (rho_j*u + v)*X_{j,t+1} + u*eps_{j,t}
    yields the two independent terms |rho_j*u + v|^alpha and |u|^alpha.

    Selection de branche : le test etait `if alpha != 1`, une EGALITE
    FLOTTANTE.  A alpha = 1 + 1e-9 on tombait donc dans la branche generale
    ou tan(pi*alpha/2) ~ -6.4e8, ce qui detruit silencieusement le critere.
    Le test passe par cf_weights.is_alpha_one (tolerance ALPHA1_TOL), et
    ALPHA_BOUNDS place desormais le domaine d'estimation au-dessus de 1.
    """
    rhos, varsigmas = params[:J], params[J:2*J]
    alpha = params[2*J] if alpha_fixed is None else alpha_fixed
    sigma = np.sum(varsigmas)
    pis = varsigmas / sigma
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
    exponent = np.zeros_like(U, dtype=complex)
    if not is_alpha_one(alpha):
        tan_t = np.tan(np.pi * alpha / 2)
        for j in range(J):
            a1 = rhos[j] * U + V          # rho_j*u + v
            exponent += -pis[j]**alpha * sigma**alpha * (
                np.abs(a1)**alpha * (1 - 1j * beta_val * np.sign(a1) * tan_t)
                / (1 - np.abs(rhos[j])**alpha)
                + np.abs(U)**alpha * (1 - 1j * beta_val * np.sign(U) * tan_t))
    else:
        # BRANCHE alpha = 1 : HORS du domaine de l'Assumption C2 et donc de
        # la Proposition 2.1.  Inatteignable quand alpha est estime (borne
        # basse ALPHA_BOUNDS[0] > 1) ; conservee pour alpha IMPOSE, cas du
        # Panel A (Cauchy) de la Table S.1 du supplement.
        for j in range(J):
            exponent += logcf_alpha1_component(rhos[j] * U + V, U,
                                               varsigmas[j], rhos[j], beta_val)
    return np.exp(exponent)


def gaussian_weights(u_grid, v_grid, kappa=KAPPA_DEFAULT):
    """
    Quadrature weights of Assumption C2:
        W[i,j] = (kappa/pi) exp(-kappa (u_i^2 + v_j^2)) * du * dv,
    a Riemann sum of the criterion (see cf_weights.quadrature_weights).

    Args:
        u_grid, v_grid: evaluation grids
        kappa: decay parameter (> 0); None gives uniform weights

    Returns:
        2D array of quadrature weights
    """
    return quadrature_weights(u_grid, v_grid, kappa)


def fast_objective(params, ecf, u_grid, v_grid, J, alpha_fixed, beta_val,
                   kappa=KAPPA_DEFAULT):
    """
    Weighted MDE criterion (Assumption C2).

    D(theta) ~= sum_{u,v} w(u,v) |varphi_n(u,v) - varphi(u,v;theta)|^2 du dv
    with w(u,v) = (kappa/pi) exp(-kappa (u^2 + v^2)).
    """
    for j in range(J):
        if params[j] <= 0 or params[j] >= 1:
            return 1e10
    for j in range(J, 2*J):
        if params[j] <= 0:
            return 1e10
    a = params[2*J] if alpha_fixed is None else alpha_fixed
    # Domaine de l'Assumption C2 quand alpha est libre ; simple validite de la
    # loi stable quand il est impose (cf. cf_weights.alpha_admissible).
    if not alpha_admissible(a, estimated=alpha_fixed is None):
        return 1e10

    tcf = vectorized_tcf(params, u_grid, v_grid, J, alpha_fixed, beta_val)
    diff = np.abs(ecf - tcf)**2
    
    # Apply Gaussian weights
    weights = gaussian_weights(u_grid, v_grid, kappa)
    
    return float(np.sum(weights * diff))


# =============================================================================
# One estimator for the full sample and for the blocks
# =============================================================================
#
# Subsampling (Politis and Romano, 1994) requires theta_b to be the same functional
# applied to the subsample as theta_n to the full sample: same starting values,
# bounds, maximum number of iterations and tolerances, no data-dependent warm start.
MDE_STARTS_J2 = (
    (0.75, 0.25, 0.50, 0.50, 1.50),
    (0.25, 0.75, 0.50, 0.50, 1.50),   # miroir de la precedente
    (0.85, 0.45, 0.30, 0.90, 1.40),
    (0.45, 0.85, 0.90, 0.30, 1.60),   # miroir de la precedente
)
N_STARTS_DEFAULT = 2


def mde_starts(J, fix_alpha, fix_beta, beta_val, n_starts=N_STARTS_DEFAULT):
    """Points de depart deterministes, en paires miroir (voir MDE_STARTS_J2)."""
    if J != 2:
        raise NotImplementedError("MDE_STARTS_J2 ne couvre que le cas J = 2")
    starts = []
    for row in MDE_STARTS_J2[:max(1, int(n_starts))]:
        x = list(row[:4])
        if not fix_alpha:
            x.append(float(np.clip(row[4], *ALPHA_BOUNDS)))
        if not fix_beta:
            x.append(float(beta_val))
        starts.append(np.array(x, dtype=float))
    return starts


def canonical_order(raw, J):
    """
    Ordering restriction of Assumption C1 (it rules out label switching).

    Components are put in DECREASING order of rho (most persistent first), the order
    of the simulation design (psi_1 = 0.8 > psi_2 = 0.3).
    """
    raw = np.asarray(raw, dtype=float)
    if raw.size < 2 * J or not np.all(np.isfinite(raw[:2 * J])):
        return raw
    order = np.argsort(-raw[:J], kind="stable")
    out = raw.copy()
    out[:J] = raw[:J][order]
    out[J:2 * J] = raw[J:2 * J][order]
    return out


def mde(ecf, starts, u_grid, v_grid, J, bounds, alpha_fixed, beta_val, kappa,
        maxiter=MDE_MAXITER):
    """
    L'estimateur MDE : minimiseur de `fast_objective` sur le meme critere, la
    meme grille, le meme kappa, les memes tolerances et les memes departs, que
    l'entree soit l'echantillon complet ou un bloc.

    Les tolerances viennent de cf_weights.lbfgsb_options : `gtol` par defaut de
    scipy vaut 1e-5 en ABSOLU alors que le critere pondere vaut ~1e-6, si bien
    que L-BFGS-B s'arretait avant d'avoir bouge dans la direction pi_1.
    """
    best_f, best_x = np.inf, None
    for x0 in starts:
        try:
            res = optimize.minimize(
                fast_objective, x0,
                args=(ecf, u_grid, v_grid, J, alpha_fixed, beta_val, kappa),
                bounds=bounds, method='L-BFGS-B',
                options=lbfgsb_options(maxiter))
        except Exception:
            continue
        if np.isfinite(res.fun) and res.fun < best_f and np.all(np.isfinite(res.x)):
            best_f, best_x = float(res.fun), np.asarray(res.x, dtype=float)
    if best_x is None:
        return np.full(len(starts[0]), np.nan)
    return canonical_order(best_x, J)


def estimate_sub(Y_sub, starts, u_grid, v_grid, J, bounds,
                 alpha_fixed, beta_val, kappa, maxiter=MDE_MAXITER):
    """MDE sur un sous-echantillon -- strictement la meme fonctionnelle que
    sur l'echantillon complet (cf. `mde`)."""
    try:
        ecf = vectorized_ecf(Y_sub, u_grid, v_grid)
    except Exception:
        return np.full(len(starts[0]), np.nan)
    return mde(ecf, starts, u_grid, v_grid, J, bounds, alpha_fixed, beta_val,
               kappa, maxiter)


def _transform_params(raw, J, fix_alpha, fix_beta):
    """Transform raw MDE output to reparameterized space.

    Input raw: (rho_1, ..., rho_J, varsigma_1, ..., varsigma_J, [alpha], [beta])
    Output:    (rho_1, ..., rho_J, sigma, pi_1, ..., pi_{J-1}, [alpha], [beta])

    For J=2 with fix_beta=True, fix_alpha=False:
        raw    = (rho_1, rho_2, varsigma_1, varsigma_2, alpha)
        output = (rho_1, rho_2, sigma, pi_1, alpha)
    """
    rhos = raw[:J]
    varsigmas = raw[J:2*J]
    sigma = np.sum(varsigmas)
    pis = varsigmas / sigma if sigma > 0 else varsigmas * 0

    out = list(rhos)
    out.append(sigma)
    # Only J-1 pi's needed (they sum to 1)
    for j in range(J - 1):
        out.append(pis[j])

    idx = 2 * J
    if not fix_alpha:
        out.append(raw[idx])
        idx += 1
    if not fix_beta:
        out.append(raw[idx])
        idx += 1

    return np.array(out)


def _reparam_names(J, fix_alpha, fix_beta):
    """Parameter names in the reparameterized space."""
    names = [f'rho_{j+1}' for j in range(J)]
    names.append('sigma')
    for j in range(J - 1):
        names.append(f'pi_{j+1}')
    if not fix_alpha:
        names.append('alpha')
    if not fix_beta:
        names.append('beta')
    return names


def _reparam_true(tp_full, J, fix_alpha, fix_beta):
    """True parameter values in the reparameterized space."""
    return _transform_params(
        np.array(list(tp_full[:2*J])
                 + ([tp_full[2*J]] if not fix_alpha else [])
                 + ([tp_full[2*J+1]] if not fix_beta else [])),
        J, fix_alpha, fix_beta
    )


def _one_rep_reparam(seed, n, b, tp_full, tp_reparam, J, av, bv, fa, fb,
                     ug, vg, bounds, pnames_reparam, clevels, ci_step, kappa,
                     n_starts=N_STARTS_DEFAULT):
    """One Monte Carlo replication, post-estimation reparameterization.

    theta_n and the theta_b are produced by the same call to `mde`, with the same
    fixed starting values, bounds, maximum number of iterations and tolerances.
    """
    from stablemixture import StableMixtureDeconvolution
    af = av if fa else None
    p = len(pnames_reparam)
    starts = mde_starts(J, fa, fb, bv, n_starts)

    mdl = StableMixtureDeconvolution(
        num_components=J, alpha_value=av, beta_value=bv,
        fix_alpha=fa, fix_beta=fb, verbose=False)
    Y, _ = mdl.generate(T=n, params=tp_full, seed=seed)

    # --- Full-sample estimate in original space, then transform ---
    tn_raw = mde(vectorized_ecf(Y, ug, vg), starts, ug, vg, J, bounds,
                 af, bv, kappa)
    if not np.all(np.isfinite(tn_raw)):
        return {'theta_n': np.full(p, np.nan), 'sd_no': np.full((0, p), np.nan),
                'ci_res': {lv: {q: {'covers': np.nan, 'width': np.nan}
                                for q in pnames_reparam} for lv in clevels},
                'ci_norm_res': {lv: {q: {'covers': np.nan, 'width': np.nan}
                                     for q in pnames_reparam} for lv in clevels},
                'norm_res': {q: {k: np.nan for k in
                                 ['mean', 'std', 'skewness', 'kurtosis',
                                  'ks_rej', 'sw_rej', 'jb_rej', 'n_blocks']}
                             for q in pnames_reparam},
                'n_no': 0, 'n_ov': 0}
    tn = _transform_params(tn_raw, J, fa, fb)

    # --- Non-overlapping blocks ---
    nno = n // b
    no_raw = np.array([estimate_sub(Y[i*b:(i+1)*b], starts, ug, vg, J,
                                    bounds, af, bv, kappa) for i in range(nno)])
    vld = ~np.isnan(no_raw).any(axis=1)
    no_raw = no_raw[vld]
    # Transform each subsample estimate
    if len(no_raw) >= 3:
        no_t = np.array([_transform_params(r, J, fa, fb) for r in no_raw])
        sd_no = np.sqrt(b) * (no_t - tn)
    else:
        sd_no = np.full((0, p), np.nan)

    # --- Overlapping blocks (thinned by ci_step, cf. _EngineReparam.mc) ---
    oi = list(range(0, n - b + 1, max(int(ci_step), 1)))
    ov_raw = np.array([estimate_sub(Y[i:i+b], starts, ug, vg, J,
                                    bounds, af, bv, kappa) for i in oi])
    vld2 = ~np.isnan(ov_raw).any(axis=1)
    ov_raw = ov_raw[vld2]
    if len(ov_raw) >= 3:
        ov_t = np.array([_transform_params(r, J, fa, fb) for r in ov_raw])
        sd_ov = np.sqrt(b) * (ov_t - tn)
    else:
        sd_ov = np.full((0, p), np.nan)

    # --- CIs: quantile-based AND normal ---
    ci = {}
    ci_norm = {}
    for lv in clevels:
        ci[lv] = {}
        ci_norm[lv] = {}
        a = 1 - lv
        z_crit = stats.norm.ppf(1 - a / 2)
        for k in range(p):
            if sd_ov.shape[0] >= 3:
                zk = sd_ov[:, k]
                zk = zk[np.isfinite(zk)]
                if len(zk) >= 3:
                    # Quantile-based CI
                    ql = np.quantile(zk, a / 2)
                    qh = np.quantile(zk, 1 - a / 2)
                    clo = tn[k] - qh / np.sqrt(n)
                    chi = tn[k] - ql / np.sqrt(n)
                    ci[lv][pnames_reparam[k]] = {
                        'covers': float(tp_reparam[k] >= clo
                                        and tp_reparam[k] <= chi),
                        'width': chi - clo}
                    # Normal CI
                    V_hat = float(np.var(zk, ddof=1))
                    se_hat = np.sqrt(V_hat / n)
                    clo_n = tn[k] - z_crit * se_hat
                    chi_n = tn[k] + z_crit * se_hat
                    ci_norm[lv][pnames_reparam[k]] = {
                        'covers': float(tp_reparam[k] >= clo_n
                                        and tp_reparam[k] <= chi_n),
                        'width': chi_n - clo_n}
                    continue
            ci[lv][pnames_reparam[k]] = {'covers': np.nan, 'width': np.nan}
            ci_norm[lv][pnames_reparam[k]] = {'covers': np.nan, 'width': np.nan}

    # --- Normality tests (on non-overlapping block deviations) ---
    nr = {}
    for k in range(p):
        if sd_no.shape[0] >= 4:
            zk = sd_no[:, k]
            zk = zk[np.isfinite(zk)]
            if len(zk) >= 4:
                mu = float(np.mean(zk))
                sg = float(np.std(zk, ddof=1))
                zs = (zk - mu) / sg if sg > 0 else zk
                ksp = stats.kstest(zs, 'norm')[1]
                swp = stats.shapiro(zs)[1] if len(zs) <= 5000 else np.nan
                jbp = stats.jarque_bera(zk)[1]
                nr[pnames_reparam[k]] = {
                    'mean': mu, 'std': sg,
                    'skewness': float(stats.skew(zk)),
                    'kurtosis': float(stats.kurtosis(zk)),
                    'ks_rej': float(ksp < 0.05),
                    'sw_rej': float(swp < 0.05) if not np.isnan(swp) else np.nan,
                    'jb_rej': float(jbp < 0.05),
                    'n_blocks': len(zk)}
                continue
        nr[pnames_reparam[k]] = {
            s: np.nan for s in
            ['mean', 'std', 'skewness', 'kurtosis', 'ks_rej', 'sw_rej', 'jb_rej']}
        nr[pnames_reparam[k]]['n_blocks'] = 0

    return {'theta_n': tn, 'sd_no': sd_no,
            'ci_res': ci, 'ci_norm_res': ci_norm,
            'norm_res': nr, 'n_no': sd_no.shape[0], 'n_ov': sd_ov.shape[0]}


# =============================================================================
# Engine (reparameterized version with Gaussian weights)
# =============================================================================

class _EngineReparam:
    """Core engine with post-estimation reparameterization and Gaussian weights."""

    def __init__(self, J=2, true_params=(0.8, 0.3, 0.7, 0.9, 1.5, 0),
                 alpha=1.5, beta=0.0, fix_alpha=False, fix_beta=True, kappa=1.0,
                 n_starts=N_STARTS_DEFAULT):
        from stablemixture import StableMixtureDeconvolution
        self.J = J
        self.n_starts = int(n_starts)
        self.alpha, self.beta = alpha, beta
        self.fix_alpha, self.fix_beta = fix_alpha, fix_beta
        self.true_params = true_params
        self.kappa = kappa

        # Original-space bounds (estimation unchanged)
        self.bounds = [(0.01, 0.99)] * J + [(0.005, 5.0)] * J
        if not fix_alpha:
            self.bounds.append(ALPHA_BOUNDS)   # (1, 2) strict, Assumption C2
        if not fix_beta:
            self.bounds.append((-0.99, 0.99))

        # Grids
        self.u_grid = np.linspace(-1.5, 1.5, 15)
        self.v_grid = np.linspace(-1.5, 1.5, 15)

        # Reparameterized names and true values
        self.param_names = _reparam_names(J, fix_alpha, fix_beta)
        self.tp_reparam = _reparam_true(true_params, J, fix_alpha, fix_beta)

    def mc(self, n, M, b, seed=42, conf_levels=(0.90, 0.95),
           ci_step=None, n_cores=-1, backend='loky'):
        """Run M replications in parallel with Gaussian weights."""
        # Overlapping blocks for the confidence intervals: the step is chosen to target
        # OV_BLOCKS_TARGET blocks (ci_step = 1, i.e. all n - b + 1 blocks, for n <= 1000).
        # The value used is printed and recorded.
        if ci_step is None:
            ci_step = max(1, int(np.ceil((n - b + 1) / OV_BLOCKS_TARGET)))
        ci_step = max(int(ci_step), 1)
        self.ci_step_used = ci_step

        nno = n // b
        nov = len(range(0, n - b + 1, ci_step))
        print(f"  n={n}, b={b}, M={M}, κ={self.kappa}, starts={self.n_starts} | "
              f"non-ov={nno}, ov(step={ci_step})={nov}, est/rep={nno + nov}")

        t0 = time.time()
        reps = Parallel(n_jobs=n_cores, backend=backend, verbose=5)(
            delayed(_one_rep_reparam)(
                seed + m, n, b, self.true_params, self.tp_reparam,
                self.J, self.alpha, self.beta,
                self.fix_alpha, self.fix_beta,
                self.u_grid, self.v_grid, self.bounds,
                self.param_names, conf_levels, ci_step, self.kappa,
                self.n_starts)
            for m in range(M))
        print(f"  Done in {time.time() - t0:.0f}s")
        return self._agg(reps, n, b, M, conf_levels)

    def _agg(self, reps, n, b, M, cl):
        """Aggregate results across replications."""
        pnames = self.param_names
        a = {'n': n, 'b': b, 'M': M, 'kappa': self.kappa,
             'ci_step': getattr(self, 'ci_step_used', None),
             'n_starts': self.n_starts,
             'param_names': pnames, 'conf_levels': cl,
             'moments': {p: {} for p in pnames},
             'rej_rates': {p: {} for p in pnames},
             'coverage': {l: {} for l in cl},
             'ci_width': {l: {} for l in cl},
             'pooled_no': {p: [] for p in pnames}}

        # Full-sample estimates (in reparam space)
        ta = np.array([r['theta_n'] for r in reps
                       if not np.any(np.isnan(r['theta_n']))])
        a['theta_n_mean'] = {}
        a['theta_n_std'] = {}
        a['true_params_est'] = {}
        for k, p in enumerate(pnames):
            a['true_params_est'][p] = float(self.tp_reparam[k])
            a['theta_n_mean'][p] = float(np.mean(ta[:, k])) if len(ta) > 0 else np.nan
            a['theta_n_std'][p] = float(np.std(ta[:, k])) if len(ta) > 0 else np.nan

        # Moments and rejection rates from normality tests
        for p in pnames:
            for s in ['mean', 'std', 'skewness', 'kurtosis']:
                v = [r['norm_res'][p][s] for r in reps
                     if not np.isnan(r['norm_res'][p].get(s, np.nan))]
                a['moments'][p][s] = float(np.mean(v)) if v else np.nan
            for t in ['ks_rej', 'sw_rej', 'jb_rej']:
                v = [r['norm_res'][p][t] for r in reps
                     if not np.isnan(r['norm_res'][p].get(t, np.nan))]
                a['rej_rates'][p][t] = float(np.mean(v)) if v else np.nan

        # Quantile CI coverage
        for l in cl:
            for p in pnames:
                c = [r['ci_res'][l][p]['covers'] for r in reps
                     if not np.isnan(r['ci_res'][l][p].get('covers', np.nan))]
                w = [r['ci_res'][l][p]['width'] for r in reps
                     if not np.isnan(r['ci_res'][l][p].get('width', np.nan))]
                a['coverage'][l][p] = float(np.mean(c)) if c else np.nan
                a['ci_width'][l][p] = float(np.mean(w)) if w else np.nan

        # Normal CI coverage
        a['cov_norm'] = {l: {} for l in cl}
        a['w_norm'] = {l: {} for l in cl}
        for l in cl:
            for p in pnames:
                c = [r['ci_norm_res'][l][p]['covers'] for r in reps
                     if not np.isnan(r['ci_norm_res'][l][p].get('covers', np.nan))]
                w = [r['ci_norm_res'][l][p]['width'] for r in reps
                     if not np.isnan(r['ci_norm_res'][l][p].get('width', np.nan))]
                a['cov_norm'][l][p] = float(np.mean(c)) if c else np.nan
                a['w_norm'][l][p] = float(np.mean(w)) if w else np.nan

        # Pool non-overlapping deviations
        for r in reps:
            sd = r['sd_no']
            if sd.shape[0] > 0:
                for k, p in enumerate(pnames):
                    v = sd[:, k]
                    a['pooled_no'][p].extend(v[np.isfinite(v)].tolist())

        # Dev.Mean / Dev.Std / Skew / Kurt are computed on the non-overlapping block
        # deviations pooled across all Monte Carlo replications.
        a['moments_pooled'] = {}
        for p in pnames:
            z = np.asarray(a['pooled_no'][p], dtype=float)
            z = z[np.isfinite(z)]
            skew_trim, outlier_frac = robust_shape_stats(z)
            a['moments_pooled'][p] = {
                'mean': float(np.mean(z)) if z.size >= 2 else np.nan,
                'std': float(np.std(z, ddof=1)) if z.size >= 2 else np.nan,
                'skewness': float(stats.skew(z)) if z.size >= 3 else np.nan,
                'kurtosis': float(stats.kurtosis(z)) if z.size >= 4 else np.nan,
                'skewness_trim': skew_trim,
                'outlier_frac': outlier_frac,
                'n_pooled': int(z.size)}
        return a

    def table(self, agg):
        """Summary DataFrame."""
        n_val = agg['n']
        rows = []
        mom = agg.get('moments_pooled', agg['moments'])
        for p in agg['param_names']:
            se = agg['theta_n_std'][p]
            ds = mom[p]['std']
            ratio = (ds / (np.sqrt(n_val) * se)
                     if se > 0 and not np.isnan(se) and not np.isnan(ds)
                     else np.nan)
            row = {'Parameter': p,
                   'True': agg['true_params_est'][p],
                   'Avg Est.': agg['theta_n_mean'][p],
                   'Std Est.': se,
                   'Dev.Mean': mom[p]['mean'],
                   'Dev.Std': ds,
                   'Ratio': ratio,
                   'Skew': mom[p]['skewness'],
                   'Kurt': mom[p]['kurtosis'],
                   'Skew.Trim': mom[p].get('skewness_trim', np.nan),
                   'Outlier%': mom[p].get('outlier_frac', np.nan),
                   'SW Rej.': agg['rej_rates'][p]['sw_rej'],
                   'JB Rej.': agg['rej_rates'][p]['jb_rej']}
            for l in agg['conf_levels']:
                pct = int(l * 100)
                row[f'qCov{pct}'] = agg['coverage'][l][p]
                row[f'nCov{pct}'] = agg['cov_norm'][l][p]
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_diag(self, agg, path=None):
        """Diagnostic histograms + QQ + KDE."""
        ps = agg['param_names']
        np_ = len(ps)
        fig, ax = plt.subplots(np_, 3, figsize=(15, 4 * np_))
        if np_ == 1:
            ax = ax.reshape(1, -1)
        fig.suptitle(
            f"Diagnostics (reparam) n={agg['n']}, b={agg['b']}, M={agg['M']}, κ={agg['kappa']}",
            fontsize=13)
        labels = {'rho_1': 'ψ₁', 'rho_2': 'ψ₂', 'sigma': 'σ', 'pi_1': 'π₁', 'alpha': 'α'}
        for i, p in enumerate(ps):
            z = np.array(agg['pooled_no'][p])
            z = z[np.isfinite(z)]
            if len(z) < 10:
                continue
            mu, sg = np.mean(z), np.std(z)
            zs = (z - mu) / sg if sg > 0 else z
            xr = np.linspace(zs.min(), zs.max(), 200)
            ax[i, 0].hist(zs, bins=40, density=True, alpha=.7, edgecolor='k')
            ax[i, 0].plot(xr, stats.norm.pdf(xr), 'r-', lw=2)
            ax[i, 0].set_title(f'{labels.get(p, p)}: Hist')
            stats.probplot(zs, dist='norm', plot=ax[i, 1])
            ax[i, 1].set_title(f'{labels.get(p, p)}: QQ')
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(zs)
            ax[i, 2].plot(xr, kde(xr), 'b-', lw=2, label='KDE')
            ax[i, 2].plot(xr, stats.norm.pdf(xr), 'r--', lw=2, label='N(0,1)')
            ax[i, 2].set_title(f'{labels.get(p, p)}: KDE')
            ax[i, 2].legend()
            for j in range(3):
                ax[i, j].grid(True, alpha=.3)
        # Leave room for the suptitle.
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        if path:
            plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()


# =============================================================================
# Public API
# =============================================================================

def run_study_reparam(sample_sizes: List[int] = [250, 500, 1000, 10000],
                      M: int = 200,
                      b_exponent: float = 2/3,
                      kappa: float = 1.0,
                      n_cores: int = -1,
                      seed: int = 42,
                      ci_step: Optional[int] = None,
                      output_dir: str = 'results_reparam',
                      **engine_kwargs):
    """
    Full convergence study with (sigma, pi_1) reparameterization and Gaussian weights.

    Estimation is performed in the original (varsigma_1, varsigma_2) space.
    All diagnostics and CIs use the transformed (sigma, pi_1) space.

    Args:
        sample_sizes: List of n values.
        M: MC replications.
        b_exponent: Block size exponent (b = floor(n^exp)). Default 2/3.
        kappa: Gaussian weight decay parameter (default: 1.0).
        n_cores: Cores for parallelization (-1 = all).
        seed: Base random seed.
        ci_step: Override thinning step (None = auto ci_step=1).
        output_dir: Where to save .txt, .csv, .png files.
        **engine_kwargs: Passed to _EngineReparam (true_params, alpha, etc.)

    Returns:
        Dict {n: summary_DataFrame}.
    """
    os.makedirs(output_dir, exist_ok=True)
    eng = _EngineReparam(kappa=kappa, **engine_kwargs)
    print("Critere MDE (Assumption C2):")
    check_grid(eng.u_grid, eng.v_grid, eng.kappa)
    all_df = {}
    all_agg = {}

    print(f"\n{'='*70}")
    print(f"SUBSAMPLING WITH REPARAMETERIZATION AND GAUSSIAN WEIGHTS")
    print(f"  Weight function: w(u,v) = (kappa/pi) exp(-kappa(u^2+v^2)), "
          f"kappa={kappa}  [Assumption C2]")
    print(f"  Reparameterization: (sigma, pi_1) instead of (varsigma_1, varsigma_2)")
    print(f"{'='*70}")

    for n in sample_sizes:
        b = int(np.floor(n ** b_exponent))
        print(f"\n{'=' * 65}")
        print(f"n={n}, b={b} (n^{b_exponent:.4f}) [REPARAM: sigma, pi_1]")

        agg = eng.mc(n, M, b, seed=seed, ci_step=ci_step, n_cores=n_cores)
        df = eng.table(agg)
        all_df[n] = df
        all_agg[n] = agg

        print(df.to_string(index=False))

        # Save per-n results
        with open(os.path.join(output_dir, f'summary_n{n}.txt'), 'w') as f:
            f.write(f"Subsampling Results (reparam, Assumption C2, kappa={kappa})\n")
            f.write(f"n={n}, b={b}, M={M}\n")
            f.write("=" * 80 + "\n")
            f.write(df.to_string(index=False) + "\n")
        df.to_csv(os.path.join(output_dir, f'summary_n{n}.csv'), index=False)
        eng.plot_diag(agg, os.path.join(output_dir, f'diagnostics_n{n}.png'))

    # Combined summary
    with open(os.path.join(output_dir, 'summary_all.txt'), 'w') as f:
        f.write("SUBSAMPLING CONVERGENCE STUDY (REPARAMETERIZED + GAUSSIAN WEIGHTS)\n")
        f.write(f"Weight function: w(u,v) = (kappa/pi) exp(-kappa(u^2+v^2)), "
                f"kappa={kappa}  [Assumption C2]\n")
        f.write("=" * 80 + "\n\n")
        for n in sample_sizes:
            a = all_agg[n]
            f.write(f"n={n}  (b={a['b']}, M={a['M']}, κ={a['kappa']})\n")
            f.write("-" * 80 + "\n")
            f.write(all_df[n].to_string(index=False) + "\n\n")

    # Convergence plot
    if len(sample_sizes) >= 2:
        ns = sorted(sample_sizes)
        params = all_df[ns[0]]['Parameter'].tolist()
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        panels = [
            ('Dev.Mean', 'Mean of Scaled Deviations', 0),
            ('Ratio', 'Ratio Dev.Std / (√n · Std Est.)', 1.0),
            ('Skew', 'Skewness', 0),
            ('Kurt', 'Excess Kurtosis', 0),
            ('qCov90', 'Quantile CI Coverage 90%', 0.90),
            ('qCov95', 'Quantile CI Coverage 95%', 0.95),
            ('nCov90', 'Normal CI Coverage 90%', 0.90),
            ('nCov95', 'Normal CI Coverage 95%', 0.95),
        ]
        for i, (col, title, ref) in enumerate(panels):
            ax = axes[i // 4, i % 4]
            for p in params:
                vals = [float(all_df[nn][all_df[nn]['Parameter'] == p][col].values[0])
                        for nn in ns if nn in all_df]
                ax.plot(ns, vals, marker='o', lw=2, ms=7, label=p)
            ax.axhline(ref, color='red', ls='--', lw=2, label=f'Target ({ref})')
            ax.set_xlabel('n')
            ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=.3)
            ax.set_xscale('log')
            if 'Cov' in col:
                ax.set_ylim([0.0, 1.05])
        plt.suptitle(f'Reparameterized + Gaussian weights (κ={kappa}): (rho_1, rho_2, sigma, pi_1, alpha)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'convergence_plot.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nAll results in {output_dir}/")
    return all_df