"""
Subsampling for MAR(1,1) stable aggregates (Case 2 of the paper) with
Gaussian weights and post-estimation reparameterization.

Created during the 2026-07 audit. Architecture copied from
AggregatedMAR-main/subsampling_reparam.py (Case 1), adapted to the
mixed causal-noncausal MAR(1,1) aggregate of Section "Case 2" in
main_v16.tex (eq:aggcf_mar11_skewed), in the symmetric case beta = 0.

Model (per component j = 1..J):
    (1 - phi_j L)(1 - psi_j L^{-1}) X_{j,t} = eps_{j,t},
    0 < phi_j < 1, 0 < psi_j < 1, eps ~ S(alpha, 0, 1, 0),
    X_t = sum_j varsigma_j X_{j,t}   (varsigma_j = sigma * pi_j).

Joint log-CF of (X_t, X_{t+1}) -- SaS case, paper's Case 2 formula:
    log phi(u,v) = - sum_j varsigma_j^alpha / |1 - phi_j psi_j|^alpha *
        [ |u + v*phi_j|^alpha / (1 - phi_j^alpha)
        + |u*psi_j + v|^alpha / (1 - psi_j^alpha) ].

Convention (u, v)  [IMPORTANT -- source of the pre-audit bug]:
    u is associated with X_t   (Y[:-1])
    v is associated with X_{t+1} (Y[1:])
This matches theoretical_cf_mar11 in monte_carlo_mar11_study.py (after its
2026-07 bugfix) and vectorized_ecf in subsampling_reparam.py.

Raw estimation space (theta_raw):
    (psi_1, .., psi_J, varsigma_1, .., varsigma_J, phi_1, .., phi_J, [alpha])
    beta fixed at 0.
Post-estimation reparameterization (analogue of eq:reparam, extended):
    (psi_1, .., psi_J, phi_1, .., phi_J, sigma, pi_1, .., pi_{J-1}, [alpha])
    with sigma = sum_j varsigma_j and pi_j = varsigma_j / sigma.

Design choice (lesson from the Case-1 audit): the FULL-SAMPLE estimate and
the SUBSAMPLE estimates use the SAME objective (same grid, same Gaussian
weights w(u,v) = exp(-kappa*(u^2+v^2)) of Assumption C2). This avoids the
inconsistency found in Case 1 where theta_n came from stablemixture.py
(different pairing/weights) while theta_b came from fast_objective.

Local usage (light validation run):
    from subsampling_mar11 import run_study_mar11
    run_study_mar11(sample_sizes=[500, 1000, 2000], M=20, n_cores=4,
                    ci_step=10)

Cluster usage (full campaign):
    run_study_mar11(sample_sizes=[250, 500, 1000, 10000, 50000, 100000],
                    M=200, n_cores=-1, ci_step=None)   # ci_step=None -> 1
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
from typing import Dict, List, Optional, Tuple
from joblib import Parallel, delayed
import warnings
import os
import time

warnings.filterwarnings('ignore')


# =============================================================================
# DGP: fast truncated MA(inf) simulation of a MAR(1,1) aggregate
# =============================================================================

def _mar11_kernel(phi: float, psi: float, max_lag: int) -> np.ndarray:
    """MA(inf) coefficients d_k, k = -max_lag..max_lag, of a MAR(1,1).

    d_k = psi^k / (1 - phi*psi)   for k >= 0   (noncausal side)
    d_k = phi^|k| / (1 - phi*psi) for k <  0   (causal side)
    Same representation as generate_mar11_process in
    monte_carlo_mar11_study.py, but assembled as one kernel so that the
    simulation reduces to a single correlation (vectorized, ~100x faster
    than the pure-Python double loop).
    Index j of the returned array corresponds to k = j - max_lag.
    """
    norm = 1.0 / (1.0 - phi * psi)
    k = np.arange(-max_lag, max_lag + 1)
    d = np.where(k >= 0, psi ** np.maximum(k, 0), phi ** np.maximum(-k, 0))
    return norm * d


def generate_mar11_aggregate_fast(n: int, phis, psis, varsigmas,
                                  alpha: float, beta: float = 0.0,
                                  max_lag: int = 300,
                                  random_state: Optional[int] = None
                                  ) -> np.ndarray:
    """Simulate X_t = sum_j varsigma_j X_{j,t} via truncated MA(inf).

    X_{j,t} = sum_{|k| <= max_lag} d_{j,k} eps_{j,t+k}: the exact stationary
    representation truncated at max_lag (truncation error O(rho^max_lag),
    negligible for max_lag = 300 and |phi|,|psi| <= 0.95). No burn-in is
    needed since the representation is already stationary.
    """
    J = len(phis)
    X = np.zeros(n)
    for j in range(J):
        seed_j = (random_state + 1000 * j) if random_state is not None else None
        eps = stats.levy_stable.rvs(alpha=alpha, beta=beta, loc=0, scale=1.0,
                                    size=n + 2 * max_lag, random_state=seed_j)
        ker = _mar11_kernel(phis[j], psis[j], max_lag)
        # X_{j,t} = sum_k d_k eps_{t+k}  ->  correlation of eps with kernel
        Xj = np.correlate(eps, ker, mode='valid')
        X += varsigmas[j] * Xj
    return X


# =============================================================================
# Characteristic functions and MDE objective (top-level for joblib pickling)
# =============================================================================

def vectorized_ecf(Y, u_grid, v_grid):
    """Empirical joint CF on the (u,v) grid.

    Convention: u <-> X_t = Y[:-1], v <-> X_{t+1} = Y[1:]  (paper's Case 2,
    same convention as subsampling_reparam.py and as theoretical_cf_mar11
    after the 2026-07 bugfix).
    """
    Y0, Y1 = Y[:-1], Y[1:]
    phase = (u_grid[:, None, None] * Y0[None, None, :]
             + v_grid[None, :, None] * Y1[None, None, :])
    return np.mean(np.exp(1j * phase), axis=2)


def vectorized_tcf_mar11(params, u_grid, v_grid, J, alpha_fixed):
    """Theoretical joint CF of the SaS MAR(1,1) aggregate on the (u,v) grid.

    Vectorized version of theoretical_cf_mar11 (beta = 0), implementing
    eq:aggcf_mar11_skewed of main_v16.tex with beta = 0:
        log phi = - sum_j vs_j^a / |1-phi_j psi_j|^a
                  * [ |U + phi_j V|^a/(1-phi_j^a) + |psi_j U + V|^a/(1-psi_j^a) ]
    params = (psi_1..psi_J, varsigma_1..varsigma_J, phi_1..phi_J, [alpha]).
    """
    psis = params[:J]
    varsigmas = params[J:2*J]
    phis = params[2*J:3*J]
    a = params[3*J] if alpha_fixed is None else alpha_fixed
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
    expo = np.zeros_like(U, dtype=float)
    for j in range(J):
        norm = varsigmas[j] ** a / abs(1.0 - phis[j] * psis[j]) ** a
        causal = np.abs(U + V * phis[j]) ** a / (1.0 - abs(phis[j]) ** a)
        noncausal = np.abs(psis[j] * U + V) ** a / (1.0 - abs(psis[j]) ** a)
        expo -= norm * (causal + noncausal)
    return np.exp(expo).astype(complex)


def gaussian_weights(u_grid, v_grid, kappa=0.5):
    """Gaussian weights w(u,v) = exp(-kappa*(u^2+v^2)) (Assumption C2)."""
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
    return np.exp(-kappa * (U ** 2 + V ** 2))


def fast_objective(params, ecf, weights, u_grid, v_grid, J, alpha_fixed):
    """Weighted MDE objective sum_{u,v} w(u,v)|phi_n - phi(theta)|^2."""
    for j in range(J):                      # psi_j in (0,1)
        if not (0.0 < params[j] < 1.0):
            return 1e10
    for j in range(J, 2*J):                 # varsigma_j > 0
        if params[j] <= 0:
            return 1e10
    for j in range(2*J, 3*J):               # phi_j in (0,1)
        if not (0.0 < params[j] < 1.0):
            return 1e10
    a = params[3*J] if alpha_fixed is None else alpha_fixed
    if a <= 1.0 or a >= 2.0:
        return 1e10
    tcf = vectorized_tcf_mar11(params, u_grid, v_grid, J, alpha_fixed)
    return float(np.sum(weights * np.abs(ecf - tcf) ** 2))


def estimate_sub(Y_sub, x0, u_grid, v_grid, weights, J, bounds,
                 alpha_fixed, maxiter=200):
    """Warm-started MDE on a subsample (single L-BFGS-B run)."""
    try:
        ecf = vectorized_ecf(Y_sub, u_grid, v_grid)
        res = optimize.minimize(
            fast_objective, x0,
            args=(ecf, weights, u_grid, v_grid, J, alpha_fixed),
            bounds=bounds, method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-10})
        return res.x
    except Exception:
        return np.full(len(x0), np.nan)


def estimate_full(Y, u_grid, v_grid, weights, J, bounds, alpha_fixed,
                  true_raw=None, n_starts=4, seed=0, maxiter=400):
    """Full-sample MDE with multi-start (same objective as estimate_sub).

    Starts: the true parameters if provided (standard in MC designs to
    avoid attributing multi-modality of the objective to the method),
    plus (n_starts - 1) random draws within the bounds.
    """
    rng = np.random.default_rng(seed)
    ecf = vectorized_ecf(Y, u_grid, v_grid)
    starts = []
    if true_raw is not None:
        starts.append(np.asarray(true_raw, dtype=float))
    while len(starts) < n_starts:
        s = np.array([rng.uniform(lo + 0.02, hi - 0.02) if np.isfinite(hi)
                      else rng.uniform(lo + 0.02, lo + 2.0)
                      for (lo, hi) in bounds])
        starts.append(s)
    best_x, best_f = None, np.inf
    for s in starts:
        try:
            res = optimize.minimize(
                fast_objective, s,
                args=(ecf, weights, u_grid, v_grid, J, alpha_fixed),
                bounds=bounds, method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-12})
            if res.fun < best_f:
                best_x, best_f = res.x, res.fun
        except Exception:
            continue
    return best_x if best_x is not None else np.full(len(bounds), np.nan)


# =============================================================================
# Sorting (label switching) and reparameterization
# =============================================================================

def _sort_components(raw, J, fix_alpha):
    """Sort components by psi descending (as sort_components_by_psi in the
    MC study) to resolve label switching before comparing/transforming."""
    psis = np.asarray(raw[:J])
    varsigmas = np.asarray(raw[J:2*J])
    phis = np.asarray(raw[2*J:3*J])
    order = np.argsort(-psis)
    out = list(psis[order]) + list(varsigmas[order]) + list(phis[order])
    if not fix_alpha:
        out.append(raw[3*J])
    return np.array(out)


def _transform_params(raw, J, fix_alpha):
    """Raw -> reparameterized space.

    Input : (psi_1..psi_J, varsigma_1..varsigma_J, phi_1..phi_J, [alpha])
    Output: (psi_1..psi_J, phi_1..phi_J, sigma, pi_1..pi_{J-1}, [alpha])
    """
    raw = _sort_components(raw, J, fix_alpha)
    psis = raw[:J]
    varsigmas = raw[J:2*J]
    phis = raw[2*J:3*J]
    sigma = float(np.sum(varsigmas))
    pis = varsigmas / sigma if sigma > 0 else varsigmas * 0
    out = list(psis) + list(phis) + [sigma] + list(pis[:J-1])
    if not fix_alpha:
        out.append(raw[3*J])
    return np.array(out)


def _reparam_names(J, fix_alpha):
    names = [f'psi_{j+1}' for j in range(J)]
    names += [f'phi_{j+1}' for j in range(J)]
    names.append('sigma')
    names += [f'pi_{j+1}' for j in range(J - 1)]
    if not fix_alpha:
        names.append('alpha')
    return names


# =============================================================================
# One Monte Carlo replication
# =============================================================================

def _one_rep_mar11(seed, n, b, tp_raw, tp_reparam, J, alpha_val, fix_alpha,
                   ug, vg, wts, bounds, pnames, clevels, ci_step,
                   n_starts, max_lag):
    """Single replication: simulate, full-sample fit, subsample fits,
    scaled deviations, CIs, normality tests. Mirrors _one_rep_reparam."""
    af = alpha_val if fix_alpha else None
    p = len(pnames)
    mxi = 100 if b < 80 else 60

    psis = np.array(tp_raw[:J])
    varsigmas = np.array(tp_raw[J:2*J])
    phis = np.array(tp_raw[2*J:3*J])
    Y = generate_mar11_aggregate_fast(
        n, phis, psis, varsigmas, alpha=alpha_val, beta=0.0,
        max_lag=max_lag, random_state=seed)

    # Full-sample estimate: SAME objective as the subsamples
    tn_raw = estimate_full(Y, ug, vg, wts, J, bounds, af,
                           true_raw=tp_raw if fix_alpha else list(tp_raw),
                           n_starts=n_starts, seed=seed)
    if np.any(np.isnan(tn_raw)):
        return None
    tn = _transform_params(tn_raw, J, fix_alpha)

    # Non-overlapping blocks
    nno = n // b
    no_raw = np.array([estimate_sub(Y[i*b:(i+1)*b], tn_raw, ug, vg, wts, J,
                                    bounds, af, mxi) for i in range(nno)])
    vld = ~np.isnan(no_raw).any(axis=1)
    no_raw = no_raw[vld]
    if len(no_raw) >= 3:
        no_t = np.array([_transform_params(r, J, fix_alpha) for r in no_raw])
        sd_no = np.sqrt(b) * (no_t - tn)
    else:
        sd_no = np.full((0, p), np.nan)

    # Overlapping blocks (thinned by ci_step)
    Nb = n - b + 1
    oi = list(range(0, Nb, max(ci_step, 1)))
    ov_raw = np.array([estimate_sub(Y[i:i+b], tn_raw, ug, vg, wts, J,
                                    bounds, af, mxi) for i in oi])
    vld2 = ~np.isnan(ov_raw).any(axis=1)
    ov_raw = ov_raw[vld2]
    if len(ov_raw) >= 3:
        ov_t = np.array([_transform_params(r, J, fix_alpha) for r in ov_raw])
        sd_ov = np.sqrt(b) * (ov_t - tn)
    else:
        sd_ov = np.full((0, p), np.nan)

    # Confidence intervals (quantile + normal), as in Case 1
    ci, ci_norm = {}, {}
    for lv in clevels:
        ci[lv], ci_norm[lv] = {}, {}
        a = 1 - lv
        z_crit = stats.norm.ppf(1 - a / 2)
        for k in range(p):
            if sd_ov.shape[0] >= 3:
                zk = sd_ov[:, k]
                zk = zk[np.isfinite(zk)]
                if len(zk) >= 3:
                    ql, qh = np.quantile(zk, a / 2), np.quantile(zk, 1 - a / 2)
                    clo, chi = tn[k] - qh / np.sqrt(n), tn[k] - ql / np.sqrt(n)
                    ci[lv][pnames[k]] = {
                        'covers': float(clo <= tp_reparam[k] <= chi),
                        'width': chi - clo}
                    se_hat = np.sqrt(np.var(zk, ddof=1) / n)
                    clo_n, chi_n = tn[k] - z_crit * se_hat, tn[k] + z_crit * se_hat
                    ci_norm[lv][pnames[k]] = {
                        'covers': float(clo_n <= tp_reparam[k] <= chi_n),
                        'width': chi_n - clo_n}
                    continue
            ci[lv][pnames[k]] = {'covers': np.nan, 'width': np.nan}
            ci_norm[lv][pnames[k]] = {'covers': np.nan, 'width': np.nan}

    # Normality diagnostics on non-overlapping deviations
    nr = {}
    for k in range(p):
        if sd_no.shape[0] >= 4:
            zk = sd_no[:, k]
            zk = zk[np.isfinite(zk)]
            if len(zk) >= 4:
                mu, sg = float(np.mean(zk)), float(np.std(zk, ddof=1))
                zs = (zk - mu) / sg if sg > 0 else zk
                swp = stats.shapiro(zs)[1] if len(zs) <= 5000 else np.nan
                jbp = stats.jarque_bera(zk)[1]
                nr[pnames[k]] = {
                    'mean': mu, 'std': sg,
                    'skewness': float(stats.skew(zk)),
                    'kurtosis': float(stats.kurtosis(zk)),
                    'sw_rej': float(swp < 0.05) if not np.isnan(swp) else np.nan,
                    'jb_rej': float(jbp < 0.05),
                    'n_blocks': len(zk)}
                continue
        nr[pnames[k]] = {s: np.nan for s in
                         ['mean', 'std', 'skewness', 'kurtosis',
                          'sw_rej', 'jb_rej']}
        nr[pnames[k]]['n_blocks'] = 0

    return {'theta_n': tn, 'sd_no': sd_no, 'ci_res': ci,
            'ci_norm_res': ci_norm, 'norm_res': nr}


# =============================================================================
# Engine
# =============================================================================

class _EngineMAR11:
    """Subsampling engine for the MAR(1,1) aggregate (SaS, beta = 0)."""

    def __init__(self, J=2,
                 true_params=(0.7, 0.4, 0.8, 0.6, 0.3, 0.5, 1.5),
                 alpha=1.5, fix_alpha=False, kappa=0.5,
                 u_max=0.5, n_grid=12, n_starts=4, max_lag=300):
        """true_params = (psi_1..psi_J, varsigma_1..varsigma_J,
        phi_1..phi_J, alpha) -- default matches the MC study main():
        psi=(0.7,0.4), varsigma=(0.8,0.6), phi=(0.3,0.5), alpha=1.5."""
        self.J = J
        self.alpha = alpha
        self.fix_alpha = fix_alpha
        self.kappa = kappa
        self.n_starts = n_starts
        self.max_lag = max_lag

        self.tp_raw = list(true_params[:3*J]) + \
            ([] if fix_alpha else [true_params[3*J]])

        # Bounds in raw space: psi, phi in (0,1); varsigma > 0; alpha in (1,2)
        self.bounds = ([(0.01, 0.99)] * J + [(0.005, 5.0)] * J
                       + [(0.01, 0.99)] * J)
        if not fix_alpha:
            self.bounds.append((1.05, 1.95))

        # Grid + Gaussian weights: defaults follow the MAR(1,1) MC study
        # (linspace on [-u_max, u_max], kappa = 0.5)
        self.u_grid = np.linspace(-u_max, u_max, n_grid)
        self.v_grid = np.linspace(-u_max, u_max, n_grid)
        self.weights = gaussian_weights(self.u_grid, self.v_grid, kappa)

        self.param_names = _reparam_names(J, fix_alpha)
        self.tp_reparam = _transform_params(
            np.array(self.tp_raw), J, fix_alpha)

    def mc(self, n, M, b, seed=42, conf_levels=(0.90, 0.95),
           ci_step=None, n_cores=-1, backend='loky'):
        if ci_step is None:
            ci_step = 1
        nno = n // b
        nov = len(range(0, n - b + 1, max(ci_step, 1)))
        print(f"  n={n}, b={b}, M={M}, kappa={self.kappa} | non-ov={nno}, "
              f"ov(step={ci_step})={nov}, est/rep={nno + nov + 1}")
        t0 = time.time()
        reps = Parallel(n_jobs=n_cores, backend=backend, verbose=0)(
            delayed(_one_rep_mar11)(
                seed + m, n, b, self.tp_raw, self.tp_reparam, self.J,
                self.alpha, self.fix_alpha, self.u_grid, self.v_grid,
                self.weights, self.bounds, self.param_names, conf_levels,
                ci_step, self.n_starts, self.max_lag)
            for m in range(M))
        reps = [r for r in reps if r is not None]
        print(f"  Done in {time.time() - t0:.0f}s ({len(reps)}/{M} valid)")
        return self._agg(reps, n, b, M, conf_levels)

    def _agg(self, reps, n, b, M, cl):
        pnames = self.param_names
        a = {'n': n, 'b': b, 'M': M, 'kappa': self.kappa,
             'param_names': pnames, 'conf_levels': cl,
             'moments': {p: {} for p in pnames},
             'rej_rates': {p: {} for p in pnames},
             'coverage': {l: {} for l in cl},
             'cov_norm': {l: {} for l in cl},
             'pooled_no': {p: [] for p in pnames}}
        ta = np.array([r['theta_n'] for r in reps
                       if not np.any(np.isnan(r['theta_n']))])
        a['theta_n_mean'], a['theta_n_std'], a['true_params_est'] = {}, {}, {}
        for k, p in enumerate(pnames):
            a['true_params_est'][p] = float(self.tp_reparam[k])
            a['theta_n_mean'][p] = float(np.mean(ta[:, k])) if len(ta) else np.nan
            a['theta_n_std'][p] = float(np.std(ta[:, k])) if len(ta) else np.nan
        for p in pnames:
            for s in ['mean', 'std', 'skewness', 'kurtosis']:
                v = [r['norm_res'][p][s] for r in reps
                     if not np.isnan(r['norm_res'][p].get(s, np.nan))]
                a['moments'][p][s] = float(np.mean(v)) if v else np.nan
            for t in ['sw_rej', 'jb_rej']:
                v = [r['norm_res'][p][t] for r in reps
                     if not np.isnan(r['norm_res'][p].get(t, np.nan))]
                a['rej_rates'][p][t] = float(np.mean(v)) if v else np.nan
        for l in cl:
            for p in pnames:
                c = [r['ci_res'][l][p]['covers'] for r in reps
                     if not np.isnan(r['ci_res'][l][p].get('covers', np.nan))]
                a['coverage'][l][p] = float(np.mean(c)) if c else np.nan
                cn = [r['ci_norm_res'][l][p]['covers'] for r in reps
                      if not np.isnan(r['ci_norm_res'][l][p].get('covers', np.nan))]
                a['cov_norm'][l][p] = float(np.mean(cn)) if cn else np.nan
        for r in reps:
            sd = r['sd_no']
            if sd.shape[0] > 0:
                for k, p in enumerate(pnames):
                    v = sd[:, k]
                    a['pooled_no'][p].extend(v[np.isfinite(v)].tolist())
        return a

    def table(self, agg):
        n_val = agg['n']
        rows = []
        for p in agg['param_names']:
            se = agg['theta_n_std'][p]
            ds = agg['moments'][p]['std']
            ratio = (ds / (np.sqrt(n_val) * se)
                     if se and se > 0 and not np.isnan(ds) else np.nan)
            row = {'Parameter': p,
                   'True': agg['true_params_est'][p],
                   'Avg Est.': agg['theta_n_mean'][p],
                   'Std Est.': se,
                   'Dev.Mean': agg['moments'][p]['mean'],
                   'Dev.Std': ds,
                   'Ratio': ratio,
                   'Skew': agg['moments'][p]['skewness'],
                   'Kurt': agg['moments'][p]['kurtosis'],
                   'SW Rej.': agg['rej_rates'][p]['sw_rej'],
                   'JB Rej.': agg['rej_rates'][p]['jb_rej']}
            for l in agg['conf_levels']:
                pct = int(l * 100)
                row[f'qCov{pct}'] = agg['coverage'][l][p]
                row[f'nCov{pct}'] = agg['cov_norm'][l][p]
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_diag(self, agg, path=None):
        ps = agg['param_names']
        np_ = len(ps)
        fig, ax = plt.subplots(np_, 3, figsize=(15, 4 * np_))
        if np_ == 1:
            ax = ax.reshape(1, -1)
        fig.suptitle(f"Diagnostics MAR(1,1) n={agg['n']}, b={agg['b']}, "
                     f"M={agg['M']}, kappa={agg['kappa']}", fontsize=13)
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
            ax[i, 0].set_title(f'{p}: Hist')
            stats.probplot(zs, dist='norm', plot=ax[i, 1])
            ax[i, 1].set_title(f'{p}: QQ')
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(zs)
            ax[i, 2].plot(xr, kde(xr), 'b-', lw=2, label='KDE')
            ax[i, 2].plot(xr, stats.norm.pdf(xr), 'r--', lw=2, label='N(0,1)')
            ax[i, 2].set_title(f'{p}: KDE')
            ax[i, 2].legend()
            for jj in range(3):
                ax[i, jj].grid(True, alpha=.3)
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()


# =============================================================================
# Self-test against the reference implementation
# =============================================================================

def _self_test(verbose=True):
    """Check vectorized_tcf_mar11 against theoretical_cf_mar11 (reference,
    scalar) and the fast DGP kernel against generate_mar11_process's
    coefficients. Run before any study."""
    from monte_carlo_mar11_study import theoretical_cf_mar11
    params = dict(psi1=0.7, psi2=0.4, varsigma1=0.8, varsigma2=0.6,
                  phi1=0.3, phi2=0.5, alpha=1.5, beta=0.0)
    raw = np.array([0.7, 0.4, 0.8, 0.6, 0.3, 0.5, 1.5])
    ug = np.array([0.4, -0.2, 0.33])
    vg = np.array([-0.15, 0.5, 0.27])
    tcf_vec = vectorized_tcf_mar11(raw, ug, vg, 2, None)
    err = 0.0
    for i, u in enumerate(ug):
        for k, v in enumerate(vg):
            ref = theoretical_cf_mar11(float(u), float(v), params, J=2)
            err = max(err, abs(tcf_vec[i, k] - ref))
    assert err < 1e-12, f"TCF mismatch vs reference: {err}"
    # Kernel check: d_0 = 1/(1-phi*psi), d_1 = psi*d_0, d_{-1} = phi*d_0
    ker = _mar11_kernel(0.3, 0.7, 5)
    n0 = 1 / (1 - 0.3 * 0.7)
    assert abs(ker[5] - n0) < 1e-14 and abs(ker[6] - 0.7 * n0) < 1e-14 \
        and abs(ker[4] - 0.3 * n0) < 1e-14
    if verbose:
        print(f"Self-test OK (max TCF error vs reference: {err:.2e})")


# =============================================================================
# Public API
# =============================================================================

def run_study_mar11(sample_sizes: List[int] = [500, 1000, 2000],
                    M: int = 20,
                    b_exponent: float = 2/3,
                    kappa: float = 0.5,
                    n_cores: int = 4,
                    seed: int = 42,
                    ci_step: Optional[int] = 10,
                    output_dir: str = 'results_mar11_sub',
                    **engine_kwargs):
    """Subsampling convergence study for the MAR(1,1) aggregate.

    Defaults are LOCAL-LIGHT (M=20, ci_step=10, n_cores=4). For the
    cluster campaign use: sample_sizes=[250,...,100000], M=200,
    n_cores=-1, ci_step=None (i.e. step 1, all overlapping blocks).
    """
    _self_test()
    os.makedirs(output_dir, exist_ok=True)
    eng = _EngineMAR11(kappa=kappa, **engine_kwargs)
    all_df, all_agg = {}, {}

    print(f"\n{'='*70}\nSUBSAMPLING MAR(1,1) (SaS, beta=0) + GAUSSIAN WEIGHTS")
    print(f"  w(u,v) = exp(-{kappa}*(u^2+v^2)) | grid: {len(eng.u_grid)}x"
          f"{len(eng.v_grid)} on [{eng.u_grid[0]:.2f},{eng.u_grid[-1]:.2f}]^2")
    print(f"  reparam: (psi_j, phi_j, sigma, pi_1, alpha)\n{'='*70}")

    for n in sample_sizes:
        b = int(np.floor(n ** b_exponent))
        print(f"\nn={n}, b={b} (n^{b_exponent:.4f})")
        agg = eng.mc(n, M, b, seed=seed, ci_step=ci_step, n_cores=n_cores)
        df = eng.table(agg)
        all_df[n], all_agg[n] = df, agg
        print(df.to_string(index=False))
        with open(os.path.join(output_dir, f'summary_n{n}.txt'), 'w') as f:
            f.write(f"MAR(1,1) Subsampling (reparam + Gaussian weights "
                    f"kappa={kappa})\nn={n}, b={b}, M={M}\n")
            f.write("=" * 80 + "\n" + df.to_string(index=False) + "\n")
        df.to_csv(os.path.join(output_dir, f'summary_n{n}.csv'), index=False)
        eng.plot_diag(agg, os.path.join(output_dir, f'diagnostics_n{n}.png'))

    with open(os.path.join(output_dir, 'summary_all.txt'), 'w') as f:
        f.write("MAR(1,1) SUBSAMPLING CONVERGENCE STUDY "
                "(REPARAM + GAUSSIAN WEIGHTS)\n")
        f.write(f"w(u,v) = exp(-{kappa}*(u^2+v^2))\n" + "=" * 80 + "\n\n")
        for n in sample_sizes:
            f.write(f"n={n}  (b={all_agg[n]['b']}, M={all_agg[n]['M']})\n")
            f.write("-" * 80 + "\n" + all_df[n].to_string(index=False) + "\n\n")

    print(f"\nAll results in {output_dir}/")
    return all_df


if __name__ == "__main__":
    run_study_mar11()
