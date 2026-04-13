"""
Subsampling with Reparameterization (sigma, pi_1) and Gaussian Weights

Modified version using Gaussian weights w(u,v) = exp(-kappa*(u^2 + v^2))
as specified in Assumption C2 of main_v15.tex.

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
from typing import Dict, List, Optional, Tuple
from joblib import Parallel, delayed
import warnings
import os
import time

warnings.filterwarnings('ignore')


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
    """
    rhos, varsigmas = params[:J], params[J:2*J]
    alpha = params[2*J] if alpha_fixed is None else alpha_fixed
    sigma = np.sum(varsigmas)
    pis = varsigmas / sigma
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
    exponent = np.zeros_like(U, dtype=complex)
    if alpha != 1:
        tan_t = np.tan(np.pi * alpha / 2)
        for j in range(J):
            a1 = rhos[j] * U + V          # rho_j*u + v
            exponent += -pis[j]**alpha * sigma**alpha * (
                np.abs(a1)**alpha * (1 - 1j * beta_val * np.sign(a1) * tan_t)
                / (1 - np.abs(rhos[j])**alpha)
                + np.abs(U)**alpha * (1 - 1j * beta_val * np.sign(U) * tan_t))
    else:
        for j in range(J):
            a1 = rhos[j] * U + V          # rho_j*u + v
            exponent += -pis[j] * sigma * (
                np.abs(a1) * (1 + 1j * beta_val * np.sign(a1) * 2 / np.pi
                              * np.log(np.abs(a1) + 1e-10))
                / (1 - np.abs(rhos[j]))
                + np.abs(U) * (1 + 1j * beta_val * np.sign(U) * 2 / np.pi
                               * np.log(np.abs(U) + 1e-10)))
    return np.exp(exponent)


def gaussian_weights(u_grid, v_grid, kappa=1.0):
    """
    Compute Gaussian weights as per Assumption C2.
    
    w(u, v) = exp(-kappa * (u^2 + v^2))
    
    Args:
        u_grid: Grid of u values
        v_grid: Grid of v values
        kappa: Decay parameter (must be > 0)
        
    Returns:
        2D array of weights
    """
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
    return np.exp(-kappa * (U**2 + V**2))


def fast_objective(params, ecf, u_grid, v_grid, J, alpha_fixed, beta_val, kappa=1.0):
    """
    MDE objective with Gaussian weights.
    
    D(theta) = sum_{u,v} w(u,v) * |varphi_n(u,v) - varphi(u,v;theta)|^2
    where w(u,v) = exp(-kappa*(u^2 + v^2))
    """
    for j in range(J):
        if params[j] <= 0 or params[j] >= 1:
            return 1e10
    for j in range(J, 2*J):
        if params[j] <= 0:
            return 1e10
    a = params[2*J] if alpha_fixed is None else alpha_fixed
    if a <= 0 or a > 2:
        return 1e10
    
    tcf = vectorized_tcf(params, u_grid, v_grid, J, alpha_fixed, beta_val)
    diff = np.abs(ecf - tcf)**2
    
    # Apply Gaussian weights
    weights = gaussian_weights(u_grid, v_grid, kappa)
    
    return float(np.sum(weights * diff))


def estimate_sub(Y_sub, x0, u_grid, v_grid, J, bounds,
                 alpha_fixed, beta_val, kappa, maxiter=200):
    """Fast MDE on subsample with warm start and Gaussian weights."""
    try:
        ecf = vectorized_ecf(Y_sub, u_grid, v_grid)
        res = optimize.minimize(
            fast_objective, x0,
            args=(ecf, u_grid, v_grid, J, alpha_fixed, beta_val, kappa),
            bounds=bounds, method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-10})
        return res.x
    except Exception:
        return np.full(len(x0), np.nan)


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
                     ug, vg, bounds, pnames_reparam, clevels, ci_step, kappa):
    """Single MC replication with post-estimation reparameterization and Gaussian weights."""
    from stablemixture import StableMixtureDeconvolution
    af = av if fa else None
    p = len(pnames_reparam)
    mxi = 100 if b < 80 else 60

    mdl = StableMixtureDeconvolution(
        num_components=J, alpha_value=av, beta_value=bv,
        fix_alpha=fa, fix_beta=fb, verbose=False)
    Y, _ = mdl.generate(T=n, params=tp_full, seed=seed)

    # Full-sample estimate in original space, then transform
    tn_raw = np.array(mdl.estimate(Y, use_optimal_weights=False))
    tn = _transform_params(tn_raw, J, fa, fb)

    # --- Non-overlapping blocks ---
    nno = n // b
    no_raw = np.array([estimate_sub(Y[i*b:(i+1)*b], tn_raw, ug, vg, J,
                                    bounds, af, bv, kappa, mxi) for i in range(nno)])
    vld = ~np.isnan(no_raw).any(axis=1)
    no_raw = no_raw[vld]
    # Transform each subsample estimate
    if len(no_raw) >= 3:
        no_t = np.array([_transform_params(r, J, fa, fb) for r in no_raw])
        sd_no = np.sqrt(b) * (no_t - tn)
    else:
        sd_no = np.full((0, p), np.nan)

    # --- Overlapping blocks (thinned) ---
    Nb = n - b + 1
    oi = list(range(0, Nb, max(ci_step, 1)))
    ov_raw = np.array([estimate_sub(Y[i:i+b], tn_raw, ug, vg, J,
                                    bounds, af, bv, kappa, mxi) for i in oi])
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
                 alpha=1.5, beta=0.0, fix_alpha=False, fix_beta=True, kappa=1.0):
        from stablemixture import StableMixtureDeconvolution
        self.J = J
        self.alpha, self.beta = alpha, beta
        self.fix_alpha, self.fix_beta = fix_alpha, fix_beta
        self.true_params = true_params
        self.kappa = kappa

        # Original-space bounds (estimation unchanged)
        self.bounds = [(0.01, 0.99)] * J + [(0.005, 5.0)] * J
        if not fix_alpha:
            self.bounds.append((0.1, 1.99))
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
        if ci_step is None:
            Nb = n - b + 1
            ci_step = 1

        nno = n // b
        nov = len(range(0, n - b + 1, max(ci_step, 1)))
        print(f"  n={n}, b={b}, M={M}, κ={self.kappa} | non-ov={nno}, "
              f"ov(step={ci_step})={nov}, est/rep={nno + nov}")

        t0 = time.time()
        reps = Parallel(n_jobs=n_cores, backend=backend, verbose=5)(
            delayed(_one_rep_reparam)(
                seed + m, n, b, self.true_params, self.tp_reparam,
                self.J, self.alpha, self.beta,
                self.fix_alpha, self.fix_beta,
                self.u_grid, self.v_grid, self.bounds,
                self.param_names, conf_levels, ci_step, self.kappa)
            for m in range(M))
        print(f"  Done in {time.time() - t0:.0f}s")
        return self._agg(reps, n, b, M, conf_levels)

    def _agg(self, reps, n, b, M, cl):
        """Aggregate results across replications."""
        pnames = self.param_names
        a = {'n': n, 'b': b, 'M': M, 'kappa': self.kappa,
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
        return a

    def table(self, agg):
        """Summary DataFrame."""
        n_val = agg['n']
        rows = []
        for p in agg['param_names']:
            se = agg['theta_n_std'][p]
            ds = agg['moments'][p]['std']
            ratio = (ds / (np.sqrt(n_val) * se)
                     if se > 0 and not np.isnan(se) and not np.isnan(ds)
                     else np.nan)
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
        """Diagnostic histograms + QQ + KDE."""
        ps = agg['param_names']
        np_ = len(ps)
        fig, ax = plt.subplots(np_, 3, figsize=(15, 4 * np_))
        if np_ == 1:
            ax = ax.reshape(1, -1)
        fig.suptitle(
            f"Diagnostics (reparam) n={agg['n']}, b={agg['b']}, M={agg['M']}, κ={agg['kappa']}",
            fontsize=13)
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
            for j in range(3):
                ax[i, j].grid(True, alpha=.3)
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=300, bbox_inches='tight')
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
    all_df = {}
    all_agg = {}

    print(f"\n{'='*70}")
    print(f"SUBSAMPLING WITH REPARAMETERIZATION AND GAUSSIAN WEIGHTS")
    print(f"  Weight function: w(u,v) = exp(-{kappa}*(u^2 + v^2))")
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
            f.write(f"Subsampling Results (reparam + Gaussian weights κ={kappa})\n")
            f.write(f"n={n}, b={b}, M={M}\n")
            f.write("=" * 80 + "\n")
            f.write(df.to_string(index=False) + "\n")
        df.to_csv(os.path.join(output_dir, f'summary_n{n}.csv'), index=False)
        eng.plot_diag(agg, os.path.join(output_dir, f'diagnostics_n{n}.png'))

    # Combined summary
    with open(os.path.join(output_dir, 'summary_all.txt'), 'w') as f:
        f.write("SUBSAMPLING CONVERGENCE STUDY (REPARAMETERIZED + GAUSSIAN WEIGHTS)\n")
        f.write(f"Weight function: w(u,v) = exp(-{kappa}*(u^2 + v^2))\n")
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