"""
Restricted Subsampling: pi_1 = 0.5 imposed.

Estimation is performed in the restricted space (rho_1, rho_2, sigma, [alpha])
with the constraint varsigma_1 = varsigma_2 = sigma / 2.

All diagnostics and CIs are computed on the restricted parameter vector
    theta_R = (rho_1, rho_2, sigma, alpha)    [when fix_alpha=False]

This should exhibit substantially better convergence than the unrestricted
subsampling, because the weakly-identified pi_1 dimension is removed.

Usage:
    from subsampling_restricted import run_study_restricted
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
from typing import Dict, List, Optional
from joblib import Parallel, delayed
import warnings
import os
import time

warnings.filterwarnings('ignore')


# =========================================================================
# Low-level helpers (top-level for joblib)
# =========================================================================

def _ecf_matrix(Y, u_grid, v_grid):
    """Empirical joint CF on (u,v) grid."""
    Y1, Y0 = Y[1:], Y[:-1]
    phase = (u_grid[:, None, None] * Y1[None, None, :]
             + v_grid[None, :, None] * Y0[None, None, :])
    return np.mean(np.exp(1j * phase), axis=2)


def _tcf_restr(params_r, u_grid, v_grid, J, alpha_fixed, beta_val):
    """Theoretical CF under restriction pi_1 = ... = pi_J = 1/J.

    params_r: (rho_1, ..., rho_J, sigma, [alpha])
    Internally: varsigma_j = sigma / J for all j.
    """
    rhos = params_r[:J]
    sigma = params_r[J]
    alpha = params_r[J + 1] if alpha_fixed is None else alpha_fixed
    pi_j = 1.0 / J
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
    exponent = np.zeros_like(U, dtype=complex)
    if alpha != 1:
        tan_t = np.tan(np.pi * alpha / 2)
        for j in range(J):
            a1 = U + rhos[j] * V
            exponent += -pi_j**alpha * sigma**alpha * (
                np.abs(a1)**alpha * (1 - 1j * beta_val * np.sign(a1) * tan_t)
                / (1 - np.abs(rhos[j])**alpha)
                + np.abs(V)**alpha * (1 - 1j * beta_val * np.sign(V) * tan_t))
    else:
        for j in range(J):
            a1 = U + rhos[j] * V
            exponent += -pi_j * sigma * (
                np.abs(a1) * (1 + 1j * beta_val * np.sign(a1) * 2 / np.pi
                              * np.log(np.abs(a1) + 1e-10)) / (1 - np.abs(rhos[j]))
                + np.abs(V) * (1 + 1j * beta_val * np.sign(V) * 2 / np.pi
                               * np.log(np.abs(V) + 1e-10)))
    return np.exp(exponent)


def _obj_restr(params_r, ecf, u_grid, v_grid, J, alpha_fixed, beta_val):
    """MDE objective for restricted model."""
    rhos = params_r[:J]
    sigma = params_r[J]
    alpha = params_r[J + 1] if alpha_fixed is None else alpha_fixed
    if sigma <= 0:
        return 1e10
    for j in range(J):
        if rhos[j] <= 0 or rhos[j] >= 1:
            return 1e10
    if alpha <= 0 or alpha > 2:
        return 1e10
    tcf = _tcf_restr(params_r, u_grid, v_grid, J, alpha_fixed, beta_val)
    return float(np.mean(np.abs(ecf - tcf) ** 2))


def _estimate_sub_restr(Y_sub, x0, u_grid, v_grid, J, bounds,
                        alpha_fixed, beta_val, maxiter=200):
    """MDE on subsample with restriction pi_j = 1/J."""
    try:
        ecf = _ecf_matrix(Y_sub, u_grid, v_grid)
        res = optimize.minimize(
            _obj_restr, x0,
            args=(ecf, u_grid, v_grid, J, alpha_fixed, beta_val),
            bounds=bounds, method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-10})
        return res.x
    except Exception:
        return np.full(len(x0), np.nan)


def _restr_param_names(J, fix_alpha, fix_beta):
    """Parameter names for restricted model."""
    names = [f'rho_{j+1}' for j in range(J)]
    names.append('sigma')
    if not fix_alpha:
        names.append('alpha')
    if not fix_beta:
        names.append('beta')
    return names


def _restr_true_values(tp_full, J, fix_alpha, fix_beta):
    """True parameter values in restricted space.

    tp_full: (rho_1,...,rho_J, varsigma_1,...,varsigma_J, alpha, beta)
    output:  (rho_1,...,rho_J, sigma, [alpha], [beta])
    """
    rhos = list(tp_full[:J])
    varsigmas = tp_full[J:2*J]
    sigma = float(np.sum(varsigmas))
    out = rhos + [sigma]
    idx = 2 * J
    if not fix_alpha:
        out.append(tp_full[idx])
        idx += 1
    if not fix_beta:
        out.append(tp_full[idx])
    return np.array(out)


# =========================================================================
# Single MC replication
# =========================================================================

def _one_rep_restricted(seed, n, b, tp_full, tp_restr, J, av, bv, fa, fb,
                        ug, vg, bounds_restr, pnames, clevels, ci_step):
    """Single MC replication for restricted subsampling."""
    from stablemixture import StableMixtureDeconvolution

    af = av if fa else None
    p = len(pnames)
    mxi = 100 if b < 80 else 60

    # Generate data from the TRUE model (which may have pi_1 != 0.5)
    mdl = StableMixtureDeconvolution(
        num_components=J, alpha_value=av, beta_value=bv,
        fix_alpha=fa, fix_beta=fb, verbose=False)
    Y, _ = mdl.generate(T=n, params=tp_full, seed=seed)

    # --- Full-sample restricted estimation ---
    ecf_full = _ecf_matrix(Y, ug, vg)
    # Initial guess: use rhos from free model, sigma = sum(varsigmas)
    free_raw = np.array(mdl.estimate(Y, use_optimal_weights=False))
    x0_r = list(free_raw[:J])
    x0_r.append(float(np.sum(free_raw[J:2*J])))
    if af is None:
        x0_r.append(free_raw[2*J])
    x0_r = np.array(x0_r)

    res_full = optimize.minimize(
        _obj_restr, x0_r,
        args=(ecf_full, ug, vg, J, af, bv),
        bounds=bounds_restr, method='L-BFGS-B',
        options={'maxiter': 300, 'ftol': 1e-10})
    tn = res_full.x

    # --- Non-overlapping blocks ---
    nno = n // b
    no_est = np.array([_estimate_sub_restr(Y[i*b:(i+1)*b], tn, ug, vg, J,
                                           bounds_restr, af, bv, mxi)
                       for i in range(nno)])
    vld = ~np.isnan(no_est).any(axis=1)
    no_est = no_est[vld]
    if len(no_est) >= 3:
        sd_no = np.sqrt(b) * (no_est - tn)
    else:
        sd_no = np.full((0, p), np.nan)

    # --- Overlapping blocks (thinned) for CIs ---
    Nb = n - b + 1
    oi = list(range(0, Nb, max(ci_step, 1)))
    ov_est = np.array([_estimate_sub_restr(Y[i:i+b], tn, ug, vg, J,
                                           bounds_restr, af, bv, mxi)
                       for i in oi])
    vld2 = ~np.isnan(ov_est).any(axis=1)
    ov_est = ov_est[vld2]
    if len(ov_est) >= 3:
        sd_ov = np.sqrt(b) * (ov_est - tn)
    else:
        sd_ov = np.full((0, p), np.nan)

    # --- CIs ---
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
                    ql = np.quantile(zk, a / 2)
                    qh = np.quantile(zk, 1 - a / 2)
                    clo = tn[k] - qh / np.sqrt(n)
                    chi = tn[k] - ql / np.sqrt(n)
                    ci[lv][pnames[k]] = {
                        'covers': float(tp_restr[k] >= clo and tp_restr[k] <= chi),
                        'width': chi - clo}
                    V_hat = float(np.var(zk, ddof=1))
                    se_hat = np.sqrt(V_hat / n)
                    clo_n = tn[k] - z_crit * se_hat
                    chi_n = tn[k] + z_crit * se_hat
                    ci_norm[lv][pnames[k]] = {
                        'covers': float(tp_restr[k] >= clo_n and tp_restr[k] <= chi_n),
                        'width': chi_n - clo_n}
                    continue
            ci[lv][pnames[k]] = {'covers': np.nan, 'width': np.nan}
            ci_norm[lv][pnames[k]] = {'covers': np.nan, 'width': np.nan}

    # --- Normality tests (non-overlapping) ---
    nr = {}
    for k in range(p):
        if sd_no.shape[0] >= 4:
            zk = sd_no[:, k]
            zk = zk[np.isfinite(zk)]
            if len(zk) >= 4:
                mu = float(np.mean(zk))
                sg = float(np.std(zk, ddof=1))
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
        nr[pnames[k]] = {
            s: np.nan for s in
            ['mean', 'std', 'skewness', 'kurtosis', 'sw_rej', 'jb_rej']}
        nr[pnames[k]]['n_blocks'] = 0

    return {'theta_n': tn, 'sd_no': sd_no,
            'ci_res': ci, 'ci_norm_res': ci_norm,
            'norm_res': nr, 'n_no': sd_no.shape[0], 'n_ov': sd_ov.shape[0]}


# =========================================================================
# Engine
# =========================================================================

class _EngineRestricted:
    """Core engine for restricted subsampling (pi_1 = 1/J)."""

    def __init__(self, J=2, true_params=(0.8, 0.3, 0.7, 0.9, 1.5, 0),
                 alpha=1.5, beta=0.0, fix_alpha=False, fix_beta=True):
        self.J = J
        self.alpha, self.beta = alpha, beta
        self.fix_alpha, self.fix_beta = fix_alpha, fix_beta
        self.true_params = true_params

        # Restricted bounds: (rho_1,...,rho_J, sigma, [alpha], [beta])
        self.bounds = [(0.01, 0.99)] * J + [(0.01, 10.0)]
        if not fix_alpha:
            self.bounds.append((0.1, 1.99))
        if not fix_beta:
            self.bounds.append((-0.99, 0.99))

        self.u_grid = np.linspace(-1.5, 1.5, 15)
        self.v_grid = np.linspace(-1.5, 1.5, 15)

        self.param_names = _restr_param_names(J, fix_alpha, fix_beta)
        self.tp_restr = _restr_true_values(true_params, J, fix_alpha, fix_beta)

    def mc(self, n, M, b, seed=42, conf_levels=(0.90, 0.95),
           ci_step=None, n_cores=-1, backend='loky'):
        """Run M replications in parallel."""
        if ci_step is None:
            Nb = n - b + 1
            ci_step = 1 #max(Nb // 100, 1)

        nno = n // b
        nov = len(range(0, n - b + 1, max(ci_step, 1)))
        print(f"  n={n}, b={b}, M={M} | non-ov={nno}, "
              f"ov(step={ci_step})={nov}, est/rep={nno + nov}")

        t0 = time.time()
        reps = Parallel(n_jobs=n_cores, backend=backend, verbose=5)(
            delayed(_one_rep_restricted)(
                seed + m, n, b, self.true_params, self.tp_restr,
                self.J, self.alpha, self.beta,
                self.fix_alpha, self.fix_beta,
                self.u_grid, self.v_grid, self.bounds,
                self.param_names, conf_levels, ci_step)
            for m in range(M))
        print(f"  Done in {time.time() - t0:.0f}s")
        return self._agg(reps, n, b, M, conf_levels)

    def _agg(self, reps, n, b, M, cl):
        """Aggregate results across replications."""
        pnames = self.param_names
        a = {'n': n, 'b': b, 'M': M,
             'param_names': pnames, 'conf_levels': cl,
             'moments': {p: {} for p in pnames},
             'rej_rates': {p: {} for p in pnames},
             'coverage': {l: {} for l in cl},
             'ci_width': {l: {} for l in cl},
             'pooled_no': {p: [] for p in pnames}}

        ta = np.array([r['theta_n'] for r in reps
                       if not np.any(np.isnan(r['theta_n']))])
        a['theta_n_mean'] = {}
        a['theta_n_std'] = {}
        a['true_params_est'] = {}
        for k, p in enumerate(pnames):
            a['true_params_est'][p] = float(self.tp_restr[k])
            a['theta_n_mean'][p] = float(np.mean(ta[:, k])) if len(ta) > 0 else np.nan
            a['theta_n_std'][p] = float(np.std(ta[:, k])) if len(ta) > 0 else np.nan

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
                w = [r['ci_res'][l][p]['width'] for r in reps
                     if not np.isnan(r['ci_res'][l][p].get('width', np.nan))]
                a['coverage'][l][p] = float(np.mean(c)) if c else np.nan
                a['ci_width'][l][p] = float(np.mean(w)) if w else np.nan

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
            f"Diagnostics (restricted, pi_1=0.5) n={agg['n']}, b={agg['b']}, M={agg['M']}",
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


# =========================================================================
# Public API
# =========================================================================

def run_study_restricted(
    sample_sizes: List[int] = [250, 500, 1000, 10000],
    M: int = 200,
    b_exponent: float = 2 / 3,
    n_cores: int = -1,
    seed: int = 42,
    ci_step: Optional[int] = None,
    output_dir: str = 'results_restricted',
    **engine_kwargs
) -> Dict:
    """Full convergence study with pi_1 = 0.5 restriction imposed.

    Args:
        sample_sizes: List of n values.
        M: MC replications.
        b_exponent: Block size exponent.
        n_cores: Cores for parallelization (-1 = all).
        seed: Base random seed.
        ci_step: Override thinning step.
        output_dir: Where to save results.
        **engine_kwargs: Passed to _EngineRestricted.

    Returns:
        Dict {n: summary_DataFrame}.
    """
    os.makedirs(output_dir, exist_ok=True)
    eng = _EngineRestricted(**engine_kwargs)
    all_df = {}
    all_agg = {}

    for n in sample_sizes:
        b = int(np.floor(n ** b_exponent))
        print(f"\n{'=' * 65}")
        print(f"n={n}, b={b} (n^{b_exponent:.4f}) [RESTRICTED: pi_1=0.5]")

        agg = eng.mc(n, M, b, seed=seed, ci_step=ci_step, n_cores=n_cores)
        df = eng.table(agg)
        all_df[n] = df
        all_agg[n] = agg

        print(df.to_string(index=False))

        with open(os.path.join(output_dir, f'summary_n{n}.txt'), 'w') as f:
            f.write(f"Restricted Subsampling (pi_1=0.5) for n={n} (b={b}, M={M})\n")
            f.write("=" * 80 + "\n")
            f.write(df.to_string(index=False) + "\n")
        df.to_csv(os.path.join(output_dir, f'summary_n{n}.csv'), index=False)
        eng.plot_diag(agg, os.path.join(output_dir, f'diagnostics_restr_n{n}.png'))

    # Combined summary
    with open(os.path.join(output_dir, 'summary_all.txt'), 'w') as f:
        f.write("RESTRICTED SUBSAMPLING (pi_1 = 0.5)\n")
        f.write("=" * 80 + "\n\n")
        for n in sample_sizes:
            a = all_agg[n]
            f.write(f"n={n}  (b={a['b']}, M={a['M']})\n")
            f.write("-" * 80 + "\n")
            f.write(all_df[n].to_string(index=False) + "\n\n")

    # Convergence plot
    if len(sample_sizes) >= 2:
        ns = sorted(sample_sizes)
        params = all_df[ns[0]]['Parameter'].tolist()
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        panels = [
            ('Dev.Mean', 'Mean of Scaled Deviations', 0),
            ('Ratio', 'Ratio Dev.Std / (sqrt(n) * Std Est.)', 1.0),
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
                        for nn in ns]
                ax.plot(ns, vals, marker='o', lw=2, ms=7, label=p)
            ax.axhline(ref, color='red', ls='--', lw=2, label=f'Target ({ref})')
            ax.set_xlabel('n')
            ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=.3)
            ax.set_xscale('log')
            if 'Cov' in col:
                ax.set_ylim([0.0, 1.05])
        plt.suptitle('Restricted: (rho_1, rho_2, sigma, alpha) with pi_1 = 0.5',
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'convergence_restr.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nAll results in {output_dir}/")
    return all_df
