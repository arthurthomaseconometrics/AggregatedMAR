"""
Interactive Subsampling Functions for VSCode

Usage:
    from subsampling_interactive import run_sensitivity, run_study

    # Block size sensitivity (fast, M=100)
    run_sensitivity(sample_sizes=[500, 1000, 2000], M=100, n_cores=64)

    # Full convergence study
    run_study(sample_sizes=[250, 500, 1000, 10000, 50000],
              M=500, b_exponent=0.75, n_cores=64)
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
    """Compute empirical joint CF on (u,v) grid via broadcasting."""
    Y1, Y0 = Y[1:], Y[:-1]
    phase = (u_grid[:, None, None] * Y1[None, None, :]
             + v_grid[None, :, None] * Y0[None, None, :])
    return np.mean(np.exp(1j * phase), axis=2)


def vectorized_tcf(params, u_grid, v_grid, J, alpha_fixed, beta_val):
    """Compute theoretical joint CF on (u,v) grid."""
    rhos, varsigmas = params[:J], params[J:2*J]
    alpha = params[2*J] if alpha_fixed is None else alpha_fixed
    sigma = np.sum(varsigmas)
    pis = varsigmas / sigma
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
    exponent = np.zeros_like(U, dtype=complex)
    if alpha != 1:
        tan_t = np.tan(np.pi * alpha / 2)
        for j in range(J):
            a1 = U + rhos[j] * V
            exponent += -pis[j]**alpha * sigma**alpha * (
                np.abs(a1)**alpha * (1 - 1j*beta_val*np.sign(a1)*tan_t)
                / (1 - np.abs(rhos[j])**alpha)
                + np.abs(V)**alpha * (1 - 1j*beta_val*np.sign(V)*tan_t))
    else:
        for j in range(J):
            a1 = U + rhos[j] * V
            exponent += -pis[j]*sigma*(
                np.abs(a1)*(1+1j*beta_val*np.sign(a1)*2/np.pi
                            *np.log(np.abs(a1)+1e-10))/(1-np.abs(rhos[j]))
                + np.abs(V)*(1+1j*beta_val*np.sign(V)*2/np.pi
                             *np.log(np.abs(V)+1e-10)))
    return np.exp(exponent)


def fast_objective(params, ecf, u_grid, v_grid, J, alpha_fixed, beta_val):
    """MDE objective."""
    for j in range(J):
        if params[j] <= 0 or params[j] >= 1: return 1e10
    for j in range(J, 2*J):
        if params[j] <= 0: return 1e10
    a = params[2*J] if alpha_fixed is None else alpha_fixed
    if a <= 0 or a > 2: return 1e10
    tcf = vectorized_tcf(params, u_grid, v_grid, J, alpha_fixed, beta_val)
    return float(np.mean(np.abs(ecf - tcf)**2))


def estimate_sub(Y_sub, x0, u_grid, v_grid, J, bounds,
                 alpha_fixed, beta_val, maxiter=60):
    """Fast MDE on subsample with warm start."""
    try:
        ecf = vectorized_ecf(Y_sub, u_grid, v_grid)
        res = optimize.minimize(
            fast_objective, x0,
            args=(ecf, u_grid, v_grid, J, alpha_fixed, beta_val),
            bounds=bounds, method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-10})
        return res.x
    except Exception:
        return np.full(len(x0), np.nan)


def _true_est(tp, J, fa, fb):
    """Map full params (2J+2) to estimated subset."""
    v = list(tp[:2*J])
    if not fa: v.append(tp[2*J])
    if not fb: v.append(tp[2*J+1])
    return np.array(v)


def _one_rep(seed, n, b, tp_full, tp_est, J, av, bv, fa, fb,
             ug, vg, bounds, pnames, clevels, ci_step):
    """Single MC replication."""
    from stablemixture import StableMixtureDeconvolution
    af = av if fa else None
    p = len(pnames)
    mxi = 100 if b < 80 else 60

    mdl = StableMixtureDeconvolution(
        num_components=J, alpha_value=av, beta_value=bv,
        fix_alpha=fa, fix_beta=fb, verbose=False)
    Y, _ = mdl.generate(T=n, params=tp_full, seed=seed)
    tn = np.array(mdl.estimate(Y, use_optimal_weights=False))

    # Non-overlapping blocks
    nno = n // b
    no_e = np.array([estimate_sub(Y[i*b:(i+1)*b], tn, ug, vg, J,
                                  bounds, af, bv, mxi) for i in range(nno)])
    vld = ~np.isnan(no_e).any(axis=1); no_e = no_e[vld]
    sd_no = np.sqrt(b)*(no_e - tn) if len(no_e) >= 3 else np.full((0,p), np.nan)

    # Overlapping blocks (thinned)
    Nb = n - b + 1
    oi = list(range(0, Nb, max(ci_step, 1)))
    ov_e = np.array([estimate_sub(Y[i:i+b], tn, ug, vg, J,
                                  bounds, af, bv, mxi) for i in oi])
    vld2 = ~np.isnan(ov_e).any(axis=1); ov_e = ov_e[vld2]
    sd_ov = np.sqrt(b)*(ov_e - tn) if len(ov_e) >= 3 else np.full((0,p), np.nan)

    # CIs: quantile-based AND normal (variance-based)
    ci = {}
    ci_norm = {}
    for lv in clevels:
        ci[lv] = {}
        ci_norm[lv] = {}
        a = 1 - lv
        z_crit = stats.norm.ppf(1 - a/2)
        for k in range(p):
            if sd_ov.shape[0] >= 3:
                zk = sd_ov[:,k]; zk = zk[np.isfinite(zk)]
                if len(zk) >= 3:
                    # Quantile-based CI (standard subsampling)
                    ql, qh = np.quantile(zk, a/2), np.quantile(zk, 1-a/2)
                    clo, chi = tn[k]-qh/np.sqrt(n), tn[k]-ql/np.sqrt(n)
                    ci[lv][pnames[k]] = {
                        'covers': float(tp_est[k]>=clo and tp_est[k]<=chi),
                        'width': chi-clo}
                    # Normal CI (variance-based, bias-robust)
                    V_hat = float(np.var(zk, ddof=1))  # centered variance
                    se_hat = np.sqrt(V_hat / n)
                    clo_n = tn[k] - z_crit * se_hat
                    chi_n = tn[k] + z_crit * se_hat
                    ci_norm[lv][pnames[k]] = {
                        'covers': float(tp_est[k]>=clo_n and tp_est[k]<=chi_n),
                        'width': chi_n - clo_n}
                    continue
            ci[lv][pnames[k]] = {'covers': np.nan, 'width': np.nan}
            ci_norm[lv][pnames[k]] = {'covers': np.nan, 'width': np.nan}

    # Normality tests
    nr = {}
    for k in range(p):
        if sd_no.shape[0] >= 4:
            zk = sd_no[:,k]; zk = zk[np.isfinite(zk)]
            if len(zk) >= 4:
                mu, sg = float(np.mean(zk)), float(np.std(zk, ddof=1))
                zs = (zk-mu)/sg if sg > 0 else zk
                ksp = stats.kstest(zs,'norm')[1]
                swp = stats.shapiro(zs)[1] if len(zs)<=5000 else np.nan
                jbp = stats.jarque_bera(zk)[1]
                nr[pnames[k]] = {
                    'mean':mu, 'std':sg,
                    'skewness':float(stats.skew(zk)),
                    'kurtosis':float(stats.kurtosis(zk)),
                    'ks_rej':float(ksp<0.05),
                    'sw_rej':float(swp<0.05) if not np.isnan(swp) else np.nan,
                    'jb_rej':float(jbp<0.05), 'n_blocks':len(zk)}
                continue
        nr[pnames[k]] = {s: np.nan for s in
            ['mean','std','skewness','kurtosis','ks_rej','sw_rej','jb_rej']}
        nr[pnames[k]]['n_blocks'] = 0

    return {'theta_n':tn, 'sd_no':sd_no, 'ci_res':ci, 'ci_norm_res':ci_norm,
            'norm_res':nr, 'n_no':sd_no.shape[0], 'n_ov':sd_ov.shape[0]}


# =============================================================================
# Engine
# =============================================================================

class _Engine:
    """Core engine shared by run_study and run_sensitivity."""

    def __init__(self, J=2, true_params=(0.8,0.3,0.7,0.9,1.5,0),
                 alpha=1.5, beta=0.0, fix_alpha=False, fix_beta=True):
        from stablemixture import StableMixtureDeconvolution
        self.J, self.alpha, self.beta = J, alpha, beta
        self.fix_alpha, self.fix_beta = fix_alpha, fix_beta
        self.true_params = true_params
        tmp = StableMixtureDeconvolution(
            num_components=J, alpha_value=alpha, beta_value=beta,
            fix_alpha=fix_alpha, fix_beta=fix_beta, verbose=False)
        self.param_names = tmp.params_names
        self.u_grid = np.linspace(-1.5, 1.5, 15)
        self.v_grid = np.linspace(-1.5, 1.5, 15)
        self.bounds = ([(0.01,0.99)]*J + [(0.005,5.0)]*J)
        if not fix_alpha: self.bounds.append((0.1, 1.99))
        if not fix_beta:  self.bounds.append((-0.99, 0.99))
        self.tp_est = _true_est(true_params, J, fix_alpha, fix_beta)

    def mc(self, n, M, b, seed=42, conf_levels=(0.90,0.95),
           ci_step=None, n_cores=-1, backend='loky'):
        """Run M replications in parallel."""
        if ci_step is None:
            Nb = n - b + 1
            ci_step = max(Nb // 100, 1)

        nno = n // b
        nov = len(range(0, n-b+1, max(ci_step,1)))
        print(f"  n={n}, b={b}, M={M} | non-ov={nno}, ov(step={ci_step})={nov}, "
              f"est/rep={nno+nov}")

        t0 = time.time()
        reps = Parallel(n_jobs=n_cores, backend=backend, verbose=5)(
            delayed(_one_rep)(
                seed+m, n, b, self.true_params, self.tp_est,
                self.J, self.alpha, self.beta, self.fix_alpha, self.fix_beta,
                self.u_grid, self.v_grid, self.bounds, self.param_names,
                conf_levels, ci_step)
            for m in range(M))
        print(f"  Done in {time.time()-t0:.0f}s")
        return self._agg(reps, n, b, M, conf_levels)

    def _agg(self, reps, n, b, M, cl):
        a = {'n':n, 'b':b, 'M':M, 'param_names':self.param_names, 'conf_levels':cl,
             'moments':{p:{} for p in self.param_names},
             'rej_rates':{p:{} for p in self.param_names},
             'coverage':{l:{} for l in cl}, 'ci_width':{l:{} for l in cl},
             'pooled_no':{p:[] for p in self.param_names}}

        ta = np.array([r['theta_n'] for r in reps if not np.any(np.isnan(r['theta_n']))])
        a['theta_n_mean'], a['theta_n_std'], a['true_params_est'] = {},{},{}
        for k,p in enumerate(self.param_names):
            a['true_params_est'][p] = float(self.tp_est[k])
            a['theta_n_mean'][p] = float(np.mean(ta[:,k])) if len(ta)>0 else np.nan
            a['theta_n_std'][p]  = float(np.std(ta[:,k]))  if len(ta)>0 else np.nan

        for p in self.param_names:
            for s in ['mean','std','skewness','kurtosis']:
                v=[r['norm_res'][p][s] for r in reps if not np.isnan(r['norm_res'][p].get(s,np.nan))]
                a['moments'][p][s] = float(np.mean(v)) if v else np.nan
            for t in ['ks_rej','sw_rej','jb_rej']:
                v=[r['norm_res'][p][t] for r in reps if not np.isnan(r['norm_res'][p].get(t,np.nan))]
                a['rej_rates'][p][t] = float(np.mean(v)) if v else np.nan

        for l in cl:
            for p in self.param_names:
                # Quantile-based CI
                c=[r['ci_res'][l][p]['covers'] for r in reps if not np.isnan(r['ci_res'][l][p].get('covers',np.nan))]
                w=[r['ci_res'][l][p]['width']  for r in reps if not np.isnan(r['ci_res'][l][p].get('width',np.nan))]
                a['coverage'][l][p] = float(np.mean(c)) if c else np.nan
                a['ci_width'][l][p] = float(np.mean(w)) if w else np.nan

        # Normal CI (variance-based)
        a['cov_norm'] = {l:{} for l in cl}
        a['w_norm'] = {l:{} for l in cl}
        for l in cl:
            for p in self.param_names:
                c=[r['ci_norm_res'][l][p]['covers'] for r in reps if not np.isnan(r['ci_norm_res'][l][p].get('covers',np.nan))]
                w=[r['ci_norm_res'][l][p]['width']  for r in reps if not np.isnan(r['ci_norm_res'][l][p].get('width',np.nan))]
                a['cov_norm'][l][p] = float(np.mean(c)) if c else np.nan
                a['w_norm'][l][p] = float(np.mean(w)) if w else np.nan

        for r in reps:
            sd = r['sd_no']
            if sd.shape[0]>0:
                for k,p in enumerate(self.param_names):
                    v=sd[:,k]; a['pooled_no'][p].extend(v[np.isfinite(v)].tolist())
        return a

    def table(self, agg):
        """Summary DataFrame with both quantile-based and normal CIs."""
        n_val = agg['n']
        rows = []
        for p in agg['param_names']:
            se = agg['theta_n_std'][p]
            ds = agg['moments'][p]['std']
            ratio = ds/(np.sqrt(n_val)*se) if se>0 and not np.isnan(se) and not np.isnan(ds) else np.nan
            row = {'Parameter':p, 'True':agg['true_params_est'][p],
                   'Avg Est.':agg['theta_n_mean'][p], 'Std Est.':se,
                   'Dev.Mean':agg['moments'][p]['mean'], 'Dev.Std':ds,
                   'Ratio':ratio,
                   'Skew':agg['moments'][p]['skewness'],
                   'Kurt':agg['moments'][p]['kurtosis'],
                   'SW Rej.':agg['rej_rates'][p]['sw_rej'],
                   'JB Rej.':agg['rej_rates'][p]['jb_rej']}
            for l in agg['conf_levels']:
                pct = int(l*100)
                row[f'qCov{pct}'] = agg['coverage'][l][p]       # quantile CI
                row[f'nCov{pct}'] = agg['cov_norm'][l][p]       # normal CI
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_diag(self, agg, path=None):
        """Diagnostic histograms + QQ + KDE."""
        ps = agg['param_names']; np_ = len(ps)
        fig, ax = plt.subplots(np_, 3, figsize=(15, 4*np_))
        if np_==1: ax = ax.reshape(1,-1)
        fig.suptitle(f"Diagnostics n={agg['n']}, b={agg['b']}, M={agg['M']}", fontsize=13)
        for i,p in enumerate(ps):
            z = np.array(agg['pooled_no'][p]); z = z[np.isfinite(z)]
            if len(z)<10: continue
            mu,sg = np.mean(z), np.std(z)
            zs = (z-mu)/sg if sg>0 else z; xr = np.linspace(zs.min(), zs.max(), 200)
            ax[i,0].hist(zs,bins=40,density=True,alpha=.7,edgecolor='k')
            ax[i,0].plot(xr,stats.norm.pdf(xr),'r-',lw=2); ax[i,0].set_title(f'{p}: Hist')
            stats.probplot(zs,dist='norm',plot=ax[i,1]); ax[i,1].set_title(f'{p}: QQ')
            from scipy.stats import gaussian_kde
            kde=gaussian_kde(zs); ax[i,2].plot(xr,kde(xr),'b-',lw=2,label='KDE')
            ax[i,2].plot(xr,stats.norm.pdf(xr),'r--',lw=2,label='N(0,1)')
            ax[i,2].set_title(f'{p}: KDE'); ax[i,2].legend()
            for j in range(3): ax[i,j].grid(True,alpha=.3)
        plt.tight_layout()
        if path: plt.savefig(path,dpi=300,bbox_inches='tight')
        plt.close()


# =============================================================================
# Public API
# =============================================================================

def run_study(sample_sizes: List[int] = [250, 500, 1000, 10000],
              M: int = 500,
              b_exponent: float = 0.75,
              n_cores: int = -1,
              seed: int = 42,
              ci_step: Optional[int] = None,
              output_dir: str = 'results',
              **engine_kwargs):
    """
    Full convergence study across sample sizes.

    Args:
        sample_sizes: List of n values.
        M: MC replications (500 is enough; more does NOT improve coverage).
        b_exponent: Block size exponent (b = floor(n^exp)). Default 0.75.
        n_cores: Cores for parallelization (-1 = all).
        seed: Base random seed.
        ci_step: Override thinning step (None = auto ~100 blocks).
        output_dir: Where to save .txt, .csv, .png files.
        **engine_kwargs: Passed to _Engine (true_params, alpha, etc.)

    Returns:
        Dict {n: summary_DataFrame}.
    """
    os.makedirs(output_dir, exist_ok=True)
    eng = _Engine(**engine_kwargs)
    all_df = {}
    all_agg = {}

    for n in sample_sizes:
        b = int(np.floor(n ** b_exponent))
        print(f"\n{'='*65}")
        print(f"n={n}, b={b} (n^{b_exponent})")

        agg = eng.mc(n, M, b, seed=seed, ci_step=ci_step, n_cores=n_cores)
        df = eng.table(agg)
        all_df[n] = df
        all_agg[n] = agg

        print(df.to_string(index=False))

        # Save .txt
        with open(os.path.join(output_dir, f'summary_n{n}.txt'), 'w') as f:
            f.write(f"Subsampling Results for n={n} (b={b}, M={M})\n")
            f.write("="*80 + "\n")
            f.write(df.to_string(index=False) + "\n")

        # Save .csv
        df.to_csv(os.path.join(output_dir, f'summary_n{n}.csv'), index=False)

        # Save .png
        eng.plot_diag(agg, os.path.join(output_dir, f'diagnostics_n{n}.png'))

    # Combined .txt
    with open(os.path.join(output_dir, 'summary_all.txt'), 'w') as f:
        f.write("SUBSAMPLING CONVERGENCE STUDY\n" + "="*80 + "\n\n")
        for n in sample_sizes:
            a = all_agg[n]
            f.write(f"n={n}  (b={a['b']}, M={a['M']})\n" + "-"*80 + "\n")
            f.write(all_df[n].to_string(index=False) + "\n\n")

    # Convergence plot (8 panels: diagnostics + both CI types)
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
            ax = axes[i//4, i%4]
            for p in params:
                vals = [float(all_df[nn][all_df[nn]['Parameter']==p][col].values[0])
                        for nn in ns if nn in all_df]
                ax.plot(ns, vals, marker='o', lw=2, ms=7, label=p)
            ax.axhline(ref, color='red', ls='--', lw=2, label=f'Target ({ref})')
            ax.set_xlabel('n'); ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=7); ax.grid(True, alpha=.3); ax.set_xscale('log')
            if 'Cov' in col: ax.set_ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'convergence_plot.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nAll results in {output_dir}/")
    return all_df


def run_sensitivity(sample_sizes: List[int] = [500, 1000, 2000],
                    M: int = 100,
                    exponents: List[float] = [0.60, 2/3, 0.70, 0.75, 0.80],
                    n_cores: int = -1,
                    seed: int = 42,
                    output_dir: str = 'results_sensitivity',
                    **engine_kwargs):
    """
    Block size sensitivity analysis.

    For each (n, exponent), runs M replications and reports avg Ratio
    and avg coverage. Helps pick the best b exponent.

    Args:
        sample_sizes: List of n values.
        M: Replications per (n, exponent) — 100 is enough.
        exponents: Block size exponents to test.
        n_cores: Cores.
        seed: Base seed.
        output_dir: Where to save results.
        **engine_kwargs: Passed to _Engine.

    Returns:
        Dict {n: sensitivity_DataFrame}.
    """
    os.makedirs(output_dir, exist_ok=True)
    eng = _Engine(**engine_kwargs)
    all_sens = {}

    for n in sample_sizes:
        print(f"\n{'#'*65}")
        print(f"SENSITIVITY: n={n}")
        print(f"{'#'*65}")

        rows = []
        for exp in exponents:
            b = int(np.floor(n ** exp))
            if b < 20 or b >= n - 10:
                print(f"  exp={exp:.3f}, b={b}: SKIPPED (too small/large)")
                continue

            agg = eng.mc(n, M, b, seed=seed, n_cores=n_cores)
            df = eng.table(agg)

            avg_ratio = df['Ratio'].mean()
            ratio_dev = (df['Ratio'] - 1).abs().mean()
            avg_cov90 = df['nCov90'].mean()
            avg_cov95 = df['nCov95'].mean()

            rows.append({
                'exp': exp, 'b': b, 'n/b': n//b,
                'avg_ratio': round(avg_ratio, 3),
                '|ratio-1|': round(ratio_dev, 3),
                'avg_cov90': round(avg_cov90, 3),
                'avg_cov95': round(avg_cov95, 3)
            })
            print(f"  exp={exp:.3f}, b={b}: ratio={avg_ratio:.3f}, "
                  f"cov95={avg_cov95:.3f}")

        sens_df = pd.DataFrame(rows)
        all_sens[n] = sens_df

        with open(os.path.join(output_dir, f'sensitivity_n{n}.txt'), 'w') as f:
            f.write(f"Block Size Sensitivity for n={n} (M={M})\n")
            f.write("="*60 + "\n")
            f.write(sens_df.to_string(index=False) + "\n")
        print(f"\n{sens_df.to_string(index=False)}")

    print(f"\nAll results in {output_dir}/")
    return all_sens