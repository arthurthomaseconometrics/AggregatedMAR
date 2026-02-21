"""
Cluster-Optimized Subsampling Analysis for MDE of Stable AR(1) Aggregates

Parallelization strategy:
- Level 1 (outer): M replications distributed across cores via joblib.
- Level 2 (inner): subsample estimations within each replication are sequential
  (avoids nested parallelism overhead; each replication is lightweight with
  non-overlapping blocks).

On a cluster with C cores, the M replications are split into ~M/C batches.
Each replication involves ~n/b + n/(step) subsample estimations with vectorized
ECF, so individual replications are fast.

Usage on a SLURM cluster:
    # In your SLURM script:
    python subsampling_cluster.py --n_cores 64 --M 500 --sample_sizes 250 500 1000 2000 3000

Author: Revised analysis
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
from scipy import stats, optimize
from typing import List, Dict, Optional, Tuple
from joblib import Parallel, delayed
import warnings
import os
import time
import argparse

warnings.filterwarnings('ignore')


# =============================================================================
# Standalone functions (top-level for pickling with joblib)
# =============================================================================

def vectorized_ecf(Y: np.ndarray,
                   u_grid: np.ndarray,
                   v_grid: np.ndarray) -> np.ndarray:
    """
    Compute the empirical joint CF on the full (u,v) grid via broadcasting.

    Args:
        Y: Time series of length n.
        u_grid: 1D array of u points (nu,).
        v_grid: 1D array of v points (nv,).

    Returns:
        Complex array of shape (nu, nv).
    """
    Y1 = Y[1:]
    Y0 = Y[:-1]
    phase = (u_grid[:, None, None] * Y1[None, None, :]
             + v_grid[None, :, None] * Y0[None, None, :])
    return np.mean(np.exp(1j * phase), axis=2)


def vectorized_tcf(params: np.ndarray,
                   u_grid: np.ndarray,
                   v_grid: np.ndarray,
                   J: int,
                   alpha_fixed: Optional[float],
                   beta_val: float) -> np.ndarray:
    """
    Compute the theoretical joint CF on the full (u,v) grid (vectorized).

    Args:
        params: Parameter array.
        u_grid: 1D array of u points.
        v_grid: 1D array of v points.
        J: Number of components.
        alpha_fixed: Fixed alpha or None.
        beta_val: Beta value.

    Returns:
        Complex array of shape (nu, nv).
    """
    rhos = params[:J]
    varsigmas = params[J:2 * J]
    idx = 2 * J
    alpha = params[idx] if alpha_fixed is None else alpha_fixed
    beta = beta_val
    sigma = np.sum(varsigmas)
    pis = varsigmas / sigma

    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
    exponent = np.zeros_like(U, dtype=complex)

    if alpha != 1:
        tan_term = np.tan(np.pi * alpha / 2)
        for j in range(J):
            arg1 = U + rhos[j] * V
            comp = -pis[j]**alpha * sigma**alpha * (
                np.abs(arg1)**alpha * (1 - 1j * beta * np.sign(arg1) * tan_term)
                / (1 - np.abs(rhos[j])**alpha)
                + np.abs(V)**alpha * (1 - 1j * beta * np.sign(V) * tan_term)
            )
            exponent += comp
    else:
        for j in range(J):
            arg1 = U + rhos[j] * V
            comp = -pis[j] * sigma * (
                np.abs(arg1) * (1 + 1j * beta * np.sign(arg1) * 2 / np.pi
                                * np.log(np.abs(arg1) + 1e-10))
                / (1 - np.abs(rhos[j]))
                + np.abs(V) * (1 + 1j * beta * np.sign(V) * 2 / np.pi
                               * np.log(np.abs(V) + 1e-10))
            )
            exponent += comp

    return np.exp(exponent)


def fast_objective(params: np.ndarray,
                   ecf: np.ndarray,
                   u_grid: np.ndarray,
                   v_grid: np.ndarray,
                   J: int,
                   alpha_fixed: Optional[float],
                   beta_val: float) -> float:
    """
    MDE objective: mean squared distance between empirical and theoretical CF.

    Args:
        params: Parameter array.
        ecf: Precomputed empirical CF (nu, nv).
        u_grid: u grid points.
        v_grid: v grid points.
        J: Number of components.
        alpha_fixed: Fixed alpha or None.
        beta_val: Beta value.

    Returns:
        Scalar objective.
    """
    for j in range(J):
        if params[j] <= 0 or params[j] >= 1:
            return 1e10
    for j in range(J, 2 * J):
        if params[j] <= 0:
            return 1e10
    idx = 2 * J
    a = params[idx] if alpha_fixed is None else alpha_fixed
    if a <= 0 or a > 2:
        return 1e10

    tcf = vectorized_tcf(params, u_grid, v_grid, J, alpha_fixed, beta_val)
    return np.mean(np.abs(ecf - tcf)**2)


def estimate_on_subsample(Y_sub: np.ndarray,
                          x0: np.ndarray,
                          u_grid: np.ndarray,
                          v_grid: np.ndarray,
                          J: int,
                          bounds: list,
                          alpha_fixed: Optional[float],
                          beta_val: float,
                          maxiter: int = 60) -> np.ndarray:
    """
    Fast MDE on a subsample: precompute ECF, warm-start L-BFGS-B.

    Args:
        Y_sub: Subsample array.
        x0: Initial guess (warm start).
        u_grid: u grid.
        v_grid: v grid.
        J: Number of components.
        bounds: Parameter bounds.
        alpha_fixed: Fixed alpha or None.
        beta_val: Beta value.
        maxiter: Max iterations.

    Returns:
        Estimated parameter array (NaN on failure).
    """
    try:
        ecf = vectorized_ecf(Y_sub, u_grid, v_grid)
        res = optimize.minimize(
            fast_objective, x0,
            args=(ecf, u_grid, v_grid, J, alpha_fixed, beta_val),
            bounds=bounds, method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return np.full(len(x0), np.nan)


def _build_true_params_est(true_params_full: tuple,
                           J: int,
                           fix_alpha: bool,
                           fix_beta: bool) -> np.ndarray:
    """
    Map the full DGP parameter tuple (2J+2 elements, always includes alpha
    and beta) to the subset actually estimated (drops fixed parameters).

    Args:
        true_params_full: (rho_1,...,rho_J, varsigma_1,...,varsigma_J, alpha, beta).
        J: Number of components.
        fix_alpha: Whether alpha is fixed during estimation.
        fix_beta: Whether beta is fixed during estimation.

    Returns:
        1D array aligned with param_names / estimated vector.
    """
    vals = list(true_params_full[:2 * J])
    if not fix_alpha:
        vals.append(true_params_full[2 * J])
    if not fix_beta:
        vals.append(true_params_full[2 * J + 1])
    return np.array(vals)


def run_single_replication(seed: int,
                           n: int,
                           b: int,
                           true_params_full: tuple,
                           true_params_est: np.ndarray,
                           J: int,
                           alpha_val: float,
                           beta_val: float,
                           fix_alpha: bool,
                           fix_beta: bool,
                           u_grid: np.ndarray,
                           v_grid: np.ndarray,
                           bounds: list,
                           param_names: list,
                           conf_levels: Tuple[float, ...],
                           ci_step: int) -> Dict:
    """
    Execute one full MC replication (top-level for joblib pickling).

    Args:
        seed: Random seed for this replication.
        n: Sample size.
        b: Block size.
        true_params_full: Full DGP parameters (2J+2, for data generation).
        true_params_est: True values aligned with estimated vector (p elements).
        J: Number of components.
        alpha_val: Alpha value.
        beta_val: Beta value.
        fix_alpha: Whether alpha is fixed.
        fix_beta: Whether beta is fixed.
        u_grid: CF evaluation grid (u).
        v_grid: CF evaluation grid (v).
        bounds: Parameter bounds (length p).
        param_names: List of parameter names (length p).
        conf_levels: CI confidence levels.
        ci_step: Step for thinning overlapping blocks.

    Returns:
        Result dictionary for this replication.
    """
    from stablemixture import StableMixtureDeconvolution

    alpha_fixed = alpha_val if fix_alpha else None
    p = len(param_names)

    # Adapt maxiter to block size: more iterations for small blocks
    maxiter_sub = 100 if b < 80 else 60

    # 1. Generate data
    model = StableMixtureDeconvolution(
        num_components=J, alpha_value=alpha_val, beta_value=beta_val,
        fix_alpha=fix_alpha, fix_beta=fix_beta, verbose=False
    )
    Y, _ = model.generate(T=n, params=true_params_full, seed=seed)

    # 2. Full-sample estimate
    theta_n = np.array(model.estimate(Y, use_optimal_weights=False))

    # 3. Non-overlapping blocks (normality diagnostics)
    n_no = n // b
    no_ests = []
    for i in range(n_no):
        blk = Y[i * b:(i + 1) * b]
        est = estimate_on_subsample(blk, theta_n, u_grid, v_grid, J,
                                    bounds, alpha_fixed, beta_val,
                                    maxiter=maxiter_sub)
        no_ests.append(est)
    no_ests = np.array(no_ests)
    valid_no = ~np.isnan(no_ests).any(axis=1)
    no_ests = no_ests[valid_no]
    # Lower threshold to 3 (for small n, we may have only 6 blocks)
    sd_no = (np.sqrt(b) * (no_ests - theta_n)
             if len(no_ests) >= 3 else np.full((0, p), np.nan))

    # 4. Thinned overlapping blocks (CI construction)
    N_b = n - b + 1
    ov_indices = list(range(0, N_b, max(ci_step, 1)))
    ov_ests = []
    for i in ov_indices:
        blk = Y[i:i + b]
        est = estimate_on_subsample(blk, theta_n, u_grid, v_grid, J,
                                    bounds, alpha_fixed, beta_val,
                                    maxiter=maxiter_sub)
        ov_ests.append(est)
    ov_ests = np.array(ov_ests)
    valid_ov = ~np.isnan(ov_ests).any(axis=1)
    ov_ests = ov_ests[valid_ov]
    sd_ov = (np.sqrt(b) * (ov_ests - theta_n)
             if len(ov_ests) >= 3 else np.full((0, p), np.nan))

    # 5. CIs — use true_params_est (aligned with estimated vector)
    ci_res = {}
    for lv in conf_levels:
        ci_res[lv] = {}
        a = 1 - lv
        for k in range(p):
            if sd_ov.shape[0] >= 3:
                z_k = sd_ov[:, k]
                z_k = z_k[np.isfinite(z_k)]
                if len(z_k) >= 3:
                    q_lo = np.quantile(z_k, a / 2)
                    q_hi = np.quantile(z_k, 1 - a / 2)
                    ci_lo = theta_n[k] - q_hi / np.sqrt(n)
                    ci_hi = theta_n[k] - q_lo / np.sqrt(n)
                    covers = float(true_params_est[k] >= ci_lo
                                   and true_params_est[k] <= ci_hi)
                    ci_res[lv][param_names[k]] = {
                        'covers': covers, 'width': ci_hi - ci_lo
                    }
                    continue
            ci_res[lv][param_names[k]] = {'covers': np.nan, 'width': np.nan}

    # 6. Normality tests on non-overlapping deviations
    # Lower threshold to 4 for small n (allows n=250 with ~6 blocks)
    min_blocks_test = 4
    norm_res = {}
    for k in range(p):
        if sd_no.shape[0] >= min_blocks_test:
            z_k = sd_no[:, k]
            z_k = z_k[np.isfinite(z_k)]
            if len(z_k) >= min_blocks_test:
                mu, sig = float(np.mean(z_k)), float(np.std(z_k, ddof=1))
                z_std = (z_k - mu) / sig if sig > 0 else z_k
                ks_p = stats.kstest(z_std, 'norm')[1]
                sw_p = stats.shapiro(z_std)[1] if len(z_std) <= 5000 else np.nan
                jb_p = stats.jarque_bera(z_k)[1]
                norm_res[param_names[k]] = {
                    'mean': mu, 'std': sig,
                    'skewness': float(stats.skew(z_k)),
                    'kurtosis': float(stats.kurtosis(z_k)),
                    'ks_rej': float(ks_p < 0.05),
                    'sw_rej': float(sw_p < 0.05) if not np.isnan(sw_p) else np.nan,
                    'jb_rej': float(jb_p < 0.05),
                    'n_blocks': len(z_k)
                }
                continue
        norm_res[param_names[k]] = {
            'mean': np.nan, 'std': np.nan, 'skewness': np.nan, 'kurtosis': np.nan,
            'ks_rej': np.nan, 'sw_rej': np.nan, 'jb_rej': np.nan, 'n_blocks': 0
        }

    return {
        'theta_n': theta_n,
        'sd_no': sd_no,
        'ci_res': ci_res,
        'norm_res': norm_res,
        'n_no': sd_no.shape[0],
        'n_ov': sd_ov.shape[0]
    }


# =============================================================================
# Main analysis class
# =============================================================================

class ClusterSubsamplingAnalysis:
    """
    Subsampling analysis optimized for cluster computing.

    Parallelization is at the replication level: each of the M independent
    replications is dispatched to a separate core. This avoids nested
    parallelism and maximizes throughput on multi-core clusters.

    Attributes:
        J (int): Number of components.
        true_params (tuple): True DGP parameters.
        alpha (float): Tail index.
        beta (float): Asymmetry.
        param_names (list): Parameter names.
    """

    def __init__(self,
                 num_components: int = 2,
                 true_params: Optional[tuple] = None,
                 alpha: float = 1.5,
                 beta: float = 0.0,
                 fix_alpha: bool = False,
                 fix_beta: bool = True):
        """
        Initialize.

        Args:
            num_components: Number of latent components.
            true_params: True parameter values.
            alpha: Tail index.
            beta: Asymmetry.
            fix_alpha: Fix alpha during estimation.
            fix_beta: Fix beta during estimation.
        """
        from stablemixture import StableMixtureDeconvolution

        self.J = num_components
        self.alpha = alpha
        self.beta = beta
        self.fix_alpha = fix_alpha
        self.fix_beta = fix_beta

        if true_params is None:
            self.true_params = (0.8, 0.3, 0.7, 0.9, alpha, beta)
        else:
            self.true_params = true_params

        tmp = StableMixtureDeconvolution(
            num_components=num_components, alpha_value=alpha, beta_value=beta,
            fix_alpha=fix_alpha, fix_beta=fix_beta, verbose=False
        )
        self.param_names = tmp.params_names

        self.u_grid = np.linspace(-1.5, 1.5, 15)
        self.v_grid = np.linspace(-1.5, 1.5, 15)

        self.bounds = [(0.01, 0.99)] * num_components + [(0.005, 5.0)] * num_components
        if not fix_alpha:
            self.bounds.append((0.1, 1.99))
        if not fix_beta:
            self.bounds.append((-0.99, 0.99))

    def monte_carlo(self,
                    n: int,
                    M: int = 500,
                    b: Optional[int] = None,
                    seed: int = 42,
                    conf_levels: Tuple[float, ...] = (0.90, 0.95),
                    ci_step: Optional[int] = None,
                    n_cores: int = -1,
                    backend: str = 'loky') -> Dict:
        """
        Run the full MC study, parallelized over replications.

        Args:
            n: Sample size.
            M: Number of MC replications.
            b: Block size (default: floor(n^{2/3})).
            seed: Base seed.
            conf_levels: CI confidence levels.
            ci_step: Thinning step for overlapping blocks (default: b//2).
            n_cores: Number of cores (-1 = all available).
            backend: joblib backend ('loky' or 'multiprocessing').

        Returns:
            Aggregated results dictionary.
        """
        if b is None:
            b = int(np.floor(n ** (2 / 3)))
        if ci_step is None:
            ci_step = 1  # Use ALL overlapping blocks for CI quantile accuracy

        n_no = n // b
        n_ov_thinned = len(range(0, n - b + 1, ci_step))
        est_per_rep = n_no + n_ov_thinned

        print(f"\n{'=' * 65}")
        print(f"CLUSTER MC SUBSAMPLING: n={n}, M={M}, b={b}, cores={n_cores}")
        print(f"  Non-overlapping blocks/rep: {n_no}")
        print(f"  Thinned overlapping blocks/rep: {n_ov_thinned}")
        print(f"  Estimations/rep: {est_per_rep}")
        print(f"  Total estimations: {M * est_per_rep}")
        print(f"{'=' * 65}")

        t0 = time.time()

        # Build true_params_est aligned with estimated vector
        true_params_est = _build_true_params_est(
            self.true_params, self.J, self.fix_alpha, self.fix_beta)

        # Parallel over M replications
        reps = Parallel(n_jobs=n_cores, backend=backend, verbose=10)(
            delayed(run_single_replication)(
                seed=seed + m,
                n=n, b=b,
                true_params_full=self.true_params,
                true_params_est=true_params_est,
                J=self.J,
                alpha_val=self.alpha,
                beta_val=self.beta,
                fix_alpha=self.fix_alpha,
                fix_beta=self.fix_beta,
                u_grid=self.u_grid,
                v_grid=self.v_grid,
                bounds=self.bounds,
                param_names=self.param_names,
                conf_levels=conf_levels,
                ci_step=ci_step
            )
            for m in range(M)
        )

        elapsed = time.time() - t0
        print(f"\nCompleted in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
        print(f"  Avg per replication: {elapsed / M:.2f}s")

        return self._aggregate(reps, n, b, M, conf_levels)

    def _aggregate(self, reps: List[Dict], n: int, b: int, M: int,
                   conf_levels: Tuple[float, ...]) -> Dict:
        """
        Aggregate results across replications.

        Args:
            reps: List of per-replication result dicts.
            n, b, M: Study parameters.
            conf_levels: CI levels.

        Returns:
            Aggregated dictionary.
        """
        agg = {
            'n': n, 'b': b, 'M': M,
            'param_names': self.param_names,
            'conf_levels': conf_levels,
            'moments': {p: {} for p in self.param_names},
            'rej_rates': {p: {} for p in self.param_names},
            'coverage': {lv: {} for lv in conf_levels},
            'ci_width': {lv: {} for lv in conf_levels},
            'pooled_no': {p: [] for p in self.param_names},
        }

        # Average full-sample estimates across replications
        theta_n_all = np.array([r['theta_n'] for r in reps
                                if not np.any(np.isnan(r['theta_n']))])
        agg['theta_n_mean'] = {}
        agg['theta_n_std'] = {}
        true_est = _build_true_params_est(
            self.true_params, self.J, self.fix_alpha, self.fix_beta)
        agg['true_params_est'] = {}
        for k, param in enumerate(self.param_names):
            agg['true_params_est'][param] = float(true_est[k])
            if len(theta_n_all) > 0:
                agg['theta_n_mean'][param] = float(np.mean(theta_n_all[:, k]))
                agg['theta_n_std'][param] = float(np.std(theta_n_all[:, k]))
            else:
                agg['theta_n_mean'][param] = np.nan
                agg['theta_n_std'][param] = np.nan

        for param in self.param_names:
            for stat in ['mean', 'std', 'skewness', 'kurtosis']:
                vals = [r['norm_res'][param][stat] for r in reps
                        if not np.isnan(r['norm_res'][param].get(stat, np.nan))]
                agg['moments'][param][stat] = float(np.mean(vals)) if vals else np.nan

            for test in ['ks_rej', 'sw_rej', 'jb_rej']:
                vals = [r['norm_res'][param][test] for r in reps
                        if not np.isnan(r['norm_res'][param].get(test, np.nan))]
                agg['rej_rates'][param][test] = float(np.mean(vals)) if vals else np.nan

        for lv in conf_levels:
            for param in self.param_names:
                covs = [r['ci_res'][lv][param]['covers'] for r in reps
                        if not np.isnan(r['ci_res'][lv][param].get('covers', np.nan))]
                wids = [r['ci_res'][lv][param]['width'] for r in reps
                        if not np.isnan(r['ci_res'][lv][param].get('width', np.nan))]
                agg['coverage'][lv][param] = float(np.mean(covs)) if covs else np.nan
                agg['ci_width'][lv][param] = float(np.mean(wids)) if wids else np.nan

        for r in reps:
            sd = r['sd_no']
            if sd.shape[0] > 0:
                for k, param in enumerate(self.param_names):
                    v = sd[:, k]
                    agg['pooled_no'][param].extend(v[np.isfinite(v)].tolist())

        return agg

    # =========================================================================
    # Output
    # =========================================================================

    def summary_table(self, agg: Dict) -> pd.DataFrame:
        """
        Build summary DataFrame from aggregated results.

        Columns:
        - Parameter: parameter name.
        - True: true DGP value.
        - Avg Est.: average full-sample MDE across M replications.
        - Std Est.: std of full-sample MDE across M replications
          (proxy for sqrt(V_kk / n)).
        - Dev.Mean: average within-rep mean of sqrt(b)(theta_b^i - theta_n).
        - Dev.Std: average within-rep std of scaled deviations.
        - Ratio: Dev.Std / (sqrt(b) * Std Est.) — should be ~1 if subsampling
          variance is consistent with full-sample variance.
        - Skewness, Kurtosis: should -> 0 under normality.
        - KS/SW/JB Rej.: per-replication rejection rate (nominal 5%).
        - Cov 90%/95%: empirical coverage of subsampling CI.
        - Width 90%/95%: average CI width.

        Args:
            agg: Aggregated results.

        Returns:
            Summary DataFrame.
        """
        b = agg['b']
        rows = []
        for param in agg['param_names']:
            std_est = agg['theta_n_std'][param]
            dev_std = agg['moments'][param]['std']
            # Ratio = Dev.Std / (sqrt(b) * Std Est.)
            if std_est > 0 and not np.isnan(std_est) and not np.isnan(dev_std):
                ratio = dev_std / (np.sqrt(b) * std_est)
            else:
                ratio = np.nan

            row = {
                'Parameter': param,
                'True': agg['true_params_est'][param],
                'Avg Est.': agg['theta_n_mean'][param],
                'Std Est.': std_est,
                'Dev.Mean': agg['moments'][param]['mean'],
                'Dev.Std': dev_std,
                'Ratio': ratio,
                'Skewness': agg['moments'][param]['skewness'],
                'Kurtosis': agg['moments'][param]['kurtosis'],
                'KS Rej.': agg['rej_rates'][param]['ks_rej'],
                'SW Rej.': agg['rej_rates'][param]['sw_rej'],
                'JB Rej.': agg['rej_rates'][param]['jb_rej'],
            }
            for lv in agg['conf_levels']:
                row[f'Cov {int(lv * 100)}%'] = agg['coverage'][lv][param]
                row[f'Width {int(lv * 100)}%'] = agg['ci_width'][lv][param]
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_diagnostics(self, agg: Dict, save_path: Optional[str] = None):
        """
        Diagnostic plots from pooled non-overlapping deviations.

        Args:
            agg: Aggregated results.
            save_path: File path.
        """
        if save_path:
            d = os.path.dirname(save_path)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

        params = agg['param_names']
        n_p = len(params)
        fig, axes = plt.subplots(n_p, 3, figsize=(15, 4 * n_p))
        if n_p == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            f"Non-overlapping diagnostics (n={agg['n']}, b={agg['b']}, M={agg['M']})",
            fontsize=13, fontweight='bold')

        for idx, param in enumerate(params):
            z = np.array(agg['pooled_no'][param])
            z = z[np.isfinite(z)]
            if len(z) < 10:
                for j in range(3):
                    axes[idx, j].text(0.5, 0.5, 'Insufficient data',
                                      ha='center', va='center',
                                      transform=axes[idx, j].transAxes)
                continue
            mu, sig = np.mean(z), np.std(z)
            z_s = (z - mu) / sig if sig > 0 else z
            xr = np.linspace(z_s.min(), z_s.max(), 200)

            axes[idx, 0].hist(z_s, bins=40, density=True, alpha=0.7, edgecolor='black')
            axes[idx, 0].plot(xr, stats.norm.pdf(xr), 'r-', lw=2)
            axes[idx, 0].set_title(f'{param}: Histogram')
            axes[idx, 0].grid(True, alpha=0.3)

            stats.probplot(z_s, dist="norm", plot=axes[idx, 1])
            axes[idx, 1].set_title(f'{param}: Q-Q')
            axes[idx, 1].grid(True, alpha=0.3)

            from scipy.stats import gaussian_kde
            kde = gaussian_kde(z_s)
            axes[idx, 2].plot(xr, kde(xr), 'b-', lw=2, label='KDE')
            axes[idx, 2].plot(xr, stats.norm.pdf(xr), 'r--', lw=2, label='N(0,1)')
            axes[idx, 2].set_title(f'{param}: KDE')
            axes[idx, 2].legend()
            axes[idx, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def convergence_plot(self,
                         all_summaries: Dict[int, pd.DataFrame],
                         save_path: Optional[str] = None):
        """
        Convergence plot across sample sizes (6 panels).

        Args:
            all_summaries: {n: summary_df} for each sample size.
            save_path: File path.
        """
        if save_path:
            d = os.path.dirname(save_path)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

        ns = sorted(all_summaries.keys())
        params = all_summaries[ns[0]]['Parameter'].tolist()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        panels = [
            ('Dev.Mean', 'Mean of Scaled Deviations', 0),
            ('Skewness', 'Skewness', 0),
            ('Kurtosis', 'Excess Kurtosis', 0),
            ('KS Rej.', 'KS Rejection Rate (per rep.)', 0.05),
            ('Cov 90%', 'Subsampling CI Coverage 90%', 0.90),
            ('Cov 95%', 'Subsampling CI Coverage 95%', 0.95),
        ]

        for i, (col, title, ref) in enumerate(panels):
            ax = axes[i // 3, i % 3]
            for param in params:
                vals = []
                for nn in ns:
                    df = all_summaries[nn]
                    v = df[df['Parameter'] == param][col].values
                    vals.append(v[0] if len(v) > 0 else np.nan)
                ax.plot(ns, vals, marker='o', lw=2, ms=7, label=param)
            ax.axhline(ref, color='red', ls='--', lw=2,
                       label=f'Expected ({ref})')
            ax.set_xlabel('Sample Size (n)')
            ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            if 'Cov' in col:
                ax.set_ylim([0.5, 1.05])
            if 'Rej' in col:
                ax.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    """
    Command-line interface for cluster execution.

    Example:
        python subsampling_cluster.py --n_cores 64 --M 500 \\
            --sample_sizes 250 500 1000 2000 3000 --output_dir results
    """
    parser = argparse.ArgumentParser(
        description='Cluster-optimized subsampling analysis')
    parser.add_argument('--n_cores', type=int, default=-1,
                        help='Number of cores (-1 = all)')
    parser.add_argument('--M', type=int, default=500,
                        help='MC replications per sample size')
    parser.add_argument('--sample_sizes', type=int, nargs='+',
                        default=[250, 500, 1000, 2000, 3000])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--backend', type=str, default='loky',
                        choices=['loky', 'multiprocessing'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    analyzer = ClusterSubsamplingAnalysis(
        num_components=2,
        true_params=(0.8, 0.3, 0.7, 0.9, 1.5, 0),
        alpha=1.5, beta=0,
        fix_alpha=False, fix_beta=True
    )

    all_summaries = {}
    all_aggs = {}

    for n in args.sample_sizes:
        agg = analyzer.monte_carlo(
            n=n, M=args.M, seed=args.seed,
            n_cores=args.n_cores, backend=args.backend
        )
        all_aggs[n] = agg
        df = analyzer.summary_table(agg)
        all_summaries[n] = df

        print(f"\n--- Summary n={n} ---")
        print(df.to_string(index=False))

        # Save table as .txt (formatted) and .csv
        txt_path = os.path.join(args.output_dir, f'summary_n{n}.txt')
        with open(txt_path, 'w') as f:
            f.write(f"Subsampling Results for n={n} (b={agg['b']}, M={agg['M']})\n")
            f.write("=" * 80 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n")
        print(f"  Saved: {txt_path}")

        csv_path = os.path.join(args.output_dir, f'summary_n{n}.csv')
        df.to_csv(csv_path, index=False)

        # Save diagnostic plot as .png
        png_path = os.path.join(args.output_dir, f'diagnostics_n{n}.png')
        analyzer.plot_diagnostics(agg, save_path=png_path)
        print(f"  Saved: {png_path}")

    # Convergence plot
    conv_path = os.path.join(args.output_dir, 'convergence_plot.png')
    analyzer.convergence_plot(all_summaries, save_path=conv_path)
    print(f"\nConvergence plot saved: {conv_path}")

    # Combined summary .txt
    combined_path = os.path.join(args.output_dir, 'summary_all.txt')
    with open(combined_path, 'w') as f:
        f.write("SUBSAMPLING CONVERGENCE STUDY - COMBINED RESULTS\n")
        f.write("=" * 80 + "\n\n")
        for n in args.sample_sizes:
            agg = all_aggs[n]
            f.write(f"n={n}  (b={agg['b']}, M={agg['M']}, "
                    f"non-overlap blocks={n // agg['b']})\n")
            f.write("-" * 80 + "\n")
            f.write(all_summaries[n].to_string(index=False))
            f.write("\n\n")
    print(f"Combined summary saved: {combined_path}")

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()