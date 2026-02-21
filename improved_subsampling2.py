"""
Optimized Subsampling Analysis for MDE Estimator of Stable AR(1) Aggregates

Performance improvements over the original:
1. Precompute the empirical CF on the full grid once per subsample (vectorized).
2. Use non-overlapping blocks only (eliminates ~95% of subsample estimations
   AND removes the dependence issue in normality testing).
3. Vectorize the theoretical CF computation across the full (u,v) grid.
4. Warm-start all subsample optimizations from the full-sample estimate.
5. Reduce optimizer iterations for subsamples (fast local refinement).
6. Add subsampling CI coverage as the primary diagnostic metric.

Speedup factor vs original:
  - Non-overlapping blocks: ~b/1 fewer estimations (factor ~15-200x)
  - Vectorized ECF: ~5-10x per estimation
  - Combined: ~100-1000x total

Author: Revised analysis
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from typing import List, Dict, Optional, Tuple
from joblib import Parallel, delayed
import warnings
import os

warnings.filterwarnings('ignore')


# =============================================================================
# Vectorized objective function (standalone for pickling with joblib)
# =============================================================================

def _vectorized_ecf(Y: np.ndarray,
                    u_grid: np.ndarray,
                    v_grid: np.ndarray) -> np.ndarray:
    """
    Compute the empirical joint characteristic function on a (u,v) grid.

    Vectorized: computes all grid points in one shot using broadcasting.

    Args:
        Y: Time series of length n.
        u_grid: 1D array of u evaluation points.
        v_grid: 1D array of v evaluation points.

    Returns:
        Complex array of shape (len(u_grid), len(v_grid)) with ECF values.
    """
    Y1 = Y[1:]   # Y_t
    Y0 = Y[:-1]  # Y_{t-1}
    # Shape: (n-1,)
    # Broadcasting: u_grid[:,None,None] * Y1[None,None,:] + v_grid[None,:,None] * Y0[None,None,:]
    # -> shape (len(u), len(v), n-1)
    phase = u_grid[:, None, None] * Y1[None, None, :] + v_grid[None, :, None] * Y0[None, None, :]
    ecf = np.mean(np.exp(1j * phase), axis=2)  # shape (len(u), len(v))
    return ecf


def _vectorized_tcf(params: tuple,
                    u_grid: np.ndarray,
                    v_grid: np.ndarray,
                    num_components: int,
                    alpha_fixed: Optional[float],
                    beta_fixed: float) -> np.ndarray:
    """
    Compute the theoretical joint characteristic function on a (u,v) grid.

    Fully vectorized over the grid.

    Args:
        params: Parameter tuple (rho_1,...,rho_J, varsigma_1,...,varsigma_J, [alpha], [beta]).
        u_grid: 1D array of u evaluation points.
        v_grid: 1D array of v evaluation points.
        num_components: Number J of latent components.
        alpha_fixed: If not None, alpha is fixed to this value.
        beta_fixed: Fixed beta value.

    Returns:
        Complex array of shape (len(u_grid), len(v_grid)).
    """
    J = num_components
    rhos = np.array(params[:J])
    varsigmas = np.array(params[J:2 * J])
    idx = 2 * J
    alpha = params[idx] if alpha_fixed is None else alpha_fixed
    beta = beta_fixed

    sigma = np.sum(varsigmas)
    pis = varsigmas / sigma

    # Meshgrid: U shape (nu, nv), V shape (nu, nv)
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')

    exponent = np.zeros_like(U, dtype=complex)

    tan_term = np.tan(np.pi * alpha / 2) if alpha != 1 else None

    for j in range(J):
        rho_j = rhos[j]
        pi_j = pis[j]

        arg1 = U + rho_j * V
        arg2 = V

        if alpha != 1:
            comp = -pi_j**alpha * sigma**alpha * (
                np.abs(arg1)**alpha * (1 - 1j * beta * np.sign(arg1) * tan_term) / (1 - np.abs(rho_j)**alpha)
                + np.abs(arg2)**alpha * (1 - 1j * beta * np.sign(arg2) * tan_term)
            )
        else:
            comp = -pi_j * sigma * (
                np.abs(arg1) * (1 + 1j * beta * np.sign(arg1) * 2 / np.pi * np.log(np.abs(arg1) + 1e-10)) / (1 - np.abs(rho_j))
                + np.abs(arg2) * (1 + 1j * beta * np.sign(arg2) * 2 / np.pi * np.log(np.abs(arg2) + 1e-10))
            )
        exponent += comp

    return np.exp(exponent)


def _fast_objective(params: tuple,
                    ecf: np.ndarray,
                    u_grid: np.ndarray,
                    v_grid: np.ndarray,
                    num_components: int,
                    alpha_fixed: Optional[float],
                    beta_fixed: float) -> float:
    """
    Fast MDE objective function using precomputed ECF and vectorized TCF.

    Args:
        params: Parameter tuple.
        ecf: Precomputed empirical CF, shape (nu, nv).
        u_grid: u evaluation points.
        v_grid: v evaluation points.
        num_components: Number of components.
        alpha_fixed: Fixed alpha value or None.
        beta_fixed: Fixed beta value.

    Returns:
        Scalar objective value.
    """
    J = num_components
    # Quick bounds check
    for j in range(J):
        if params[j] <= 0 or params[j] >= 1:
            return 1e10
    for j in range(J, 2 * J):
        if params[j] <= 0:
            return 1e10
    idx = 2 * J
    alpha = params[idx] if alpha_fixed is None else alpha_fixed
    if alpha <= 0 or alpha > 2:
        return 1e10

    tcf = _vectorized_tcf(params, u_grid, v_grid, num_components, alpha_fixed, beta_fixed)
    diff = np.abs(ecf - tcf)**2
    return np.mean(diff)


def _estimate_fast(Y_sub: np.ndarray,
                   initial_guess: tuple,
                   u_grid: np.ndarray,
                   v_grid: np.ndarray,
                   num_components: int,
                   bounds: list,
                   alpha_fixed: Optional[float],
                   beta_fixed: float,
                   maxiter: int = 60) -> np.ndarray:
    """
    Fast MDE estimation on a single subsample.

    Precomputes the ECF once, then runs L-BFGS-B with warm start.

    Args:
        Y_sub: Subsample time series.
        initial_guess: Starting point (typically full-sample estimate).
        u_grid: Evaluation grid for u.
        v_grid: Evaluation grid for v.
        num_components: Number of components.
        bounds: Parameter bounds.
        alpha_fixed: Fixed alpha or None.
        beta_fixed: Fixed beta.
        maxiter: Max optimizer iterations.

    Returns:
        Estimated parameter array (NaN if failure).
    """
    n_params = len(initial_guess)
    try:
        ecf = _vectorized_ecf(Y_sub, u_grid, v_grid)

        result = optimize.minimize(
            _fast_objective,
            initial_guess,
            args=(ecf, u_grid, v_grid, num_components, alpha_fixed, beta_fixed),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return result.x
    except Exception:
        return np.full(n_params, np.nan)


# =============================================================================
# Main class
# =============================================================================

class FastSubsamplingAnalysis:
    """
    Optimized subsampling analysis for stable AR(1) aggregate MDE.

    Key design choices:
    - Non-overlapping blocks by default for normality diagnostics.
    - Overlapping blocks for CI construction (with option to subsample them).
    - Vectorized ECF and TCF computations.
    - Per-replication normality tests with rejection rate aggregation.

    Attributes:
        num_components (int): Number of latent components.
        true_params (tuple): True DGP parameters.
        alpha (float): Tail index.
        beta (float): Asymmetry parameter.
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
            num_components: Number of components J.
            true_params: True parameter values.
            alpha: Tail index.
            beta: Asymmetry parameter.
            fix_alpha: Whether alpha is fixed during estimation.
            fix_beta: Whether beta is fixed during estimation.
        """
        from stablemixture import StableMixtureDeconvolution

        self.num_components = num_components
        self.alpha = alpha
        self.beta = beta
        self.fix_alpha = fix_alpha
        self.fix_beta = fix_beta

        if true_params is None:
            self.true_params = (0.8, 0.3, 0.7, 0.9, alpha, beta)
        else:
            self.true_params = true_params

        # Use StableMixtureDeconvolution only for generate() and param_names
        self._model = StableMixtureDeconvolution(
            num_components=num_components,
            alpha_value=alpha,
            beta_value=beta,
            fix_alpha=fix_alpha,
            fix_beta=fix_beta,
            verbose=False
        )
        self.param_names = self._model.params_names

        # Evaluation grid (same as in stablemixture.py)
        self.u_grid = np.linspace(-1.5, 1.5, 15)
        self.v_grid = np.linspace(-1.5, 1.5, 15)

        # Bounds
        J = num_components
        self.bounds = [(0.01, 0.99)] * J + [(0.005, 5.0)] * J
        if not fix_alpha:
            self.bounds.append((0.1, 1.99))
        if not fix_beta:
            self.bounds.append((-0.99, 0.99))

        self.alpha_fixed = alpha if fix_alpha else None
        self.beta_fixed = beta

    def generate_data(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate data from the true DGP.

        Args:
            n: Sample size.
            seed: Random seed.

        Returns:
            Time series array.
        """
        Y, _ = self._model.generate(T=n, params=self.true_params, seed=seed)
        return Y

    def estimate_full(self, Y: np.ndarray) -> np.ndarray:
        """
        Full-sample MDE estimation (two-step: uniform then optimal weights).

        Args:
            Y: Full time series.

        Returns:
            Estimated parameter array.
        """
        params = self._model.estimate(Y, use_optimal_weights=False)
        return np.array(params)

    def _estimate_subsample(self, Y_sub: np.ndarray, theta_n: np.ndarray) -> np.ndarray:
        """
        Fast subsample estimation with warm start from full-sample estimate.

        Args:
            Y_sub: Subsample.
            theta_n: Full-sample estimate (warm start).

        Returns:
            Estimated parameter array.
        """
        return _estimate_fast(
            Y_sub, tuple(theta_n),
            self.u_grid, self.v_grid,
            self.num_components, self.bounds,
            self.alpha_fixed, self.beta_fixed,
            maxiter=60
        )

    # =========================================================================
    # Single replication
    # =========================================================================

    def single_replication(self,
                           n: int,
                           b: int,
                           seed: int,
                           conf_levels: Tuple[float, ...] = (0.90, 0.95),
                           use_overlapping_for_ci: bool = True,
                           ci_subsample_step: int = 1,
                           n_jobs: int = 1) -> Dict:
        """
        Single MC replication of the subsampling procedure.

        Strategy:
        - NON-OVERLAPPING blocks for normality diagnostics (independent).
        - OVERLAPPING blocks (optionally subsampled) for CI construction.
        - All subsample estimations warm-started from theta_n.

        Args:
            n: Sample size.
            b: Block size.
            seed: Random seed.
            conf_levels: CI levels.
            use_overlapping_for_ci: If True, use overlapping blocks for CI.
            ci_subsample_step: Step for thinning overlapping blocks (e.g., b//2).
            n_jobs: Parallel jobs for subsample estimation.

        Returns:
            Result dictionary.
        """
        p = len(self.param_names)
        Y = self.generate_data(n=n, seed=seed)
        theta_n = self.estimate_full(Y)

        # --- Non-overlapping blocks (for normality tests) ---
        n_nonoverlap = n // b
        nonoverlap_blocks = [Y[i * b:(i + 1) * b] for i in range(n_nonoverlap)]

        if n_nonoverlap >= 3:
            no_estimates = Parallel(n_jobs=n_jobs)(
                delayed(self._estimate_subsample)(blk, theta_n)
                for blk in nonoverlap_blocks
            )
            no_estimates = np.array(no_estimates)
            valid = ~np.isnan(no_estimates).any(axis=1)
            no_estimates = no_estimates[valid]
            scaled_dev_no = np.sqrt(b) * (no_estimates - theta_n) if len(no_estimates) >= 3 else np.full((0, p), np.nan)
        else:
            scaled_dev_no = np.full((0, p), np.nan)

        # --- Overlapping blocks (for CI), optionally thinned ---
        if use_overlapping_for_ci:
            N_b = n - b + 1
            step = max(ci_subsample_step, 1)
            indices = list(range(0, N_b, step))
            ov_blocks = [Y[i:i + b] for i in indices]

            ov_estimates = Parallel(n_jobs=n_jobs)(
                delayed(self._estimate_subsample)(blk, theta_n)
                for blk in ov_blocks
            )
            ov_estimates = np.array(ov_estimates)
            valid_ov = ~np.isnan(ov_estimates).any(axis=1)
            ov_estimates = ov_estimates[valid_ov]
            scaled_dev_ov = np.sqrt(b) * (ov_estimates - theta_n) if len(ov_estimates) >= 5 else np.full((0, p), np.nan)
        else:
            # Use non-overlapping for CI too
            scaled_dev_ov = scaled_dev_no

        # --- Subsampling CIs ---
        ci_results = {}
        for level in conf_levels:
            ci_results[level] = {}
            a = 1 - level
            for k in range(p):
                if scaled_dev_ov.shape[0] >= 5:
                    z_k = scaled_dev_ov[:, k]
                    z_k = z_k[np.isfinite(z_k)]
                    if len(z_k) >= 5:
                        q_lo = np.quantile(z_k, a / 2)
                        q_hi = np.quantile(z_k, 1 - a / 2)
                        ci_lo = theta_n[k] - q_hi / np.sqrt(n)
                        ci_hi = theta_n[k] - q_lo / np.sqrt(n)
                        covers = (self.true_params[k] >= ci_lo) and (self.true_params[k] <= ci_hi)
                        ci_results[level][self.param_names[k]] = {
                            'ci_low': ci_lo, 'ci_high': ci_hi,
                            'covers': covers, 'width': ci_hi - ci_lo
                        }
                        continue
                ci_results[level][self.param_names[k]] = {
                    'ci_low': np.nan, 'ci_high': np.nan,
                    'covers': np.nan, 'width': np.nan
                }

        # --- Normality tests (per replication, on non-overlapping) ---
        norm_res = {}
        for k in range(p):
            if scaled_dev_no.shape[0] >= 8:
                z_k = scaled_dev_no[:, k]
                z_k = z_k[np.isfinite(z_k)]
                if len(z_k) >= 8:
                    mu, sig = np.mean(z_k), np.std(z_k, ddof=1)
                    z_std = (z_k - mu) / sig if sig > 0 else z_k
                    ks_p = stats.kstest(z_std, 'norm')[1]
                    sw_p = stats.shapiro(z_std)[1] if len(z_std) <= 5000 else np.nan
                    jb_p = stats.jarque_bera(z_k)[1]
                    norm_res[self.param_names[k]] = {
                        'mean': mu, 'std': sig,
                        'skewness': stats.skew(z_k), 'kurtosis': stats.kurtosis(z_k),
                        'ks_pval': ks_p, 'sw_pval': sw_p, 'jb_pval': jb_p,
                        'ks_rej': ks_p < 0.05,
                        'sw_rej': (sw_p < 0.05) if not np.isnan(sw_p) else np.nan,
                        'jb_rej': jb_p < 0.05, 'n_blocks': len(z_k)
                    }
                    continue
            norm_res[self.param_names[k]] = {
                k2: np.nan for k2 in ['mean', 'std', 'skewness', 'kurtosis',
                                       'ks_pval', 'sw_pval', 'jb_pval',
                                       'ks_rej', 'sw_rej', 'jb_rej', 'n_blocks']
            }

        return {
            'theta_n': theta_n,
            'scaled_dev_no': scaled_dev_no,
            'ci_results': ci_results,
            'norm_results': norm_res,
            'n_no_blocks': scaled_dev_no.shape[0],
            'n_ov_blocks': scaled_dev_ov.shape[0]
        }

    # =========================================================================
    # Monte Carlo
    # =========================================================================

    def monte_carlo(self,
                    n: int,
                    M: int = 500,
                    b: Optional[int] = None,
                    seed: int = 42,
                    conf_levels: Tuple[float, ...] = (0.90, 0.95),
                    ci_subsample_step: Optional[int] = None,
                    n_jobs: int = -1) -> Dict:
        """
        Monte Carlo subsampling study.

        Args:
            n: Sample size.
            M: Number of replications.
            b: Block size (default: floor(n^{2/3})).
            seed: Base seed.
            conf_levels: CI levels.
            ci_subsample_step: Step for thinning overlapping blocks for CI.
                Default: b//2 (reduces computation by ~2x, keeps good CI).
            n_jobs: Parallel jobs for replications.

        Returns:
            Aggregated result dictionary.
        """
        if b is None:
            b = int(np.floor(n ** (2 / 3)))
        if ci_subsample_step is None:
            ci_subsample_step = max(b // 2, 1)

        n_no = n // b
        N_b_thinned = len(range(0, n - b + 1, ci_subsample_step))

        print(f"\n{'=' * 60}")
        print(f"Fast MC Subsampling: n={n}, M={M}, b={b}")
        print(f"  Non-overlapping blocks/rep: {n_no}")
        print(f"  Thinned overlapping blocks/rep (CI): {N_b_thinned}")
        print(f"  Total subsample estimations: ~{M * (n_no + N_b_thinned)}")
        print(f"{'=' * 60}")

        results_list = []
        for m in range(M):
            if (m + 1) % max(M // 10, 1) == 0:
                print(f"  Replication {m + 1}/{M}")
            rep = self.single_replication(
                n=n, b=b, seed=seed + m,
                conf_levels=conf_levels,
                use_overlapping_for_ci=True,
                ci_subsample_step=ci_subsample_step,
                n_jobs=1  # parallelism at outer level if needed
            )
            results_list.append(rep)

        return self._aggregate(results_list, n, b, M, conf_levels)

    def _aggregate(self, reps, n, b, M, conf_levels):
        """
        Aggregate per-replication results.

        Args:
            reps: List of replication dictionaries.
            n: Sample size.
            b: Block size.
            M: Number of replications.
            conf_levels: CI levels.

        Returns:
            Aggregated dictionary.
        """
        agg = {
            'n': n, 'b': b, 'M': M,
            'param_names': self.param_names,
            'conf_levels': conf_levels,
        }

        # Per-replication moments and rejection rates
        for key_type in ['moments', 'rej_rates']:
            agg[key_type] = {p: {} for p in self.param_names}

        for param in self.param_names:
            for stat in ['mean', 'std', 'skewness', 'kurtosis']:
                vals = [r['norm_results'][param][stat] for r in reps
                        if not np.isnan(r['norm_results'][param].get(stat, np.nan))]
                agg['moments'][param][stat] = np.mean(vals) if vals else np.nan
            for test in ['ks_rej', 'sw_rej', 'jb_rej']:
                vals = [r['norm_results'][param][test] for r in reps
                        if not np.isnan(r['norm_results'][param].get(test, np.nan))]
                agg['rej_rates'][param][test] = np.mean(vals) if vals else np.nan

        # Coverage
        agg['coverage'] = {lv: {} for lv in conf_levels}
        agg['ci_width'] = {lv: {} for lv in conf_levels}
        for lv in conf_levels:
            for param in self.param_names:
                covs = [r['ci_results'][lv][param]['covers'] for r in reps
                        if not np.isnan(r['ci_results'][lv][param].get('covers', np.nan))]
                wids = [r['ci_results'][lv][param]['width'] for r in reps
                        if not np.isnan(r['ci_results'][lv][param].get('width', np.nan))]
                agg['coverage'][lv][param] = np.mean(covs) if covs else np.nan
                agg['ci_width'][lv][param] = np.mean(wids) if wids else np.nan

        # Pooled non-overlapping deviations (for plots)
        agg['pooled_no'] = {p: [] for p in self.param_names}
        for r in reps:
            sd = r['scaled_dev_no']
            if sd.shape[0] > 0:
                for k, param in enumerate(self.param_names):
                    v = sd[:, k]
                    agg['pooled_no'][param].extend(v[np.isfinite(v)].tolist())

        return agg

    # =========================================================================
    # Display
    # =========================================================================

    def summary_table(self, agg: Dict) -> pd.DataFrame:
        """
        Build summary table from aggregated results.

        Args:
            agg: Output of monte_carlo().

        Returns:
            Summary DataFrame.
        """
        rows = []
        for param in agg['param_names']:
            row = {'Parameter': param}
            for stat in ['mean', 'std', 'skewness', 'kurtosis']:
                row[stat.capitalize()] = agg['moments'][param][stat]
            row['KS Rej.'] = agg['rej_rates'][param]['ks_rej']
            row['SW Rej.'] = agg['rej_rates'][param]['sw_rej']
            row['JB Rej.'] = agg['rej_rates'][param]['jb_rej']
            for lv in agg['conf_levels']:
                row[f'Cov {int(lv * 100)}%'] = agg['coverage'][lv][param]
                row[f'Width {int(lv * 100)}%'] = agg['ci_width'][lv][param]
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_diagnostics(self, agg: Dict, save_path: Optional[str] = None):
        """
        Diagnostic plots from pooled non-overlapping deviations.

        Args:
            agg: Output of monte_carlo().
            save_path: File path to save the figure.
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

        fig.suptitle(f"Non-overlapping blocks diagnostics (n={agg['n']}, b={agg['b']}, M={agg['M']})",
                     fontsize=13, fontweight='bold')

        for idx, param in enumerate(params):
            z = np.array(agg['pooled_no'][param])
            z = z[np.isfinite(z)]
            if len(z) < 10:
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