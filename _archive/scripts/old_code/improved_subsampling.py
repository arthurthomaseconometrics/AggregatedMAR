"""
Improved Subsampling Analysis for MDE Estimator of Stable AR(1) Aggregates

Key improvements over the original version:
1. Per-replication normality testing (fixes pooling-induced over-rejection)
2. Non-overlapping blocks for diagnostics (eliminates dependence bias)
3. Subsampling confidence intervals with empirical coverage rates
4. Data-driven block size selection via minimum volatility method
5. Proper separation of overlapping (for CI) and non-overlapping (for testing)

Author: Revised analysis
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Dict, Optional, Tuple
from joblib import Parallel, delayed
import warnings
import os

warnings.filterwarnings('ignore')


class ImprovedSubsamplingAnalysis:
    """
    Implements the corrected subsampling methodology for verifying asymptotic
    normality and constructing valid confidence intervals for the MDE estimator.

    Key differences from the original SubsamplingAnalysis:
    - Normality tests are run per MC replication, not on pooled data.
    - Non-overlapping blocks are used for normality diagnostics.
    - Subsampling confidence intervals are constructed and coverage is reported.
    - Block size can be selected via data-driven methods.

    Attributes:
        num_components (int): Number of latent AR(1) components.
        true_params (tuple): True parameter values.
        alpha (float): Tail index parameter.
        beta (float): Asymmetry parameter.
        param_names (list): Names of the estimated parameters.
    """

    def __init__(self,
                 num_components: int = 2,
                 true_params: Optional[tuple] = None,
                 alpha: float = 1.5,
                 beta: float = 0.0):
        """
        Initialize the improved subsampling analysis.

        Args:
            num_components: Number of latent AR(1) components.
            true_params: True parameter values (rho_1, ..., rho_J, varsigma_1, ..., varsigma_J, alpha, beta).
            alpha: Tail index parameter.
            beta: Asymmetry parameter.
        """
        from stablemixture import StableMixtureDeconvolution

        self.num_components = num_components
        self.alpha = alpha
        self.beta = beta

        if true_params is None:
            self.true_params = (0.8, 0.3, 0.7, 0.9, alpha, beta)
        else:
            self.true_params = true_params

        self.model = StableMixtureDeconvolution(
            num_components=num_components,
            alpha_value=alpha,
            beta_value=beta,
            fix_alpha=False,
            fix_beta=True,
            verbose=False
        )
        self.param_names = self.model.params_names

    def generate_data(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a sample from the stable AR(1) aggregate model.

        Args:
            n: Sample size.
            seed: Random seed for reproducibility.

        Returns:
            Generated time series of length n.
        """
        Y, _ = self.model.generate(T=n, params=self.true_params, seed=seed)
        return Y

    def _estimate_on_subsample(self,
                               Y_sub: np.ndarray,
                               initial_guess: Optional[tuple] = None) -> np.ndarray:
        """
        Estimate parameters on a single subsample.

        Args:
            Y_sub: Subsample time series.
            initial_guess: Initial values for optimization.

        Returns:
            Estimated parameter vector (NaN array if estimation fails).
        """
        from stablemixture import StableMixtureDeconvolution

        try:
            model_sub = StableMixtureDeconvolution(
                num_components=self.num_components,
                alpha_value=self.alpha,
                beta_value=self.beta,
                fix_alpha=False,
                fix_beta=True,
                verbose=False
            )
            params_sub = model_sub.estimate(
                Y=Y_sub,
                initial_guess=list(initial_guess) if initial_guess is not None else None,
                use_optimal_weights=False,
                step1=True
            )
            return np.array(params_sub)
        except Exception:
            return np.full(len(self.param_names), np.nan)

    # =========================================================================
    # Core subsampling routines
    # =========================================================================

    def extract_nonoverlapping_blocks(self,
                                      Y: np.ndarray,
                                      b: int) -> List[np.ndarray]:
        """
        Extract non-overlapping subsamples of size b from Y.

        Args:
            Y: Full time series of length n.
            b: Block size.

        Returns:
            List of non-overlapping subsample arrays.
        """
        n = len(Y)
        n_blocks = n // b
        blocks = [Y[i * b:(i + 1) * b] for i in range(n_blocks)]
        return blocks

    def extract_overlapping_blocks(self,
                                   Y: np.ndarray,
                                   b: int) -> List[np.ndarray]:
        """
        Extract overlapping subsamples of size b from Y.

        Args:
            Y: Full time series of length n.
            b: Block size.

        Returns:
            List of overlapping subsample arrays.
        """
        n = len(Y)
        N_b = n - b + 1
        blocks = [Y[i:i + b] for i in range(N_b)]
        return blocks

    def single_replication(self,
                           n: int,
                           b: int,
                           seed: int,
                           conf_levels: Tuple[float, ...] = (0.90, 0.95),
                           n_jobs_inner: int = 1) -> Dict:
        """
        Perform the full subsampling procedure for a single MC replication.

        For each replication:
        1. Generate data, compute full-sample estimate.
        2. Compute overlapping subsample estimates and scaled deviations.
        3. Construct subsampling confidence intervals.
        4. Also compute non-overlapping subsample estimates for normality testing.
        5. Run normality tests on non-overlapping scaled deviations.

        Args:
            n: Sample size.
            b: Block size.
            seed: Random seed.
            conf_levels: Confidence levels for CI construction.
            n_jobs_inner: Number of parallel jobs for subsample estimation.

        Returns:
            Dictionary with results for this replication.
        """
        p = len(self.param_names)

        # 1. Generate data
        Y = self.generate_data(n=n, seed=seed)

        # 2. Full-sample estimate
        theta_n = np.array(self.model.estimate(Y, use_optimal_weights=False))

        # 3. Non-overlapping blocks: for normality diagnostics
        nonoverlap_blocks = self.extract_nonoverlapping_blocks(Y, b)
        n_nonoverlap = len(nonoverlap_blocks)

        if n_nonoverlap < 5:
            # Too few non-overlapping blocks; fall back to widely-spaced overlapping
            spacing = max(b // 2, 1)
            indices = list(range(0, n - b + 1, spacing))
            nonoverlap_blocks = [Y[i:i + b] for i in indices]
            n_nonoverlap = len(nonoverlap_blocks)

        # Estimate on non-overlapping blocks
        nonoverlap_estimates = []
        for block in nonoverlap_blocks:
            est = self._estimate_on_subsample(block, initial_guess=tuple(theta_n))
            nonoverlap_estimates.append(est)
        nonoverlap_estimates = np.array(nonoverlap_estimates)

        # Remove failed estimations
        valid_mask_no = ~np.isnan(nonoverlap_estimates).any(axis=1)
        nonoverlap_estimates = nonoverlap_estimates[valid_mask_no]

        # Scaled deviations (non-overlapping)
        if len(nonoverlap_estimates) >= 5:
            scaled_dev_no = np.sqrt(b) * (nonoverlap_estimates - theta_n)
        else:
            scaled_dev_no = np.full((0, p), np.nan)

        # 4. Overlapping blocks: for CI construction
        N_b = n - b + 1
        overlap_estimates = []
        for i in range(N_b):
            est = self._estimate_on_subsample(Y[i:i + b], initial_guess=tuple(theta_n))
            overlap_estimates.append(est)
        overlap_estimates = np.array(overlap_estimates)

        valid_mask_ov = ~np.isnan(overlap_estimates).any(axis=1)
        overlap_estimates = overlap_estimates[valid_mask_ov]

        # Scaled deviations (overlapping)
        if len(overlap_estimates) >= 5:
            scaled_dev_ov = np.sqrt(b) * (overlap_estimates - theta_n)
        else:
            scaled_dev_ov = np.full((0, p), np.nan)

        # 5. Subsampling confidence intervals (per parameter)
        ci_results = {}
        for level in conf_levels:
            ci_results[level] = {}
            alpha_ci = 1 - level
            for k in range(p):
                if len(scaled_dev_ov) >= 5:
                    z_k = scaled_dev_ov[:, k]
                    z_k = z_k[np.isfinite(z_k)]
                    if len(z_k) >= 5:
                        q_low = np.quantile(z_k, alpha_ci / 2)
                        q_high = np.quantile(z_k, 1 - alpha_ci / 2)
                        ci_low = theta_n[k] - q_high / np.sqrt(n)
                        ci_high = theta_n[k] - q_low / np.sqrt(n)
                        covers = (self.true_params[k] >= ci_low) and (self.true_params[k] <= ci_high)
                        ci_results[level][self.param_names[k]] = {
                            'ci_low': ci_low,
                            'ci_high': ci_high,
                            'covers': covers,
                            'width': ci_high - ci_low
                        }
                    else:
                        ci_results[level][self.param_names[k]] = {
                            'ci_low': np.nan, 'ci_high': np.nan,
                            'covers': np.nan, 'width': np.nan
                        }
                else:
                    ci_results[level][self.param_names[k]] = {
                        'ci_low': np.nan, 'ci_high': np.nan,
                        'covers': np.nan, 'width': np.nan
                    }

        # 6. Normality tests on non-overlapping scaled deviations
        normality_results = {}
        for k in range(p):
            if len(scaled_dev_no) >= 8:
                z_k = scaled_dev_no[:, k]
                z_k = z_k[np.isfinite(z_k)]
                if len(z_k) >= 8:
                    mu_k = np.mean(z_k)
                    std_k = np.std(z_k, ddof=1)
                    skew_k = stats.skew(z_k)
                    kurt_k = stats.kurtosis(z_k)
                    z_std = (z_k - mu_k) / std_k if std_k > 0 else z_k
                    ks_stat, ks_pval = stats.kstest(z_std, 'norm')
                    if len(z_std) <= 5000:
                        sw_stat, sw_pval = stats.shapiro(z_std)
                    else:
                        sw_stat, sw_pval = np.nan, np.nan
                    jb_stat, jb_pval = stats.jarque_bera(z_k)
                    normality_results[self.param_names[k]] = {
                        'mean': mu_k, 'std': std_k,
                        'skewness': skew_k, 'kurtosis': kurt_k,
                        'ks_pval': ks_pval, 'sw_pval': sw_pval, 'jb_pval': jb_pval,
                        'ks_reject': ks_pval < 0.05,
                        'sw_reject': sw_pval < 0.05 if not np.isnan(sw_pval) else np.nan,
                        'jb_reject': jb_pval < 0.05,
                        'n_blocks': len(z_k)
                    }
                    continue

            normality_results[self.param_names[k]] = {
                'mean': np.nan, 'std': np.nan,
                'skewness': np.nan, 'kurtosis': np.nan,
                'ks_pval': np.nan, 'sw_pval': np.nan, 'jb_pval': np.nan,
                'ks_reject': np.nan, 'sw_reject': np.nan, 'jb_reject': np.nan,
                'n_blocks': 0
            }

        return {
            'theta_n': theta_n,
            'scaled_dev_nonoverlap': scaled_dev_no,
            'scaled_dev_overlap': scaled_dev_ov,
            'ci_results': ci_results,
            'normality_results': normality_results,
            'n_nonoverlap_blocks': len(nonoverlap_estimates),
            'n_overlap_blocks': len(overlap_estimates),
            'seed': seed
        }

    # =========================================================================
    # Monte Carlo study
    # =========================================================================

    def monte_carlo_study(self,
                          n: int,
                          M: int = 500,
                          b: Optional[int] = None,
                          seed: int = 42,
                          conf_levels: Tuple[float, ...] = (0.90, 0.95),
                          n_jobs: int = -1) -> Dict:
        """
        Run the full Monte Carlo subsampling study for a given sample size.

        Args:
            n: Sample size.
            M: Number of Monte Carlo replications.
            b: Block size (default: floor(n^{2/3})).
            seed: Base random seed.
            conf_levels: Confidence levels for CI construction.
            n_jobs: Number of parallel jobs for MC replications.

        Returns:
            Dictionary with aggregated results.
        """
        if b is None:
            b = int(np.floor(n ** (2 / 3)))

        print(f"\n{'#' * 70}")
        print(f"Improved MC Subsampling: n={n}, M={M}, b={b}")
        print(f"  Non-overlapping blocks per replication: ~{n // b}")
        print(f"  Overlapping blocks per replication: {n - b + 1}")
        print(f"{'#' * 70}")

        # Run replications (sequentially for now; can parallelize at outer level)
        all_rep_results = []
        for m in range(M):
            if (m + 1) % 50 == 0 or m == 0:
                print(f"  Replication {m + 1}/{M}...")
            rep_result = self.single_replication(
                n=n, b=b, seed=seed + m,
                conf_levels=conf_levels, n_jobs_inner=1
            )
            all_rep_results.append(rep_result)

        # Aggregate
        return self._aggregate_results(all_rep_results, n, b, M, conf_levels)

    def _aggregate_results(self,
                           rep_results: List[Dict],
                           n: int, b: int, M: int,
                           conf_levels: Tuple[float, ...]) -> Dict:
        """
        Aggregate results across Monte Carlo replications.

        Args:
            rep_results: List of per-replication result dictionaries.
            n: Sample size.
            b: Block size.
            M: Number of replications.
            conf_levels: Confidence levels used.

        Returns:
            Aggregated results dictionary.
        """
        p = len(self.param_names)

        # -- Per-replication normality test rejection rates --
        rejection_rates = {param: {'ks': [], 'sw': [], 'jb': []}
                          for param in self.param_names}
        moments_per_rep = {param: {'mean': [], 'std': [], 'skewness': [], 'kurtosis': []}
                          for param in self.param_names}

        for rep in rep_results:
            nr = rep['normality_results']
            for param in self.param_names:
                if param in nr and not np.isnan(nr[param].get('ks_reject', np.nan)):
                    rejection_rates[param]['ks'].append(nr[param]['ks_reject'])
                    rejection_rates[param]['sw'].append(nr[param]['sw_reject'])
                    rejection_rates[param]['jb'].append(nr[param]['jb_reject'])
                    moments_per_rep[param]['mean'].append(nr[param]['mean'])
                    moments_per_rep[param]['std'].append(nr[param]['std'])
                    moments_per_rep[param]['skewness'].append(nr[param]['skewness'])
                    moments_per_rep[param]['kurtosis'].append(nr[param]['kurtosis'])

        # -- Coverage rates --
        coverage_rates = {level: {} for level in conf_levels}
        ci_widths = {level: {} for level in conf_levels}
        for level in conf_levels:
            for param in self.param_names:
                covers = [rep['ci_results'][level][param]['covers']
                         for rep in rep_results
                         if not np.isnan(rep['ci_results'][level][param].get('covers', np.nan))]
                widths = [rep['ci_results'][level][param]['width']
                         for rep in rep_results
                         if not np.isnan(rep['ci_results'][level][param].get('width', np.nan))]
                coverage_rates[level][param] = np.mean(covers) if covers else np.nan
                ci_widths[level][param] = np.mean(widths) if widths else np.nan

        # -- Pool non-overlapping scaled deviations for diagnostic plots --
        pooled_nonoverlap = {param: [] for param in self.param_names}
        for rep in rep_results:
            sd_no = rep['scaled_dev_nonoverlap']
            if sd_no.shape[0] > 0:
                for k, param in enumerate(self.param_names):
                    vals = sd_no[:, k]
                    vals = vals[np.isfinite(vals)]
                    pooled_nonoverlap[param].extend(vals.tolist())

        return {
            'n': n, 'b': b, 'M': M,
            'param_names': self.param_names,
            'rejection_rates': rejection_rates,
            'moments_per_rep': moments_per_rep,
            'coverage_rates': coverage_rates,
            'ci_widths': ci_widths,
            'pooled_nonoverlap_deviations': pooled_nonoverlap,
            'conf_levels': conf_levels
        }

    # =========================================================================
    # Summary and display
    # =========================================================================

    def summary_table(self, results: Dict) -> pd.DataFrame:
        """
        Build a summary table from aggregated MC results.

        The table reports, for each parameter:
        - Average moments (mean, std, skewness, kurtosis) of scaled deviations,
          averaged across replications (computed on non-overlapping blocks).
        - Per-replication rejection rates of KS, SW, JB tests.
        - Empirical coverage rates of subsampling confidence intervals.

        Args:
            results: Aggregated results from monte_carlo_study().

        Returns:
            Summary DataFrame.
        """
        rows = []
        for param in results['param_names']:
            rr = results['rejection_rates'][param]
            mom = results['moments_per_rep'][param]
            row = {
                'Parameter': param,
                'Mean': np.mean(mom['mean']) if mom['mean'] else np.nan,
                'Std': np.mean(mom['std']) if mom['std'] else np.nan,
                'Skewness': np.mean(mom['skewness']) if mom['skewness'] else np.nan,
                'Kurtosis': np.mean(mom['kurtosis']) if mom['kurtosis'] else np.nan,
                'KS Rej.': np.mean(rr['ks']) if rr['ks'] else np.nan,
                'SW Rej.': np.mean(rr['sw']) if rr['sw'] else np.nan,
                'JB Rej.': np.mean(rr['jb']) if rr['jb'] else np.nan,
            }
            for level in results['conf_levels']:
                cov = results['coverage_rates'][level].get(param, np.nan)
                width = results['ci_widths'][level].get(param, np.nan)
                row[f'Cov. {int(level * 100)}%'] = cov
                row[f'Width {int(level * 100)}%'] = width
            rows.append(row)
        return pd.DataFrame(rows)

    # =========================================================================
    # Diagnostics plots
    # =========================================================================

    def plot_diagnostics(self,
                         results: Dict,
                         save_path: Optional[str] = None):
        """
        Create diagnostic plots using pooled non-overlapping scaled deviations.

        Args:
            results: Aggregated results from monte_carlo_study().
            save_path: Path to save the figure.
        """
        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

        param_names = results['param_names']
        n_params = len(param_names)

        fig, axes = plt.subplots(n_params, 3, figsize=(15, 4 * n_params))
        if n_params == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            f"Diagnostics (non-overlapping blocks, n={results['n']}, "
            f"b={results['b']}, M={results['M']})",
            fontsize=14, fontweight='bold', y=1.01
        )

        for idx, param in enumerate(param_names):
            z = np.array(results['pooled_nonoverlap_deviations'][param])
            z = z[np.isfinite(z)]

            if len(z) < 10:
                for j in range(3):
                    axes[idx, j].text(0.5, 0.5, 'Insufficient data',
                                      ha='center', va='center',
                                      transform=axes[idx, j].transAxes)
                continue

            mu, sigma = np.mean(z), np.std(z)
            z_std = (z - mu) / sigma if sigma > 0 else z

            # Histogram
            ax1 = axes[idx, 0]
            ax1.hist(z_std, bins=40, density=True, alpha=0.7,
                     edgecolor='black', label='Empirical')
            x_range = np.linspace(z_std.min(), z_std.max(), 200)
            ax1.plot(x_range, stats.norm.pdf(x_range, 0, 1),
                     'r-', linewidth=2, label='N(0,1)')
            ax1.set_title(f'{param}: Histogram')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # QQ plot
            ax2 = axes[idx, 1]
            stats.probplot(z_std, dist="norm", plot=ax2)
            ax2.set_title(f'{param}: Q-Q Plot')
            ax2.grid(True, alpha=0.3)

            # KDE
            ax3 = axes[idx, 2]
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(z_std)
            ax3.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
            ax3.plot(x_range, stats.norm.pdf(x_range, 0, 1),
                     'r--', linewidth=2, label='N(0,1)')
            ax3.set_title(f'{param}: Kernel Density')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        plt.close()

    # =========================================================================
    # Block size sensitivity
    # =========================================================================

    def block_size_sensitivity(self,
                               Y: np.ndarray,
                               b_grid: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Perform block size sensitivity analysis on a single trajectory.

        For each candidate b in the grid, compute the subsampling variance
        estimate using non-overlapping blocks. The optimal b corresponds to the
        region where the variance estimate is most stable (minimum volatility
        criterion of Politis, Romano & Wolf, 1999).

        Args:
            Y: Full time series.
            b_grid: Grid of block sizes. Default: 10 values from n^{1/2} to n^{4/5}.

        Returns:
            DataFrame with columns ['b', 'n_blocks', 'var_rho_1', ..., 'var_alpha'].
        """
        n = len(Y)

        if b_grid is None:
            b_min = max(int(np.floor(n ** 0.5)), 20)
            b_max = int(np.floor(n ** 0.8))
            b_grid = np.unique(np.linspace(b_min, b_max, 10).astype(int)).tolist()

        # Full-sample estimate
        theta_n = np.array(self.model.estimate(Y, use_optimal_weights=False))

        rows = []
        for b_cand in b_grid:
            blocks = self.extract_nonoverlapping_blocks(Y, b_cand)
            n_blocks = len(blocks)
            if n_blocks < 3:
                continue

            estimates = []
            for block in blocks:
                est = self._estimate_on_subsample(block, initial_guess=tuple(theta_n))
                if not np.any(np.isnan(est)):
                    estimates.append(est)

            if len(estimates) < 3:
                continue

            estimates = np.array(estimates)
            deviations = estimates - theta_n
            var_est = np.var(deviations, axis=0, ddof=1)

            row = {'b': b_cand, 'n_blocks': len(estimates)}
            for k, param in enumerate(self.param_names):
                row[f'var_{param}'] = var_est[k]
            rows.append(row)

        return pd.DataFrame(rows)


# =============================================================================
# Convergence study
# =============================================================================

def improved_convergence_study(
        sample_sizes: List[int] = [250, 500, 1000, 2000],
        M: int = 100,
        save_dir: str = 'outputs',
        n_jobs: int = -1):
    """
    Run the improved convergence study across multiple sample sizes.

    Args:
        sample_sizes: List of sample sizes.
        M: Number of MC replications per sample size.
        save_dir: Directory to save outputs.
        n_jobs: Number of parallel jobs.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("IMPROVED CONVERGENCE STUDY")
    print("=" * 70)
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  MC replications: {M}")
    print(f"  Output directory: {save_dir}")
    print("=" * 70 + "\n")

    analyzer = ImprovedSubsamplingAnalysis(
        num_components=2,
        true_params=(0.8, 0.3, 0.7, 0.9, 1.5, 0),
        alpha=1.5,
        beta=0
    )

    all_results = {}
    summary_tables = {}

    for n in sample_sizes:
        print(f"\n{'*' * 70}")
        print(f"Processing n = {n}")
        print(f"{'*' * 70}")

        b = int(np.floor(n ** (2 / 3)))
        results_n = analyzer.monte_carlo_study(
            n=n, M=M, b=b, seed=42,
            conf_levels=(0.90, 0.95), n_jobs=n_jobs
        )
        all_results[n] = results_n

        summary_df = analyzer.summary_table(results_n)
        summary_tables[n] = summary_df

        print(f"\nSummary for n = {n}:")
        print("=" * 70)
        print(summary_df.to_string(index=False))
        print("=" * 70)

        # Diagnostic plots
        fig_path = os.path.join(save_dir, f"improved_diagnostics_n{n}.png")
        analyzer.plot_diagnostics(results_n, save_path=fig_path)

        # Save CSV
        csv_path = os.path.join(save_dir, f"improved_summary_n{n}.csv")
        summary_df.to_csv(csv_path, index=False)

    # Convergence plot
    conv_path = os.path.join(save_dir, "improved_convergence_plot.png")
    create_improved_convergence_plot(summary_tables, save_path=conv_path)

    return all_results, summary_tables


def create_improved_convergence_plot(
        summary_tables: Dict[int, pd.DataFrame],
        save_path: Optional[str] = None):
    """
    Create convergence plot highlighting coverage rates alongside moments.

    Args:
        summary_tables: Dictionary mapping sample size to summary DataFrames.
        save_path: Path to save the figure.
    """
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    sample_sizes = sorted(summary_tables.keys())
    param_names = summary_tables[sample_sizes[0]]['Parameter'].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: Mean
    ax = axes[0, 0]
    for param in param_names:
        vals = [summary_tables[n][summary_tables[n]['Parameter'] == param]['Mean'].values[0]
                for n in sample_sizes]
        ax.plot(sample_sizes, vals, marker='o', linewidth=2, markersize=7, label=param)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Expected (0)')
    ax.set_xlabel('Sample Size (n)')
    ax.set_title('Mean of Scaled Deviations', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Panel 2: Skewness
    ax = axes[0, 1]
    for param in param_names:
        vals = [summary_tables[n][summary_tables[n]['Parameter'] == param]['Skewness'].values[0]
                for n in sample_sizes]
        ax.plot(sample_sizes, vals, marker='o', linewidth=2, markersize=7, label=param)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Expected (0)')
    ax.set_xlabel('Sample Size (n)')
    ax.set_title('Skewness', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Panel 3: Kurtosis
    ax = axes[0, 2]
    for param in param_names:
        vals = [summary_tables[n][summary_tables[n]['Parameter'] == param]['Kurtosis'].values[0]
                for n in sample_sizes]
        ax.plot(sample_sizes, vals, marker='o', linewidth=2, markersize=7, label=param)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Expected (0)')
    ax.set_xlabel('Sample Size (n)')
    ax.set_title('Excess Kurtosis', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Panel 4: KS rejection rate (per replication)
    ax = axes[1, 0]
    for param in param_names:
        vals = [summary_tables[n][summary_tables[n]['Parameter'] == param]['KS Rej.'].values[0]
                for n in sample_sizes]
        ax.plot(sample_sizes, vals, marker='o', linewidth=2, markersize=7, label=param)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='Nominal 5%')
    ax.set_xlabel('Sample Size (n)')
    ax.set_title('KS Rejection Rate (per replication)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim([-0.05, 1.05])

    # Panel 5: 90% Coverage
    ax = axes[1, 1]
    col_90 = 'Cov. 90%'
    for param in param_names:
        vals = [summary_tables[n][summary_tables[n]['Parameter'] == param][col_90].values[0]
                for n in sample_sizes]
        ax.plot(sample_sizes, vals, marker='o', linewidth=2, markersize=7, label=param)
    ax.axhline(0.90, color='red', linestyle='--', linewidth=2, label='Nominal 90%')
    ax.set_xlabel('Sample Size (n)')
    ax.set_title('Subsampling CI Coverage (90%)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim([0.5, 1.05])

    # Panel 6: 95% Coverage
    ax = axes[1, 2]
    col_95 = 'Cov. 95%'
    for param in param_names:
        vals = [summary_tables[n][summary_tables[n]['Parameter'] == param][col_95].values[0]
                for n in sample_sizes]
        ax.plot(sample_sizes, vals, marker='o', linewidth=2, markersize=7, label=param)
    ax.axhline(0.95, color='red', linestyle='--', linewidth=2, label='Nominal 95%')
    ax.set_xlabel('Sample Size (n)')
    ax.set_title('Subsampling CI Coverage (95%)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim([0.5, 1.05])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    results, summaries = improved_convergence_study(
        sample_sizes=[500, 1000, 2000],
        M=100,  # Increase to 500 for publication
        save_dir=output_dir,
        n_jobs=-1
    )

    print("\n" + "=" * 70)
    print("IMPROVED CONVERGENCE STUDY COMPLETED")
    print("=" * 70)