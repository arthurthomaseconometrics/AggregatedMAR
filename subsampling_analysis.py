"""
Subsampling Analysis for MDE Estimator of Stable AR(1) Aggregates

This module implements the subsampling methodology to verify the asymptotic 
normality result stated in Proposition prop:tcl2 of the paper. The approach 
estimates parameters on overlapping subsamples and analyzes the distribution 
of scaled deviations to assess convergence to normality.

Author: Analysis for stable mixture deconvolution
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, List, Dict, Optional
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Import the stable mixture class
import sys
sys.path.append('AggregatedMAR/')
from stablemixture import StableMixtureDeconvolution


class SubsamplingAnalysis:
    """
    Implements subsampling methodology to verify asymptotic normality of the MDE estimator.
    
    The class provides tools to:
    1. Generate data from stable AR(1) aggregates
    2. Estimate parameters on overlapping subsamples
    3. Compute scaled deviations and test for normality
    4. Visualize convergence to the asymptotic distribution
    
    Attributes:
        num_components (int): Number of latent AR(1) components
        true_params (tuple): True parameter values (rho_1, ..., rho_J, varsigma_1, ..., varsigma_J, alpha, beta)
        alpha (float): Tail index parameter
        beta (float): Asymmetry parameter
    """
    
    def __init__(self, 
                 num_components: int = 2,
                 true_params: Optional[tuple] = None,
                 alpha: float = 1.5,
                 beta: float = 0.3):
        """
        Initialize the subsampling analysis.
        
        Args:
            num_components: Number of latent AR(1) components (default: 2)
            true_params: True parameter values. If None, uses default values from paper
            alpha: Tail index parameter (default: 1.5)
            beta: Asymmetry parameter (default: 0.3)
        """
        self.num_components = num_components
        self.alpha = alpha
        self.beta = beta
        
        # Default parameters from paper (Table 3: General alpha-stable case)
        if true_params is None:
            # rho_1=0.8, rho_2=0.3, varsigma_1=0.7, varsigma_2=0.9, alpha=1.5, beta=0.3
            self.true_params = (0.8, 0.3, 0.7, 0.9, alpha, beta)
        else:
            self.true_params = true_params
            
        # Initialize the model
        self.model = StableMixtureDeconvolution(
            num_components=num_components,
            alpha_value=alpha,
            beta_value=beta,
            fix_alpha=False,
            fix_beta=True,
            verbose=False
        )
        
        # Parameter names
        self.param_names = self.model.params_names
        
    
    def generate_data(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a sample from the stable AR(1) aggregate model.
        
        Args:
            n: Sample size
            seed: Random seed for reproducibility
            
        Returns:
            Generated time series of length n
        """
        Y, _ = self.model.generate(T=n, params=self.true_params, seed=seed)
        return Y
    
    
    def estimate_subsample(self, 
                          Y_subsample: np.ndarray, 
                          initial_guess: Optional[tuple] = None) -> tuple:
        """
        Estimate parameters from a subsample using the MDE method.
        
        Args:
            Y_subsample: Subsample time series
            initial_guess: Initial values for optimization (uses full sample estimate if None)
            
        Returns:
            Estimated parameters for the subsample
        """
        try:
            # Create a fresh model instance for each subsample
            model_sub = StableMixtureDeconvolution(
                num_components=self.num_components,
                alpha_value=self.alpha,
                beta_value=self.beta,
                fix_alpha=False,
                fix_beta=True,
                verbose=False
            )
            
            # Estimate with the subsample
            params_sub = model_sub.estimate(
                Y=Y_subsample, 
                initial_guess=list(initial_guess) if initial_guess is not None else None,
                use_optimal_weights=False,  # Use uniform weights for speed
                step1=True  # Skip step 1 if we have good initial values
            )
            
            return params_sub
            
        except Exception as e:
            # Return NaN if estimation fails
            print(f"Warning: Subsample estimation failed with error: {e}")
            return tuple([np.nan] * len(self.param_names))
    
    
    def subsample_analysis(self, 
                          Y: np.ndarray, 
                          b: Optional[int] = None,
                          n_jobs: int = -1) -> Dict[str, np.ndarray]:
        """
        Perform subsampling analysis on a single time series.
        
        Args:
            Y: Full sample time series
            b: Subsample size (default: n^(2/3))
            n_jobs: Number of parallel jobs (-1 uses all cores)
            
        Returns:
            Dictionary containing:
                - 'theta_n': Full sample estimate
                - 'theta_subsamples': Array of subsample estimates (N_b x p)
                - 'scaled_deviations': Scaled deviations sqrt(b)(theta_i^b - theta_n)
                - 'subsample_size': Value of b used
        """
        n = len(Y)
        
        # Choose subsample size if not provided
        if b is None:
            b = int(np.floor(n ** (2/3)))
        
        # Ensure b is not too large
        b = min(b, n - 10)
        
        print(f"\n{'='*60}")
        print(f"Subsampling Analysis:")
        print(f"  Full sample size: n = {n}")
        print(f"  Subsample size: b = {b}")
        print(f"  Number of subsamples: N_b = {n - b + 1}")
        print(f"{'='*60}\n")
        
        # Step 1: Estimate with full sample
        print("Step 1: Estimating parameters from full sample...")
        theta_n = self.model.estimate(Y, use_optimal_weights=False)
        print(f"Full sample estimate: {theta_n}")
        
        # Step 2: Extract subsamples and estimate
        N_b = n - b + 1
        print(f"\nStep 2: Estimating parameters from {N_b} subsamples...")
        print("This may take several minutes depending on sample size...")
        
        # Create subsamples (do not materialize all at once to save memory)
        def process_subsample(i):
            Y_sub = Y[i:i+b]
            # Use independent initial guess for each subsample (not theta_n)
            #init_guess = self.model.generate_initial_guess()
            return self.estimate_subsample(Y_sub, initial_guess=theta_n)
        
        # Parallel estimation of subsamples
        theta_subsamples = Parallel(n_jobs=n_jobs)(
            delayed(process_subsample)(i) for i in range(N_b)
        )
        
        # Convert to numpy array
        theta_subsamples = np.array(theta_subsamples)
        
        # Remove failed estimations (NaN rows)
        valid_mask = ~np.isnan(theta_subsamples).any(axis=1)
        if not valid_mask.all():
            print(f"\nWarning: {(~valid_mask).sum()} subsample estimations failed and were removed")
            theta_subsamples = theta_subsamples[valid_mask]
        
        # Step 3: Compute scaled deviations
        scaled_deviations = np.sqrt(b) * (theta_subsamples - theta_n)
        
        print(f"\nCompleted subsample analysis.")
        print(f"  Valid subsamples: {len(theta_subsamples)}")
        
        return {
            'theta_n': theta_n,
            'theta_subsamples': theta_subsamples,
            'scaled_deviations': scaled_deviations,
            'subsample_size': b,
            'n_valid': len(theta_subsamples)
        }
    
    
    def test_normality(self, 
                      scaled_deviations: np.ndarray,
                      param_idx: int) -> Dict[str, float]:
        """
        Perform statistical tests for normality on scaled deviations for a single parameter.
        
        Args:
            scaled_deviations: Array of scaled deviations (N_b x p)
            param_idx: Index of parameter to test
            
        Returns:
            Dictionary with test statistics and p-values
        """
        z = scaled_deviations[:, param_idx]
        
        # Remove any NaN or inf values
        z = z[np.isfinite(z)]
        
        if len(z) < 20:
            return {
                'mean': np.nan,
                'std': np.nan,
                'skewness': np.nan,
                'kurtosis': np.nan,
                'ks_statistic': np.nan,
                'ks_pvalue': np.nan,
                'sw_statistic': np.nan,
                'sw_pvalue': np.nan,
                'jb_statistic': np.nan,
                'jb_pvalue': np.nan
            }
        
        # Compute moments
        mean = np.mean(z)
        std = np.std(z, ddof=1)
        skewness = stats.skew(z)
        kurtosis = stats.kurtosis(z)  # Excess kurtosis
        
        # Standardize
        z_std = (z - mean) / std if std > 0 else z
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.kstest(z_std, 'norm')
        
        # Shapiro-Wilk test (only for sample size <= 5000)
        if len(z) <= 5000:
            sw_stat, sw_pval = stats.shapiro(z_std)
        else:
            sw_stat, sw_pval = np.nan, np.nan
        
        # Jarque-Bera test
        jb_stat, jb_pval = stats.jarque_bera(z)
        
        return {
            'mean': mean,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'sw_statistic': sw_stat,
            'sw_pvalue': sw_pval,
            'jb_statistic': jb_stat,
            'jb_pvalue': jb_pval
        }
    
    
    def monte_carlo_subsample(self,
                            n: int,
                            M: int = 100,
                            b: Optional[int] = None,
                            seed: int = 42,
                            n_jobs: int = -1) -> Dict:
        """
        Perform Monte Carlo study with subsampling for a given sample size.
        
        Args:
            n: Sample size
            M: Number of Monte Carlo replications
            b: Subsample size (default: n^(2/3))
            seed: Random seed
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with aggregated results across MC replications
        """
        if b is None:
            b = int(np.floor(n ** (2/3)))
        
        print(f"\n{'#'*70}")
        print(f"Monte Carlo Subsampling Study: n={n}, M={M}, b={b}")
        print(f"{'#'*70}")
        
        # Storage for results
        all_test_results = {param: [] for param in self.param_names}
        all_scaled_deviations = {param: [] for param in self.param_names}
        
        for m in range(M):
            if (m + 1) % 10 == 0:
                print(f"\nReplication {m+1}/{M}...")
            
            # Generate data
            Y = self.generate_data(n=n, seed=seed + m)
            
            # Perform subsampling analysis
            try:
                results = self.subsample_analysis(Y, b=b, n_jobs=n_jobs)
                scaled_dev = results['scaled_deviations']
                
                # Test normality for each parameter
                for idx, param_name in enumerate(self.param_names):
                    test_res = self.test_normality(scaled_dev, idx)
                    all_test_results[param_name].append(test_res)
                    
                    # Store scaled deviations for this parameter
                    z_param = scaled_dev[:, idx]
                    z_param = z_param[np.isfinite(z_param)]
                    all_scaled_deviations[param_name].extend(z_param.tolist())
                    
            except Exception as e:
                print(f"Warning: Replication {m+1} failed with error: {e}")
                continue
        
        # Aggregate results
        aggregated_results = {
            'n': n,
            'b': b,
            'M': M,
            'param_names': self.param_names,
            'test_results': all_test_results,
            'scaled_deviations': all_scaled_deviations
        }
        
        return aggregated_results
    
    
    def plot_diagnostics(self, 
                        results: Dict,
                        save_path: Optional[str] = None):
        """
        Create diagnostic plots for subsampling analysis.
        
        Args:
            results: Results dictionary from monte_carlo_subsample()
            save_path: Path to save the figure (if None, displays only)
        """
        param_names = results['param_names']
        n_params = len(param_names)
        
        fig, axes = plt.subplots(n_params, 3, figsize=(15, 4*n_params))
        if n_params == 1:
            axes = axes.reshape(1, -1)
        
        for idx, param_name in enumerate(param_names):
            # Get scaled deviations for this parameter
            z = np.array(results['scaled_deviations'][param_name])
            z = z[np.isfinite(z)]
            
            if len(z) < 10:
                continue
            
            # Standardize
            z_std = (z - np.mean(z)) / np.std(z)
            
            # Plot 1: Histogram with normal overlay
            ax1 = axes[idx, 0]
            ax1.hist(z_std, bins=50, density=True, alpha=0.7, 
                    edgecolor='black', label='Empirical')
            x_range = np.linspace(z_std.min(), z_std.max(), 100)
            ax1.plot(x_range, stats.norm.pdf(x_range, 0, 1), 
                    'r-', linewidth=2, label='N(0,1)')
            ax1.set_xlabel('Standardized Scaled Deviation')
            ax1.set_ylabel('Density')
            ax1.set_title(f'{param_name}: Histogram')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Q-Q plot
            ax2 = axes[idx, 1]
            stats.probplot(z_std, dist="norm", plot=ax2)
            ax2.set_title(f'{param_name}: Q-Q Plot')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Kernel density estimate
            ax3 = axes[idx, 2]
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(z_std)
            x_range = np.linspace(z_std.min(), z_std.max(), 200)
            ax3.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
            ax3.plot(x_range, stats.norm.pdf(x_range, 0, 1), 
                    'r--', linewidth=2, label='N(0,1)')
            ax3.set_xlabel('Standardized Scaled Deviation')
            ax3.set_ylabel('Density')
            ax3.set_title(f'{param_name}: Kernel Density')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")
        
        plt.show()
        
        
    def summarize_results(self, results: Dict) -> pd.DataFrame:
        """
        Create a summary table of test statistics across parameters.
        
        Args:
            results: Results dictionary from monte_carlo_subsample()
            
        Returns:
            DataFrame with summary statistics
        """
        param_names = results['param_names']
        test_results = results['test_results']
        
        summary_data = []
        
        for param_name in param_names:
            tests = test_results[param_name]
            
            if len(tests) == 0:
                continue
            
            # Average across MC replications
            avg_test = {
                'Parameter': param_name,
                'Mean': np.mean([t['mean'] for t in tests if not np.isnan(t['mean'])]),
                'Std': np.mean([t['std'] for t in tests if not np.isnan(t['std'])]),
                'Skewness': np.mean([t['skewness'] for t in tests if not np.isnan(t['skewness'])]),
                'Kurtosis': np.mean([t['kurtosis'] for t in tests if not np.isnan(t['kurtosis'])]),
                'KS p-value': np.mean([t['ks_pvalue'] for t in tests if not np.isnan(t['ks_pvalue'])]),
                'SW p-value': np.mean([t['sw_pvalue'] for t in tests if not np.isnan(t['sw_pvalue'])]),
                'JB p-value': np.mean([t['jb_pvalue'] for t in tests if not np.isnan(t['jb_pvalue'])]),
                'KS Reject (5%)': np.mean([t['ks_pvalue'] < 0.05 for t in tests if not np.isnan(t['ks_pvalue'])]),
                'SW Reject (5%)': np.mean([t['sw_pvalue'] < 0.05 for t in tests if not np.isnan(t['sw_pvalue'])]),
                'JB Reject (5%)': np.mean([t['jb_pvalue'] < 0.05 for t in tests if not np.isnan(t['jb_pvalue'])])
            }
            
            summary_data.append(avg_test)
        
        df = pd.DataFrame(summary_data)
        return df


def convergence_study(sample_sizes: List[int] = [250, 500, 1000],
                     M: int = 100,
                     save_dir: str = 'AggregatedMAR/outputs',
                     n_jobs: int = -1):
    """
    Perform convergence study across multiple sample sizes.
    
    Args:
        sample_sizes: List of sample sizes to test
        M: Number of Monte Carlo replications per sample size
        save_dir: Directory to save results
        n_jobs: Number of parallel jobs
    """
    print("\n" + "="*70)
    print("CONVERGENCE STUDY: Asymptotic Normality Verification")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  MC replications per n: {M}")
    print(f"  Model: General alpha-stable AR(1) aggregates")
    print(f"  True parameters: rho_1=0.8, rho_2=0.3, varsigma_1=0.7, varsigma_2=0.9")
    print(f"                   alpha=1.5, beta=0.3")
    print("="*70 + "\n")
    
    # Initialize analysis
    analyzer = SubsamplingAnalysis(
        num_components=2,
        true_params=(0.8, 0.3, 0.7, 0.9, 1.5, 0),
        alpha=1.5,
        beta=0
    )
    
    # Storage for results across sample sizes
    all_results = {}
    summary_tables = {}
    
    for n in sample_sizes:
        print(f"\n\n{'*'*70}")
        print(f"Processing sample size n = {n}")
        print(f"{'*'*70}")
        
        # Run Monte Carlo subsampling
        results_n = analyzer.monte_carlo_subsample(
            n=n,
            M=M,
            b=int(np.floor(n ** (2/3))),
            seed=42,
            n_jobs=n_jobs
        )
        
        all_results[n] = results_n
        
        # Create summary table
        summary_df = analyzer.summarize_results(results_n)
        summary_tables[n] = summary_df
        
        print(f"\n\nSummary for n = {n}:")
        print("="*70)
        print(summary_df.to_string(index=False))
        print("="*70)
        
        # Create diagnostic plots
        fig_path = f"{save_dir}/diagnostics_n{n}.png"
        analyzer.plot_diagnostics(results_n, save_path=fig_path)
    
    # Create convergence plot
    create_convergence_plot(summary_tables, save_path=f"{save_dir}/convergence_plot.png")
    
    # Save summary tables to CSV
    for n, df in summary_tables.items():
        csv_path = f"{save_dir}/summary_n{n}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSummary table saved: {csv_path}")
    
    return all_results, summary_tables


def create_convergence_plot(summary_tables: Dict[int, pd.DataFrame],
                           save_path: Optional[str] = None):
    """
    Create plots showing convergence of test statistics as n increases.
    
    Args:
        summary_tables: Dictionary mapping sample size to summary DataFrames
        save_path: Path to save the figure
    """
    sample_sizes = sorted(summary_tables.keys())
    param_names = summary_tables[sample_sizes[0]]['Parameter'].tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['Mean', 'Skewness', 'Kurtosis', 'KS p-value']
    titles = ['Mean of Scaled Deviations', 'Skewness', 'Excess Kurtosis', 
              'KS Test p-value']
    expected = [0, 0, 0, 0.5]  # Expected values under normality
    
    for idx, (metric, title, exp_val) in enumerate(zip(metrics, titles, expected)):
        ax = axes[idx // 2, idx % 2]
        
        for param_name in param_names:
            values = []
            for n in sample_sizes:
                df = summary_tables[n]
                val = df[df['Parameter'] == param_name][metric].values
                values.append(val[0] if len(val) > 0 else np.nan)
            
            ax.plot(sample_sizes, values, marker='o', linewidth=2, 
                   markersize=8, label=param_name)
        
        ax.axhline(y=exp_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Expected ({exp_val})')
        ax.set_xlabel('Sample Size (n)', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConvergence plot saved to: {save_path}")
    
    plt.show()


# Main execution function
if __name__ == "__main__":
    """
    Main execution: Run convergence study across multiple sample sizes.
    """
    import os
    
    # Create output directory if it doesn't exist
    output_dir = 'AggregatedMAR/outputs/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run convergence study with multiple sample sizes
    # Note: For full replication, use M=500 and sample_sizes=[250, 500, 1000, 2000]
    # Here we use smaller values for demonstration
    results, summaries = convergence_study(
        sample_sizes=[250, 500, 1000,2000],  # Can add 2000 for more evidence
        M=10,  # Number of MC replications (increase to 500 for publication)
        save_dir=output_dir,
        n_jobs=-1  # Use all available cores
    )
    
    print("\n" + "="*70)
    print("CONVERGENCE STUDY COMPLETED")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")
    print("\nFiles generated:")
    print(f"  - diagnostics_n*.png: Diagnostic plots for each sample size")
    print(f"  - convergence_plot.png: Convergence across sample sizes")
