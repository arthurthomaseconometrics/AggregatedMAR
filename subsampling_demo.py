"""
Quick Demonstration of Subsampling Analysis

This script provides a minimal working example to demonstrate the subsampling
methodology for verifying asymptotic normality of the MDE estimator.

Usage:
    python subsampling_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/arthurthomas/Library/CloudStorage/Dropbox/AggregatedMAR/code Python_arthur/subsampling/')
from subsampling_analysis import SubsamplingAnalysis

def quick_demo():
    """
    Quick demonstration with a single sample size.
    
    This function:
    1. Generates a single time series
    2. Performs subsampling analysis
    3. Tests for normality
    4. Creates diagnostic plots
    """
    print("\n" + "="*70)
    print("QUICK DEMONSTRATION: Subsampling Analysis")
    print("="*70)
    
    # Initialize the analysis
    analyzer = SubsamplingAnalysis(
        num_components=2,
        true_params=(0.8, 0.3, 0.7, 0.9, 1.5, 0),
        alpha=1.5,
        beta=0
    )
    
    # Generate a single time series
    n = 500
    print(f"\n1. Generating time series with n = {n}...")
    Y = analyzer.generate_data(n=n, seed=42)
    print(f"   Sample generated: mean = {Y.mean():.3f}, std = {Y.std():.3f}")
    
    # Perform subsampling analysis
    print(f"\n2. Performing subsampling analysis...")
    b = int(np.floor(n ** (2/3)))  # Subsample size
    results = analyzer.subsample_analysis(Y, b=b, n_jobs=-1)
    
    # Test normality for each parameter
    print(f"\n3. Testing for normality...")
    print("="*70)
    scaled_dev = results['scaled_deviations']
    
    for idx, param_name in enumerate(analyzer.param_names):
        print(f"\n{param_name}:")
        print("-" * 40)
        test_results = analyzer.test_normality(scaled_dev, idx)
        
        print(f"  Mean:            {test_results['mean']:8.4f}  (expected: 0)")
        print(f"  Std:             {test_results['std']:8.4f}")
        print(f"  Skewness:        {test_results['skewness']:8.4f}  (expected: 0)")
        print(f"  Excess Kurtosis: {test_results['kurtosis']:8.4f}  (expected: 0)")
        print(f"  KS test p-value: {test_results['ks_pvalue']:8.4f}")
        print(f"  SW test p-value: {test_results['sw_pvalue']:8.4f}")
        print(f"  JB test p-value: {test_results['jb_pvalue']:8.4f}")
        
        # Interpretation
        if test_results['ks_pvalue'] > 0.05:
            print(f"  → KS test: Cannot reject normality (p > 0.05)")
        else:
            print(f"  → KS test: Rejects normality (p < 0.05)")
    
    print("\n" + "="*70)
    
    # Create diagnostic plots
    print(f"\n4. Creating diagnostic plots...")
    
    # Prepare results in the format expected by plot_diagnostics
    plot_results = {
        'param_names': analyzer.param_names,
        'scaled_deviations': {}
    }
    
    for idx, param_name in enumerate(analyzer.param_names):
        z_param = scaled_dev[:, idx]
        z_param = z_param[np.isfinite(z_param)]
        plot_results['scaled_deviations'][param_name] = z_param.tolist()
    
    # Create plots
    analyzer.plot_diagnostics(plot_results, save_path='/Users/arthurthomas/Library/CloudStorage/Dropbox/AggregatedMAR/code Python_arthur/subsampling/demo_diagnostics.png')
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETED")
    print("="*70)
    print("\nInterpretation:")
    print("  - If p-values > 0.05: Consistent with asymptotic normality")
    print("  - Mean close to 0: No systematic bias in subsample estimates")
    print("  - Skewness close to 0: Symmetric distribution")
    print("  - Excess kurtosis close to 0: Tails similar to normal distribution")
    print("\nDiagnostic plots saved to: /outputs/demo_diagnostics.png")


def mini_convergence_demo():
    """
    Mini convergence study with just two sample sizes.
    
    This demonstrates how convergence improves with larger samples,
    but uses fewer replications for speed.
    """
    print("\n" + "="*70)
    print("MINI CONVERGENCE DEMO")
    print("="*70)
    
    from subsampling_analysis import convergence_study
    
    # Run with small sample sizes and few replications for speed
    results, summaries = convergence_study(
        sample_sizes=[250, 500],  # Just two sample sizes
        M=50,  # Reduced number of replications
        save_dir='/Users/arthurthomas/Library/CloudStorage/Dropbox/AggregatedMAR/code Python_arthur/subsampling/outputs/',
        n_jobs=-1
    )
    
    print("\n" + "="*70)
    print("MINI CONVERGENCE DEMO COMPLETED")
    print("="*70)
    print("\nKey observations to look for:")
    print("  1. Means should get closer to 0 as n increases")
    print("  2. Skewness and kurtosis should decrease in magnitude")
    print("  3. Test p-values should increase (less rejection of normality)")
    print("\nResults saved to: /mnt/user-data/outputs/")


if __name__ == "__main__":
    """
    Main execution: Choose between quick demo or mini convergence study.
    """
    import sys
    
    print("\nSubsampling Analysis Demonstration")
    print("="*70)
    print("\nChoose an option:")
    print("  1. Quick demo (single sample, fast)")
    print("  2. Mini convergence study (2 sample sizes, moderate speed)")
    print("  3. Both")
    
    # For automatic execution in this context, run quick demo
    print("\nRunning quick demonstration...\n")
    quick_demo()
    
    # Uncomment to also run mini convergence demo:
    # print("\n\nRunning mini convergence demonstration...\n")
    # mini_convergence_demo()