"""
Example Execution: Subsampling Analysis for Verifying Asymptotic Normality - CORRECTED VERSION

This script demonstrates a complete workflow for verifying the asymptotic 
normality result of Proposition prop:tcl2 using subsampling methodology.

IMPORTANT: This is a minimal example for demonstration. For publication-quality
results, increase M (number of MC replications) and add more sample sizes.

Estimated runtime: ~30-60 minutes on a modern multi-core CPU
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from subsampling_analysis import SubsamplingAnalysis, convergence_study

# ==============================================================================
# STEP 1: Single Sample Analysis
# ==============================================================================
print("\n" + "="*80)
print("STEP 1: SINGLE SAMPLE ANALYSIS")
print("="*80)
print("\nThis demonstrates the subsampling procedure on a single time series.")
print("We will:")
print("  1. Generate a time series from the true model")
print("  2. Extract overlapping subsamples")
print("  3. Estimate parameters from each subsample")
print("  4. Compute scaled deviations and test for normality")

# Initialize analyzer with parameters from Table 3 (General α-stable case)
print("\n" + "-"*80)
print("Configuration:")
print("-"*80)
true_params = {
    'rho_1': 0.8,
    'rho_2': 0.3,
    'varsigma_1': 0.7,
    'varsigma_2': 0.9,
    'alpha': 1.5,
    'beta': 0
}
print(f"True parameters: {true_params}")

analyzer = SubsamplingAnalysis(
    num_components=2,
    true_params=(0.8, 0.3, 0.7, 0.9, 1.5, 0),
    alpha=1.5,
    beta=0
)

# Generate time series
n = 2000
print(f"\nSample size: n = {n}")
Y = analyzer.generate_data(n=n, seed=42)
print(f"Generated data: mean = {Y.mean():.4f}, std = {Y.std():.4f}")

# Perform subsampling analysis
b = int(np.floor(n ** (2/3)))  # Subsample size: n^(2/3)
print(f"Subsample size: b = {b}")
print(f"Number of subsamples: N_b = {n - b + 1}")

print("\nRunning subsampling analysis (this may take a few minutes)...")
results = analyzer.subsample_analysis(Y, b=b, n_jobs=-1)

# Display results for each parameter
print("\n" + "="*80)
print("NORMALITY TEST RESULTS")
print("="*80)

for idx, param_name in enumerate(analyzer.param_names):
    print(f"\n{param_name} (True value: {true_params.get(param_name, 'N/A')})")
    print("-"*80)
    
    test_res = analyzer.test_normality(results['scaled_deviations'], idx)
    
    print(f"  Sample moments of scaled deviations:")
    print(f"    Mean:            {test_res['mean']:8.4f}  (expected: 0.0000)")
    print(f"    Std Dev:         {test_res['std']:8.4f}")
    print(f"    Skewness:        {test_res['skewness']:8.4f}  (expected: 0.0000)")
    print(f"    Excess Kurtosis: {test_res['kurtosis']:8.4f}  (expected: 0.0000)")
    
    print(f"\n  Normality test p-values:")
    print(f"    Kolmogorov-Smirnov: {test_res['ks_pvalue']:.4f}")
    print(f"    Shapiro-Wilk:       {test_res['sw_pvalue']:.4f}")
    print(f"    Jarque-Bera:        {test_res['jb_pvalue']:.4f}")
    
    # Interpretation
    print(f"\n  Interpretation (at 5% significance level):")
    reject_count = sum([
        test_res['ks_pvalue'] < 0.05,
        test_res['sw_pvalue'] < 0.05,
        test_res['jb_pvalue'] < 0.05
    ])
    
    if reject_count == 0:
        print(f"    ✓ All tests accept normality (0/3 rejections)")
    elif reject_count == 1:
        print(f"    ~ Mixed evidence for normality (1/3 rejections)")
    elif reject_count == 2:
        print(f"    ⚠ Weak evidence against normality (2/3 rejections)")
    else:
        print(f"    ✗ Strong evidence against normality (3/3 rejections)")

# Save diagnostic plots
print("\n" + "="*80)
print("Creating diagnostic plots...")

# Create outputs directory if it doesn't exist
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

plot_results = {
    'param_names': analyzer.param_names,
    'scaled_deviations': {}
}

for idx, param_name in enumerate(analyzer.param_names):
    z_param = results['scaled_deviations'][:, idx]
    z_param = z_param[np.isfinite(z_param)]
    plot_results['scaled_deviations'][param_name] = z_param.tolist()

step1_path = os.path.join(output_dir, 'step1_diagnostics.png')
analyzer.plot_diagnostics(plot_results, save_path=step1_path)
print(f"✓ Diagnostic plots saved to: {step1_path}")


# ==============================================================================
# STEP 2: Convergence Study (Optional - More Time Consuming)
# ==============================================================================
print("\n\n" + "="*80)
print("STEP 2: CONVERGENCE STUDY (OPTIONAL)")
print("="*80)
print("\nThis step verifies that normality improves as sample size increases.")
print("It performs Monte Carlo simulations for multiple sample sizes.")
print("\nWARNING: This is computationally intensive!")
print("  - For demonstration: M=50 replications, 2 sample sizes (~30-60 min)")
print("  - For publication: M=500 replications, 4 sample sizes (~10-20 hours)")

# Ask user if they want to run convergence study
run_convergence = input("\nRun convergence study? (y/n): ").lower().strip()
# Uncomment the line below for interactive mode:


if run_convergence == 'y':
    print("\nStarting convergence study...")
    print("You can monitor progress in the console output.")
    print("Press Ctrl+C to interrupt if needed.\n")
    
    try:
        # Run with reduced parameters for demonstration
        # To replicate paper results, use:
        #   sample_sizes=[250, 500, 1000, 2000]
        #   M=500
        results_conv, summaries_conv = convergence_study(
            sample_sizes=[500],  # Reduced for demo
            M=10,  # Reduced for demo (use 500 for publication)
            save_dir=output_dir,  # Use output_dir instead of '/outputs'
            n_jobs=-1
        )
        
        print("\n" + "="*80)
        print("CONVERGENCE STUDY COMPLETED")
        print("="*80)
        print("\nKey results to examine:")
        print(f"  1. {output_dir}/convergence_plot.png - Shows improvement as n increases")
        print(f"  2. {output_dir}/summary_n*.csv - Detailed statistics for each sample size")
        print(f"  3. {output_dir}/diagnostics_n*.png - Visual diagnostics for each n")
        
        print("\nWhat to look for in the convergence plot:")
        print("  ✓ Mean converging to 0")
        print("  ✓ Skewness and kurtosis decreasing in magnitude")
        print("  ✓ Test p-values increasing (moving away from rejection)")
        
    except KeyboardInterrupt:
        print("\n\nConvergence study interrupted by user.")
    except Exception as e:
        print(f"\n\nError in convergence study: {e}")
        print("Check the error message above for details.")
        import traceback
        traceback.print_exc()
        
else:
    print("\nSkipping convergence study.")
    print("To run it later, execute:")
    print("  from subsampling_analysis_fixed import convergence_study")
    print("  convergence_study(sample_sizes=[250, 500, 1000], M=100, save_dir='outputs')")


# ==============================================================================
# STEP 3: Summary and Interpretation
# ==============================================================================
print("\n\n" + "="*80)
print("SUMMARY AND INTERPRETATION")
print("="*80)

print("\n1. THEORETICAL EXPECTATION (from Proposition prop:tcl2):")
print("   Under correct model specification and sufficient sample size:")
print("   √n(θ̂ₙ - θ₀) →ᵈ N(0, Σ⁻¹ΩΣ⁻¹)")
print("\n   This implies that scaled subsample deviations √b(θ̂ᵢᵇ - θ̂ₙ)")
print("   should approximately follow a normal distribution.")

print("\n2. WHAT WE OBSERVED:")
print("   - Check the diagnostic plots for visual assessment")
print("   - Examine test p-values (>0.05 supports normality)")
print("   - Compare moments to normal distribution expectations")

print("\n3. FINITE SAMPLE CONSIDERATIONS:")
print("   - Smaller samples may show deviations from normality")
print("   - AR coefficients ρ₁, ρ₂ typically estimate best")
print("   - Scale parameters ς₁, ς₂ have higher variance")
print("   - Tail index α usually estimated precisely")
print("   - Asymmetry β is most challenging to estimate")

print("\n4. WHEN TO BE CONCERNED:")
print("   ⚠ If all tests strongly reject normality for large n")
print("   ⚠ If moments don't improve with increasing n")
print("   ⚠ If Q-Q plots show systematic departures from linearity")

print("\n5. NEXT STEPS:")
print("   - Examine diagnostic plots carefully")
print("   - Compare results across different sample sizes")
print("   - Check if specific parameters are problematic")
print("   - Consider model misspecification if issues persist")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll results saved to: {output_dir}/")
print("\nFiles generated:")
print(f"  - {output_dir}/step1_diagnostics.png: Diagnostic plots from Step 1")
if run_convergence == 'y':
    print(f"  - {output_dir}/convergence_plot.png: Convergence visualization")
    print(f"  - {output_dir}/diagnostics_n*.png: Diagnostics for each sample size")
    print(f"  - {output_dir}/summary_n*.csv: Summary statistics tables")

print("\n" + "="*80)
print("For more information, see:")
print("  - README.md: Comprehensive usage guide")
print("  - subsampling_theory.tex: Mathematical foundations")
print("  - subsampling_analysis_fixed.py: Implementation details")
print("="*80 + "\n")
