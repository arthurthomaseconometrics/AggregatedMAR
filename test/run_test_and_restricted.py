"""
Run script for:
  1. Criterion Difference Test  (H0: pi_1 = 0.5)
  2. Restricted Subsampling     (pi_1 = 0.5 imposed)

Adjust sample_sizes, M, and n_cores to your computational budget.

Usage:
    python run_test_and_restricted.py
"""

from criterion_difference_test import run_mc_criterion_test, test_single_series
from subsampling_restricted import run_study_restricted

# ========================================================================
# Common settings
# ========================================================================
TRUE_PARAMS = (0.8, 0.3, 0.7, 0.9, 1.5, 0.0)
ALPHA = 1.5
BETA = 0.0
J = 2
SEED = 42

# ========================================================================
# 1. Criterion Difference Test (Monte Carlo)
# ========================================================================
print("\n" + "=" * 80)
print("PART 1: CRITERION DIFFERENCE TEST  H0: pi_1 = 0.5")
print("=" * 80)

criterion_results = run_mc_criterion_test(
    sample_sizes=[500, 1000, 10000, 50000],
    M=200,
    b_exponent=2/3,
    n_cores=-1,
    seed=SEED,
    output_dir='results_criterion_test',
    true_params=TRUE_PARAMS,
    alpha=ALPHA,
    beta=BETA,
    fix_alpha=False,
    fix_beta=True,
    J=J,
)

# ========================================================================
# 2. Restricted Subsampling (pi_1 = 0.5 imposed)
# ========================================================================
print("\n" + "=" * 80)
print("PART 2: RESTRICTED SUBSAMPLING  (pi_1 = 0.5)")
print("=" * 80)

restricted_results = run_study_restricted(
    sample_sizes=[250, 500, 1000, 10000, 50000, 100000],
    M=200,
    b_exponent=2/3,
    n_cores=-1,
    seed=SEED,
    output_dir='results_restricted',
    true_params=TRUE_PARAMS,
    alpha=ALPHA,
    beta=BETA,
    fix_alpha=False,
    fix_beta=True,
    J=J,
)

print("\n" + "=" * 80)
print("ALL DONE.")
print("  Criterion test results: results_criterion_test/")
print("  Restricted subsampling: results_restricted/")
print("=" * 80)
