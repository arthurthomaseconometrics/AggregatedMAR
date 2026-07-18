"""
Subsampling study for MAR(1,1) aggregates (Case 2 of the paper) -- CLUSTER
launcher, full-scale settings. Mirrors the style of
AggregatedMAR-main/main_sub.py for Case 1 (M=200, ci_step=None -> all
overlapping blocks, n_cores=-1).

True parameters (psi1=0.7, psi2=0.4, varsigma1=0.8, varsigma2=0.6,
phi1=0.3, phi2=0.5, alpha=1.5) and grid settings (n_grid=15, kappa=0.5,
u_max=0.5) match main_mc_mar11.py for consistency across the Case-2
pipeline. beta is fixed at 0 (SaS case) -- subsampling_mar11.py does not
currently support an asymmetric (beta != 0) MAR(1,1) aggregate.

Run on the cluster:
    python3 main_sub_mar11.py
Adjust n_cores to the number of cores allocated to the job (-1 = all).

For a quick local sanity check before committing to this full campaign,
use subsampling_mar11.run_study_mar11 directly with lighter settings,
e.g.:
    from subsampling_mar11 import run_study_mar11
    run_study_mar11(sample_sizes=[500, 1000, 2000], M=20, n_cores=4,
                    ci_step=10)
"""

from subsampling_mar11 import run_study_mar11

run_study_mar11(
    sample_sizes=[250, 500, 1000, 10000, 50000, 100000],
    M=200,
    b_exponent=2/3,
    kappa=0.5,
    n_cores=-1,
    seed=42,
    ci_step=None,               # None -> step 1, all overlapping blocks
    output_dir='results_mar11_sub',
    true_params=(0.7, 0.4, 0.8, 0.6, 0.3, 0.5, 1.5),
    alpha=1.5,
    fix_alpha=False,
    u_max=0.5,
    n_grid=15,
    n_starts=8,
)
