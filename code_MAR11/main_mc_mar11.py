"""
Monte Carlo study for MAR(1,1) aggregates (Case 2 of the paper) -- CLUSTER
launcher, full-scale settings. Mirrors the style of
AggregatedMAR-main/main_sub.py for Case 1.

True parameters (psi1=0.7, psi2=0.4, varsigma1=0.8, varsigma2=0.6,
phi1=0.3, phi2=0.5, alpha=1.5, beta=0) and grid settings (n_grid=15,
kappa=0.5, u_max=0.5, linspace) are the ones already validated locally
(see monte_carlo_mar11_study.main() and the 2026-07 audit): beta is kept
fixed at 0 (SaS case, matching the symmetric restriction of Case 2 in
main_v16.tex and of subsampling_mar11.py) for consistency with the
subsampling companion script (main_sub_mar11.py).

Run on the cluster:
    python3 main_mc_mar11.py
Adjust n_jobs to the number of cores allocated to the job (-1 = all).
"""

from monte_carlo_mar11_study import run_monte_carlo_study, sort_components_by_psi
import matplotlib.pyplot as plt
from monte_carlo_mar11_study import plot_mc_results

true_params = dict(
    psi1=0.7,      psi2=0.4,
    varsigma1=0.8, varsigma2=0.6,
    phi1=0.3,      phi2=0.5,
    alpha=1.5,     beta=0.0,
)

df_raw, df_summary = run_monte_carlo_study(
    **true_params,
    # estimation
    estimate_alpha=True,
    estimate_beta=False,       # SaS case (beta fixed at 0), see docstring above
    estimate_phi=True,
    # MC -- full-scale campaign (n_replications=1000 matches the "1,000
    # replications" convention of Table S.1 for Case 1; extend
    # sample_sizes to 50000/100000 if the campaign budget allows)
    n_replications=1000,
    sample_sizes=[250, 500, 1000, 10000],
    n_jobs=-1,
    seed_base=42,
    # grid -- matching the paper's Case-2 design (linspace on [-0.5,0.5],
    # kappa=0.5, 15x15 -> 225 moment conditions for 7 free parameters)
    n_grid=15,
    kappa=0.5,
    grid_type='linspace',
    u_max=0.5,
    scale=1.0,                 # irrelevant for linspace
    # multi-start
    n_starts=8,
    # output
    verbose=True,
    output_dir='results_mc_mar11',
)

print("\n" + "=" * 70)
print("Summary Statistics")
print("=" * 70)
print(df_summary.to_string(index=False))

true_params_sorted = sort_components_by_psi(true_params)
fig = plot_mc_results(
    df_raw,
    true_params=true_params_sorted,
    estimate_phi=True,
    estimate_alpha=True,
    estimate_beta=False,
    save_path='results_mc_mar11/mc_results_mar11.pdf',
)
plt.close(fig)
print("\nAll results in results_mc_mar11/")
