"""
Criterion Difference Test for H0: pi_1 = 0.5 (i.e., varsigma_1 = varsigma_2).

The test statistic is:
    Delta_n = n * [Q_n(theta_hat_restr) - Q_n(theta_hat_free)]

where Q_n is the MDE objective function.  Under H0 and standard regularity,
Delta_n --> chi^2(1).  We also construct a subsampling reference distribution
as a robustness check.

Two modes:
  1. Monte Carlo:  assess size and power over M replications at each n.
  2. Empirical:    apply the test to a single observed series.

Usage:
    from criterion_difference_test import run_mc_criterion_test, test_single_series
"""

import numpy as np
from scipy import stats, optimize
from cf_weights import KAPPA_DEFAULT, quadrature_weights, weighted_sse, check_grid
from joblib import Parallel, delayed
from typing import List, Optional, Tuple, Dict
import time
import os
import warnings
warnings.filterwarnings('ignore')


# =========================================================================
# Low-level helpers (top-level for joblib pickling)
# =========================================================================

def _ecf_matrix(Y, u_grid, v_grid):
    """Compute empirical joint CF on (u,v) grid via broadcasting.

    Convention: u is associated with X_t = Y[:-1],
                v is associated with X_{t+1} = Y[1:].
    This is consistent with the theoretical CF where
    E[exp(i(u*X_t + v*X_{t+1}))] decomposes as
    (rho_j*u + v)*X_{j,t+1} + u*eps_{j,t}.
    """
    Y0, Y1 = Y[:-1], Y[1:]   # Y0 = X_t, Y1 = X_{t+1}
    phase = (u_grid[:, None, None] * Y0[None, None, :]
             + v_grid[None, :, None] * Y1[None, None, :])
    return np.mean(np.exp(1j * phase), axis=2)


def _tcf_matrix(params_raw, u_grid, v_grid, J, alpha_fixed, beta_val):
    """Theoretical joint CF on (u,v) grid.

    params_raw: (rho_1,...,rho_J, varsigma_1,...,varsigma_J, [alpha])
    """
    rhos = params_raw[:J]
    varsigmas = params_raw[J:2*J]
    alpha = params_raw[2*J] if alpha_fixed is None else alpha_fixed
    sigma = np.sum(varsigmas)
    pis = varsigmas / sigma if sigma > 0 else varsigmas * 0
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
    exponent = np.zeros_like(U, dtype=complex)
    if alpha != 1:
        tan_t = np.tan(np.pi * alpha / 2)
        for j in range(J):
            a1 = rhos[j] * U + V          # rho_j*u + v
            exponent += -pis[j]**alpha * sigma**alpha * (
                np.abs(a1)**alpha * (1 - 1j * beta_val * np.sign(a1) * tan_t)
                / (1 - np.abs(rhos[j])**alpha)
                + np.abs(U)**alpha * (1 - 1j * beta_val * np.sign(U) * tan_t))
    else:
        for j in range(J):
            a1 = rhos[j] * U + V          # rho_j*u + v
            exponent += -pis[j] * sigma * (
                np.abs(a1) * (1 + 1j * beta_val * np.sign(a1) * 2 / np.pi
                              * np.log(np.abs(a1) + 1e-10)) / (1 - np.abs(rhos[j]))
                + np.abs(U) * (1 + 1j * beta_val * np.sign(U) * 2 / np.pi
                               * np.log(np.abs(U) + 1e-10)))
    return np.exp(exponent)


def _obj_free(params, ecf, u_grid, v_grid, J, alpha_fixed, beta_val, W):
    """MDE objective — free (unrestricted) model.

    PATCH (Assumption C2, main_v23.tex) : approximation de Riemann de
    l'integrale (2.14) contre w(u,v) = (kappa/pi) exp(-kappa(u^2+v^2)).
    L'ancienne version renvoyait `np.mean(|ecf - tcf|^2)`, dont l'echelle est
    arbitraire — ce qui invalidait la comparaison de Delta_n a un quantile
    chi2 fixe.  W is None retablit l'ancien comportement.
    """
    for j in range(J):
        if params[j] <= 0 or params[j] >= 1:
            return 1e10
    for j in range(J, 2 * J):
        if params[j] <= 0:
            return 1e10
    a = params[2 * J] if alpha_fixed is None else alpha_fixed
    if a <= 0 or a > 2:
        return 1e10
    tcf = _tcf_matrix(params, u_grid, v_grid, J, alpha_fixed, beta_val)
    if W is None:
        return float(np.mean(np.abs(ecf - tcf) ** 2))
    return weighted_sse(ecf, tcf, W)


def _obj_restr(params_r, ecf, u_grid, v_grid, J, alpha_fixed, beta_val, W):
    """MDE objective — restricted model with pi_1 = 0.5 (varsigma_1 = varsigma_2 = sigma/2).

    params_r: (rho_1, rho_2, sigma, [alpha])  — dimension = J + 1 [+ 1]
    We internally reconstruct varsigma_1 = varsigma_2 = sigma / 2.
    """
    rhos = params_r[:J]
    sigma = params_r[J]
    idx = J + 1
    alpha = params_r[idx] if alpha_fixed is None else alpha_fixed
    if sigma <= 0:
        return 1e10
    for j in range(J):
        if rhos[j] <= 0 or rhos[j] >= 1:
            return 1e10
    if alpha <= 0 or alpha > 2:
        return 1e10
    # Reconstruct full param vector with varsigma_1 = varsigma_2 = sigma/2
    varsigmas = np.full(J, sigma / J)
    full = np.concatenate([rhos, varsigmas, [alpha] if alpha_fixed is None else []])
    tcf = _tcf_matrix(full, u_grid, v_grid, J, alpha_fixed, beta_val)
    if W is None:
        return float(np.mean(np.abs(ecf - tcf) ** 2))
    return weighted_sse(ecf, tcf, W)


def _estimate_free(Y, x0, u_grid, v_grid, J, bounds_free, alpha_fixed,
                   beta_val, W, maxiter=200):
    """Estimate free (unrestricted) model on a given series."""
    ecf = _ecf_matrix(Y, u_grid, v_grid)
    res = optimize.minimize(
        _obj_free, x0,
        args=(ecf, u_grid, v_grid, J, alpha_fixed, beta_val, W),
        bounds=bounds_free, method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-10})
    return res.x, res.fun


def _estimate_restr(Y, x0_r, u_grid, v_grid, J, bounds_restr, alpha_fixed,
                    beta_val, W, maxiter=200):
    """Estimate restricted model (pi_1 = 0.5) on a given series."""
    ecf = _ecf_matrix(Y, u_grid, v_grid)
    res = optimize.minimize(
        _obj_restr, x0_r,
        args=(ecf, u_grid, v_grid, J, alpha_fixed, beta_val, W),
        bounds=bounds_restr, method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-10})
    return res.x, res.fun


def _free_to_restr_x0(theta_free, J, alpha_fixed):
    """Convert free-model estimate to initial guess for restricted model.

    free: (rho_1,...,rho_J, varsigma_1,...,varsigma_J, [alpha])
    restr: (rho_1,...,rho_J, sigma, [alpha])
    """
    rhos = theta_free[:J]
    varsigmas = theta_free[J:2*J]
    sigma = np.sum(varsigmas)
    out = list(rhos) + [sigma]
    if alpha_fixed is None:
        out.append(theta_free[2 * J])
    return np.array(out)


def _restr_to_free_x0(theta_restr, J, alpha_fixed):
    """Plonge une solution restreinte dans l'espace libre.

    restr: (rho_1,...,rho_J, sigma, [alpha])
    free : (rho_1,...,rho_J, sigma/J,...,sigma/J, [alpha])

    Le modele restreint (pi_j = 1/J) est un SOUS-MODELE du modele libre : tout
    point restreint est admissible dans l'espace libre.  C'est ce qui garantit
    Q_free <= Q_restr, donc Delta_n >= 0.
    """
    rhos = list(theta_restr[:J])
    sigma = float(theta_restr[J])
    out = rhos + [sigma / J] * J
    if alpha_fixed is None:
        out.append(theta_restr[J + 1])
    return np.array(out)


def _delta_nested(Y, ecf, x0_free, u_grid, v_grid, J, bounds_free,
                  bounds_restr, alpha_fixed, beta_val, W, scale, maxiter):
    """Statistique de difference de critere, avec emboitement garanti.

    Renvoie (theta_free, Q_free, theta_restr, Q_restr, delta).

    PATCH.  La version initiale evaluait Q_free au point renvoye par
    stablemixture.estimate() — qui minimise un AUTRE critere — puis calculait
    Q_restr par optimisation effective, d'ou des Delta_n NEGATIFS (medianes
    -0.001 et -0.002 aux tables S.2.4 pour n=50k et n=100k), impossibles pour
    une restriction emboitee.

    Repondere le critere ne suffit pas : meme reoptimise, le modele libre reste
    parfois piege dans un minimum local situe AU-DESSUS de l'optimum contraint
    (le critere MDE est non convexe et la direction pi_1 est quasi plate).
    On impose donc l'emboitement numeriquement : le modele libre est reoptimise
    une seconde fois en partant de la solution restreinte plongee dans l'espace
    libre, et l'on retient le meilleur des deux optima.  Le point restreint
    etant admissible dans l'espace libre, Q_free <= Q_restr devient vrai par
    construction, a la tolerance de l'optimiseur pres.
    """
    # 1er optimum libre (depart : warm start externe)
    th_f, Q_f = _estimate_free(Y, x0_free, u_grid, v_grid, J, bounds_free,
                               alpha_fixed, beta_val, W, maxiter=maxiter)
    # optimum contraint, chauffe depuis l'optimum libre
    th_r, Q_r = _estimate_restr(Y, _free_to_restr_x0(th_f, J, alpha_fixed),
                                u_grid, v_grid, J, bounds_restr,
                                alpha_fixed, beta_val, W, maxiter=maxiter)

    # 2e optimum libre : depart = solution restreinte plongee dans l'espace libre
    x0_emb = _restr_to_free_x0(th_r, J, alpha_fixed)
    Q_emb = _obj_free(x0_emb, ecf, u_grid, v_grid, J, alpha_fixed, beta_val, W)
    th_f2, Q_f2 = _estimate_free(Y, x0_emb, u_grid, v_grid, J, bounds_free,
                                 alpha_fixed, beta_val, W, maxiter=maxiter)

    for th_c, Q_c in ((x0_emb, Q_emb), (th_f2, Q_f2)):
        if Q_c < Q_f:
            th_f, Q_f = np.asarray(th_c, dtype=float), Q_c

    return th_f, Q_f, th_r, Q_r, scale * (Q_r - Q_f)


# =========================================================================
# Single-replication worker (for MC study)
# =========================================================================

def _one_rep_criterion(seed, n, b, tp_full, J, av, bv, fa, fb,
                       ug, vg, bounds_free, bounds_restr,
                       kappa=KAPPA_DEFAULT):
    """Single MC replication: compute Delta_n and subsample reference distribution.

    Returns dict with:
        'delta_n':  the full-sample criterion difference statistic n * (Q_restr - Q_free)
        'delta_b':  array of subsample criterion differences b * (Q_restr^(i) - Q_free^(i))
        'pval_chi2': asymptotic p-value from chi2(1)
        'pval_sub':  subsampling p-value
        'theta_free': full-sample free estimate
        'theta_restr': full-sample restricted estimate
        'Q_free': full-sample free objective
        'Q_restr': full-sample restricted objective
    """
    from stablemixture import StableMixtureDeconvolution

    af = av if fa else None
    W = quadrature_weights(ug, vg, kappa)

    mdl = StableMixtureDeconvolution(
        num_components=J, alpha_value=av, beta_value=bv,
        fix_alpha=fa, fix_beta=fb, verbose=False)
    Y, _ = mdl.generate(T=n, params=tp_full, seed=seed)

    # --- Full-sample estimation ---
    # PATCH : le modele LIBRE est desormais reoptimise sur le critere
    # _obj_free lui-meme, comme le modele restreint, au lieu d'etre repris tel
    # quel de mdl.estimate() (qui minimise un AUTRE critere : poids uniformes
    # de stablemixture.objective_function, maxiter=100, depart fixe).
    # Cette asymetrie produisait Q_restr < Q_free, donc des Delta_n NEGATIFS
    # (medianes -0.001 et -0.002 aux Tables S.2.4 pour n=50k et n=100k), ce
    # qui est impossible pour une restriction emboitee correctement optimisee.
    # mdl.estimate() ne sert plus que de point de depart (warm start).
    x0_free = np.array(mdl.estimate(Y, use_optimal_weights=False))
    ecf_full = _ecf_matrix(Y, ug, vg)
    (theta_free_raw, Q_free,
     theta_restr_raw, Q_restr, delta_n) = _delta_nested(
        Y, ecf_full, x0_free, ug, vg, J, bounds_free, bounds_restr,
        af, bv, W, scale=n, maxiter=300)

    pval_chi2 = 1.0 - stats.chi2.cdf(max(delta_n, 0), df=1)

    # --- Subsample criterion differences (non-overlapping blocks) ---
    nno = n // b
    delta_b_list = []
    mxi = 100 if b < 80 else 60

    for i in range(nno):
        Y_sub = Y[i * b:(i + 1) * b]
        try:
            ecf_sub = _ecf_matrix(Y_sub, ug, vg)
            # Meme construction emboitee que sur l'echantillon complet, sans
            # quoi la distribution de reference de subsampling herite du meme
            # biais d'optimisation que la statistique qu'elle doit calibrer.
            *_, d_sub = _delta_nested(
                Y_sub, ecf_sub, theta_free_raw, ug, vg, J, bounds_free,
                bounds_restr, af, bv, W, scale=b, maxiter=mxi)
            delta_b_list.append(d_sub)
        except Exception:
            pass

    delta_b = np.array(delta_b_list)

    # Subsampling p-value: fraction of subsample statistics >= full-sample statistic
    if len(delta_b) >= 3:
        pval_sub = float(np.mean(delta_b >= delta_n))
    else:
        pval_sub = np.nan

    return {
        'delta_n': delta_n,
        # Diagnostic : doit valoir False pour toutes les replications.  Un
        # Delta_n < 0 signale que l'optimum libre n'est pas atteint, donc que
        # le test n'est pas interpretable pour cette replication.
        'delta_n_negative': bool(delta_n < 0),
        'frac_delta_b_negative': (float(np.mean(delta_b < 0))
                                  if len(delta_b) else np.nan),
        'delta_b': delta_b,
        'pval_chi2': pval_chi2,
        'pval_sub': pval_sub,
        'theta_free': theta_free_raw,
        'theta_restr': theta_restr_raw,
        'Q_free': Q_free,
        'Q_restr': Q_restr,
    }


# =========================================================================
# Public API — Monte Carlo study
# =========================================================================

def run_mc_criterion_test(
    sample_sizes: List[int] = [500, 1000, 10000],
    M: int = 200,
    b_exponent: float = 2 / 3,
    n_cores: int = -1,
    seed: int = 42,
    output_dir: str = 'results_criterion_test',
    true_params: tuple = (0.8, 0.3, 0.7, 0.9, 1.5, 0.0),
    alpha: float = 1.5,
    beta: float = 0.0,
    fix_alpha: bool = False,
    fix_beta: bool = True,
    J: int = 2,
    kappa: float = KAPPA_DEFAULT,
) -> Dict:
    """Run Monte Carlo study of the criterion difference test for H0: pi_1 = 0.5.

    Args:
        sample_sizes: List of sample sizes n.
        M: Number of MC replications.
        b_exponent: Block size exponent (b = floor(n^exp)).
        n_cores: Number of cores for parallelization (-1 = all).
        seed: Base random seed.
        output_dir: Directory for output files.
        true_params: True DGP parameters (rho_1,...,rho_J, varsigma_1,...,varsigma_J, alpha, beta).
        alpha, beta: Stability and skewness parameters.
        fix_alpha, fix_beta: Whether to fix alpha/beta during estimation.
        J: Number of latent components.

    Returns:
        Dict keyed by n with aggregated results.
    """
    os.makedirs(output_dir, exist_ok=True)
    af = alpha if fix_alpha else None

    # Grids
    ug = np.linspace(-1.5, 1.5, 15)
    vg = np.linspace(-1.5, 1.5, 15)

    print("Critere MDE (Assumption C2, main_v23.tex):")
    check_grid(ug, vg, kappa)

    # Bounds — free model
    bounds_free = [(0.01, 0.99)] * J + [(0.005, 5.0)] * J
    if not fix_alpha:
        bounds_free.append((0.1, 1.99))
    if not fix_beta:
        bounds_free.append((-0.99, 0.99))

    # Bounds — restricted model: (rho_1,...,rho_J, sigma, [alpha])
    bounds_restr = [(0.01, 0.99)] * J + [(0.01, 10.0)]
    if not fix_alpha:
        bounds_restr.append((0.1, 1.99))
    if not fix_beta:
        bounds_restr.append((-0.99, 0.99))

    # Compute true pi_1
    varsigmas_true = [true_params[J + j] for j in range(J)]
    sigma_true = sum(varsigmas_true)
    pi1_true = varsigmas_true[0] / sigma_true

    results = {}
    lines = []
    lines.append("CRITERION DIFFERENCE TEST: H0: pi_1 = 0.5")
    lines.append(f"True pi_1 = {pi1_true:.4f}")
    lines.append(f"Weight: w(u,v) = (kappa/pi) exp(-kappa(u^2+v^2)), "
                 f"kappa={kappa}  [Assumption C2, main_v23.tex]")
    lines.append("=" * 80)

    for n in sample_sizes:
        b = int(np.floor(n ** b_exponent))
        nno = n // b
        print(f"\n{'=' * 65}")
        print(f"n={n}, b={b}, blocks/rep={nno}, M={M}")

        t0 = time.time()
        reps = Parallel(n_jobs=n_cores, backend='loky', verbose=5)(
            delayed(_one_rep_criterion)(
                seed + m, n, b, true_params, J, alpha, beta,
                fix_alpha, fix_beta, ug, vg, bounds_free, bounds_restr,
                kappa)
            for m in range(M))
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s")

        # Aggregate
        deltas = np.array([r['delta_n'] for r in reps])
        pvals_chi2 = np.array([r['pval_chi2'] for r in reps])
        pvals_sub = np.array([r['pval_sub'] for r in reps
                              if not np.isnan(r['pval_sub'])])

        rej_chi2_05 = float(np.mean(pvals_chi2 < 0.05))
        rej_chi2_10 = float(np.mean(pvals_chi2 < 0.10))
        rej_sub_05 = float(np.mean(pvals_sub < 0.05)) if len(pvals_sub) > 0 else np.nan
        rej_sub_10 = float(np.mean(pvals_sub < 0.10)) if len(pvals_sub) > 0 else np.nan

        # Diagnostic d'optimisation : Delta_n >= 0 par construction.
        frac_neg_n = float(np.mean([r['delta_n_negative'] for r in reps]))
        fb_neg = [r['frac_delta_b_negative'] for r in reps
                  if not np.isnan(r['frac_delta_b_negative'])]
        frac_neg_b = float(np.mean(fb_neg)) if fb_neg else np.nan

        agg = {
            'n': n, 'b': b, 'M': M,
            'delta_mean': float(np.mean(deltas)),
            'delta_median': float(np.median(deltas)),
            'delta_std': float(np.std(deltas)),
            'pval_chi2_mean': float(np.mean(pvals_chi2)),
            'rej_chi2_05': rej_chi2_05,
            'rej_chi2_10': rej_chi2_10,
            'rej_sub_05': rej_sub_05,
            'rej_sub_10': rej_sub_10,
            'frac_delta_n_negative': frac_neg_n,
            'frac_delta_b_negative': frac_neg_b,
            'kappa': kappa,
        }
        results[n] = agg

        summary = (
            f"n={n}: Delta_n mean={agg['delta_mean']:.4f}, "
            f"median={agg['delta_median']:.4f}, std={agg['delta_std']:.4f}\n"
            f"  chi2(1) p-val mean={agg['pval_chi2_mean']:.4f}, "
            f"rej@5%={rej_chi2_05:.3f}, rej@10%={rej_chi2_10:.3f}\n"
            f"  subsampling rej@5%={rej_sub_05:.3f}, rej@10%={rej_sub_10:.3f}\n"
            f"  [diag] frac(Delta_n < 0)={frac_neg_n:.3f}, "
            f"mean frac(Delta_b < 0)={frac_neg_b:.3f}  "
            f"(doivent etre ~0 ; sinon l'optimum libre n'est pas atteint)"
        )
        print(summary)
        lines.append(f"\nn={n}  (b={b}, M={M}, kappa={kappa})")
        lines.append("-" * 80)
        lines.append(summary)

    # Save
    with open(os.path.join(output_dir, 'criterion_test_results.txt'), 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nResults saved in {output_dir}/")
    return results


# =========================================================================
# Public API — single series test
# =========================================================================

def test_single_series(
    Y: np.ndarray,
    b_exponent: float = 2 / 3,
    alpha: float = 1.5,
    beta: float = 0.0,
    fix_alpha: bool = False,
    fix_beta: bool = True,
    J: int = 2,
    initial_guess: Optional[List[float]] = None,
    kappa: float = KAPPA_DEFAULT,
) -> Dict:
    """Apply the criterion difference test to a single observed series.

    Args:
        Y: Observed time series.
        b_exponent: Block size exponent.
        alpha, beta: Stability parameters.
        fix_alpha, fix_beta: Fix during estimation.
        J: Number of components.
        initial_guess: Optional initial guess for free model.

    Returns:
        Dict with delta_n, p-values (chi2 and subsampling), estimates, profile.
    """
    from stablemixture import StableMixtureDeconvolution

    n = len(Y)
    b = int(np.floor(n ** b_exponent))
    af = alpha if fix_alpha else None

    ug = np.linspace(-1.5, 1.5, 15)
    vg = np.linspace(-1.5, 1.5, 15)
    W = quadrature_weights(ug, vg, kappa)

    bounds_free = [(0.01, 0.99)] * J + [(0.005, 5.0)] * J
    if not fix_alpha:
        bounds_free.append((0.1, 1.99))
    if not fix_beta:
        bounds_free.append((-0.99, 0.99))

    bounds_restr = [(0.01, 0.99)] * J + [(0.01, 10.0)]
    if not fix_alpha:
        bounds_restr.append((0.1, 1.99))
    if not fix_beta:
        bounds_restr.append((-0.99, 0.99))

    # --- Free model ---
    mdl = StableMixtureDeconvolution(
        num_components=J, alpha_value=alpha, beta_value=beta,
        fix_alpha=fix_alpha, fix_beta=fix_beta, verbose=False)
    # PATCH : mdl.estimate() ne fournit qu'un warm start ; le modele libre est
    # reoptimise sur _obj_free, comme le modele restreint (cf. commentaire dans
    # _one_rep_criterion).
    x0_free = np.array(mdl.estimate(Y, initial_guess=initial_guess,
                                    use_optimal_weights=False))
    ecf_full = _ecf_matrix(Y, ug, vg)

    # --- Free and restricted models, avec emboitement garanti ---
    (theta_free, Q_free,
     theta_restr, Q_restr, delta_n) = _delta_nested(
        Y, ecf_full, x0_free, ug, vg, J, bounds_free, bounds_restr,
        af, beta, W, scale=n, maxiter=300)

    pval_chi2 = 1.0 - stats.chi2.cdf(max(delta_n, 0), df=1)

    # --- Subsampling reference distribution ---
    nno = n // b
    delta_b_list = []
    mxi = 100 if b < 80 else 60
    for i in range(nno):
        Y_sub = Y[i * b:(i + 1) * b]
        try:
            ecf_sub = _ecf_matrix(Y_sub, ug, vg)
            *_, d_sub = _delta_nested(
                Y_sub, ecf_sub, theta_free, ug, vg, J, bounds_free,
                bounds_restr, af, beta, W, scale=b, maxiter=mxi)
            delta_b_list.append(d_sub)
        except Exception:
            pass
    delta_b = np.array(delta_b_list)
    pval_sub = float(np.mean(delta_b >= delta_n)) if len(delta_b) >= 3 else np.nan

    # --- Profile of Q_n in pi_1 direction ---
    pi1_grid = np.linspace(0.1, 0.9, 17)
    profile = []
    for pi1_val in pi1_grid:
        # Fix pi_1 = pi1_val => varsigma_1 = pi1_val * sigma, varsigma_2 = (1-pi1_val) * sigma
        # Optimize over (rho_1, rho_2, sigma, [alpha]) with this constraint
        def _obj_profile(p, _ecf=ecf_full, _pi1=pi1_val):
            rhos_p = p[:J]
            sigma_p = p[J]
            a_p = p[J + 1] if af is None else af
            if sigma_p <= 0:
                return 1e10
            for jj in range(J):
                if rhos_p[jj] <= 0 or rhos_p[jj] >= 1:
                    return 1e10
            if a_p <= 0 or a_p > 2:
                return 1e10
            vs = np.array([_pi1 * sigma_p, (1 - _pi1) * sigma_p])
            full_p = np.concatenate([rhos_p, vs, [a_p] if af is None else []])
            tcf = _tcf_matrix(full_p, ug, vg, J, af, beta)
            if W is None:
                return float(np.mean(np.abs(_ecf - tcf) ** 2))
            return weighted_sse(_ecf, tcf, W)

        x0_prof = list(theta_restr[:J]) + [theta_restr[J]]
        if af is None:
            x0_prof.append(theta_restr[J + 1])
        bounds_prof = [(0.01, 0.99)] * J + [(0.01, 10.0)]
        if af is None:
            bounds_prof.append((0.1, 1.99))

        try:
            res_prof = optimize.minimize(
                _obj_profile, np.array(x0_prof),
                bounds=bounds_prof, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10})
            profile.append((pi1_val, res_prof.fun))
        except Exception:
            profile.append((pi1_val, np.nan))

    # Derived quantities from free estimate
    varsigmas_free = theta_free[J:2*J]
    sigma_free = np.sum(varsigmas_free)
    pi1_free = varsigmas_free[0] / sigma_free if sigma_free > 0 else np.nan

    return {
        'n': n, 'b': b,
        'delta_n': delta_n,
        'pval_chi2': pval_chi2,
        'pval_sub': pval_sub,
        'theta_free': theta_free,
        'theta_restr': theta_restr,
        'Q_free': Q_free,
        'Q_restr': Q_restr,
        'pi1_hat_free': pi1_free,
        'sigma_hat_free': sigma_free,
        'delta_b': delta_b,
        'profile': profile,
    }