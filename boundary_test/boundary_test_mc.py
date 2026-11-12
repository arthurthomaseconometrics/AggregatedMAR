"""Monte Carlo for the boundary test of H0: min_j psi_j = 0 (section_boundary_test).

DGP: Case 1 of main_v24 with J = 2 and beta = 0 (symmetric alpha-stable),
    X_t = varsigma_1 X_{1,t} + varsigma_2 X_{2,t},  X_{j,t} = psi_j X_{j,t+1} + eps_{j,t},
with eps_{j,t} i.i.d. S(alpha, 0, 1, 0).  psi_j = 0 makes X_{j,t} = eps_{j,t}.

Estimator: criterion (2.14) of main_v24 evaluated with the paper's quadrature
(15 x 15 grid on [-1.5, 1.5]^2, weight (kappa/pi) exp(-kappa(u^2 + v^2)) du dv,
kappa = 1, as in cf_weights.quadrature_weights), minimised over the extended
space psi_j in [0, PSI_MAX].  The labels are sorted afterwards so that
psi_1 < psi_2, which is the ordering of Assumption 1.  The restricted
estimator imposes psi_1 = 0.

Tests of H0, based on S_n = sqrt(m) psi_hat_1 with m = n - 1:
  * asymptotic t-tests t = S_n / sqrt(V_11), with V = Sigma^{-1} Omega Sigma^{-1}
    (same quadrature for Sigma, Bartlett HAC for Omega) evaluated either at the
    unrestricted or at the restricted estimate, against z_{1-a};
  * restricted parametric bootstrap, applied to S_n and to the t-ratio with
    V at the unrestricted estimate (bootstrap-t): the statistic is compared
    with the quantile of its value on samples simulated from the restricted
    estimate.  Rejection rates use the warp-speed method of Giacomini, Politis
    and White (2013): one bootstrap sample per replication, critical value
    from the M bootstrap statistics.

Usage: python boundary_test_mc.py [--reps 2000] [--workers 12] [--sizes ...] [--append]
"""
import argparse
import itertools
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

# Quadrature of the paper (cf_weights.py: U_MAX_DEFAULT, N_GRID_DEFAULT, KAPPA_DEFAULT)
KAPPA, U_MAX, N_GRID = 1.0, 1.5, 15
_g = np.linspace(-U_MAX, U_MAX, N_GRID)
_U, _V = np.meshgrid(_g, _g, indexing="ij")
U, V = _U.ravel(), _V.ravel()
STEP = _g[1] - _g[0]
W = (KAPPA / np.pi) * np.exp(-KAPPA * (U**2 + V**2)) * STEP**2

# Extended parameter space: theta = (varsigma_1, varsigma_2, psi_1, psi_2, alpha)
PSI_MAX = 0.98
BOUNDS = [(0.05, 5.0), (0.05, 5.0), (0.0, PSI_MAX), (0.0, PSI_MAX), (1.05, 1.95)]
BOUNDS_H0 = BOUNDS[:2] + [(0.0, 0.0)] + BOUNDS[3:]
STARTS = [(0.1, 0.6), (0.3, 0.9), (0.02, 0.8), (0.5, 0.95)]
STARTS_H0 = [(0.0, 0.6), (0.0, 0.9)]
# Optimiser tolerances of the paper (cf_weights.MDE_GTOL / MDE_FTOL / MDE_MAXITER)
LBFGSB = dict(gtol=1e-12, ftol=1e-16, maxiter=2000)

# Design
ALPHA = 1.5
PSI_LARGE, VARSIGMA_LARGE = 0.8, 0.7
VARSIGMA_SMALL = 0.9
PSI_SMALL = [0.0, 0.05, 0.1, 0.2, 0.3]
SAMPLE_SIZES = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
BURN = 300
BASE_SEED = 20261106
LEVELS = (0.05, 0.10)


def _xlogx(x, a):
    """x^a log x, extended by 0 at x = 0."""
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = x[pos] ** a * np.log(x[pos])
    return out


def cf_and_grad(theta):
    """phi(u, v; theta) on the grid and its gradient G (p x K), beta = 0."""
    vs, psi, a = theta[:2], theta[2:4], theta[4]
    abs_u = np.abs(U)
    B, dB = abs_u**a, _xlogx(abs_u, a)
    L = np.zeros_like(U)
    dL = np.zeros((5, U.size))
    for j in range(2):
        s = U * psi[j] + V
        abs_s = np.abs(s)
        c = 1.0 / (1.0 - psi[j] ** a)
        A = abs_s**a * c
        plog = psi[j] ** a * np.log(psi[j]) if psi[j] > 0 else 0.0
        L -= vs[j] ** a * (A + B)
        dL[j] = -a * vs[j] ** (a - 1) * (A + B)
        dL[2 + j] = -vs[j] ** a * (
            a * U * abs_s ** (a - 1) * np.sign(s) * c + abs_s**a * a * psi[j] ** (a - 1) * c**2
        )
        dL[4] -= vs[j] ** a * (
            np.log(vs[j]) * (A + B) + _xlogx(abs_s, a) * c + abs_s**a * plog * c**2 + dB
        )
    phi = np.exp(L)
    return phi, phi * dL


def criterion(theta, ecf):
    phi, G = cf_and_grad(theta)
    f = ecf - phi
    return float(np.sum(W * np.abs(f) ** 2)), -2.0 * np.sum(W * np.real(np.conj(f) * G), axis=1)


def simulate(n, theta, rng):
    """Aggregate with parameters theta = (varsigma_1, varsigma_2, psi_1, psi_2, alpha)."""
    x = np.zeros(n)
    for vs, psi in ((theta[0], theta[2]), (theta[1], theta[3])):
        eps = stats.levy_stable.rvs(theta[4], 0.0, size=n + BURN, random_state=rng)
        # X_t = psi X_{t+1} + eps_t: a causal filter on the time-reversed series,
        # whose transient sits at the end of the array and is discarded
        x += vs * signal.lfilter([1.0], [1.0, -psi], eps[::-1])[::-1][:n]
    return x


def _minimise(ecf, starts, bounds):
    best = None
    for p1, p2 in starts:
        res = optimize.minimize(
            criterion, [0.8, 0.8, p1, p2, 1.5], args=(ecf,), jac=True,
            method="L-BFGS-B", bounds=bounds, options=LBFGSB,
        )
        if best is None or res.fun < best.fun:
            best = res
    return best


CHUNK = 20_000  # pairs per block when forming exp(i(u X_t + v X_{t+1})), to bound memory


def _pair_blocks(x):
    """exp(i(u X_t + v X_{t+1})) on the grid, by blocks of pairs t = 1, ..., n-1."""
    for s in range(0, x.size - 1, CHUNK):
        e = min(s + CHUNK, x.size - 1)
        yield np.exp(1j * (np.outer(x[s:e], U) + np.outer(x[s + 1:e + 1], V)))


def ecf(x):
    return sum(Zb.sum(axis=0) for Zb in _pair_blocks(x)) / (x.size - 1)


def estimate(x):
    """Unrestricted estimate, relabelled so that psi_1 < psi_2."""
    res = _minimise(ecf(x), STARTS, BOUNDS)
    theta = res.x.copy()
    if theta[2] > theta[3]:
        theta = theta[[1, 0, 3, 2, 4]]
    return theta, res


def estimate_h0(x):
    """Estimate under H0: psi_1 = 0."""
    return _minimise(ecf(x), STARTS_H0, BOUNDS_H0).x


def sandwich(theta, x):
    """Sigma^{-1} Omega Sigma^{-1}, Omega by Bartlett HAC on the scores K_t(theta)."""
    phi, G = cf_and_grad(theta)
    WG = (W * np.conj(G)).T
    K = np.vstack([np.real((Zb - phi) @ WG) for Zb in _pair_blocks(x)])
    m = K.shape[0]
    Sigma = np.real((G * W) @ np.conj(G).T)
    K -= K.mean(axis=0)
    Omega = K.T @ K / m
    lags = int(np.floor(4 * (m / 100) ** (2 / 9)))
    for h in range(1, lags + 1):
        Gam = K[h:].T @ K[:-h] / m
        Omega += (1 - h / (lags + 1)) * (Gam + Gam.T)
    Si = np.linalg.inv(Sigma)
    return Si @ Omega @ Si


def one_rep(task):
    n, psi_small, rep = task
    seed = np.random.SeedSequence([BASE_SEED, n, int(round(1000 * psi_small)), rep])
    rng = np.random.default_rng(seed)
    theta0 = np.array([VARSIGMA_SMALL, VARSIGMA_LARGE, psi_small, PSI_LARGE, ALPHA])
    x = simulate(n, theta0, rng)
    m = n - 1
    theta, res = estimate(x)
    theta_h0 = estimate_h0(x)
    v11 = sandwich(theta, x)[2, 2]
    v11_h0 = sandwich(theta_h0, x)[2, 2]
    S = np.sqrt(m) * theta[2]
    # warp-speed bootstrap: one sample drawn from the restricted estimate
    x_star = simulate(n, theta_h0, rng)
    theta_star, _ = estimate(x_star)
    v11_star = sandwich(theta_star, x_star)[2, 2]
    return dict(n=n, psi_small=psi_small, rep=rep, psi1_hat=theta[2], psi2_hat=theta[3],
                alpha_hat=theta[4], S=S, S_star=np.sqrt(m) * theta_star[2],
                t_star=np.sqrt(m) * theta_star[2] / np.sqrt(v11_star) if v11_star > 0 else np.nan,
                t_unres=S / np.sqrt(v11) if v11 > 0 else np.nan,
                t_h0=S / np.sqrt(v11_h0) if v11_h0 > 0 else np.nan,
                sd_h0=np.sqrt(v11_h0) if v11_h0 > 0 else np.nan,
                at_boundary=theta[2] < 1e-6, converged=bool(res.success))


def summarise(df):
    rows = []
    for (n, p), g in df.groupby(["n", "psi_small"]):
        r = dict(n=n, psi_small=p, reps=len(g), share_boundary=g.at_boundary.mean(),
                 mean_psi1_hat=g.psi1_hat.mean(), median_sd_h0=g.sd_h0.median(),
                 share_converged=g.converged.mean(), nan_t_unres=int(g.t_unres.isna().sum()))
        for a in LEVELS:
            z = stats.norm.ppf(1 - a)
            r[f"rej_t_unres_{a:.2f}"] = np.mean(g.t_unres > z)
            r[f"rej_t_h0_{a:.2f}"] = np.mean(g.t_h0 > z)
            r[f"rej_boot_{a:.2f}"] = np.mean(g.S > np.quantile(g.S_star, 1 - a))
            r[f"rej_boot_t_{a:.2f}"] = np.mean(g.t_unres > np.nanquantile(g.t_star, 1 - a))
        r["rej_two_sided_0.05"] = np.mean(np.abs(g.t_unres) > stats.norm.ppf(0.975))
        rows.append(r)
    return pd.DataFrame(rows)


def _n(n):
    return f"{int(n):,}".replace(",", "{,}")


def latex_table(summ):
    """Panel A: size of the t-test under H0 (psi_1 = 0).  Panel B: power at 5% under H1."""
    h0 = summ[summ.psi_small == 0.0]
    alts = [p for p in PSI_SMALL if p > 0]
    lines = [
        r"\begin{tabular}{rcccccc}", r"\toprule",
        r"& \multicolumn{2}{c}{Asymptotic} & \multicolumn{2}{c}{Bootstrap} & Two-sided & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"$n$ & $a = 5\%$ & $a = 10\%$ & $a = 5\%$ & $a = 10\%$ & $a = 5\%$ & $\Pr(\hat\psi_1 = 0)$ \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Panel A. Rejection rates under $H_0$: $\psi_1 = 0$}} \\",
        r"\midrule",
    ]
    for _, r in h0.iterrows():
        lines.append(f"{_n(r.n)} & {r['rej_t_unres_0.05']:.3f} & {r['rej_t_unres_0.10']:.3f} & "
                     f"{r['rej_boot_t_0.05']:.3f} & {r['rej_boot_t_0.10']:.3f} & "
                     f"{r['rej_two_sided_0.05']:.3f} & {r.share_boundary:.3f} \\\\")
    lines += [
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Panel B. Rejection rates of the bootstrap test at 5\% under $H_1$: $\psi_1 > 0$}} \\",
        r"\midrule",
        r"$n$ & " + " & ".join(f"$\\psi_1={p:g}$" for p in alts) + r" & & \\",
        r"\midrule",
    ]
    for n in SAMPLE_SIZES:
        cells = [f"{summ[(summ.n == n) & (summ.psi_small == p)].iloc[0]['rej_boot_t_0.05']:.3f}"
                 for p in alts]
        lines.append(_n(n) + " & " + " & ".join(cells) + r" & & \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sizes", type=int, nargs="*", default=SAMPLE_SIZES)
    ap.add_argument("--append", action="store_true",
                    help="keep the replications of the other sample sizes in results/")
    ap.add_argument("--table-only", action="store_true",
                    help="rewrite tab_boundary_mc.tex from results/summary.csv")
    args = ap.parse_args()
    if args.table_only:
        (OUT / "tab_boundary_mc.tex").write_text(latex_table(pd.read_csv(OUT / "summary.csv")))
        return
    OUT.mkdir(exist_ok=True)
    tasks = list(itertools.product(args.sizes, PSI_SMALL, range(args.reps)))
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = pool.map(one_rep, tasks, chunksize=1)
    df = pd.DataFrame(rows)
    if args.append and (OUT / "replications.csv").exists():
        old = pd.read_csv(OUT / "replications.csv")
        df = pd.concat([old[~old.n.isin(args.sizes)], df], ignore_index=True)
    df.to_csv(OUT / "replications.csv", index=False)
    summ = summarise(df)
    summ.to_csv(OUT / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.round(3).to_string(index=False))
    print(f"{len(tasks)} replications in {time.time() - t0:.0f} s")
    if sorted(df.n.unique()) == SAMPLE_SIZES:
        (OUT / "tab_boundary_mc.tex").write_text(latex_table(summ))


if __name__ == "__main__":
    main()
