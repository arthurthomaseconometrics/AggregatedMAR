"""Subsampling version of the test of H0: psi_1 = 0, reported in Section S.4.2 of the
Online Supplement of de Truchis, Fries and Thomas (2026) (alternative critical values).

For each replication, the statistic computed on the full sample is compared with
the (1 - a)-quantile of the same statistic computed on overlapping blocks of
size b = floor(n^{2/3}), with the null value psi_1 = 0 plugged in (Politis,
Romano and Wolf, 1999, Ch. 3):
  * S:  sqrt(n) psi_hat_1   against the quantile of sqrt(b) psi_hat_{1,b}
  * t:  t_n (V at theta_hat) against the quantile of t_b (V at theta_hat_b)
The blocks are thinned to N_BLOCKS evenly spaced starting points.

Usage: python subsampling_check.py [--reps 500] [--workers 12]
"""
import argparse
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

import boundary_test_mc as b

N_BLOCKS = 250
LEVELS = (0.05, 0.10)


def t_ratio(theta, x):
    v = b.sandwich(theta, x)[2, 2]
    return np.sqrt(x.size - 1) * theta[2] / np.sqrt(v) if v > 0 else np.nan


def one_rep(task):
    n, psi_small, rep = task
    seed = np.random.SeedSequence([b.BASE_SEED + 1, n, int(round(1000 * psi_small)), rep])
    rng = np.random.default_rng(seed)
    theta0 = np.array([b.VARSIGMA_SMALL, b.VARSIGMA_LARGE, psi_small, b.PSI_LARGE, b.ALPHA])
    x = b.simulate(n, theta0, rng)
    theta, _ = b.estimate(x)
    S, t = np.sqrt(n - 1) * theta[2], t_ratio(theta, x)
    bs = int(np.floor(n ** (2 / 3)))
    starts = np.unique(np.linspace(0, n - bs, N_BLOCKS).astype(int))
    Sb, tb = [], []
    for s in starts:
        xb = x[s:s + bs]
        th, _ = b.estimate(xb)
        Sb.append(np.sqrt(bs - 1) * th[2])
        tb.append(t_ratio(th, xb))
    out = dict(n=n, psi_small=psi_small, rep=rep, b=bs, S=S, t=t)
    for a in LEVELS:
        out[f"rej_S_{a:.2f}"] = S > np.quantile(Sb, 1 - a)
        out[f"rej_t_{a:.2f}"] = t > np.nanquantile(tb, 1 - a)
        out[f"cv_S_{a:.2f}"] = np.quantile(Sb, 1 - a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    tasks = list(itertools.product([1_000, 5_000], [0.0, 0.3], range(args.reps)))
    with Pool(args.workers) as pool:
        df = pd.DataFrame(pool.map(one_rep, tasks, chunksize=2))
    b.OUT.mkdir(exist_ok=True)
    df.to_csv(b.OUT / "subsampling_check.csv", index=False)
    cols = [c for c in df.columns if c.startswith(("rej_", "cv_"))]
    print(df.groupby(["n", "psi_small", "b"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
