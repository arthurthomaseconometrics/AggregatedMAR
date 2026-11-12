#!/usr/bin/env python3
r"""
Figure 1 of the paper (\label{fig:contagg}, figures/discretestableagg.pdf):
"Simulated stable aggregate dynamics with two components".

Two purely anticipative alpha-stable AR(1) components,
    X_{j,t} = psi_j X_{j,t+1} + eps_{j,t},   eps_{j,t} ~iid S(alpha, beta, 1, 0),
with (psi_1, pi_1) = (0.90, 0.40) and (psi_2, pi_2) = (0.70, 0.60), aggregated as
    calX_t = sigma * (pi_1 X_{1,t} + pi_2 X_{2,t}).

Panels: top left, the weights pi_j of the two components; top right, the
aggregate calX_t; bottom, the contribution sigma * pi_j * X_{j,t} of each latent
component (the two bottom panels add up to the top-right one).

The figure is illustrative.  The script that produced the published PDF was
not archived; this one reproduces its design (parameters, layout, T = 500),
but the simulated path depends on the random draw, so the series will not be
identical to the published one.

Usage
-----
    python figure1_stable_aggregate.py                 # seed 42
    python figure1_stable_aggregate.py --seed 7 --alpha 1.5

Output: results_figure1/discretestableagg.{pdf,png}, run_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results_figure1")


def anticipative_ar1(n: int, psi: float, alpha: float, beta: float,
                     rng: np.random.Generator, burn_in: int = 500) -> np.ndarray:
    """X_t = psi X_{t+1} + eps_t, built backwards; the burn-in is on the
    future side (it is discarded at the END of the recursion)."""
    m = n + burn_in
    eps = stats.levy_stable.rvs(alpha, beta, loc=0.0, scale=1.0, size=m,
                                random_state=rng)
    x = np.zeros(m)
    x[-1] = eps[-1]
    for t in range(m - 2, -1, -1):
        x[t] = psi * x[t + 1] + eps[t]
    return x[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--psi", type=float, nargs=2, default=[0.90, 0.70])
    ap.add_argument("--pi", type=float, nargs=2, default=[0.40, 0.60])
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ss = np.random.SeedSequence(args.seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(2)]  # independent streams
    comps = [args.sigma * p * anticipative_ar1(args.T, psi, args.alpha, args.beta, r)
             for psi, p, r in zip(args.psi, args.pi, rngs)]
    agg = comps[0] + comps[1]
    t = np.arange(args.T)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    a = ax[0, 0]
    a.bar([0, 1], args.pi, width=0.6, color="#ee6a5f", alpha=0.95)
    for i, (p, psi) in enumerate(zip(args.pi, args.psi)):
        a.text(i, p + 0.01, f"π{i + 1}={p:.2f}", ha="center", va="bottom")
    a.set_xticks([0, 1], [f"ψ₁={args.psi[0]:g}", f"ψ₂={args.psi[1]:g}"])
    a.set_ylim(0, max(args.pi) + 0.1)
    a.set_ylabel("π")
    a.set_title("Discrete mixture weights")

    ax[0, 1].plot(t, agg, color="black", lw=0.8, alpha=0.8)
    ax[0, 1].set_title(r"Stable aggregate process $\mathcal{X}_t$")
    for j, (c, col) in enumerate(zip(comps, ["blue", "red"])):
        a = ax[1, j]
        a.plot(t, c, color=col, lw=0.8, alpha=0.8)
        sub = "₁₂"[j]
        a.set_title(f"Latent component {j + 1}\nψ{sub}={args.psi[j]:g}, π{sub}={args.pi[j]:.2f}")
    for a in ax.flat:
        a.grid(alpha=0.3)
    for a in (ax[0, 1], ax[1, 0], ax[1, 1]):
        a.set_xlabel("Time")
        a.set_ylabel("Value")
    fig.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, "discretestableagg.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "discretestableagg.png"), dpi=150, bbox_inches="tight")
    cfg = dict(vars(args), target="main paper :: fig:contagg (figures/discretestableagg.pdf)",
               python_version=platform.python_version(), numpy_version=np.__version__,
               scipy_version=scipy.__version__)
    with open(os.path.join(OUTDIR, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Written to {OUTDIR}/discretestableagg.pdf")


if __name__ == "__main__":
    main()
