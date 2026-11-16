#!/usr/bin/env python3
r"""
Figure S.6 of the Online Supplement of de Truchis, Fries and Thomas (2026): deconvolution of the OVX index
from the G-alpha-S two-component model (Section S.5.2).

Three stacked panels: the observed detrended OVX series, followed by the two
latent components filtered with the particle MCMC smoother of de Truchis, Thomas and Vaudree (2026)
(`stablemixture.StableMixtureDeconvolution.mcmc_filter`).

Outputs (results_ovx_deconvolution/)
--------------------------------------------------------------------------
    figure_ovx_deconvolution.pdf / .png   the figure
    filtered_components.csv               observed series + filtered X_1, X_2
    filtering_diagnostics.csv             aggregation residual, per-component
                                          moments, share of variation
    run_config.json                       settings and environment

Filter settings
--------------------------------------------------------------------------
1000 particles, MCMC window of 5, seed 11 (`ovx_common.FILTER_SETTINGS`), applied
to the G-alpha-S estimates of Table S.14.

Reproducibility
--------------------------------------------------------------------------
`mcmc_filter` is stochastic and parallelized; at a fixed seed it is
reproducible for a given number of workers, but task splitting depends on
`--n-jobs`. The value used is recorded in run_config.json.

Usage
--------------------------------------------------------------------------
    python figure_ovx_deconvolution.py
    python figure_ovx_deconvolution.py --n-particles 2000 --n-jobs 8
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ovx_common as oc

OUTDIR_NAME = "results_ovx_deconvolution"


def build_model(params: dict, verbose: bool = False):
    """Instantiate the two-component G-alpha-S model at the given parameters."""
    from stablemixture import StableMixtureDeconvolution
    model = StableMixtureDeconvolution(
        num_components=2,
        alpha_value=params["alpha"], beta_value=params["beta"],
        fix_alpha=False, fix_beta=False, verbose=verbose)
    # Raw parameter vector expected by mcmc_filter:
    # (rho_1, rho_2, varsigma_1, varsigma_2, alpha, beta)
    raw = (params["rho_1"], params["rho_2"],
           params["varsigma_1"], params["varsigma_2"],
           params["alpha"], params["beta"])
    return model, raw


def make_figure(df: pd.DataFrame, X: list, params: dict) -> plt.Figure:
    """Reproduce the layout of `visualize_filtered_components`.

    Top panel: observed aggregate in black. Following panels: each filtered
    component, dashed, titled by (rho_j, pi_j, varsigma_j).
    """
    plt.style.use("ggplot")
    n_rows = len(X) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, 4 * n_rows))

    axes[0].plot(df.index, df["detrended"].to_numpy(), "k-", linewidth=1.0, alpha=0.7)
    axes[0].set_title("Mixed Alpha-Stable Process (OVX)")
    axes[0].grid(True, alpha=0.3)

    colors = ["b", "r"]
    for j, xj in enumerate(X):
        ax = axes[j + 1]
        ax.plot(df.index, xj, color=colors[j % len(colors)], linestyle="--",
                label="Filtered")
        ax.set_title(
            f"Component {j+1}: "
            f"ψ={params[f'rho_{j+1}']:.4f}, "
            f"π={params[f'pi_{j+1}']:.4f}, "
            f"ς={params[f'varsigma_{j+1}']:.4f}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    return fig


def diagnostics(df: pd.DataFrame, X: list) -> pd.DataFrame:
    """Deconvolution quality checks.

    `residual` is Y - sum_j X_j: the filtering procedure of de Truchis, Thomas and Vaudree (2026)
    does not constrain this sum to equal Y exactly, so the residual is a
    diagnostic to report, not an error.
    """
    y = df["detrended"].to_numpy()
    total = np.sum(X, axis=0)
    resid = y - total
    rows = [{
        "quantity": "residual (Y - sum_j X_j)",
        "mean": float(np.mean(resid)),
        "std": float(np.std(resid, ddof=1)),
        "min": float(np.min(resid)),
        "max": float(np.max(resid)),
        "share_of_Y_variation": float(np.std(resid, ddof=1) / np.std(y, ddof=1)),
    }]
    for j, xj in enumerate(X):
        rows.append({
            "quantity": f"component_{j+1}",
            "mean": float(np.mean(xj)),
            "std": float(np.std(xj, ddof=1)),
            "min": float(np.min(xj)),
            "max": float(np.max(xj)),
            "share_of_Y_variation": float(np.std(xj, ddof=1) / np.std(y, ddof=1)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=oc.DEFAULT_DATA)
    ap.add_argument("--n-particles", type=int,
                    default=oc.FILTER_SETTINGS["n_particles"])
    ap.add_argument("--window-size", type=int,
                    default=oc.FILTER_SETTINGS["window_size"])
    ap.add_argument("--seed", type=int, default=oc.FILTER_SETTINGS["seed"])
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    outdir = oc.ensure_outdir(OUTDIR_NAME)
    df = oc.load_ovx(args.data)
    y = df["detrended"].to_numpy()

    print("=" * 72)
    print("Figure S.6 -- OVX deconvolution")
    print("=" * 72)
    print(f"  sample: {oc.sample_note(df)}")

    params = oc.estimated_params(oc.MAIN_SPEC)

    print(f"  parameters: rho=({params['rho_1']:.4f}, {params['rho_2']:.4f}), "
          f"alpha={params['alpha']:.4f}, beta={params['beta']:.4f}")
    print(f"  filter:     {args.n_particles} particles, window "
          f"{args.window_size}, seed {args.seed}, n_jobs={args.n_jobs}")

    model, raw = build_model(params)
    t0 = time.time()
    X = model.mcmc_filter(y, raw,
                          n_particles=args.n_particles,
                          window_size=args.window_size,
                          n_jobs=args.n_jobs,
                          show_progress=True,
                          seed=args.seed)
    print(f"  filtering completed in {time.time() - t0:.0f}s")

    X = [np.asarray(xj, dtype=float) for xj in X]

    out = pd.DataFrame({"ovx": df["ovx"].to_numpy(),
                        "detrended": y,
                        "component_1": X[0],
                        "component_2": X[1]}, index=df.index)
    out["sum_components"] = out["component_1"] + out["component_2"]
    out["residual"] = out["detrended"] - out["sum_components"]
    out.to_csv(os.path.join(outdir, "filtered_components.csv"))

    diag = diagnostics(df, X)
    diag.to_csv(os.path.join(outdir, "filtering_diagnostics.csv"), index=False)
    print("\n" + diag.to_string(index=False))

    fig = make_figure(df, X, params)
    paths = oc.save_figure(fig, outdir, "figure_ovx_deconvolution")
    plt.close(fig)

    oc.write_run_config(outdir, {
        "target": "Online Supplement, Figure S.6",
        "data_file": os.path.relpath(args.data, oc.HERE),
        "T": int(len(df)),
        "sample_note": oc.sample_note(df),
        "params": {k: float(v) for k, v in params.items()},
        "spec": oc.MAIN_SPEC,
        "n_particles": args.n_particles,
        "window_size": args.window_size,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
    })

    print(f"\nWritten to {OUTDIR_NAME}/")
    for p in paths:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
