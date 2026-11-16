#!/usr/bin/env python3
r"""Table S.14 of the Online Supplement of de Truchis, Fries and Thomas (2026):
estimation of the OVX models (GaS, SaS, S1S).

Data: ../data/OVX_History.csv, window 2015-05-23 .. 2025-05-23, weekly mean, linear
detrend, centred, rescaled so that max|Y| = 10: T = 522.

Estimator: StableMixtureDeconvolution.estimate (stablemixture.py of this folder) with
step1=False: the minimum distance criterion, evaluated on a 15 x 15 grid, is minimised
by L-BFGS-B from an initial guess.  Hessian standard errors from .inference().  Initial guesses: INIT_GAS below for GaS; the sequential
(NLS) estimates of stable_aggregates.py for SaS and S1S.  Components are ordered
psi_1 < psi_2.

Usage:  cd ovx_estimation && python run_ovx_estimation.py     (~1 min)
Output: estimation.csv   (spec, parameter, estimate, std_error, t_stat), read by
                         ../ovx_common.py for Table S.14, Figure S.6, Table S.15 and
                         Figures 2, S.7-S.8
        run_config.json
"""
from __future__ import annotations

import json
import os
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import scipy

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "..", "data", "OVX_History.csv")

SPECS = {"gas": (False, False), "sas": (False, True), "s1s": (True, True)}  # (fix_alpha, fix_beta)
ROWS = ["rho_1", "rho_2", "alpha", "beta", "sigma", "pi_1", "pi_2"]
INIT_GAS = [0.8947193483793816, 0.99, 0.02967446828050219, 0.01,
                     1.5313965112981227, 0.3039407542242554]  # rho_1, rho_2, varsigma_1, varsigma_2 (scaled data), alpha, beta


def load_series():
    df = pd.read_csv(DATA)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    df = df.set_index("DATE").sort_index().loc["2015-05-23":"2025-05-23"].dropna()
    w = df.resample("W").mean()["OVX"].dropna().values
    t = np.arange(len(w))
    y = w - np.polyval(np.polyfit(t, w, 1), t)
    y -= y.mean()
    sf = 1.0
    if np.abs(y).max() > 10.0:
        sf = 10.0 / np.abs(y).max()
        y = y * sf
    return y, sf


def sequential_start(Y, spec):
    from stable_aggregates import StableAggregates
    fix_alpha, fix_beta = SPECS[spec]
    sa = StableAggregates(model_type="discrete")
    if fix_alpha:
        e = sa.estimate_discrete_mixture_params(Y, alpha=1.0, beta=0.0, J=2, method="nls")
    else:
        e = sa.sequential_estimation(Y, params_dict={"J": 2}, m=1.0, method="nls", verbose=False)
    init = [float(x) for x in e["phis"]] + [max(0.01, float(x)) for x in e["varsigmas"]]
    if not fix_alpha:
        init.append(float(e["alpha"]))
    if not fix_beta:
        init.append(float(sa.estimate_alpha_beta(Y)[1]))
    return init


def main() -> None:
    from stablemixture import StableMixtureDeconvolution
    Y, sf = load_series()
    print(f"T = {len(Y)}, scale factor {sf:.6f}")
    rows = []
    for spec, (fix_alpha, fix_beta) in SPECS.items():
        init = INIT_GAS if spec == "gas" else sequential_start(Y, spec)
        m = StableMixtureDeconvolution(num_components=2, alpha_value=1.0, beta_value=0.0,
                                       fix_alpha=fix_alpha, fix_beta=fix_beta, verbose=False)
        est = tuple(float(x) for x in m.estimate(Y, initial_guess=list(init),
                                                 use_optimal_weights=True, step1=False))
        d = dict(m._params_to_dict(est))
        if d["rho_1"] > d["rho_2"]:      # ordering psi_1 < psi_2 (components are exchangeable)
            est_sw = list(est)
            est_sw[0], est_sw[1], est_sw[2], est_sw[3] = est[1], est[0], est[3], est[2]
            est = tuple(est_sw)
            d = dict(m._params_to_dict(est))
        vs = [d["varsigma_1"] / sf, d["varsigma_2"] / sf]
        d["sigma"] = vs[0] + vs[1]
        d["pi_1"], d["pi_2"] = vs[0] / d["sigma"], vs[1] / d["sigma"]
        se = {}
        try:
            inf = m.inference(Y, est)
            for _, r in inf.iterrows():
                se[str(r["Parameter"]).split(" ")[0]] = (complex(r["Std. Error"]).real,
                                                          complex(r["t-statistic"]).real)
        except Exception as e:
            print(f"[{spec}] inference failed: {e}")
        for p in ROWS:
            est_p = d.get(p, np.nan)
            if (spec == "sas" and p == "beta") or (spec == "s1s" and p in ("alpha", "beta")):
                est_p = np.nan
            s_, t_ = se.get(p, (np.nan, np.nan))
            if p == "sigma" and not np.isnan(s_):
                s_ = s_ / sf
                t_ = est_p / s_
            if p == "pi_2" and "pi_1" in se:          # pi_2 = 1 - pi_1: same standard error
                s_ = se["pi_1"][0]
                t_ = est_p / s_ if s_ > 0 else np.nan
            rows.append(dict(spec=spec, parameter=p, estimate=est_p, std_error=s_, t_stat=t_))
        print(f"[{spec}] " + ", ".join(f"{p}={d[p]:.4f}" for p in ROWS if not np.isnan(d.get(p, np.nan))), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "estimation.csv"), index=False)
    with open(os.path.join(HERE, "run_config.json"), "w") as f:
        json.dump(dict(T=len(Y), scale_factor=sf, estimator="MDE computed from an initial guess (step1=False)",
                       sample="2015-05-23..2025-05-23, weekly mean, linear detrend",
                       python_version=platform.python_version(), numpy_version=np.__version__,
                       scipy_version=scipy.__version__, pandas_version=pd.__version__), f, indent=2)


if __name__ == "__main__":
    main()
