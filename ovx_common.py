r"""
Shared building blocks for the replication of Section "Application to the OVX
Index" (\label{app:ovx}) of online_supplement3.tex, and of Figure
\ref{fig:combined_forecasts} of main_v23.tex.

This module produces no output on its own. It collects what the five
`table_ovx_*.py` / `figure_ovx_*.py` scripts share:

  1. loading and pre-processing the OVX index (Section
     \ref{app:ovx_estimation});
  2. the published point estimates and filtering/forecasting settings, read
     off `Latex/figures/OVXResults.xlsx`;
  3. a port of the pattern-matching machinery from `stablemar.py`
     (`kfinder`, `foreprob`), restricted to the MAR(0,1) case used by
     Algorithm \ref{alg:pattern_matching}.

Notes on the data
--------------------------------------------------------------------------
`data/OVX_History.csv` covers weekly OVX levels through 2025-04-25. After
weekly resampling the usable sample is 2015-05-31 to 2025-04-27
(T = 518).

Identification of k0
--------------------------------------------------------------------------
`kfinder` compares the normalized observed path to the normalized
theoretical pattern psi^(k-i), i = 0..m-1. This pattern equals
psi^k * psi^(-i): the factor psi^k is constant in i and cancels under
normalization, so the matching distance is independent of k, and the argmin
is decided only by floating-point noise. k0 is therefore not identified by
this criterion; the published values (K0_PUBLISHED) are used throughout.
This has no effect on the forecast trajectory or the crash horizon h*, since
the historical pattern is renormalized on |X_{t0}| (which also cancels
psi^{k0}). Its only visible effect is which of the first k0 - 1 crash
probabilities are masked ("--") in Table \ref{tab:bubble_forecast}.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "OVX_History.csv")


# =============================================================================
# 1. Data: weekly detrended OVX (Section \ref{app:ovx_estimation})
# =============================================================================

SAMPLE_START = "2015-05-23"
SAMPLE_END = "2025-05-23"
DETREND_ORDER = 1          # linear detrending, Section \ref{app:ovx_estimation}
RESAMPLE_RULE = "W"        # weekly
RESAMPLE_HOW = "mean"      # within-week average (aggregate_data in stablemixture_app)


def load_ovx(path: str = DEFAULT_DATA,
             start: str = SAMPLE_START,
             end: str = SAMPLE_END,
             rule: str = RESAMPLE_RULE,
             how: str = RESAMPLE_HOW,
             detrend_order: int = DETREND_ORDER) -> pd.DataFrame:
    """Load, resample, and detrend the OVX series.

    Mirrors the `load_data -> prepare_data -> aggregate_data -> remove_trend`
    chain of `stablemixture_app.py`.

    Returns:
        DataFrame indexed by date, with columns `ovx` (aggregated level),
        `trend` (fitted polynomial trend), and `detrended` (series used
        throughout the analysis).
    """
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    df = df.set_index("DATE").sort_index()
    df = df.loc[start:end].dropna()

    resampler = df["OVX"].resample(rule)
    agg = resampler.mean() if how == "mean" else resampler.last()
    agg = agg.dropna()

    y = agg.to_numpy(dtype=float)
    t = np.arange(len(y))
    coeffs = np.polyfit(t, y, detrend_order)
    trend = np.polyval(coeffs, t)

    out = pd.DataFrame({"ovx": y, "trend": trend, "detrended": y - trend},
                       index=agg.index)
    out.index.name = "date"
    out.attrs["detrend_coeffs"] = coeffs.tolist()
    return out


def sample_note(df: pd.DataFrame) -> str:
    """One-line summary of the sample actually used."""
    return (f"T = {len(df)}, {df.index[0]:%Y-%m-%d} to "
            f"{df.index[-1]:%Y-%m-%d}.")


# =============================================================================
# 2. Published point estimates and run settings
# =============================================================================
#
# Source: Latex/figures/OVXResults.xlsx, sheet "Feuil1", which archives the
# output of the `stablemixture_app.py` session that produced Table
# \ref{tab:ovx_estimation}, Table \ref{tab:bubble_forecast}, and the
# accompanying figures. Columns used here are Estimate / Std. / t-stat.

PUBLISHED_ESTIMATION: Dict[str, Dict[str, Optional[Tuple[float, float, float]]]] = {
    "gas": {
        "rho_1": (0.7989, 0.0673, 11.862),
        "rho_2": (0.8470, 0.0668, 12.678),
        "alpha": (1.4686, 0.0995, 14.764),
        "beta": (-0.1275, 0.0684, -1.863),
        "sigma": (2.0932, 0.2400, 8.723),
        "pi_1": (0.2790, 0.0403, 6.930),
        "pi_2": (0.7210, 0.0409, 17.622),
    },
    "sas": {
        "rho_1": (0.2507, 0.0077, 32.477),
        "rho_2": (0.9865, 0.0040, 244.560),
        "alpha": (1.2405, 0.0084, 147.613),
        "beta": None,
        "sigma": (0.8964, 0.0212, 42.226),
        "pi_1": (0.8915, 0.0052, 171.084),
        "pi_2": (0.1085, 0.0182, 5.957),
    },
    "s1s": {
        "rho_1": (0.9226, 0.0082, 112.824),
        "rho_2": (0.9346, 0.0074, 126.404),
        "alpha": None,
        "beta": None,
        "sigma": (0.1966, 0.0294, 6.692),
        "pi_1": (0.5029, 0.0511, 9.833),
        "pi_2": (0.4971, 0.0545, 9.121),
    },
}

SPEC_LABELS = {
    "gas": r"$\mathcal{G}\alpha\mathcal{S}$",
    "sas": r"$\mathcal{S}\alpha\mathcal{S}$",
    "s1s": r"$\mathcal{S}1\mathcal{S}$",
}

# Specification used for filtering and forecasting throughout the supplement.
MAIN_SPEC = "gas"


def published_params(spec: str = MAIN_SPEC) -> Dict[str, float]:
    """Point estimates for `spec`, augmented with the varsigma_j scales."""
    tbl = PUBLISHED_ESTIMATION[spec]
    out = {k: v[0] for k, v in tbl.items() if v is not None}
    out.setdefault("alpha", 1.0)      # S1S: alpha = 1 imposed
    out.setdefault("beta", 0.0)       # SaS / S1S: beta = 0 imposed
    for j in (1, 2):
        out[f"varsigma_{j}"] = out["sigma"] * out[f"pi_{j}"]
    return out


# MCMC filter settings of \cite{dtv25} ("Filtration tuning parameters"
# column of OVXResults.xlsx).
FILTER_SETTINGS = {
    "n_particles": 1000,
    "window_size": 5,
    "seed": 11,
}

# Forecasting exercise settings (Section \ref{app:ovx_forecast}).
FORECAST_CUTOFF = "2020-01-05"
FORECAST_M = 20            # pattern-matching depth, "m = 20 throughout"
FORECAST_H = 20            # forecast horizon
FORECAST_VARTHETA = 1      # positive bubble
FORECAST_KMAX = 20
THRESHOLDS = (0.90, 0.95, 0.99)

# Component-wise k0 used throughout (see the module docstring on
# identification).
K0_PUBLISHED = {1: 3, 2: 1}

# Level of the detrended series at the forecast cut-off (2020-01-05),
# consistent with the full announced sample; used to anchor the
# pattern-matching trajectory.
ANCHOR_PUBLISHED = 11.271398


# =============================================================================
# 3. Pattern matching and crash probabilities (port of stablemar.py)
# =============================================================================
#
# Restricted to the purely anticipative MAR(0,1) case, the only case covered
# by Algorithm \ref{alg:pattern_matching}: order (r, s) = (0, 1), so the
# causal polynomial is empty and a crash sends the trajectory to zero.

def kfinder(x: Sequence[float], tau: int, psi: float, m: int = FORECAST_M,
            kmax: int = FORECAST_KMAX, vartheta: int = 1,
            kmin: int = 1) -> Tuple[int, float, np.ndarray]:
    """Step 2 of Algorithm \\ref{alg:pattern_matching} (pattern matching).

    Port of `StableMAR.kfinder` for an anticipative AR(1).

    Returns:
        (k0, quality, distances), where `distances[k]` is the L1 distance
        between the normalized observed path and the normalized pattern of
        index k. `distances` is constant in k (see the module docstring);
        `k0` is returned only for consistency with the original code.
    """
    x = np.asarray(x, dtype=float)
    xm = np.flip(np.array([x[tau - i] for i in range(m)]))
    xnorm = xm / np.linalg.norm(xm)

    distances = np.full(kmax + 1, np.nan)
    for k in range(kmin, kmax + 1):
        pattern = np.array([psi ** (k - i) for i in range(m)])
        pattern = pattern / np.linalg.norm(pattern)
        distances[k] = np.sum(np.abs(xnorm - vartheta * pattern))

    k0 = int(np.nanargmin(distances[kmin:]) + kmin)
    quality = float(1.0 - distances[k0] / (2 * m))
    return k0, quality, distances


def k0_is_degenerate(distances: np.ndarray, tol: float = 1e-9) -> bool:
    """True if the matching distance does not separate k (expected here)."""
    d = distances[np.isfinite(distances)]
    return d.size > 1 and float(np.nanmax(d) - np.nanmin(d)) < tol


def foreprob(x: Sequence[float], k0: int, h: int, maxp: float, t: int,
             psi: float, alpha: float, vartheta: int = 1,
             m: int = FORECAST_M,
             anchor: Optional[float] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """Steps 3-8 of Algorithm \\ref{alg:pattern_matching}.

    Port of `StableMAR.foreprob` for MAR(0,1). Probabilities follow
    Proposition 3.3(a):
        Crash Before h = |psi|^(alpha (h-1)) (1 - |psi|^alpha)
        Survive at h   = |psi|^(alpha h)
        Crash at h     = 1 - |psi|^(alpha h)
    and the trajectory is X_hat_{t+h} = X_hat_{t+h-1} / |psi| as long as
    Crash at h <= maxp, and zero afterwards (the causal polynomial being
    empty).

    Args:
        x: series under analysis (detrended series).
        k0: pattern shift index; only affects which of the first k0 - 1
            probabilities are masked (see the module docstring).
        h: maximum horizon.
        maxp: risk threshold p.
        t: index of the cut-off t_0.
        psi, alpha: parameters of the selected component.
        vartheta: sign of the extreme event.
        m: depth of the past pattern.
        anchor: if given, replaces |x[t]| as the anchoring level.

    Returns:
        (df_prob, full_traj), where `full_traj` has length m + 1 + h and
        covers dates t-m .. t+h, and `df_prob` is indexed by h = 1..h.
    """
    x = np.asarray(x, dtype=float)
    psi_abs = abs(psi)

    proba = np.empty((h, 4))
    for i in range(1, h + 1):
        proba[i - 1, 0] = i
        proba[i - 1, 1] = psi_abs ** (alpha * (i - 1)) * (1 - psi_abs ** alpha)
        proba[i - 1, 2] = psi_abs ** (alpha * i)
        proba[i - 1, 3] = 1 - psi_abs ** (alpha * i)

    level = abs(float(x[t])) if anchor is None else abs(float(anchor))

    full_traj = np.zeros(m + 1 + h)
    # Past pattern: psi^(k0+m-i), renormalized to equal |X_{t0}| at i = m.
    # The psi^k0 factor cancels under this renormalization, which is why
    # the trajectory does not depend on k0.
    past_pattern = np.array([psi ** (k0 + m - i) for i in range(m + 1)])
    scale = level / abs(past_pattern[-1]) if past_pattern[-1] != 0 else 1.0
    full_traj[:m + 1] = vartheta * past_pattern * scale

    for i in range(1, h + 1):
        idx = m + i
        if proba[i - 1, 3] < maxp:
            full_traj[idx] = (full_traj[m] if i == 1 else full_traj[idx - 1]) / psi_abs
        else:
            full_traj[idx] = 0.0   # r = 0: no causal part, crash to zero

    proba[0, 1] = np.nan
    if k0 > 1:
        proba[0:k0 - 1, 2:4] = np.nan

    df_prob = pd.DataFrame(proba,
                           columns=["h", "Crash Before h", "Survive at h",
                                    "Crash at h"]).set_index("h")
    return df_prob, full_traj


def crash_horizon(psi: float, alpha: float, threshold: float,
                  hmax: int = 200) -> Optional[int]:
    """First h such that 1 - |psi|^(alpha h) >= threshold (h* of the algorithm)."""
    for h in range(1, hmax + 1):
        if 1 - abs(psi) ** (alpha * h) >= threshold:
            return h
    return None


def forecast_dates(index: pd.DatetimeIndex, t: int, m: int, h: int) -> pd.DatetimeIndex:
    """Dates covering t-m .. t+h, extended weekly beyond the sample if needed."""
    last_in = min(t + h, len(index) - 1)
    dates = list(index[t - m:last_in + 1])
    n_extra = (t + h) - last_in
    if n_extra > 0:
        step = index[-1] - index[-2]
        dates += [index[last_in] + (i + 1) * step for i in range(n_extra)]
    return pd.DatetimeIndex(dates)


# =============================================================================
# 4. Output helpers (same conventions as results_table_s1/)
# =============================================================================

def ensure_outdir(name: str) -> str:
    """Create (if needed) and return the `name` output folder."""
    path = os.path.join(HERE, name)
    os.makedirs(path, exist_ok=True)
    return path


def write_run_config(outdir: str, config: Dict, name: str = "run_config.json") -> str:
    """Write run_config.json with the environment, as in results_table_s1/."""
    import scipy
    payload = dict(config)
    payload.update({
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    path = os.path.join(outdir, name)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def tex_num(value: Optional[float], nd: int = 4, dash: str = "--") -> str:
    """LaTeX number formatting with a proper math minus sign."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return dash
    s = f"{value:.{nd}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def save_figure(fig, outdir: str, stem: str, dpi: int = 200) -> List[str]:
    """Save a figure as PDF (for the .tex build) and PNG (for quick viewing)."""
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    return paths
