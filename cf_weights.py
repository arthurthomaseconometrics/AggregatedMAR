r"""Quadrature of the minimum distance criterion on the characteristic function.

The criterion of de Truchis, Fries and Thomas (2026), Assumption C2, is

    D(theta) = \iint |phi_n(u,v) - phi(u,v;theta)|^2 w(u,v) du dv,
    w(u, v)  = (kappa / pi) * exp(-kappa * (u^2 + v^2)),   kappa > 0.

`quadrature_weights` returns W[i, j] = w(u_i, v_j) du dv on a regular grid, so that
sum(W * |ecf - tcf|^2) is a Riemann sum of D(theta).  With the 15 x 15 grid on
[-1.5, 1.5]^2 and kappa = 1, the weights sum to 0.956.  `check_grid` warns when the
grid captures too little of the mass of w.  `kappa=None` gives uniform weights (the
mean of |ecf - tcf|^2), used by the `--uniform-weights` option of the Monte Carlo
scripts.

The module also holds the estimation bounds of alpha (ALPHA_BOUNDS), the L-BFGS-B
tolerances shared by all estimators (MDE_GTOL, MDE_FTOL, MDE_MAXITER) and the
seeding helper `stable_seed`.
"""

import numpy as np

# Value of kappa used throughout.
KAPPA_DEFAULT = 1.0

# Grid: 15 x 15 points on [-1.5, 1.5]^2.
U_MAX_DEFAULT = 1.5
N_GRID_DEFAULT = 15

# =============================================================================
# Estimation space of alpha
# =============================================================================
#
# Assumption C2 restricts alpha to the open interval (1, 2): the characteristic
# function changes form at alpha = 1 when beta != 0, and the C^2 regularity of the
# criterion requires alpha > 1.  ALPHA_MIN > 1 and ALPHA_MAX < 2 make the estimation
# space compact and included in (1, 2).  These bounds apply only when alpha is
# estimated; a specification with alpha imposed (e.g. alpha = 1, Cauchy) is not
# restricted: use `alpha_admissible(alpha, estimated=False)`.
ALPHA_MIN = 1.01
ALPHA_MAX = 1.99
ALPHA_BOUNDS = (ALPHA_MIN, ALPHA_MAX)

# Tolerance used to select the alpha = 1 branch of the characteristic functions.
ALPHA1_TOL = 1e-8

# =============================================================================
# L-BFGS-B tolerances
# =============================================================================
#
# scipy's L-BFGS-B stops on absolute criteria (gtol on the projected gradient, ftol on
# the decrease of f).  The weighted criterion is of order 1e-6 at the optimum, so the
# default values stop the optimizer before it has moved along weakly curved
# directions (pi_1).  These tighter values are used by every estimator of the package.
MDE_GTOL = 1e-12
MDE_FTOL = 1e-16
MDE_MAXITER = 2000


def lbfgsb_options(maxiter=MDE_MAXITER, **overrides):
    """L-BFGS-B options shared by all MDE modules (see MDE_GTOL)."""
    opts = {"maxiter": int(maxiter), "ftol": MDE_FTOL, "gtol": MDE_GTOL}
    opts.update(overrides)
    return opts


def stable_seed(*parts, base=0):
    """
    Deterministic seed, reproducible across processes.

    Python's `hash()` is randomized per process (PYTHONHASHSEED), so
    `seed_base + hash((key, T, r))` does NOT give the same draws from one run
    to the next.  zlib.crc32 on the UTF-8 representation is stable.
    """
    import zlib
    key = "|".join(repr(p) for p in parts).encode("utf-8")
    return int((base + zlib.crc32(key)) % (2 ** 32 - 1))

_CACHE = {}


def alpha_admissible(alpha, estimated=True):
    """
    Is the tail index in the domain where the criterion is evaluated?

    Parameters
    ----------
    alpha : float
    estimated : bool
        True  : alpha is a FREE parameter and the domain is that of
                Assumption C2, i.e. [ALPHA_MIN, ALPHA_MAX].
        False : alpha is IMPOSED by the user; only the validity of the
                alpha-stable law itself is required, alpha in (0, 2].  This
                allows the Cauchy case (alpha = 1) of Panel A of Table S.1.

    Returns
    -------
    bool
    """
    if not np.isfinite(alpha):
        return False
    if estimated:
        # slack: L-BFGS-B respects the bounds, but not always to the last bit
        return ALPHA_MIN - 1e-9 <= alpha <= ALPHA_MAX + 1e-9
    return 0.0 < alpha <= 2.0


def is_alpha_one(alpha):
    """
    Should the alpha = 1 branch of the Samorodnitsky-Taqqu representation be
    used?  Tested with a tolerance, never by floating-point equality.
    """
    return abs(alpha - 1.0) <= ALPHA1_TOL


def logcf_alpha1_component(a_geo, a_iid, varsigma, rho, beta):
    r"""
    Log-CF of an aggregated MAR(0,1) component when alpha = 1.

    OUTSIDE THE DOMAIN OF ASSUMPTION C2 / PROPOSITION 2.1.  This branch is
    reached only when alpha is IMPOSED (see `alpha_admissible`).  It is used
    for Panel A of Table S.1 of the supplement (Cauchy aggregate) and for any
    diagnostic with alpha fixed.

    Component
    ---------
        Z = varsigma * [ a_geo * sum_{k>=0} rho^k eps_{k} + a_iid * eps_* ]

    with eps iid S_1(1, beta, 0).  In the subsampling modules
    a_geo = rho*u + v and a_iid = u; in stablemixture.char_function the
    (u, v) convention is transposed, a_geo = u + rho*v and a_iid = v.  The
    function returns log E[exp(i Z)].

    Formula
    -------
    For Z = sum_k c_k eps_k with eps ~ S_1(1, beta, 0),

        log phi_Z(1) = - sum_k |c_k|  -  i beta (2/pi) sum_k c_k ln|c_k| .

    The logarithm thus applies COEFFICIENT BY COEFFICIENT: the factor varsigma
    enters the logarithms and, for geometric c_k, sum_k c_k ln|c_k| produces
    the drift term -i beta (2/pi) a_geo * rho ln|rho| / (1-rho)^2.  This is
    the quantity sum_k d_k ln|d_k| whose finiteness is guaranteed by
    condition (assump:coefa1) of the paper, sum_k |d_k| |ln|d_k|| < infinity.
    When beta = 0, the imaginary part is identically zero.

    Notes
    -----
    Geometric sums used (|rho| < 1):
        sum_k |rho|^k = 1/(1-|rho|),  sum_k rho^k = 1/(1-rho),
        sum_k k rho^k = rho/(1-rho)^2.
    The terms a*ln(varsigma|a|) are extended by 0 at a = 0, their limit.
    """
    a_geo = np.asarray(a_geo, dtype=float)
    a_iid = np.asarray(a_iid, dtype=float)

    # real part: - sum_k |c_k|
    re = -varsigma * (np.abs(a_geo) / (1.0 - abs(rho)) + np.abs(a_iid))

    if beta == 0.0:
        return re.astype(complex)

    def _x_log_x(a):
        """a * ln(varsigma*|a|), extended by 0 at a = 0."""
        m = np.abs(a) > 0.0
        return np.where(m, a * np.log(np.where(m, varsigma * np.abs(a), 1.0)), 0.0)

    # sum_k c_k ln|c_k|, where c_k = varsigma * a_geo * rho^k, then varsigma * a_iid
    drift_geo = rho * np.log(abs(rho)) / (1.0 - rho) ** 2 if rho != 0.0 else 0.0
    s = varsigma * (_x_log_x(a_geo) / (1.0 - rho) + a_geo * drift_geo
                    + _x_log_x(a_iid))

    return re - 1j * beta * (2.0 / np.pi) * s


def default_grid(n_grid=N_GRID_DEFAULT, u_max=U_MAX_DEFAULT):
    """(u, v) grid shared by all modules."""
    g = np.linspace(-u_max, u_max, n_grid)
    return g, g.copy()


def _grid_step(g):
    """Step of a regular grid; 1.0 if the grid is degenerate."""
    g = np.asarray(g, dtype=float)
    if g.size < 2:
        return 1.0
    return float(g[1] - g[0])


def quadrature_weights(u_grid, v_grid, kappa=KAPPA_DEFAULT):
    """
    Quadrature matrix W such that sum(W * |ecf - tcf|^2) approximates (2.14).

    Parameters
    ----------
    u_grid, v_grid : np.ndarray
        Regular evaluation grids.
    kappa : float or None
        Parameter of the Gaussian weight of Assumption C2.  Must be > 0.
        `None` gives uniform weights, i.e. the mean of |ecf - tcf|^2.

    Returns
    -------
    np.ndarray, shape (len(u_grid), len(v_grid))
    """
    u_grid = np.asarray(u_grid, dtype=float)
    v_grid = np.asarray(v_grid, dtype=float)

    key = (u_grid.tobytes(), v_grid.tobytes(), kappa)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    if kappa is None:
        W = np.full((u_grid.size, v_grid.size), 1.0 / (u_grid.size * v_grid.size))
    else:
        if kappa <= 0:
            raise ValueError("kappa must be strictly positive (Assumption C2).")
        U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
        du, dv = _grid_step(u_grid), _grid_step(v_grid)
        W = (kappa / np.pi) * np.exp(-kappa * (U ** 2 + V ** 2)) * du * dv

    W.flags.writeable = False
    _CACHE[key] = W
    return W


def weighted_sse(ecf, tcf, W):
    """Discretized MDE criterion: approximation of (2.14)."""
    return float(np.sum(W * np.abs(ecf - tcf) ** 2))


def captured_mass(u_grid, v_grid, kappa=KAPPA_DEFAULT):
    """
    Share of the mass of w captured by the bounded grid.

    Equal to erf(sqrt(kappa) * u_max)^2 for a symmetric grid, up to the
    discretization error; the function returns the actual sum of the
    quadrature weights, which is the relevant quantity in practice.
    """
    if kappa is None:
        return 1.0
    return float(np.sum(quadrature_weights(u_grid, v_grid, kappa)))


def check_grid(u_grid, v_grid, kappa=KAPPA_DEFAULT, min_mass=0.90, verbose=True):
    """
    Grid diagnostic.  Returns the captured mass and warns when it is too
    small for the Riemann sum to represent (2.14) faithfully.
    """
    m = captured_mass(u_grid, v_grid, kappa)
    if verbose:
        lbl = "uniform" if kappa is None else f"Gaussian kappa={kappa}"
        print(f"  Weights: {lbl} | grid {len(u_grid)}x{len(v_grid)} "
              f"on [{u_grid[0]:.2f}, {u_grid[-1]:.2f}]^2 | "
              f"captured mass = {m:.4f}")
    if kappa is not None and m < min_mass:
        import warnings
        warnings.warn(
            f"The grid captures only {m:.3f} of the mass of the Gaussian weight "
            f"(kappa={kappa}).  Increase u_max or decrease kappa so that the "
            f"Riemann sum approximates the integral (2.14) correctly.",
            RuntimeWarning, stacklevel=2)
    return m
