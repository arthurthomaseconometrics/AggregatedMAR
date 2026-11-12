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
    """Options L-BFGS-B communes a tous les modules MDE (voir MDE_GTOL)."""
    opts = {"maxiter": int(maxiter), "ftol": MDE_FTOL, "gtol": MDE_GTOL}
    opts.update(overrides)
    return opts


def stable_seed(*parts, base=0):
    """
    Graine deterministe et reproductible entre processus.

    `hash()` de Python est randomise par processus (PYTHONHASHSEED), donc
    `seed_base + hash((key, T, r))` ne redonne PAS les memes tirages d'une
    execution a l'autre.  zlib.crc32 sur la representation UTF-8 est stable.
    """
    import zlib
    key = "|".join(repr(p) for p in parts).encode("utf-8")
    return int((base + zlib.crc32(key)) % (2 ** 32 - 1))

_CACHE = {}


def alpha_admissible(alpha, estimated=True):
    """
    Le parametre de queue est-il dans le domaine ou le critere est evalue ?

    Parameters
    ----------
    alpha : float
    estimated : bool
        True  : alpha est un parametre LIBRE, le domaine est celui de
                l'Assumption C2, soit [ALPHA_MIN, ALPHA_MAX].
        False : alpha est IMPOSE par l'utilisateur ; on n'exige plus que la
                validite de la loi alpha-stable elle-meme, alpha dans (0, 2].
                C'est ce qui laisse tourner le cas Cauchy (alpha = 1) du
                Panel A de la Table S.1.

    Returns
    -------
    bool
    """
    if not np.isfinite(alpha):
        return False
    if estimated:
        # slack : L-BFGS-B respecte les bornes, mais pas toujours au bit pres
        return ALPHA_MIN - 1e-9 <= alpha <= ALPHA_MAX + 1e-9
    return 0.0 < alpha <= 2.0


def is_alpha_one(alpha):
    """
    Faut-il utiliser la branche alpha = 1 de la representation de
    Samorodnitsky-Taqqu ?  Test par tolerance, jamais par egalite flottante.
    """
    return abs(alpha - 1.0) <= ALPHA1_TOL


def logcf_alpha1_component(a_geo, a_iid, varsigma, rho, beta):
    r"""
    Log-CF d'une composante MAR(0,1) agregee dans le cas alpha = 1.

    HORS DU DOMAINE DE L'ASSUMPTION C2 / DE LA PROPOSITION 2.1.  Cette
    branche n'est atteignable qu'avec alpha IMPOSE (cf. `alpha_admissible`).
    Elle est conservee pour le Panel A de la Table S.1 du supplement
    (agregat de Cauchy) et pour tout diagnostic a alpha fixe.

    Composante representee
    ----------------------
        Z = varsigma * [ a_geo * sum_{k>=0} rho^k eps_{k} + a_iid * eps_* ]

    avec eps iid de loi S_1(1, beta, 0).  Dans les modules de subsampling
    a_geo = rho*u + v et a_iid = u ; dans stablemixture.char_function la
    convention (u, v) est transposee, a_geo = u + rho*v et a_iid = v.  La
    fonction renvoie log E[exp(i Z)].

    Formule
    -------
    Pour Z = sum_k c_k eps_k avec eps ~ S_1(1, beta, 0),

        log phi_Z(1) = - sum_k |c_k|  -  i beta (2/pi) sum_k c_k ln|c_k| .

    Le ln s'applique donc COEFFICIENT PAR COEFFICIENT.  C'est le point que
    l'implementation precedente manquait : elle ecrivait

        -varsigma [ |a_geo|(1 + i beta (2/pi) sign(a_geo) ln|a_geo|)/(1-|rho|)
                    + |a_iid|(1 + i beta (2/pi) sign(a_iid) ln|a_iid|) ]

    c'est-a-dire qu'elle appliquait ln a l'argument AGREGE.  Il manquait
    (i) le facteur varsigma a l'interieur des logarithmes et (ii) le terme
    de derive  -i beta (2/pi) a_geo * rho ln|rho| / (1-rho)^2  issu de
    sum_k c_k ln|c_k| pour c_k geometrique.  C'est exactement la quantite
    sum_k d_k ln|d_k| dont la condition (assump:coefa1) du papier,
    sum_k |d_k| |ln|d_k|| < infini, garantit la finitude.

    L'erreur etait sans effet sur les resultats publies (beta = 0 dans
    toutes les simulations : la partie imaginaire est alors identiquement
    nulle et les deux formules coincident), mais elle piegeait toute
    specification S1S ASYMETRIQUE.

    Notes
    -----
    Somme geometriques utilisees (|rho| < 1) :
        sum_k |rho|^k = 1/(1-|rho|),  sum_k rho^k = 1/(1-rho),
        sum_k k rho^k = rho/(1-rho)^2.
    Les termes a*ln(varsigma|a|) sont prolonges par 0 en a = 0, qui est
    leur limite.
    """
    a_geo = np.asarray(a_geo, dtype=float)
    a_iid = np.asarray(a_iid, dtype=float)

    # partie reelle : - sum_k |c_k|
    re = -varsigma * (np.abs(a_geo) / (1.0 - abs(rho)) + np.abs(a_iid))

    if beta == 0.0:
        return re.astype(complex)

    def _x_log_x(a):
        """a * ln(varsigma*|a|), prolonge par 0 en a = 0."""
        m = np.abs(a) > 0.0
        return np.where(m, a * np.log(np.where(m, varsigma * np.abs(a), 1.0)), 0.0)

    # sum_k c_k ln|c_k|, ou c_k = varsigma * a_geo * rho^k puis varsigma * a_iid
    drift_geo = rho * np.log(abs(rho)) / (1.0 - rho) ** 2 if rho != 0.0 else 0.0
    s = varsigma * (_x_log_x(a_geo) / (1.0 - rho) + a_geo * drift_geo
                    + _x_log_x(a_iid))

    return re - 1j * beta * (2.0 / np.pi) * s


def default_grid(n_grid=N_GRID_DEFAULT, u_max=U_MAX_DEFAULT):
    """Grille (u, v) commune a tous les modules."""
    g = np.linspace(-u_max, u_max, n_grid)
    return g, g.copy()


def _grid_step(g):
    """Pas d'une grille reguliere ; 1.0 si la grille est degeneree."""
    g = np.asarray(g, dtype=float)
    if g.size < 2:
        return 1.0
    return float(g[1] - g[0])


def quadrature_weights(u_grid, v_grid, kappa=KAPPA_DEFAULT):
    """
    Matrice de quadrature W telle que sum(W * |ecf - tcf|^2) approxime (2.14).

    Parameters
    ----------
    u_grid, v_grid : np.ndarray
        Grilles regulieres d'evaluation.
    kappa : float or None
        Parametre du poids gaussien de l'Assumption C2.  Doit etre > 0.
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
            raise ValueError("kappa doit etre strictement positif (Assumption C2).")
        U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
        du, dv = _grid_step(u_grid), _grid_step(v_grid)
        W = (kappa / np.pi) * np.exp(-kappa * (U ** 2 + V ** 2)) * du * dv

    W.flags.writeable = False
    _CACHE[key] = W
    return W


def weighted_sse(ecf, tcf, W):
    """Critere MDE discretise : approximation de (2.14)."""
    return float(np.sum(W * np.abs(ecf - tcf) ** 2))


def captured_mass(u_grid, v_grid, kappa=KAPPA_DEFAULT):
    """
    Fraction de la masse de w capturee par la grille bornee.

    Vaut exactement erf(sqrt(kappa) * u_max)^2 pour une grille symetrique,
    a l'erreur de discretisation pres ; on renvoie ici la somme effective
    des poids de quadrature, qui est la quantite pertinente en pratique.
    """
    if kappa is None:
        return 1.0
    return float(np.sum(quadrature_weights(u_grid, v_grid, kappa)))


def check_grid(u_grid, v_grid, kappa=KAPPA_DEFAULT, min_mass=0.90, verbose=True):
    """
    Diagnostic de grille.  Renvoie la masse capturee et avertit si elle est
    trop faible pour que la somme de Riemann represente honnetement (2.14).
    """
    m = captured_mass(u_grid, v_grid, kappa)
    if verbose:
        lbl = "uniforme" if kappa is None else f"gaussien kappa={kappa}"
        print(f"  Poids: {lbl} | grille {len(u_grid)}x{len(v_grid)} "
              f"sur [{u_grid[0]:.2f}, {u_grid[-1]:.2f}]^2 | "
              f"masse capturee = {m:.4f}")
    if kappa is not None and m < min_mass:
        import warnings
        warnings.warn(
            f"La grille ne capture que {m:.3f} de la masse du poids gaussien "
            f"(kappa={kappa}).  Elargir u_max ou reduire kappa pour que la "
            f"somme de Riemann approxime correctement l'integrale (2.14).",
            RuntimeWarning, stacklevel=2)
    return m
