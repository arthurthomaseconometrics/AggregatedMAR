r"""
Poids de quadrature pour le critere MDE sur fonction caracteristique.

Motivation
----------
L'Assumption C2 de main_v23.tex impose la forme EXACTE du poids

    w(u, v) = (kappa / pi) * exp(-kappa * (u^2 + v^2)),   kappa > 0,

et cette forme est utilisee dans les preuves elles-memes (passages en
coordonnees polaires, constantes C_{w,alpha} et I_r(alpha,kappa) des
Lemmes 2.1 et 4.1).  Le critere de l'equation (2.14) est l'integrale

    D(theta) = \iint_{R^2} |phi_n(u,v) - phi(u,v;theta)|^2 w(u,v) du dv.

Plusieurs modules de ce depot minimisaient a la place une MOYENNE non
ponderee sur la grille, `np.mean(|ecf - tcf|^2)`, qui n'est pas le critere
couvert par la Proposition 2.1.  Ce module fournit la quadrature commune
permettant d'aligner tous les modules sur (2.14).

Discretisation
--------------
`quadrature_weights` renvoie la matrice

    W[i, j] = w(u_i, v_j) * du * dv

de sorte que `np.sum(W * |ecf - tcf|^2)` soit une somme de Riemann de
l'integrale (2.14), et non une quantite d'echelle arbitraire.  La
normalisation kappa/pi fait de w une densite de probabilite
N(0, (2 kappa)^{-1} I_2) : c'est sans effet sur l'argmin (le footnote de
l'Assumption C2 le note), mais c'est indispensable des qu'on compare une
DIFFERENCE de criteres a un quantile fixe, comme dans le test de la
Section S.2.4 du supplement.

La grille etant bornee, une fraction de la masse de w est perdue.
`captured_mass` la quantifie ; `check_grid` emet un avertissement si elle
tombe sous un seuil.  Avec la grille historique 15 x 15 sur [-1.5, 1.5]^2
et kappa = 1.0, la somme des poids de quadrature vaut 0.956
(la valeur exacte de l'integrale tronquee etant erf(1.5)^2 = 0.933 ; l'ecart
est l'erreur de discretisation de la somme de Riemann).

Mode legacy
-----------
`quadrature_weights(..., kappa=None)` renvoie des poids uniformes
normalises reproduisant exactement `np.mean(...)`.  Cela permet de
regenerer les resultats archives (Latex/resultats_*) avec le code patche.
"""

import numpy as np

# Valeur de kappa retenue pour l'ensemble des exercices du supplement.
# Elle etait deja celle de subsampling_reparam.py / subsampling_interactive.py
# (seuls modules qui ponderaient), on la fixe ici pour tous les modules.
KAPPA_DEFAULT = 1.0

# Grille historique, conservee pour que les resultats restent comparables
# aux tables archivees ; seul le poids change.
U_MAX_DEFAULT = 1.5
N_GRID_DEFAULT = 15

_CACHE = {}


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
        `None` active le mode legacy (poids uniformes normalises, soit
        exactement l'ancien `np.mean(|ecf - tcf|^2)`).

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
        lbl = "uniforme (legacy)" if kappa is None else f"gaussien kappa={kappa}"
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
