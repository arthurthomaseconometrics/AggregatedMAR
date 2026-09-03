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

Domaine du parametre de queue
-----------------------------
Ce module porte aussi `ALPHA_BOUNDS`, les bornes de l'espace d'ESTIMATION
de alpha.  Voir le commentaire de `ALPHA_BOUNDS` ci-dessous : l'Assumption
C2 impose alpha dans (1, 2) OUVERT, alors que le code utilisait partout
(0.1, 1.99), un intervalle qui traverse alpha = 1.
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

# =============================================================================
# Domaine du parametre de queue (Assumption C2 de main_v23.tex)
# =============================================================================
#
# L'Assumption C2 impose alpha dans (1, 2) STRICTEMENT, pour deux raisons
# donnees explicitement dans le papier :
#
#   (a) la representation de Samorodnitsky-Taqqu change de forme
#       fonctionnelle en alpha = 1 des que beta != 0 : le facteur
#       beta*tan(pi*alpha/2)*sign(u) est remplace par -beta*(2/pi)*ln|u|.
#       La famille parametrique n'est donc meme pas CONTINUE en alpha = 1,
#       et tan(pi*alpha/2) diverge quand alpha -> 1.
#   (b) l'analyse C^2 du Lemme 2.1 exige alpha - 2 > -1, soit alpha > 1.
#
# Le code utilisait (0.1, 1.99) partout, un intervalle qui CONTIENT alpha = 1
# et descend sous 1 : l'espace de parametres du code ne coincidait pas avec
# le Theta compact de l'Assumption C1-C2.  La borne basse est desormais
# ALPHA_MIN > 1, ce qui rend Theta compact ET inclus dans l'ouvert (1, 2).
#
# ATTENTION.  Ces bornes contraignent l'espace d'ESTIMATION uniquement.  Une
# specification ou alpha est IMPOSE (fix_alpha=True / estimate_alpha=False)
# reste libre de prendre alpha = 1 : c'est le cas du Panel A de la Table S.1
# du supplement (aggregat de Cauchy), qui n'estime pas alpha et sort donc
# simplement du domaine de validite de la Prop. 2.1 sans etre interdit.
# Utiliser `alpha_admissible(alpha, estimated=False)` dans ce cas.
ALPHA_MIN = 1.01
ALPHA_MAX = 1.99
ALPHA_BOUNDS = (ALPHA_MIN, ALPHA_MAX)

# Tolerance de selection de la branche alpha = 1 des fonctions
# caracteristiques.  Les modules testaient `if alpha != 1`, une egalite
# FLOTTANTE : a alpha = 1 + 1e-9 on retombait dans la branche generale ou
# tan(pi*alpha/2) ~ -6.4e8, ce qui detruit le critere sans rien signaler.
# Avec ALPHA_MIN = 1.01 le cas est hors d'atteinte en estimation, mais le
# test par tolerance protege les appels a alpha impose.
ALPHA1_TOL = 1e-8

# =============================================================================
# Tolerances de l'optimiseur (audit 2026-09-03)
# =============================================================================
#
# scipy.optimize.minimize(method='L-BFGS-B') arrete sur des criteres ABSOLUS :
# `gtol` (defaut 1e-5) porte sur max|gradient projete|, et `ftol` (defaut
# 2.22e-9) sur la decroissance de f rapportee a max(|f|, 1) -- donc absolue
# des que |f| < 1.  Or le critere (2.14) normalise par l'Assumption C2 vaut
# ~1e-6 a l'optimum et son gradient est du meme ordre : avec les defauts,
# L-BFGS-B S'ARRETE AVANT d'avoir bouge dans les directions faiblement
# courbees (ici pi_1).
#
# Verification a BRUIT NUL (phi_n remplacee par phi(.;theta_0) exacte, i.e.
# n = infini), depart (0.8, 0.2, 0.5, 0.5, 1.5), critere kappa = 1 :
#
#   defauts scipy   -> (psi1,psi2,sigma,pi1,alpha) = (0.7699, 0.2358, 1.6119,
#                       0.5090, 1.5017),  D = 1.19e-06,  |grad| = 9.7e-06 < gtol
#   avec MDE_GTOL   -> (0.8000, 0.3000, 1.6000, 0.4375, 1.5000) = theta_0,
#                       D = 8.9e-15
#
# Le decrochage est un deplacement FIXE : il ne decroit pas avec n, alors que
# l'ecart-type d'echantillonnage decroit en n^{-1/2}.  Le biais exprime en
# ecarts-types croit donc comme sqrt(n), ce qui reproduit exactement la derive
# de Dev.Mean et l'effondrement de couverture attribues jusqu'ici a une
# non-identification de pi_1.
#
# ATTENTION.  La normalisation kappa/pi * du * dv qui aligne le critere sur
# l'Assumption C2 est sans effet sur l'argmin (footnote de C2) mais divise le
# gradient par ~68 : elle fait donc mordre le gtol absolu ~68x plus tot.
# Toute renormalisation du critere DOIT s'accompagner d'un reglage de ces
# tolerances.
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
