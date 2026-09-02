r"""
Relance des exercices du supplement en ligne avec le critere de la
Proposition 2.1 de main_v23.tex.

Ce que ce script relance
------------------------
  S.2.2 / S.2.3  subsampling non restreint (espace original + reparametrise)
  S.2.4          test de difference de critere pour H0 : pi_1 = 1/2
  S.2.5          subsampling restreint sous pi_1 = 1/2
  S.2.6          subsampling restreint sous pi_1 = 0.4375 (valeur vraie)

Ce qui a change par rapport aux resultats archives (Latex/resultats_*)
---------------------------------------------------------------------
1. POIDS.  S.2.4, S.2.5 et S.2.6 minimisaient `np.mean(|ecf - tcf|^2)`,
   c'est-a-dire un critere a poids UNIFORMES.  L'Assumption C2 impose
   w(u,v) = (kappa/pi) exp(-kappa(u^2+v^2)), et c'est cette forme qui est
   utilisee dans les preuves.  Les quatre modules passent desormais par
   cf_weights.quadrature_weights, qui renvoie une somme de Riemann de
   l'integrale (2.14).  Le resultat phare cite dans le texte principal
   (couverture nominale a n = 100 000 sous restriction correcte) portera
   donc enfin sur l'estimateur couvert par la Proposition 2.1.

2. CENTRAGE.  Dans subsampling_reparam, theta_n venait de
   stablemixture.estimate() (poids uniformes, maxiter=100, depart fixe)
   alors que les theta_b venaient de fast_objective (poids gaussiens) :
   deux statistiques differentes, alors que le subsampling exige la meme
   a deux echelles.  theta_n est maintenant le minimiseur du meme critere.
   C'est ce run qui dira si la derive observee a n = 100 000 (Dev. Mean
   croissante, couverture a 0.145) est reelle ou instrumentale.

3. EMBOITEMENT (S.2.4 seulement).  Delta_n = n[Q(restr) - Q(libre)] etait
   parfois NEGATIF (medianes -0.001 et -0.002 pour n = 50k et 100k), ce qui
   est impossible pour une restriction emboitee et signalait que l'optimum
   libre n'etait pas atteint.  Le modele libre est desormais reoptimise
   depuis la solution restreinte plongee dans l'espace libre.  Le diagnostic
   frac(Delta_n < 0) est imprime : il doit valoir 0.000.

Reproduction des anciens resultats
----------------------------------
Passer KAPPA = None retablit exactement les poids uniformes d'origine
(les corrections 2 et 3 restent actives, elles ne sont pas optionnelles).

Usage
-----
    python run_supplement_rerun.py                # tout, design du papier
    python run_supplement_rerun.py --quick        # n <= 1000, M = 20
    python run_supplement_rerun.py --only s26     # une seule section
    python run_supplement_rerun.py --kappa 2.0    # sensibilite au poids
"""

import argparse
import os
import time

# =========================================================================
# Design du papier (Section 4 de main_v23.tex, Table S.2 du supplement)
# =========================================================================

TRUE_PARAMS = (0.8, 0.3, 0.7, 0.9, 1.5, 0.0)  # rho_1, rho_2, vs_1, vs_2, alpha, beta
ALPHA = 1.5
BETA = 0.0
J = 2
FIX_ALPHA = False
FIX_BETA = True
SEED = 42

SAMPLE_SIZES = [250, 500, 1000, 10000, 50000, 100000]
SAMPLE_SIZES_CRIT = [500, 1000, 10000, 50000, 100000]
M = 200
B_EXPONENT = 2 / 3

# Poids de l'Assumption C2.  None => poids uniformes (mode legacy).
KAPPA = 1.0

OUT_ROOT = 'results_v23'

MODEL_KW = dict(true_params=TRUE_PARAMS, alpha=ALPHA, beta=BETA,
                fix_alpha=FIX_ALPHA, fix_beta=FIX_BETA, J=J)


def _banner(tag, title):
    print("\n" + "=" * 78)
    print(f"  [{tag}]  {title}")
    print("=" * 78)


def run_s23(sizes, m, kappa, n_cores, out_root):
    """S.2.2 / S.2.3 : subsampling non restreint, centrage corrige."""
    _banner("S.2.2-S.2.3", "Subsampling non restreint (centrage corrige)")
    from subsampling_reparam import run_study_reparam
    return run_study_reparam(
        sample_sizes=sizes, M=m, b_exponent=B_EXPONENT, kappa=kappa,
        n_cores=n_cores, seed=SEED,
        output_dir=os.path.join(out_root, 'unrestricted_reparam'),
        **MODEL_KW)


def run_s24(sizes, m, kappa, n_cores, out_root):
    """S.2.4 : test de difference de critere, H0 : pi_1 = 1/2."""
    _banner("S.2.4", "Test de difference de critere (H0 : pi_1 = 1/2)")
    from criterion_difference_test import run_mc_criterion_test
    res = run_mc_criterion_test(
        sample_sizes=sizes, M=m, b_exponent=B_EXPONENT, n_cores=n_cores,
        seed=SEED, output_dir=os.path.join(out_root, 'criterion_test'),
        kappa=kappa, **MODEL_KW)

    bad = {n: a['frac_delta_n_negative'] for n, a in res.items()
           if a['frac_delta_n_negative'] > 0}
    if bad:
        print("\n  !! ATTENTION : Delta_n < 0 subsiste pour " + str(bad))
        print("     L'optimum libre n'est pas atteint ; le test n'est pas")
        print("     interpretable en l'etat (augmenter maxiter ou multi-start).")
    else:
        print("\n  OK : Delta_n >= 0 pour toutes les replications et toutes")
        print("       les tailles — l'emboitement est numeriquement respecte.")
    return res


def run_s25(sizes, m, kappa, n_cores, out_root):
    """S.2.5 : subsampling restreint sous pi_1 = 1/2."""
    _banner("S.2.5", "Subsampling restreint sous pi_1 = 1/2")
    from subsampling_restricted import run_study_restricted
    return run_study_restricted(
        sample_sizes=sizes, M=m, b_exponent=B_EXPONENT, n_cores=n_cores,
        seed=SEED, output_dir=os.path.join(out_root, 'restricted_pi_half'),
        kappa=kappa, **MODEL_KW)


def run_s26(sizes, m, kappa, n_cores, out_root):
    """S.2.6 : subsampling restreint sous pi_1 = 0.4375 (resultat phare)."""
    _banner("S.2.6", "Subsampling restreint sous pi_1 = 0.4375 (valeur vraie)")
    from subsampling_restricted_truepi import run_study_restricted
    return run_study_restricted(
        sample_sizes=sizes, M=m, b_exponent=B_EXPONENT, n_cores=n_cores,
        seed=SEED, output_dir=os.path.join(out_root, 'restricted_pi_true'),
        kappa=kappa, **MODEL_KW)


SECTIONS = {
    's23': (run_s23, 'SAMPLE_SIZES'),
    's24': (run_s24, 'SAMPLE_SIZES_CRIT'),
    's25': (run_s25, 'SAMPLE_SIZES'),
    's26': (run_s26, 'SAMPLE_SIZES'),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(SECTIONS),
                    help="Sections a relancer (defaut : toutes).")
    ap.add_argument('--kappa', type=float, default=KAPPA,
                    help="Parametre du poids gaussien (defaut : 1.0).")
    ap.add_argument('--uniform-weights', action='store_true',
                    help="Mode legacy : poids uniformes (kappa = None).")
    ap.add_argument('--M', type=int, default=M, help="Replications MC.")
    ap.add_argument('--n-cores', type=int, default=-1, help="-1 = tous.")
    ap.add_argument('--out', default=OUT_ROOT, help="Racine des sorties.")
    ap.add_argument('--quick', action='store_true',
                    help="Test rapide : n <= 1000, M = 20.")
    args = ap.parse_args()

    kappa = None if args.uniform_weights else args.kappa
    sizes = SAMPLE_SIZES
    sizes_crit = SAMPLE_SIZES_CRIT
    m = args.M
    if args.quick:
        sizes, sizes_crit, m = [250, 500, 1000], [500, 1000], 20

    os.makedirs(args.out, exist_ok=True)

    print("=" * 78)
    print("  RELANCE DU SUPPLEMENT — critere de la Proposition 2.1 (main_v23)")
    print("=" * 78)
    print(f"  Poids       : {'UNIFORMES (legacy)' if kappa is None else f'gaussiens, kappa = {kappa}'}")
    print(f"  DGP         : rho = (0.8, 0.3), varsigma = (0.7, 0.9), "
          f"alpha = {ALPHA}, beta = {BETA}")
    print(f"  pi_1 vrai   : {TRUE_PARAMS[2] / (TRUE_PARAMS[2] + TRUE_PARAMS[3]):.4f}")
    print(f"  Tailles     : {sizes}")
    print(f"  M           : {m}   |   b = floor(n^{B_EXPONENT:.4f})")
    print(f"  Sorties     : {args.out}/")

    todo = args.only or sorted(SECTIONS)
    t0 = time.time()
    for tag in todo:
        fn, which = SECTIONS[tag]
        s = sizes_crit if which == 'SAMPLE_SIZES_CRIT' else sizes
        fn(s, m, kappa, args.n_cores, args.out)

    print("\n" + "=" * 78)
    print(f"  TERMINE en {(time.time() - t0) / 60:.1f} min — resultats dans {args.out}/")
    print("=" * 78)
    print("""
  Points a verifier dans les sorties :
   1. S.2.4 : frac(Delta_n < 0) doit valoir 0.000 partout.
   2. S.2.3 : la Dev. Mean doit-elle toujours croitre avec n ?  Si elle se
      stabilise, la derive de la version archivee etait instrumentale (dis-
      cordance de centrage) ; si elle persiste, elle est reelle et reflete
      la crete d'identification dans la direction pi_1.
   3. S.2.6 : la couverture nominale a n = 100 000 doit se maintenir sous
      le poids gaussien.  C'est le resultat cite dans le texte principal.""")


if __name__ == '__main__':
    main()
