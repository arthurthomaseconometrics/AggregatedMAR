"""
Diagnostic autonome a lancer EN PREMIER sur le supercalculateur, avant toute
campagne de subsampling (M=200, tous les n), pour verifier que la copie
locale de stablemixture.py n'a pas la regression d'appariement (u,v) trouvee
lors de l'audit du 2026-07 sur la copie du poste local.

Rappel du bug : char_function(u,v,params) calcule la CF theorique de la paire
(Y_t, Y_{t-1}) avec u <-> observation POSTERIEURE et v <-> observation
ANTERIEURE. objective_function (et compute_optimal_weights) doivent utiliser
exactement le meme appariement pour l'ECF, sinon l'objectif MDE est mal
specifie (la CF d'une paire non-causale n'est PAS symetrique en (u,v) des
que alpha < 2), ce qui biaise fortement l'estimation plein-echantillon
(observe : rho_1 -> 0.94 au lieu de 0.80, rho_2 -> 0.01 au lieu de 0.30
sur une serie simulee n=20 000 avec le bug present).

Ce script ne depend que de numpy/scipy (pas de joblib) et tourne en
quelques secondes : deux etapes independantes du fix reellement applique
dans le fichier .py, donc le diagnostic est valide meme si vous n'avez pas
encore copie la version corrigee sur le cluster.

Usage :
    cd AggregatedMAR-main/   (sur le cluster, a cote de stablemixture.py)
    python3 check_stablemixture_fix.py
"""

import inspect
import numpy as np
import sys

sys.path.insert(0, '.')
from stablemixture import StableMixtureDeconvolution


def check_source_pairing():
    """Etape 1 (instantanee) : inspecte le CODE SOURCE de objective_function
    et compute_optimal_weights pour voir quel appariement (u,v) est utilise,
    sans rien executer. Fiable tant que le code n'a pas ete obfusque."""
    print("=" * 70)
    print("ETAPE 1 - Inspection du code source (instantanee)")
    print("=" * 70)
    src_obj = inspect.getsource(StableMixtureDeconvolution.objective_function)
    src_w = inspect.getsource(StableMixtureDeconvolution.compute_optimal_weights)

    def verdict(src, name):
        # Ne regarder que les lignes de CODE (pas les commentaires/docstrings,
        # qui peuvent citer l'ancien motif a titre explicatif et fausser la
        # detection par simple recherche de sous-chaine).
        code_lines = [ln for ln in src.splitlines()
                      if "cf_emp = np.mean" in ln or "cf_values = np.exp" in ln]
        code_lines = [ln for ln in code_lines if not ln.strip().startswith("#")]
        blob = "\n".join(code_lines)
        buggy = "Y[:-1] + v * Y[1:]" in blob or "Y[:-1] + v*Y[1:]" in blob
        fixed = "Y[1:] + v * Y[:-1]" in blob or "Y[1:] + v*Y[:-1]" in blob
        if fixed and not buggy:
            print(f"  [{name}] OK - appariement corrige (u<->futur, v<->passe)")
            return True
        elif buggy and not fixed:
            print(f"  [{name}] BUG PRESENT - appariement u<->passe, v<->futur "
                  "(incoherent avec char_function)")
            return False
        else:
            print(f"  [{name}] INDETERMINE - motif non reconnu dans les "
                  f"lignes de code ({len(code_lines)} trouvee(s)) - se fier "
                  "aux etapes 2/3.")
            return None

    ok1 = verdict(src_obj, "objective_function")
    ok2 = verdict(src_w, "compute_optimal_weights")
    return ok1, ok2


def check_numeric_pairing(T=20000, seed=123):
    """Etape 2 (~10-20s) : test numerique independant du texte du code.
    Simule une serie a parametres connus, evalue char_function(u,v) au
    meme point que les DEUX appariements ECF possibles, et rapporte lequel
    colle le mieux. Ne regarde PAS le code source: verifie le comportement
    reel de char_function tel qu'importe, donc detecte aussi une eventuelle
    incoherence si char_function elle-meme a change entre-temps."""
    print("\n" + "=" * 70)
    print(f"ETAPE 2 - Test numerique (T={T}, ~10-20s)")
    print("=" * 70)

    true_params = (0.8, 0.3, 0.7, 0.9, 1.5, 0.0)
    mdl = StableMixtureDeconvolution(num_components=2, alpha_value=1.5,
                                     beta_value=0.0, fix_alpha=False,
                                     fix_beta=True, verbose=False)
    Y, _ = mdl.generate(T=T, params=true_params, seed=seed)

    pts = [(1.0, -0.5), (0.5, 1.0), (-1.0, 0.3), (0.7, -1.2)]
    err_current = 0.0   # convention actuellement cablee dans objective_function
    err_swapped = 0.0   # convention inverse
    print(f"{'(u,v)':>14} | {'char_function':>20} | {'ECF u<->Y[:-1]':>20} "
          f"| {'ECF u<->Y[1:]':>20}")
    for (u, v) in pts:
        cf_theo = mdl.char_function(u, v, true_params)
        ecf_a = np.mean(np.exp(1j * (u * Y[:-1] + v * Y[1:])))   # u<->passe
        ecf_b = np.mean(np.exp(1j * (u * Y[1:] + v * Y[:-1])))   # u<->futur
        err_current += abs(ecf_a - cf_theo)
        err_swapped += abs(ecf_b - cf_theo)
        print(f"({u:5.2f},{v:5.2f}) | {cf_theo.real:8.4f}{cf_theo.imag:+7.4f}j "
              f"| {ecf_a.real:8.4f}{ecf_a.imag:+7.4f}j "
              f"| {ecf_b.real:8.4f}{ecf_b.imag:+7.4f}j")

    print(f"\nErreur cumulee |ECF - char_function| :")
    print(f"  u<->Y[:-1] (ancienne convention) : {err_current:.4f}")
    print(f"  u<->Y[1:]  (convention corrigee) : {err_swapped:.4f}")

    if err_swapped < err_current / 3:
        print("=> char_function suit la convention u<->FUTUR (celle du papier).")
        needs = "u<->Y[1:] + v<->Y[:-1]"
    elif err_current < err_swapped / 3:
        print("=> char_function suit la convention u<->PASSE (inhabituel, a "
              "verifier contre main_v16.tex Case 1).")
        needs = "u<->Y[:-1] + v<->Y[1:]"
    else:
        print("=> AMBIGU (les deux erreurs sont du meme ordre) - augmenter T "
              "ou ne pas se fier a ce test seul.")
        needs = None
    return needs


def check_full_sample_estimate(T=20000, seed=7):
    """Etape 3 (optionnelle, ~15-30s selon le CPU) : lance UNE estimation
    plein-echantillon (pas de subsampling, pas de parallelisme) et compare
    aux vrais parametres. C'est le test le plus direct : si le bug est
    present, rho_2 s'effondre typiquement pres de 0 et rho_1 sature pres
    de 1 (observe localement : rho_2 -> 0.01, rho_1 -> 0.94)."""
    print("\n" + "=" * 70)
    print(f"ETAPE 3 - Une estimation plein-echantillon (T={T}, optionnelle)")
    print("=" * 70)
    true_params = (0.8, 0.3, 0.7, 0.9, 1.5, 0.0)
    mdl = StableMixtureDeconvolution(num_components=2, alpha_value=1.5,
                                     beta_value=0.0, fix_alpha=False,
                                     fix_beta=True, verbose=False)
    Y, _ = mdl.generate(T=T, params=true_params, seed=seed)
    est = mdl.estimate(Y, use_optimal_weights=False)
    names = ["rho_1", "rho_2", "varsigma_1", "varsigma_2", "alpha"]
    print(f"{'':>12} " + " ".join(f"{n:>10}" for n in names))
    print(f"{'true':>12} " + " ".join(f"{v:10.4f}" for v in true_params[:5]))
    print(f"{'estimate':>12} " + " ".join(f"{v:10.4f}" for v in est))

    # Signature du bug: rho_2 tres proche de 0 ET rho_1 tres proche de 1
    bug_signature = est[1] < 0.05 or est[0] > 0.93
    if bug_signature:
        print("\n=> SIGNATURE DU BUG DETECTEE (rho_2 proche de 0 ou rho_1 "
              "proche de 1). NE PAS lancer la campagne complete tant que ce "
              "n'est pas corrige.")
    else:
        print("\n=> Pas de signature evidente du bug sur cette replication "
              "(pas une preuve formelle, mais un bon signe).")
    return not bug_signature


if __name__ == "__main__":
    # Etape 3 (comportementale, sur objective_function/estimate() tels
    # qu'importes) est le juge de paix : c'est la seule qui teste le
    # pipeline reellement utilise par le subsampling, sans dependre du
    # texte du code. Les etapes 1 et 2 ne servent qu'a corroborer / a
    # localiser le probleme si l'etape 3 detecte une anomalie.
    ok1, ok2 = check_source_pairing()
    needs = check_numeric_pairing()
    ok3 = check_full_sample_estimate()

    print("\n" + "=" * 70)
    print("VERDICT GLOBAL")
    print("=" * 70)

    if ok1 is False or ok2 is False:
        print("SOURCE: pattern buggy detecte dans le code -> BUG QUASI CERTAIN.")
    if ok3 is False:
        print("COMPORTEMENT (etape 3, la plus fiable): signature du bug "
              "detectee sur l'estimation plein-echantillon.")

    if ok3 is True and ok1 is not False and ok2 is not False:
        print("\nOK - aucune signature du bug detectee (les 3 etapes "
              "concordent ou sont neutres). Vous pouvez passer a un run "
              "leger de subsampling (ex. M=20, un seul n) avant la "
              "campagne complete du supplement.")
    else:
        print("\nBUG PROBABLE - ne PAS lancer la campagne complete. "
              "Patch a appliquer (2 fichiers) :")
        print("""
Dans objective_function (recherche: "cf_emp = np.mean"), remplacer:
    cf_emp = np.mean(np.exp(1j * (u * Y[:-1] + v * Y[1:])))
par:
    cf_emp = np.mean(np.exp(1j * (u * Y[1:] + v * Y[:-1])))

Dans compute_optimal_weights (deux occurrences, "cf_emp" et "cf_values"):
    cf_emp = np.mean(np.exp(1j * (u*Y[:-1] + v*Y[1:])))
    cf_values = np.exp(1j * (u*Y[:-1] + v*Y[1:]))
remplacer par:
    cf_emp = np.mean(np.exp(1j * (u*Y[1:] + v*Y[:-1])))
    cf_values = np.exp(1j * (u*Y[1:] + v*Y[:-1]))

Puis relancer ce script pour confirmer.
""")
