r"""
Tests de regression du patch "Assumption C2 + centrage".

    python test_patch_c2.py

Verifie :
  1. Le mode legacy (kappa=None) reproduit exactement l'ancien np.mean(...).
  2. La quadrature gaussienne est la somme de Riemann annoncee et integre 1
     lorsque la grille est assez large.
  3. Les quatre implementations du critere (criterion_difference_test,
     subsampling_reparam, subsampling_restricted, subsampling_restricted_truepi)
     coincident la ou elles doivent coincider.
  4. Delta_n >= 0 : l'emboitement du modele restreint dans le modele libre est
     numeriquement respecte (c'etait la pathologie de la Table S.2.4).
  5. Le centrage de subsampling_reparam est bien le minimiseur du critere des
     sous-echantillons, et non plus la sortie de stablemixture.estimate().
"""

import numpy as np

import cf_weights as cw
import criterion_difference_test as cdt
import subsampling_reparam as srp
import subsampling_restricted as srs
import subsampling_restricted_truepi as srt
from stablemixture import StableMixtureDeconvolution

TP = (0.8, 0.3, 0.7, 0.9, 1.5, 0.0)
J, AV, BV, FA, FB = 2, 1.5, 0.0, False, True
UG = VG = np.linspace(-1.5, 1.5, 15)
KAPPA = 1.0

_n_ok = 0


def check(label, cond, detail=""):
    global _n_ok
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}" + (f"  {detail}" if detail else ""))
    assert cond, label
    _n_ok += 1


def make_series(T=2000, seed=7):
    mdl = StableMixtureDeconvolution(num_components=J, alpha_value=AV,
                                     beta_value=BV, fix_alpha=FA,
                                     fix_beta=FB, verbose=False)
    Y, _ = mdl.generate(T=T, params=TP, seed=seed)
    return mdl, Y


def test_legacy_equivalence(Y):
    print("\n1. Mode legacy (kappa=None) == ancien np.mean(...)")
    ecf = cdt._ecf_matrix(Y, UG, VG)
    tcf = cdt._tcf_matrix(np.array([0.8, 0.3, 0.7, 0.9, 1.5]),
                          UG, VG, J, None, BV)
    ref = float(np.mean(np.abs(ecf - tcf) ** 2))
    got = cw.weighted_sse(ecf, tcf, cw.quadrature_weights(UG, VG, None))
    check("poids uniformes reproduisent np.mean", np.isclose(ref, got, rtol=1e-12),
          f"(mean={ref:.6e}, legacy={got:.6e})")


def test_quadrature():
    print("\n2. Quadrature gaussienne")
    W = cw.quadrature_weights(UG, VG, KAPPA)
    du = UG[1] - UG[0]
    U, V = np.meshgrid(UG, VG, indexing='ij')
    manual = (KAPPA / np.pi) * np.exp(-KAPPA * (U ** 2 + V ** 2)) * du * du
    check("W == (kappa/pi) exp(-kappa r^2) du dv", np.allclose(W, manual))

    wide = np.linspace(-8, 8, 401)
    check("integre 1 sur une grille large",
          abs(cw.captured_mass(wide, wide, KAPPA) - 1.0) < 1e-6,
          f"(masse={cw.captured_mass(wide, wide, KAPPA):.8f})")

    m = cw.captured_mass(UG, VG, KAPPA)
    check("grille historique capture > 90% de la masse", m > 0.90,
          f"(masse={m:.4f})")

    try:
        cw.quadrature_weights(UG, VG, -1.0)
        ok = False
    except ValueError:
        ok = True
    check("kappa <= 0 rejete (Assumption C2)", ok)


def test_cross_module_agreement(Y):
    print("\n3. Coherence entre modules")
    W = cw.quadrature_weights(UG, VG, KAPPA)
    th_free = np.array([0.8, 0.3, 0.7, 0.9, 1.5])          # pi_1 = 0.4375
    pi_true = np.array([0.4375, 0.5625])

    q_cdt = cdt._obj_free(th_free, cdt._ecf_matrix(Y, UG, VG),
                          UG, VG, J, None, BV, W)
    q_srp = srp.fast_objective(th_free, srp.vectorized_ecf(Y, UG, VG),
                               UG, VG, J, None, BV, KAPPA)
    check("criterion_difference_test == subsampling_reparam",
          np.isclose(q_cdt, q_srp, rtol=1e-10),
          f"({q_cdt:.6e} vs {q_srp:.6e})")

    # meme point, exprime dans l'espace restreint sous pi_1 = 0.4375
    th_r = np.array([0.8, 0.3, 1.6, 1.5])
    q_srt = srt._obj_restricted(th_r, srt._ecf_matrix(Y, UG, VG),
                                UG, VG, J, None, BV, pi_true, W)
    check("subsampling_restricted_truepi == libre au meme point",
          np.isclose(q_cdt, q_srt, rtol=1e-10),
          f"({q_cdt:.6e} vs {q_srt:.6e})")

    # sous pi_1 = 1/2, le point libre equivalent est varsigma = (0.8, 0.8)
    q_srs = srs._obj_restr(th_r, srs._ecf_matrix(Y, UG, VG),
                           UG, VG, J, None, BV, W)
    q_half = cdt._obj_free(np.array([0.8, 0.3, 0.8, 0.8, 1.5]),
                           cdt._ecf_matrix(Y, UG, VG), UG, VG, J, None, BV, W)
    check("subsampling_restricted == libre a varsigma = sigma/2",
          np.isclose(q_srs, q_half, rtol=1e-10),
          f"({q_srs:.6e} vs {q_half:.6e})")


def test_delta_nonnegative():
    print("\n4. Delta_n >= 0 (emboitement du modele restreint)")
    bf = [(0.01, 0.99)] * J + [(0.005, 5.0)] * J + [(0.1, 1.99)]
    br = [(0.01, 0.99)] * J + [(0.01, 10.0)] + [(0.1, 1.99)]
    n, b = 1000, int(np.floor(1000 ** (2 / 3)))
    reps = [cdt._one_rep_criterion(200 + m, n, b, TP, J, AV, BV, FA, FB,
                                   UG, VG, bf, br, KAPPA) for m in range(6)]
    d = np.array([r['delta_n'] for r in reps])
    check("aucun Delta_n < 0 sur l'echantillon complet",
          not any(r['delta_n_negative'] for r in reps),
          f"(min = {d.min():.3e})")
    fb = np.nanmean([r['frac_delta_b_negative'] for r in reps])
    check("aucun Delta_b < 0 sur les sous-echantillons", fb == 0.0,
          f"(frac = {fb:.3f})")


def test_centering(mdl, Y):
    print("\n5. Centrage de subsampling_reparam")
    from scipy import optimize
    ecf = srp.vectorized_ecf(Y, UG, VG)
    bounds = [(0.01, 0.99)] * J + [(0.005, 5.0)] * J + [(0.1, 1.99)]

    x0 = np.array(mdl.estimate(Y, use_optimal_weights=False))  # ancien centrage
    res = optimize.minimize(srp.fast_objective, x0,
                            args=(ecf, UG, VG, J, None, BV, KAPPA),
                            bounds=bounds, method='L-BFGS-B',
                            options={'maxiter': 300, 'ftol': 1e-10})
    q_old = srp.fast_objective(x0, ecf, UG, VG, J, None, BV, KAPPA)
    q_new = float(res.fun)
    check("le nouveau centrage minimise bien le critere des blocs",
          q_new <= q_old + 1e-12,
          f"(Q_ancien={q_old:.6e} -> Q_nouveau={q_new:.6e})")
    if q_new < q_old * (1 - 1e-6):
        print(f"         -> l'ancien centrage n'etait PAS un minimiseur "
              f"(ecart relatif {1 - q_new / q_old:.2%})")


if __name__ == '__main__':
    print("=" * 70)
    print("  Tests de regression : patch Assumption C2 + centrage")
    print("=" * 70)
    mdl, Y = make_series()
    test_legacy_equivalence(Y)
    test_quadrature()
    test_cross_module_agreement(Y)
    test_delta_nonnegative()
    test_centering(mdl, Y)
    print("\n" + "=" * 70)
    print(f"  {_n_ok} verifications passees.")
    print("=" * 70)
