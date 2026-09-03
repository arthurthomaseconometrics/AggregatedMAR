#!/usr/bin/env python3
"""
Preflight check -- run this FIRST on the cluster, before burning core-hours.

It verifies that the four audit fixes of 2026-09-03 are actually active in the
environment you are about to run in.  Takes a few seconds.  Exit code 0 = all
good, 1 = at least one check failed (do NOT launch the production jobs).

    python hpc/preflight_check.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import optimize, stats

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


print("Preflight check -- audit fixes of 2026-09-03\n")
print("Environment")
print(f"  python {sys.version.split()[0]} | numpy {np.__version__} | "
      f"scipy {__import__('scipy').__version__}")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    print(f"  {v}={os.environ.get(v, '<unset>')}"
          + ("   <-- set this to 1, see run_s22.sbatch" if os.environ.get(v) != "1" else ""))

# ---------------------------------------------------------------- 1. gtol
print("\n1. Optimizer tolerances (cf_weights.MDE_GTOL)")
from cf_weights import ALPHA_BOUNDS, MDE_GTOL, lbfgsb_options
from subsampling_reparam import fast_objective, vectorized_tcf, _transform_params

ug = np.linspace(-1.5, 1.5, 15)
vg = ug.copy()
bounds = [(0.01, 0.99)] * 2 + [(0.005, 5.0)] * 2 + [ALPHA_BOUNDS]
th0 = np.array([0.8, 0.3, 0.7, 0.9, 1.5])
# noiseless: the ECF is replaced by the exact population CF at theta_0 (n = inf)
ecf = vectorized_tcf(th0, ug, vg, 2, None, 0.0)
x0 = np.array([0.8, 0.2, 0.5, 0.5, 1.5])
res = optimize.minimize(fast_objective, x0, args=(ecf, ug, vg, 2, None, 0.0, 1.0),
                        bounds=bounds, method="L-BFGS-B", options=lbfgsb_options(2000))
got = _transform_params(res.x, 2, False, True)
want = np.array([0.8, 0.3, 1.6, 0.4375, 1.5])
err = float(np.max(np.abs(got - want)))
check("noiseless criterion recovers theta_0", err < 1e-3,
      f"max|err| = {err:.2e}, pi_1 = {got[3]:.4f} (true 0.4375), gtol = {MDE_GTOL:g}")
print("         (with scipy's default gtol=1e-5 this returns pi_1 = 0.509)")

# ------------------------------------------------------- 2. burn-in side
print("\n2. DGP burn-in kept on the correct side (stablemixture.generate)")
from stablemixture import StableMixtureDeconvolution

mdl = StableMixtureDeconvolution(2, alpha_value=1.5, beta_value=0.0,
                                 fix_alpha=False, fix_beta=True, verbose=False)
tp = (0.8, 0.3, 0.7, 0.9, 1.5, 0.0)
last = np.array([mdl.generate(T=200, params=tp, seed=5000 + s)[0][-1] for s in range(60)])
check("last observation is not identically zero", np.mean(last == 0) == 0.0,
      f"fraction of replications with Y[-1] == 0: {np.mean(last == 0):.2f} (was 1.00)")

S = np.array([np.abs(mdl.generate(T=300, params=tp, seed=6000 + s)[0]) for s in range(120)])
head, tail = np.median(S[:, :50]), np.median(S[:, -50:])
check("no attenuation at the end of the series", 0.7 < tail / head < 1.4,
      f"median|Y| tail/head = {tail / head:.3f}")

# ------------------------------------------------- 3. independent seeds
print("\n3. Independent per-component seed streams")
_, c_a = mdl.generate(T=60, params=tp, seed=42)
_, c_b = mdl.generate(T=60, params=tp, seed=43)
check("component 2 @seed s != component 1 @seed s+1",
      not np.allclose(c_a[1] / 0.9, c_b[0] / 0.7),
      "these shared their innovations exactly before the fix")

# --------------------------------------- 4. C1 ordering + same functional
print("\n4. Assumption C1 ordering and the single MDE functional")
from subsampling_reparam import canonical_order, mde_starts, estimate_sub

co = canonical_order(np.array([0.3, 0.8, 0.9, 0.7, 1.5]), 2)
check("canonical_order sorts rho descending, carrying varsigma",
      np.allclose(co, [0.8, 0.3, 0.7, 0.9, 1.5]), f"-> {np.round(co, 4)}")
st = mde_starts(2, False, True, 0.0, 2)
check("starting values are data-independent and mirrored", len(st) == 2
      and np.allclose(st[0][[1, 0, 3, 2, 4]], st[1]), f"{[list(np.round(x,2)) for x in st]}")

print("\n" + ("ALL CHECKS PASSED -- safe to launch the production jobs."
              if not FAIL else f"FAILED: {FAIL}\nDo NOT launch the production jobs."))
sys.exit(1 if FAIL else 0)
