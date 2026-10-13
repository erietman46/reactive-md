"""Multi-criteria comparison of calibrated candidates.

Selection is done in two passes (never by a single least-squares number):

1. *Hard rejection* - a candidate is removed if any of these fails:
   clean landscape (one bond minimum, one barrier, one dispersion minimum),
   continuous force (C1) at every join, bond stiffness below the MD time-step
   limit, and the ability to vary the barrier at fixed K.
2. *Pareto comparison* of the survivors on the scored criteria (all "lower is
   better"): thermodynamic errors, barrier/equilibrium leakage, non-bonded
   side effects of the barrier, parameter count, identifiability.  A candidate
   is dominated when another is at least as good on every criterion and
   strictly better on one.  The non-dominated set is the shortlist; ties are
   broken by simplicity (fewest free parameters), as argued in the strategy.
"""
from __future__ import annotations

import numpy as np


def smoothness(pot, h=1e-5):
    """Jumps of U, dU/dr and d2U/dr2 across the potential's breakpoints.

    One-sided limits are estimated by linear extrapolation from two points on
    each side (so a finite slope does not register as a jump)."""
    jumps = dict(U=0.0, dU=0.0, d2U=0.0)

    def lim(g, b, side):
        x1, x2 = b + side * h, b + 2 * side * h
        return 2 * g(np.array([x1]))[0] - g(np.array([x2]))[0]

    def curv(b, side):
        d = lambda x: pot.dU(np.array([x]))[0]
        c1 = side * (d(b + 2 * side * h) - d(b + side * h)) / h      # at b + 1.5 side h
        c2 = side * (d(b + 3 * side * h) - d(b + 2 * side * h)) / h  # at b + 2.5 side h
        return c1 + 1.5 * (c1 - c2)

    for b in pot.breakpoints():
        jumps["U"] = max(jumps["U"], abs(lim(pot.U, b, 1) - lim(pot.U, b, -1)))
        jumps["dU"] = max(jumps["dU"], abs(lim(pot.dU, b, 1) - lim(pot.dU, b, -1)))
        jumps["d2U"] = max(jumps["d2U"], abs(curv(b, 1) - curv(b, -1)))
    return jumps


HARD = ("clean", "C1", "stiffness_md_ok", "barrier_tunable")
SCORED = ("rms_lnK_fit", "rms_lnK_test", "abs_dH_err", "rms_B2frozen_sigma", "orthogonality",
          "B2frozen_shift_per_decade", "n_free", "log10_condition")


def hard_pass(row):
    return all(bool(row.get(k, False)) for k in HARD)


def pareto_front(rows, keys=SCORED):
    """Indices of non-dominated rows (lower is better for every key)."""
    vals = np.array([[row.get(k, np.inf) for k in keys] for row in rows], float)
    vals = np.where(np.isfinite(vals), vals, np.inf)
    front = []
    for i in range(len(rows)):
        dominated = False
        for j in range(len(rows)):
            if i != j and np.all(vals[j] <= vals[i]) and np.any(vals[j] < vals[i]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def rank(rows):
    """Annotate rows with hard-pass, Pareto membership and a simplicity-first order."""
    for r in rows:
        r["hard_pass"] = hard_pass(r)
    survivors = [r for r in rows if r["hard_pass"]]
    front = pareto_front(survivors)
    for i, r in enumerate(survivors):
        r["pareto"] = i in front
    for r in rows:
        r.setdefault("pareto", False)
    order = sorted(
        rows,
        key=lambda r: (not r["hard_pass"], not r["pareto"], r.get("n_free", 99),
                       r.get("rms_lnK_test", np.inf)),
    )
    return order
