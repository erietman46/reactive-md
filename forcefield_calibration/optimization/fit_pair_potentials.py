"""Bounded least-squares fitting of pair potentials to thermodynamic targets.

A fit is defined by

* a *base* potential (all fixed parameters),
* a list of free parameters ``(name, lo, hi)``.  The special names ``eb`` and
  ``kb`` set the bond depth and the bond-branch curvature through
  ``with_bond`` (the branch shape parameter is solved for), so every bond
  form is fitted in the same physical coordinates; all other names are raw
  potential parameters,
* a residual function of the trial potential.

Optimisation is global-then-local: ``scipy.optimize.differential_evolution``
on the summed squares (bounded, seeded, reproducible), refined with
``scipy.optimize.least_squares`` (trust-region reflective) inside the same
bounds.  Trial parameter sets that do not give a clean bond/barrier/dispersion
landscape receive a large constant residual, which acts as a hard constraint.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from ..thermodynamics.equilibrium_integrals import bound_state
from ..thermodynamics.virial_coefficients import B2

BAD = 1e3


@dataclass
class FreeParam:
    name: str
    lo: float
    hi: float
    log: bool = False

    def to_x(self, v):
        return np.log(v) if self.log else v

    def from_x(self, x):
        return float(np.exp(x)) if self.log else float(x)

    @property
    def bounds(self):
        return (self.to_x(self.lo), self.to_x(self.hi))


def as_params(spec):
    out = []
    for s in spec:
        if isinstance(s, FreeParam):
            out.append(s)
        else:
            name, lo, hi, *rest = s
            out.append(FreeParam(name, lo, hi, bool(rest[0]) if rest else name in ("kb", "C6")))
    return out


def build(base, params, values):
    """Potential with ``values`` substituted for the free ``params``."""
    kw = {}
    bond = {}
    for p, v in zip(params, values):
        if p.name in ("eb", "kb"):
            bond[p.name] = v
        else:
            kw[p.name] = v
    pot = base.replace(**kw) if kw else base
    if bond:
        # keep the current branch curvature when only the depth changes
        pot = pot.with_bond(eb=bond.get("eb"), kb=bond.get("kb", nominal_kb(base)))
    return pot


def nominal_kb(pot):
    from ..potentials.branches import bond_k

    return float(bond_k(pot.p))


@dataclass
class FitResult:
    pot: object
    values: dict
    residuals: np.ndarray
    cost: float
    success: bool
    message: str
    n_free: int
    extra: dict = field(default_factory=dict)

    @property
    def chi2(self):
        return float(np.sum(self.residuals ** 2))


def fit(base, spec, residual_fn, seed=1, popsize=15, maxiter=80, de=True, x0=None):
    params = as_params(spec)
    kb_fixed = nominal_kb(base)

    def make(x):
        vals = [p.from_x(xi) for p, xi in zip(params, x)]
        kw, bond = {}, {}
        for p, v in zip(params, vals):
            (bond if p.name in ("eb", "kb") else kw)[p.name] = v
        pot = base.replace(**kw) if kw else base
        if bond:
            pot = pot.with_bond(eb=bond.get("eb"), kb=bond.get("kb", kb_fixed))
        return pot, vals

    n_res = None

    def resid(x):
        nonlocal n_res
        try:
            pot, _ = make(x)
            if not pot.landscape().clean or (hasattr(pot, "feasible") and not pot.feasible):
                raise ValueError("unclean")
            r = np.asarray(residual_fn(pot), float)
            if not np.all(np.isfinite(r)):
                raise ValueError("nan")
            n_res = r.size
            return r
        except Exception:
            return np.full(n_res or 1, BAD)

    bounds = [p.bounds for p in params]
    if de:
        res_de = differential_evolution(
            lambda x: float(np.sum(resid(x) ** 2)), bounds, seed=seed, popsize=popsize,
            maxiter=maxiter, tol=1e-10, polish=False, init="sobol",
        )
        x_start = res_de.x
    else:
        x_start = np.array([p.to_x(v) for p, v in zip(params, x0)]) if x0 is not None else \
            np.array([0.5 * (a + b) for a, b in bounds])
    lo, hi = np.array(bounds).T
    eps = 1e-9 * (hi - lo)
    x_start = np.clip(x_start, lo + eps, hi - eps)
    ls = least_squares(resid, x_start, bounds=(lo, hi), x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    pot, vals = make(ls.x)
    r = resid(ls.x)
    return FitResult(pot, {p.name: v for p, v in zip(params, vals)}, r, float(np.sum(r ** 2)),
                     bool(ls.success) and r[0] != BAD, ls.message, len(params))


# ------------------------------------------------------------ residuals
def lnK_residuals(ref, T_idx=None, w_K=1.0, w_H=0.0, T_H=None):
    """Residuals on ln K_p(T) (and optionally dH(T)) against a reference dict."""
    T = ref["T"] if T_idx is None else ref["T"][T_idx]
    lnK = ref["lnK"] if T_idx is None else ref["lnK"][T_idx]
    sK = ref["sigma_lnK"] if T_idx is None else ref["sigma_lnK"][T_idx]
    dH = ref["dH"] if T_idx is None else ref["dH"][T_idx]
    sH = ref["sigma_dH"] if T_idx is None else ref["sigma_dH"][T_idx]

    def fn(pot):
        bss = [bound_state(pot, t) for t in T]
        r = [w_K * (b.log_Kp - k) / s for b, k, s in zip(bss, lnK, sK)]
        if w_H:
            r += [w_H * (b.deltaH - h) / s for b, h, s in zip(bss, dH, sH)]
        return np.array(r)

    return fn


def b2_frozen_residuals(T, B2_target, sigma):
    T = np.atleast_1d(T)

    def fn(pot):
        return np.array([(B2(pot, t).frozen - b) / s for t, b, s in zip(T, B2_target, sigma)])

    return fn


def predictions(pot, T):
    """Model ln K_p, dH, dS, B2 (total/frozen) on a temperature grid."""
    out = {k: [] for k in ("T", "lnK", "dH", "dS", "B2", "B2_frozen", "Kc_L_mol")}
    for t in np.atleast_1d(T):
        bs = bound_state(pot, t)
        v = B2(pot, t)
        out["T"].append(t)
        out["lnK"].append(bs.log_Kp)
        out["dH"].append(bs.deltaH)
        out["dS"].append(bs.deltaS)
        out["B2"].append(v.total)
        out["B2_frozen"].append(v.frozen)
        out["Kc_L_mol"].append(bs.Kc_L_per_mol)
    return {k: np.array(v) for k, v in out.items()}
