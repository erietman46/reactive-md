"""Equal-equilibrium barrier families (kinetic control at fixed K).

For a calibrated potential, the barrier height (``Eh`` for the piecewise
potential, ``H`` for the additive one) is varied.  For every height:

1. the *uncompensated* change of ln K_p at T_ref measures how strongly the
   barrier leaks into the equilibrium (orthogonality of the two controls);
2. the bond depth eps_b is re-tuned (1-D root find) so that ln K_p(T_ref) is
   restored exactly - the constrained re-calibration;
3. the TST dissociation time, the radial free-energy barrier and the side
   effects (dH, frozen B2, bond geometry) of the compensated potential are
   recorded.

The kinetic estimates are TST/high-pressure-limit screening values; the real
MD relaxation times must be measured (research plan, Step 6).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ..thermodynamics.equilibrium_integrals import bound_state
from ..thermodynamics.kinetics_tst import tst_rates
from ..thermodynamics.virial_coefficients import B2
from .fit_pair_potentials import nominal_kb


def compensate_eb(pot, T, lnK_target, kb=None, span=(0.5, 2.0), keep_shape=False):
    """Return pot with eps_b re-tuned so that ln K_p(T) = lnK_target.

    By default the bond curvature k_b is kept; ``keep_shape=True`` keeps the
    branch shape parameter instead (e.g. the Morse range beta_b, so k_b scales
    with eps_b)."""
    kb = None if keep_shape else (nominal_kb(pot) if kb is None else kb)
    eb0 = pot.p["eb"]

    def g(eb):
        try:
            p = pot.with_bond(eb=eb, kb=kb)
            if not (p.landscape().clean and getattr(p, "feasible", True)):
                return np.nan
            return bound_state(p, T).log_Kp - lnK_target
        except ValueError:
            return np.nan

    eb = robust_root(g, eb0 * span[0], eb0 * span[1], x0=eb0)
    return pot.with_bond(eb=eb, kb=kb)


def robust_root(g, lo, hi, x0=None, n=41):
    """Root of g on [lo, hi] where g may be undefined (nan) in places.

    Tries a direct bracket first, then scans a grid for a sign change between
    two valid points, preferring the bracket closest to x0."""
    try:
        a, b = g(lo), g(hi)
        if np.isfinite(a) and np.isfinite(b) and a * b <= 0:
            return brentq(g, lo, hi, xtol=1e-10)
    except ValueError:
        pass
    xs = np.linspace(lo, hi, n)
    vs = np.array([g(x) for x in xs])
    cands = [i for i in range(n - 1) if np.isfinite(vs[i]) and np.isfinite(vs[i + 1]) and vs[i] * vs[i + 1] <= 0]
    if not cands:
        raise ValueError("no valid root in range")
    ref = x0 if x0 is not None else 0.5 * (lo + hi)
    i = min(cands, key=lambda k: abs(xs[k] - ref))

    def gg(x):
        v = g(x)
        if not np.isfinite(v):
            raise ValueError("invalid shape inside bracket")
        return v

    return brentq(gg, xs[i], xs[i + 1], xtol=1e-10)


def barrier_scan(pot, key, heights, T_ref, mu_gmol, lnK_target=None):
    """Rows describing the uncompensated and compensated potentials for each barrier height."""
    lnK0 = bound_state(pot, T_ref).log_Kp if lnK_target is None else lnK_target
    kb = nominal_kb(pot)
    base_tau = tst_rates(pot, T_ref, mu_gmol)
    rows = []
    for h in heights:
        row = dict(height=float(h))
        try:
            p_u = pot.replace(**{key: h})
            ok = p_u.landscape().clean and getattr(p_u, "feasible", True)
            row["clean"] = bool(ok)
            if not ok:
                rows.append(row)
                continue
            bs_u = bound_state(p_u, T_ref)
            row["dlnK_uncomp"] = bs_u.log_Kp - lnK0
            p_c = compensate_eb(p_u, T_ref, lnK0, kb=kb)
            bs_c = bound_state(p_c, T_ref)
            r = tst_rates(p_c, T_ref, mu_gmol)
            row.update(
                eb_comp=p_c.p["eb"], deb=p_c.p["eb"] - pot.p["eb"],
                U_barrier=p_c.u_barrier, r_barrier=p_c.r_barrier, u_bond=p_c.u_bond, r_bond=p_c.r_bond,
                k_bond=p_c.bond_stiffness(),
                dH=bs_c.deltaH, dS=bs_c.deltaS,
                B2_frozen=B2(p_c, T_ref).frozen,
                log_kd=r.log_kd, tau_d=r.tau_d, dF_diss=r.dF_diss, dF_assoc=r.dF_assoc,
                dln_tau=-(r.log_kd - base_tau.log_kd),
                clean_comp=bool(p_c.landscape().clean and getattr(p_c, "feasible", True)),
            )
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)
        rows.append(row)
    return rows


def orthogonality(rows):
    """|d lnK / d ln tau| from the uncompensated scan: 0 = perfectly separated controls."""
    ok = [r for r in rows if r.get("clean") and "dlnK_uncomp" in r and "dln_tau" in r]
    if len(ok) < 2:
        return np.nan
    x = np.array([r["dln_tau"] for r in ok])
    y = np.array([r["dlnK_uncomp"] for r in ok])
    if np.ptp(x) == 0:
        return np.nan
    return float(abs(np.polyfit(x, y, 1)[0]))


def heights_for_tau_ratios(pot, key, T_ref, mu_gmol, ratios, h_lo, h_hi, lnK_target=None, keep_shape=False):
    """Barrier heights (with eps_b compensation) giving tau_d = ratio * tau_d(h_lo)."""
    lnK0 = bound_state(pot, T_ref).log_Kp if lnK_target is None else lnK_target
    kb = nominal_kb(pot)

    def ln_tau(h):
        p = compensate_eb(pot.replace(**{key: h}), T_ref, lnK0, kb=kb, keep_shape=keep_shape)
        return -tst_rates(p, T_ref, mu_gmol).log_kd, p

    l0, p0 = ln_tau(h_lo)
    out = []
    for q in ratios:
        if q == 1:
            out.append((h_lo, p0))
            continue
        h = brentq(lambda x: ln_tau(x)[0] - (l0 + np.log(q)), h_lo, h_hi, xtol=1e-6)
        out.append((h, ln_tau(h)[1]))
    return out
