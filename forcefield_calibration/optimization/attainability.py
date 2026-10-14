"""Can a single-site isotropic pair potential reproduce K_p(T) AND dH together?

For a bound basin that is approximately harmonic,

    Z_b ~ 4 pi r_b^2 sqrt(2 pi RT / k_b) exp(eps_b / RT),
    <U>_b ~ -eps_b + RT/2,

so dH = <U>_b - RT fixes eps_b, after which K_p (i.e. Z_b) can only be
matched by the bond width sqrt(RT/k_b).  This module quantifies the trade-off:

* :func:`stiffness_scan` - for each fixed k_b, fit eps_b to ln K_p(T) and
  report the dH error (and vice versa);
* :func:`solve_both` - the (eps_b, k_b) that matches ln K_p and dH exactly at
  one temperature, with the bond frequency and MD time step it would imply;
* :func:`entropy_budget` - the reaction entropy of the model versus experiment;
* :func:`anchor_orientational`, :func:`width_scan` - the same questions for a
  Kern-Frenkel potential, where the patch width w (orientational entropy)
  replaces the bond width as the second handle and k_b stays physical;
* :func:`per_temperature_anchoring` - eps_b (and w) re-anchored at every
  temperature separately: a T-independent potential that describes the
  reaction needs the same parameters at every T.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, fsolve

from ..constants import C_CM, NA, R_KJ
from ..thermodynamics.equilibrium_integrals import bound_state


def _fit_eb_to_lnK(pot, kb, T, lnK_ref, w, eb_lo=1.0, eb_hi=400.0):
    """Weighted least-squares eps_b at fixed curvature (1-D, bounded, golden + Brent)."""
    from scipy.optimize import minimize_scalar

    def cost(eb):
        try:
            p = pot.with_bond(eb=eb, kb=kb)
            if not p.landscape().clean:
                return 1e12
            return float(np.sum(w * (np.array([bound_state(p, t).log_Kp for t in T]) - lnK_ref) ** 2))
        except Exception:
            return 1e12

    r = minimize_scalar(cost, bounds=(eb_lo, eb_hi), method="bounded", options=dict(xatol=1e-6))
    return float(r.x), pot.with_bond(eb=r.x, kb=kb)


def _eb_for_dH(pot, kb, T, dH_ref):
    """eps_b giving dH(T) = dH_ref; bracketed around |dH_ref| (dH ~ -eps_b + RT/2 - RT)."""
    def g(eb):
        return bound_state(pot.with_bond(eb=eb, kb=kb), T).deltaH - dH_ref

    return brentq(g, 0.6 * abs(dH_ref), 1.6 * abs(dH_ref), xtol=1e-8)


def wavenumber(kb, mu_gmol):
    return float(np.sqrt(kb * 1e3 * 1e20 / (mu_gmol * 1e-3)) / (2 * np.pi * C_CM))


def period_fs(kb, mu_gmol):
    return 1e15 / (wavenumber(kb, mu_gmol) * C_CM)


def stiffness_scan(pot, ref, kbs, mu_gmol, T_dH=298.15):
    """Rows of {kb, eb_K, rms_lnK, dH_err, eb_H, lnK_err_at_T_dH, wavenumber, period_fs}."""
    T, lnK, s = ref["T"], ref["lnK"], ref["sigma_lnK"]
    w = 1.0 / s ** 2
    i_dH = int(np.argmin(np.abs(T - T_dH)))
    rows = []
    for kb in kbs:
        row = dict(kb=float(kb), wavenumber=wavenumber(kb, mu_gmol), period_fs=period_fs(kb, mu_gmol))
        try:
            # piecewise bridges cannot always be joined monotonically to an extremely stiff well;
            # the well itself (and hence Z_b) is still represented, but flag the shape
            row["shape_valid"] = bool(getattr(pot.with_bond(kb=kb), "feasible", True))
        except Exception:  # noqa: BLE001
            row["shape_valid"] = False
        try:
            ebK, pK = _fit_eb_to_lnK(pot, kb, T, lnK, w)
            pred = np.array([bound_state(pK, t).log_Kp for t in T])
            bs = bound_state(pK, T[i_dH])
            row.update(eb_K=ebK, rms_lnK=float(np.sqrt(np.mean((pred - lnK) ** 2))),
                       dH_model_K=bs.deltaH, dH_err=bs.deltaH - ref["dH"][i_dH],
                       dS_model_K=bs.deltaS)
        except Exception as e:  # noqa: BLE001
            row.update(error_K=str(e))
        try:
            ebH = _eb_for_dH(pot, kb, T[i_dH], ref["dH"][i_dH])
            bsH = bound_state(pot.with_bond(eb=ebH, kb=kb), T[i_dH])
            row.update(eb_H=ebH, lnK_err_at_T=bsH.log_Kp - lnK[i_dH])
        except Exception as e:  # noqa: BLE001
            row.update(error_H=str(e))
        rows.append(row)
    return rows


def solve_both(pot, T, lnK_ref, dH_ref, mu_gmol, guess=(56.0, 1e10)):
    """(eps_b, k_b) matching ln K_p(T) and dH(T) simultaneously (log-space Newton)."""

    def F(x):
        eb, lkb = x
        try:
            bs = bound_state(pot.with_bond(eb=eb, kb=np.exp(lkb)), T)
            return [bs.log_Kp - lnK_ref, bs.deltaH - dH_ref]
        except Exception:  # noqa: BLE001 - shape parameter out of range for this bond form
            return [1e3, 1e3]

    x, info, ier, msg = fsolve(F, [guess[0], np.log(guess[1])], full_output=True, xtol=1e-12)
    eb, kb = float(x[0]), float(np.exp(x[1]))
    try:
        p = pot.with_bond(eb=eb, kb=kb)
    except Exception as e:  # noqa: BLE001
        return dict(eb=eb, kb=kb, converged=False, error=str(e), shape_valid=False,
                    thermal_width_A=float(np.sqrt(R_KJ * T / kb)), wavenumber=wavenumber(kb, mu_gmol),
                    period_fs=period_fs(kb, mu_gmol), md_timestep_fs_20pts=period_fs(kb, mu_gmol) / 20.0)
    width = np.sqrt(R_KJ * T / kb)
    return dict(eb=eb, kb=kb, converged=ier == 1, shape_valid=bool(getattr(p, "feasible", True)), residual=list(map(float, F(x))),
                thermal_width_A=float(width), wavenumber=wavenumber(kb, mu_gmol),
                period_fs=period_fs(kb, mu_gmol), md_timestep_fs_20pts=period_fs(kb, mu_gmol) / 20.0,
                r_bond=p.r_bond)


def entropy_budget(pot, T, dS_ref):
    """Model reaction entropy and the deficit relative to experiment (J/mol/K)."""
    bs = bound_state(pot, T)
    return dict(T=T, dS_model=bs.deltaS, dS_ref=dS_ref, dS_missing=dS_ref - bs.deltaS,
                lnK_equivalent=(dS_ref - bs.deltaS) / (R_KJ * 1e3))


# ------------------------------------------------------------ Kern-Frenkel (orientational) handle
def _eb_for_lnK(pot, T, lnK_ref, kb=None, span=(0.3, 3.0)):
    from .barrier_scan import compensate_eb

    return compensate_eb(pot, T, lnK_ref, kb=kb, span=span)


def anchor_orientational(pot, T, lnK_ref, dH_ref, kb=None, w_bounds=(0.05, 1.5)):
    """(eps_b, w) of a Kern-Frenkel potential with ln K_p(T) and dH(T) both exact at fixed k_b.

    Nested 1-D roots: eps_b(w) restores ln K for every trial width, and w is
    then solved for dH (monotonic: a wider patch keeps more orientational
    entropy, so a shallower well gives the same K and a less negative dH)."""
    from .barrier_scan import robust_root

    def at(w):
        return _eb_for_lnK(pot.replace(w=w), T, lnK_ref, kb=kb)

    def g(lw):
        try:
            return bound_state(at(np.exp(lw)), T).deltaH - dH_ref
        except ValueError:
            return np.nan

    lw = robust_root(g, np.log(w_bounds[0]), np.log(w_bounds[1]), x0=np.log(pot.p["w"]), n=25)
    return at(float(np.exp(lw)))


def width_scan(pot, ref, ws, T_dH=298.15, kb=None):
    """Kern-Frenkel analogue of :func:`stiffness_scan`: k_b fixed, patch width w scanned.

    For each w: eps_b fitted to ln K_p(T) (dH error at T_dH) and eps_b fitted to dH (ln K error)."""
    T, lnK, s = ref["T"], ref["lnK"], ref["sigma_lnK"]
    w8 = 1.0 / s ** 2
    i_dH = int(np.argmin(np.abs(T - T_dH)))
    kb = kb if kb is not None else _nominal_kb(pot)
    rows = []
    for w in ws:
        q = pot.replace(w=float(w))
        row = dict(w=float(w), w_deg=float(np.rad2deg(w)))
        try:
            ebK, pK = _fit_eb_to_lnK(q, kb, T, lnK, w8)
            pred = np.array([bound_state(pK, t).log_Kp for t in T])
            bs = bound_state(pK, T[i_dH])
            row.update(eb_K=ebK, kappa_K=pK.angular_stiffness(), rms_lnK=float(np.sqrt(np.mean((pred - lnK) ** 2))),
                       dH_model_K=bs.deltaH, dH_err=bs.deltaH - ref["dH"][i_dH], dS_model_K=bs.deltaS,
                       shape_valid=bool(pK.feasible))
        except Exception as e:  # noqa: BLE001
            row.update(error_K=str(e))
        try:
            ebH = _eb_for_dH(q, kb, T[i_dH], ref["dH"][i_dH])
            bsH = bound_state(q.with_bond(eb=ebH, kb=kb), T[i_dH])
            row.update(eb_H=ebH, lnK_err_at_T=bsH.log_Kp - lnK[i_dH])
        except Exception as e:  # noqa: BLE001
            row.update(error_H=str(e))
        rows.append(row)
    return rows


def _nominal_kb(pot):
    from .fit_pair_potentials import nominal_kb

    return nominal_kb(pot)


def per_temperature_anchoring(pot, ref, kb=None, orientational=False):
    """Anchor the potential separately at every reference temperature.

    For each T_i: eps_b(T_i) with ln K_p(T_i) exact (k_b and, for Kern-Frenkel,
    w kept); with ``orientational`` also (eps_b, w)(T_i) with ln K_p and dH
    exact.  Each anchored potential is then evaluated at every T_j, giving
    the transfer matrix of ln K errors (row i: anchored at T_i)."""
    T, lnK, dH = ref["T"], ref["lnK"], ref["dH"]
    kb = kb if kb is not None else _nominal_kb(pot)
    rows, M = [], np.full((len(T), len(T)), np.nan)
    for i, t in enumerate(T):
        row = dict(T=float(t), lnK_ref=float(lnK[i]), dH_ref=float(dH[i]))
        try:
            p = _eb_for_lnK(pot, t, lnK[i], kb=kb, span=(0.03, 3.0))
            bs = bound_state(p, t)
            row.update(eb=p.p["eb"], U_bond=p.u_bond, dH=bs.deltaH, dH_err=bs.deltaH - dH[i], dS=bs.deltaS,
                       dCp=bs.deltaCp)
            M[i] = [bound_state(p, tj).log_Kp - lj for tj, lj in zip(T, lnK)]
        except ValueError as e:
            row["error"] = str(e)
        if orientational:
            try:
                q = anchor_orientational(pot, t, lnK[i], dH[i], kb=kb)
                bq = bound_state(q, t)
                row.update(eb_KH=q.p["eb"], w_KH=q.p["w"], kappa_KH=q.angular_stiffness(),
                           dH_KH=bq.deltaH, dS_KH=bq.deltaS, dCp_KH=bq.deltaCp)
            except ValueError as e:
                row["error_KH"] = str(e)
        rows.append(row)
    ebs = np.array([r.get("eb", np.nan) for r in rows])
    summary = dict(
        eb_min=float(np.nanmin(ebs)), eb_max=float(np.nanmax(ebs)), eb_span=float(np.nanmax(ebs) - np.nanmin(ebs)),
        deb_dT=float(np.polyfit(T, ebs, 1)[0]) if np.all(np.isfinite(ebs)) else np.nan,
        rms_dH_err=float(np.sqrt(np.nanmean([r.get("dH_err", np.nan) ** 2 for r in rows]))),
        rms_lnK_transfer=[float(np.sqrt(np.nanmean(m ** 2))) for m in M],
    )
    if orientational:
        wk = np.array([r.get("w_KH", np.nan) for r in rows])
        ek = np.array([r.get("eb_KH", np.nan) for r in rows])
        summary.update(w_KH_min=float(np.nanmin(wk)), w_KH_max=float(np.nanmax(wk)),
                       eb_KH_min=float(np.nanmin(ek)), eb_KH_max=float(np.nanmax(ek)))
    return dict(T=T, rows=rows, transfer_lnK_err=M, summary=summary)
