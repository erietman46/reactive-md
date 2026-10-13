"""Three-particle energy surfaces: required penalty strength, worst trimer, traps.

Geometry convention: particle 1 is the vertex, r12 = a, r13 = b, angle theta at
particle 1, r23 = sqrt(a^2 + b^2 - 2ab cos theta).  Energies are reported
relative to an isolated bonded pair at its minimum plus a free monomer:

    dE(a, b, theta) = U2(a) + U2(b) + U2(r23) + E_mb(a, b, r23)
                      - [U2(r_b) + E_mb_pair(r_b)]

dE < 0 means that adding a second bond (a trimer) is energetically favourable
relative to a *free* monomer.  The penalty is sized against a stricter but
physically fairer reference, the dimer-monomer van der Waals complex (third
particle outside the bonded region of both partners, :func:`vdw_reference`):
a two-bond geometry is unwanted if it lies below that complex.  Using the free
monomer instead makes the required strength diverge when the barrier is low,
because a particle at the barrier top of one partner can sit in the
dispersion well of the other.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..constants import R_KJ


def third_side(a, b, theta):
    return np.sqrt(np.maximum(a * a + b * b - 2 * a * b * np.cos(theta), 1e-12))


def addition_energy(pair, mb, r13, theta, r12=None, with_mb=True):
    """Energy of adding particle 3 to a bonded pair (r12 = r_b), relative to pair + free monomer."""
    rb = pair.r_bond if r12 is None else r12
    c = third_side(rb, r13, theta)
    e = pair.U(r13) + pair.U(c)
    if with_mb and mb is not None:
        e = e + mb.triplet_energy(np.full_like(np.asarray(r13, float), rb), r13, c) - mb.pair_energy(rb)
    return e


def vdw_reference(pair, n_b=300, n_t=181):
    """Lowest energy of dimer (at r_b) + third particle outside both bonded regions,
    relative to dimer + free monomer: (E_vdw <= 0, (r13, r23, theta_deg))."""
    rb, edge = pair.r_bond, pair.r_barrier
    b = np.linspace(edge, edge + 4.0, n_b)
    th = np.deg2rad(np.linspace(0.0, 180.0, n_t))
    B, Th = np.meshgrid(b, th, indexing="ij")
    C = third_side(rb, B, Th)
    E = np.where(C >= edge, pair.U(B) + pair.U(C), np.inf)
    i = np.unravel_index(np.argmin(E), E.shape)
    e = float(min(E[i], 0.0))
    return e, (float(B[i]), float(C[i]), float(np.rad2deg(Th[i])))


def _scan_points(pair, fine=True):
    """Points (a, b, theta) of the explorer's global trimer scan (both bonds inside the bonded region)."""
    rb, edge = pair.r_bond, pair.r_barrier
    lo = 0.75 * rb
    dr, dth = (0.005, 2.0) if fine else (0.01, 5.0)
    th1 = np.deg2rad(np.arange(60.0, 180.0 + 1e-9, dth))
    b1 = np.arange(lo, edge + 1e-12, dr)
    B1, T1 = np.meshgrid(b1, th1, indexing="ij")
    A1 = np.full_like(B1, rb)
    da, dg = (0.025, 4.0) if fine else (0.05, 6.0)
    ax = np.arange(lo, edge + 1e-9, da)
    th2 = np.deg2rad(np.arange(20.0, 180.0 + 1e-9, dg))
    A2, B2, T2 = np.meshgrid(ax, ax, th2, indexing="ij")
    keep = B2 >= A2
    a = np.r_[A1.ravel(), A2[keep]]
    b = np.r_[B1.ravel(), B2[keep]]
    t = np.r_[T1.ravel(), T2[keep]]
    return a, b, t


@dataclass
class StrengthAnalysis:
    lam_min: float          # smallest strength making every two-bond geometry cost energy
    at: tuple | None        # (a, b, theta_deg) setting lam_min
    unfixable: float        # most negative dE of geometries the penalty cannot see (0 if none)
    unfixable_at: tuple | None
    worst_pair_only: tuple  # (dE, a, b, theta_deg) without the many-body term, vs free monomer
    reference: str = "vdw"
    E_ref: float = 0.0      # reference energy relative to dimer + free monomer


def required_strength(pair, mb, reference="vdw") -> StrengthAnalysis:
    """Smallest strength for which every two-bond geometry lies above the reference.

    reference = "free": dimer + free monomer (the explorer's tbAnalyse);
    reference = "vdw":  dimer + monomer in its best non-bonded position (default).
    Valid for terms that are linear in their strength."""
    a, b, t = _scan_points(pair)
    c = third_side(a, b, t)
    E_ref = vdw_reference(pair)[0] if reference == "vdw" else 0.0
    ref = pair.U(pair.r_bond) + E_ref
    E2 = pair.U(a) + pair.U(b) + pair.U(c) - ref
    unit = mb.with_strength(1.0)
    E3 = unit.triplet_energy(a, b, c) - unit.pair_energy(pair.r_bond)
    i0 = int(np.argmin(E2))
    worst0 = (float(E2[i0] + E_ref), float(a[i0]), float(b[i0]), float(np.rad2deg(t[i0])))
    neg = E2 < 0
    fix = neg & (E3 > 1e-300)
    lam = np.zeros_like(E2)
    lam[fix] = -E2[fix] / E3[fix]
    j = int(np.argmax(lam))
    need = float(lam[j]) if fix.any() else 0.0
    at = (float(a[j]), float(b[j]), float(np.rad2deg(t[j]))) if fix.any() else None
    bad = neg & ~(E3 > 1e-300)
    if bad.any():
        k = np.nonzero(bad)[0][np.argmin(E2[bad])]
        unf, unf_at = float(E2[k]), (float(a[k]), float(b[k]), float(np.rad2deg(t[k])))
    else:
        unf, unf_at = 0.0, None
    return StrengthAnalysis(need, at, unf, unf_at, worst0, reference, E_ref)


def worst_trimer(pair, mb):
    """Lowest dE over the trimer scan with the many-body term at its current strength."""
    a, b, t = _scan_points(pair)
    c = third_side(a, b, t)
    e = pair.U(a) + pair.U(b) + pair.U(c) - pair.U(pair.r_bond)
    if mb is not None:
        e = e + mb.triplet_energy(a, b, c) - mb.pair_energy(pair.r_bond)
    i = int(np.argmin(e))
    return float(e[i]), float(a[i]), float(b[i]), float(np.rad2deg(t[i]))


@dataclass
class Trap:
    r12: float
    r13: float
    r23: float
    theta_deg: float
    dE: float          # energy of the trapped trimer relative to dimer + free monomer
    barrier: float     # lowest escape barrier to a state with one particle clearly gone
    lifetime_s: float  # rough 0.1 ps * exp(barrier / RT)


def trap_check(pair, mb, T=300.0, da=0.025, dth=4.0, n_report=5):
    """Find local trimer minima and their lowest escape barriers (minimax path).

    The surface is evaluated on a grid over (a, b, theta). A local minimum with
    both bonds inside the bonded region (a, b <= r_barrier) is a trimer state.
    Its escape saddle is the lowest energy level at which the flooded
    (26-connected) basin containing it reaches a grid point where one particle
    has clearly left (max(a, b) > r_barrier + 0.8 A).
    """
    rb, edge = pair.r_bond, pair.r_barrier
    ax = np.arange(0.85 * rb, edge + 1.2 + 1e-9, da)
    th = np.deg2rad(np.arange(20.0, 180.0 + 1e-9, dth))
    A, Bm, Th = np.meshgrid(ax, ax, th, indexing="ij")
    C = third_side(A, Bm, Th)
    ref = pair.U(rb) + (mb.pair_energy(rb) if mb is not None else 0.0)
    G = pair.U(A) + pair.U(Bm) + pair.U(C) - ref
    if mb is not None:
        G = G + mb.triplet_energy(A, Bm, C)
    G = np.where(np.isfinite(G), G, 1e30)
    escaped = np.maximum(A, Bm) > edge + 0.8
    fp = np.ones((3, 3, 3), bool)
    local_min = G <= ndimage.minimum_filter(G, footprint=fp, mode="nearest")
    cand = local_min & (A <= edge) & (Bm <= edge) & (Bm >= A)
    cand[0, :, :] = cand[-1, :, :] = False
    cand[:, 0, :] = cand[:, -1, :] = False
    idx = np.argwhere(cand)
    if idx.size == 0:
        return []
    e0s = G[tuple(idx.T)]
    order = np.argsort(e0s)[: n_report]
    levels = np.unique(G)
    out = []
    for o in order:
        i, j, k = idx[o]
        e0 = G[i, j, k]
        lo = int(np.searchsorted(levels, e0))
        hi = len(levels) - 1

        def connected(level):
            lab, _ = ndimage.label(G <= level, structure=fp)
            l0 = lab[i, j, k]
            return l0 != 0 and np.any(escaped & (lab == l0))

        if not connected(levels[hi]):
            barrier = np.inf
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                if connected(levels[mid]):
                    hi = mid
                else:
                    lo = mid + 1
            barrier = float(levels[lo] - e0)
        if barrier > 0.5:
            a_, b_, t_ = ax[i], ax[j], th[k]
            out.append(Trap(float(a_), float(b_), float(third_side(a_, b_, t_)), float(np.rad2deg(t_)),
                            float(e0), barrier, float(1e-13 * np.exp(min(barrier / (R_KJ * T), 700)))))
    return out
