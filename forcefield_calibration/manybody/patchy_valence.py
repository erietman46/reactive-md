"""Valence of the smooth Kern-Frenkel model: can one patch hold two bonds?

A Kern-Frenkel bead has a single patch.  A second partner can only bond if the
patch points at both partners at once, so a narrow patch limits the valence by
itself (the classical single-bond-per-patch condition of patchy particles).
This module quantifies that with the same three-particle tools as the
isotropic coordination penalties (``trimer_energy_surface``, ``third_virial``):

* :func:`second_bond_map` - energy of adding particle 3 to a bonded pair 1-2,
  relative to the pair + a free monomer, with the partner patches pointing at
  particle 1 and the patch of particle 1 rotated (in the plane) to its best
  direction for every (r13, theta);
* :func:`relaxed_trimer` - lowest trimer energy with both bonds inside the
  bonded region, minimised over all positions *and* all three orientations
  (multi-start), for comparison with the dimer-monomer van der Waals complex;
* :func:`trimer_equilibrium_kf` - a rigorous upper bound on the trimer
  association constant K3.  Since 1 - f_i f_j >= [(1 - f_i) + (1 - f_j)]/2,
  the angular energy of a triplet is bounded below by a sum of single-patch
  terms, sum_k sum_{l != k} W(r_kl) (1 - f(n_k . e_kl)) / 2, which factorises
  over the three patches.  Each patch is then averaged exactly over its own
  two neighbours (a tabulated 2-D sphere integral H(x, y, cos gamma)), and the
  bound is symmetric, so the inclusion-exclusion integral of
  ``third_virial`` applies unchanged:

      K3 <= (1/3!) J[exp(-beta sum U_on) H_1 H_2 H_3]

  The same bound applied to an isolated pair (:func:`pair_bound_log_KV`)
  shows how loose it is: each bond is overestimated by the factor
  K_V(bound)/K_V, so K3 is overestimated by roughly the square of it.

An optional isotropic many-body term (one of ``coordination_penalties``) can
be added on top; its triplet energy depends on distances only.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from ..constants import R_KJ
from ..potentials.kern_frenkel import patch_factor
from .third_virial import TrimerEquilibrium, _combine, _vertex_integrals
from ..thermodynamics.equilibrium_integrals import bound_state

_GLX, _GLW = np.polynomial.legendre.leggauss(8)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def trimer_energy(kf, x2, x3, n1, n2, n3):
    """Total pair energy of particles at 0, x2, x3 with patch vectors n1, n2, n3 (vectorised over leading axes)."""
    z = np.zeros_like(x2)
    e12 = kf.energy(x2 - z, n1, n2)[0]
    e13 = kf.energy(x3 - z, n1, n3)[0]
    e23 = kf.energy(x3 - x2, n2, n3)[0]
    return e12 + e13 + e23


def second_bond_map(kf, r13, theta, mb=None, n_phi=361):
    """dE(r13, theta) of adding particle 3 to the aligned pair 1-2 (r12 = r_b), relative to pair + free monomer.

    Partner patches point at particle 1; the patch of particle 1 is rotated in the
    plane of the three particles to the angle that minimises the energy."""
    r13, theta = np.broadcast_arrays(np.asarray(r13, float), np.asarray(theta, float))
    rb = kf.r_bond
    shp = r13.shape
    a = r13.ravel()
    t = theta.ravel()
    x2 = np.broadcast_to(np.array([rb, 0.0, 0.0]), (a.size, 3))
    x3 = np.stack([a * np.cos(t), a * np.sin(t), np.zeros_like(a)], axis=-1)
    n2 = np.broadcast_to(np.array([-1.0, 0.0, 0.0]), x2.shape)
    n3 = -_unit(x3)
    best = np.full(a.size, np.inf)
    for phi in np.linspace(-np.pi, np.pi, n_phi):
        n1 = np.broadcast_to(np.array([np.cos(phi), np.sin(phi), 0.0]), x2.shape)
        best = np.minimum(best, trimer_energy(kf, x2, x3, n1, n2, n3))
    e = best - kf.U(rb)
    if mb is not None:
        c = np.linalg.norm(x3 - x2, axis=-1)
        e = e + mb.triplet_energy(np.full_like(a, rb), a, c) - mb.pair_energy(rb)
    return e.reshape(shp)


@dataclass
class RelaxedTrimer:
    dE: float              # lowest trimer energy relative to aligned dimer + free monomer (kJ/mol)
    r12: float
    r13: float
    theta_deg: float
    patch_angles_deg: tuple  # angle of each patch to its nearest partner direction
    n_starts: int


def relaxed_trimer(kf, mb=None, n_starts=60, seed=7):
    """Lowest energy of a trimer with both bonds on particle 1 inside the bonded region (r12, r13 <= r_div),
    minimised over (r12, r13, theta) and all three patch orientations."""
    rng = np.random.default_rng(seed)
    rb, R = kf.r_bond, kf.r_barrier
    e_ref = kf.U(rb) + (mb.pair_energy(rb) if mb is not None else 0.0)

    def unpack(x):
        a, b, t = x[:3]
        ang = x[3:].reshape(3, 2)
        n = np.stack([np.sin(ang[:, 0]) * np.cos(ang[:, 1]), np.sin(ang[:, 0]) * np.sin(ang[:, 1]),
                      np.cos(ang[:, 0])], axis=-1)
        x2 = np.array([a, 0.0, 0.0])
        x3 = np.array([b * np.cos(t), b * np.sin(t), 0.0])
        return x2, x3, n

    def f(x):
        x2, x3, n = unpack(x)
        e = float(trimer_energy(kf, x2[None], x3[None], n[0][None], n[1][None], n[2][None])[0])
        if mb is not None:
            c = np.linalg.norm(x3 - x2)
            e += float(mb.triplet_energy(np.array([x[0]]), np.array([x[1]]), np.array([c]))[0])
        return e - e_ref if np.isfinite(e) else 1e6

    bounds = [(0.8 * rb, R), (0.8 * rb, R), (np.deg2rad(20.0), np.pi)] + [(0.0, np.pi), (-np.pi, np.pi)] * 3
    best = None
    for k in range(n_starts):
        x0 = np.array([rng.uniform(rb - 0.1, rb + 0.2), rng.uniform(rb - 0.1, R), rng.uniform(0.3, np.pi)]
                      + [v for _ in range(3) for v in (np.arccos(rng.uniform(-1, 1)), rng.uniform(-np.pi, np.pi))])
        if k == 0:  # symmetric start: partners point at 1, patch 1 bisects
            x0 = np.array([rb, rb, 2.0, np.pi / 2, 1.0, np.pi / 2, np.pi, np.pi / 2, 2.0 - np.pi])
        r = minimize(f, x0, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    x2, x3, n = unpack(best.x)
    pos = [np.zeros(3), x2, x3]

    def ang(i):
        d = [_unit(pos[j] - pos[i]) for j in range(3) if j != i]
        return float(np.rad2deg(min(np.arccos(np.clip(n[i] @ v, -1, 1)) for v in d)))

    return RelaxedTrimer(float(best.fun), float(best.x[0]), float(best.x[1]), float(np.rad2deg(best.x[2])),
                         tuple(ang(i) for i in range(3)), n_starts)


# ------------------------------------------------------------ trimer constant (upper bound)
_LNX = np.linspace(np.log(1e-3), np.log(3e4), 28)


def _two_neighbour_table(w, cos_grid, n_alpha=44, n_psi=40):
    """ln H(x, y, cos g) = ln < exp(-x (1 - f(n.e_a)) - y (1 - f(n.e_b))) >_n for one patch with two neighbours
    at angle g; x, y on the grid exp(_LNX).  The sphere is split into the halves closer to e_a and to e_b, each
    integrated in polar coordinates about its own axis (refined at small polar angle)."""
    edges = np.unique(np.r_[0.0, np.geomspace(1e-6, 2.0, n_alpha)])  # in 1 - cos(alpha)
    h = 0.5 * np.diff(edges)[:, None]
    u = (0.5 * (edges[:-1] + edges[1:])[:, None] + h * _GLX).ravel()
    wu = (h * _GLW).ravel()
    ca = 1.0 - u
    sa = np.sqrt(np.maximum(1.0 - ca * ca, 0.0))
    psi = (np.arange(n_psi) + 0.5) * np.pi / n_psi
    X = np.exp(_LNX)
    out = np.empty((X.size, X.size, cos_grid.size))
    for k, cg in enumerate(cos_grid):
        sg = np.sqrt(max(1.0 - cg * cg, 0.0))
        cb = ca[:, None] * cg + sa[:, None] * sg * np.cos(psi)[None, :]
        half = np.where(ca[:, None] > cb, 1.0, np.where(ca[:, None] == cb, 0.5, 0.0))  # ties: e_a = e_b
        near = half > 0
        wt = (wu[:, None] * (np.pi / n_psi) * 2.0 / (4.0 * np.pi)) * half
        Fa = np.broadcast_to((1.0 - patch_factor(ca, w)[0])[:, None], cb.shape)[near]
        Fb = (1.0 - patch_factor(cb, w)[0])[near]
        wv = wt[near]
        # half closer to e_a: (x on F_a, y on F_b); mirror half closer to e_b: (x on F_b, y on F_a)
        e1 = -X[:, None, None] * Fa[None, None, :] - X[None, :, None] * Fb[None, None, :]
        e2 = -X[:, None, None] * Fb[None, None, :] - X[None, :, None] * Fa[None, None, :]
        m = np.maximum(e1.max(axis=-1), e2.max(axis=-1))
        sm = (np.exp(e1 - m[..., None]) + np.exp(e2 - m[..., None])) @ wv
        out[:, :, k] = m + np.log(sm)
    return out


@lru_cache(maxsize=8)
def _h_interp(w):
    cos_grid = np.linspace(-1.0, 1.0, 31)
    tab = _two_neighbour_table(w, cos_grid)
    return RegularGridInterpolator((_LNX, _LNX, cos_grid), tab)


def _ln_h(w, x, y, cg):
    """ln H for arrays x, y (>= 0) and cos g, with x, y clipped to the table (H -> 1 for x, y -> 0)."""
    lo, hi = np.exp(_LNX[0]), np.exp(_LNX[-1])
    X, Y, C = np.broadcast_arrays(x, y, cg)
    pts = np.stack([np.log(np.clip(X, lo, hi)), np.log(np.clip(Y, lo, hi)), np.clip(C, -1, 1)], axis=-1)
    return _h_interp(float(w))(pts.reshape(-1, 3)).reshape(X.shape)


def pair_bound_log_KV(kf, T):
    """ln K_V of the same symmetric bound applied to an isolated pair (measures how loose the bound is)."""
    beta = 1.0 / (R_KJ * T)
    w = kf.p["w"]
    one = lambda r: 2.0 * _ln_h(w, 0.5 * beta * kf.wall(r)[0], 0.0, 1.0)
    return bound_state(_Shifted(kf, one), T).log_KV


class _Shifted:
    """Isotropic view of a Kern-Frenkel potential whose angular term is replaced by ``log_weight(r)``."""

    def __init__(self, kf, log_weight):
        self.kf, self.lw = kf, log_weight

    def __getattr__(self, k):
        return getattr(self.kf, k)

    def angular_terms(self, r, T):
        la = self.lw(np.asarray(r, float))
        z = np.zeros_like(la)
        return la, z, z


def trimer_equilibrium_kf(kf, T, mb=None, res=None) -> TrimerEquilibrium:
    """Upper bound on the dilute-gas trimer constant K3 of a Kern-Frenkel potential (module docstring).

    Uses 1 - f_i f_j >= [(1 - f_i) + (1 - f_j)] / 2, so that the angular energy is bounded below by a sum
    of single-patch terms; each patch then averages over its own two neighbours exactly (``_ln_h``)."""
    beta = 1.0 / (R_KJ * T)
    R = kf.r_barrier
    w = kf.p["w"]

    def log_f(a, b, c):
        e = kf.U(a) + kf.U(b) + kf.U(c)
        if mb is not None:
            e = e + mb.triplet_energy(a, b, c)
        A, B, C = np.broadcast_arrays(a, b, c)
        xa, xb, xc = (0.5 * beta * kf.wall(v)[0] for v in (A, B, C))
        c1 = (A * A + B * B - C * C) / (2 * A * B)   # angle at particle 1 (neighbours 2: a, 3: b)
        c2 = (A * A + C * C - B * B) / (2 * A * C)   # angle at particle 2 (neighbours 1: a, 3: c)
        c3 = (B * B + C * C - A * A) / (2 * B * C)   # angle at particle 3 (neighbours 1: b, 2: c)
        return -beta * e + _ln_h(w, xa, xb, c1) + _ln_h(w, xa, xc, c2) + _ln_h(w, xb, xc, c3)

    lv, lt = _vertex_integrals(kf, T, R, log_f, res=res)
    log_K3 = _combine(lv, lt) - np.log(6.0)
    extra = None if mb is None else mb.pair_energy
    log_K2 = bound_state(kf, T, extra_energy=extra).log_KV
    return TrimerEquilibrium(T, float(log_K2), float(log_K3))
