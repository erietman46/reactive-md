"""Three-particle cluster integrals: trimer association constant and Delta B3.

Particle 1 at the origin, r12 = a, r13 = b, r23 = c.  For a permutation-
symmetric integrand f, the integral over configurations in which at least one
particle has both partners within a distance R is (inclusion-exclusion over
the three possible vertex particles)

    J_R[f] = 3 I_v - 2 I_tri,
    I_v   = int_{a<R} int_{b<R} f d^3r2 d^3r3,
    I_tri = same with additionally c < R,

    d^3r2 d^3r3 = (4 pi a^2 da) (2 pi b^2 db) d(cos theta).

Two quantities are evaluated with this:

* the dilute-gas trimer association constant (bond = pair inside r_div)

      K3 = rho_A3 / rho_A^3 = (1/3!) J_{r_div}[exp(-beta(U_12 + U_13 + U_23 + E_mb))]   [A^6]

  which, with K_V = K2, gives the trimer content at any (dilute) density;

* the many-body change of the third virial coefficient

      Delta B3 = -(1/3) J_reach[exp(-beta sum U2) (exp(-beta E_mb) - 1)]       [A^6]

  (>= 0 for a non-negative penalty; zero integrand unless some particle has two
  neighbours inside the penalty's reach).

The integrand is sharply peaked (thermal bond width ~ sqrt(RT/k_b)), so all
three coordinates use composite Gauss-Legendre panels refined around the bond
length; the angular panels are refined around the "third bond" configuration
c = r_b for every (a, b).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import A3_TO_CM3, NA, R_KJ
from ..thermodynamics.equilibrium_integrals import bound_state, dimer_fraction

_NG = 6
_X, _W = np.polynomial.legendre.leggauss(_NG)


def _gl(edges):
    edges = np.asarray(edges, float)
    a, b = edges[..., :-1, None], edges[..., 1:, None]
    h = 0.5 * (b - a)
    x = 0.5 * (a + b) + h * _X
    w = h * _W
    return x.reshape(*edges.shape[:-1], -1), w.reshape(*edges.shape[:-1], -1)


def _radial_nodes(pair, T, R, n_coarse=28, n_fine=18, n_sig=10.0):
    rb = pair.r_bond
    k = pair.bond_stiffness()
    w = np.sqrt(R_KJ * T / k) if k > 0 else 0.05
    beta = 1 / (R_KJ * T)
    r = np.linspace(0.3, rb, 3000)
    x = beta * (pair.U(r) - pair.u_bond)
    ok = np.nonzero(x < 60)[0]
    r_lo = r[max(ok[0] - 1, 0)] if ok.size else 0.5 * rb
    e = list(np.linspace(r_lo, R, n_coarse + 1))
    lo, hi = max(r_lo, rb - n_sig * w), min(R, rb + n_sig * w)
    if hi > lo:
        e += list(np.linspace(lo, hi, n_fine + 1))
    return _gl(np.unique(e)), w


def _cos_nodes(a, b, rb, w, n_coarse=32, n_fine=20, n_sig=10.0):
    """Per-(a, b) angular nodes in cos(theta), refined where c = r_b."""
    cstar = (a * a + b * b - rb * rb) / (2 * a * b)
    dc = n_sig * w * rb / (a * b)
    coarse = np.broadcast_to(np.linspace(-1, 1, n_coarse + 1), a.shape + (n_coarse + 1,))
    fine = np.clip(cstar[..., None] + dc[..., None] * np.linspace(-1, 1, n_fine + 1), -1, 1)
    edges = np.sort(np.concatenate([coarse, fine], axis=-1), axis=-1)
    return _gl(edges)


RESOLUTION = dict(n_coarse=16, n_fine=10, c_coarse=16, c_fine=12)  # converged to ~1e-7 in ln K3 (see tests)


def _vertex_integrals(pair, T, R, log_f, chunk=16, res=None):
    """Return (log I_v, log I_tri) for an integrand given as log f(a, b, c) (vectorised)."""
    res = res or RESOLUTION
    (ra, wa), w = _radial_nodes(pair, T, R, res["n_coarse"], res["n_fine"])
    keep = ra < R
    ra, wa = ra[keep], wa[keep]
    rb = pair.r_bond
    logs_v, logs_t = [], []
    for i0 in range(0, ra.size, chunk):
        a = ra[i0:i0 + chunk][:, None]
        b = ra[None, :]
        A, B = np.broadcast_arrays(a, b)
        cx, cw = _cos_nodes(A, B, rb, w, res["c_coarse"], res["c_fine"])
        Aa, Bb = A[..., None], B[..., None]
        C = np.sqrt(np.maximum(Aa * Aa + Bb * Bb - 2 * Aa * Bb * cx, 1e-12))
        lf = log_f(Aa, Bb, C)
        with np.errstate(divide="ignore"):
            lw = (np.log(4 * np.pi * Aa ** 2 * wa[i0:i0 + chunk][:, None, None])
                  + np.log(2 * np.pi * Bb ** 2 * wa[None, :, None]) + np.log(cw))
        tot = lf + lw
        logs_v.append(tot.ravel())
        logs_t.append(np.where(C < R, tot, -np.inf).ravel())
    lv, lt = np.concatenate(logs_v), np.concatenate(logs_t)

    def lse(x):
        m = np.max(x)
        return -np.inf if not np.isfinite(m) else m + np.log(np.sum(np.exp(x - m)))

    return lse(lv), lse(lt)


def _combine(log_v, log_t):
    """log(3 I_v - 2 I_tri) (I_tri <= I_v so the result is positive)."""
    if not np.isfinite(log_t):
        return np.log(3.0) + log_v
    return log_v + np.log(3.0 - 2.0 * np.exp(log_t - log_v))


@dataclass
class TrimerEquilibrium:
    T: float
    log_K2: float   # ln(K_V / A^3)
    log_K3: float   # ln(K3 / A^6)

    def fractions(self, rho0):
        """Fractions of monomer units in dimers and trimers at monomer-unit density rho0 (A^-3),
        dilute mass action rho0 = rho1 + 2 K2 rho1^2 + 3 K3 rho1^3 solved for rho1."""
        K2, K3 = np.exp(self.log_K2), np.exp(self.log_K3)
        lo, hi = 0.0, rho0
        for _ in range(200):
            m = 0.5 * (lo + hi)
            if m + 2 * K2 * m * m + 3 * K3 * m ** 3 > rho0:
                hi = m
            else:
                lo = m
        r1 = 0.5 * (lo + hi)
        return dict(monomer=r1 / rho0, dimer=2 * K2 * r1 ** 2 / rho0, trimer=3 * K3 * r1 ** 3 / rho0)


def trimer_equilibrium(pair, T, mb=None, res=None) -> TrimerEquilibrium:
    beta = 1 / (R_KJ * T)
    R = pair.r_barrier

    def log_f(a, b, c):
        e = pair.U(a) + pair.U(b) + pair.U(c)
        if mb is not None:
            e = e + mb.triplet_energy(a, b, c)
        return -beta * e

    lv, lt = _vertex_integrals(pair, T, R, log_f, res=res)
    log_K3 = _combine(lv, lt) - np.log(6.0)
    # K2 includes the isolated-pair contribution of the many-body term (zero for exact hinges)
    extra = None if mb is None else mb.pair_energy
    log_K2 = bound_state(pair, T, extra_energy=extra).log_KV
    return TrimerEquilibrium(T, float(log_K2), float(log_K3))


def delta_B3(pair, mb, T, res=None):
    """Many-body contribution to B3 in (cm^3/mol)^2 (>= 0 for non-negative penalties)."""
    beta = 1 / (R_KJ * T)
    R = max(mb.reach, pair.r_bond + 0.05)

    def log_f(a, b, c):
        e2 = pair.U(a) + pair.U(b) + pair.U(c)
        e3 = mb.triplet_energy(a, b, c)
        g = -np.expm1(-beta * e3)  # 1 - exp(-beta E3) >= 0
        with np.errstate(divide="ignore"):
            return -beta * e2 + np.log(np.maximum(g, 0.0))

    lv, lt = _vertex_integrals(pair, T, R, log_f, res=res)
    if not np.isfinite(lv):
        return 0.0
    val_log = _combine(lv, lt) - np.log(3.0)
    return float(np.exp(val_log) * (NA * A3_TO_CM3) ** 2)


def dimer_fraction_from(pair, T, rho0):
    return float(dimer_fraction(np.exp(bound_state(pair, T).log_KV), rho0))
