"""Bound-state configurational integrals and dilute-gas association equilibrium.

For two identical structureless particles with pair potential U(r) the bound
(dimer) configurational volume is

    Z_b(T) = 4 pi  int_0^{r_div} r^2 exp(-U(r)/RT) dr          [A^3]

with ``r_div`` the dividing surface (by default the barrier maximum r_double-dagger).
The dilute-gas association constant in number-density units is

    K_V = rho_A2 / rho_A^2 = Z_b / 2                           [A^3]

(the 1/2 is the symmetry number of a dimer of identical particles) and the
dimensionless standard-state constant is

    K_p = K_V p0 / (k_B T).

Very deep wells make exp(-U/RT) overflow, so all integrals are evaluated as
logarithms with the minimum energy factored out, on a composite Gauss-Legendre
grid that is refined around the bond minimum (resolution set by the thermal
width sqrt(RT/k_b), which can be as small as 1e-6 A in the attainability study).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import A3_TO_M3, KB, NA, P0, R_KJ

_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


def gl_nodes(edges):
    """Composite 8-point Gauss-Legendre nodes/weights for sorted panel edges."""
    edges = np.asarray(edges, float)
    a, b = edges[:-1, None], edges[1:, None]
    half = 0.5 * (b - a)
    x = (a + b) * 0.5 + half * _GL_X[None, :]
    w = half * _GL_W[None, :]
    return x.ravel(), w.ravel()


def _inner_limit(pot, T, r_hi, umin, cut=90.0):
    """Smallest r where beta (U - Umin) drops below ``cut`` (integrand negligible below)."""
    beta = 1.0 / (R_KJ * T)
    r = np.linspace(0.05, r_hi, 4000)
    with np.errstate(over="ignore", invalid="ignore"):
        x = beta * (pot.U(r) - umin)
    ok = np.nonzero(np.isfinite(x) & (x < cut))[0]
    if ok.size == 0:
        return r_hi
    return float(r[max(ok[0] - 1, 0)])


def bound_grid(pot, T, r_div=None, n_coarse=400, n_fine=160, n_sig=14.0):
    """Quadrature nodes covering the bound region [r_lo, r_div]."""
    L = pot.landscape()
    if L.bond is None:
        raise ValueError(f"{pot.name}: no bond minimum")
    rb, umin = L.bond.r, L.bond.u
    if r_div is None:
        if L.barrier is None:
            raise ValueError(f"{pot.name}: no barrier, dividing surface undefined")
        r_div = L.barrier.r
    k = pot.bond_stiffness()
    w = np.sqrt(R_KJ * T / k) if (np.isfinite(k) and k > 0) else 0.05
    r_lo = _inner_limit(pot, T, rb, umin)
    edges = list(np.linspace(r_lo, r_div, n_coarse + 1))
    lo, hi = max(r_lo, rb - n_sig * w), min(r_div, rb + n_sig * w)
    if hi > lo:
        edges += list(np.linspace(lo, hi, n_fine + 1))
    edges += [b for b in pot.breakpoints() if r_lo < b < r_div]
    edges = np.unique(np.asarray(edges))
    return gl_nodes(edges), umin, r_div


@dataclass
class BoundState:
    T: float
    r_div: float
    log_Zb: float      # ln(Z_b / A^3)
    mean_U: float      # <U>_b, kJ/mol
    var_U: float       # <U^2>_b - <U>_b^2, (kJ/mol)^2

    @property
    def log_KV(self):
        """ln(K_V / A^3)."""
        return self.log_Zb - np.log(2.0)

    @property
    def log_Kp(self):
        return self.log_KV + np.log(A3_TO_M3 * P0 / (KB * self.T))

    @property
    def Kc_L_per_mol(self):
        return np.exp(self.log_KV) * NA * 1e-27

    @property
    def deltaH(self):
        """Standard reaction enthalpy of 2A -> A2 (kJ/mol): <U>_b - RT."""
        return self.mean_U - R_KJ * self.T

    @property
    def deltaG(self):
        return -R_KJ * self.T * self.log_Kp

    @property
    def deltaS(self):
        """J/(mol K)."""
        return 1e3 * (self.deltaH - self.deltaG) / self.T

    @property
    def deltaCp(self):
        """Heat-capacity change d(dH)/dT = var(U)/(R T^2) - R  (J/(mol K))."""
        return 1e3 * (self.var_U / (R_KJ * self.T ** 2) - R_KJ)


def bound_state(pot, T, r_div=None, extra_energy=None) -> BoundState:
    """Configurational integrals of the bound basin at temperature T.

    ``extra_energy(r)`` (optional) is added to U, e.g. the isolated-pair
    contribution of a many-body penalty when checking pair invariance.
    """
    (r, w), umin, r_div = bound_grid(pot, T, r_div)
    U = pot.U(r)
    if extra_energy is not None:
        U = U + extra_energy(r)
    beta = 1.0 / (R_KJ * T)
    with np.errstate(over="ignore"):
        x = -beta * (U - umin)
    x = np.where(np.isfinite(x), x, -np.inf)
    xm = np.max(x)
    g = w * r * r * np.exp(x - xm)
    s = g.sum()
    live = g > 0  # ignore nodes deep in the repulsive wall (U may be ~1e300 there)
    mean = float((g[live] * U[live]).sum() / s)
    var = float((g[live] * (U[live] - mean) ** 2).sum() / s)
    log_Zb = float(np.log(4.0 * np.pi * s) + xm - beta * umin)
    return BoundState(T, float(r_div), log_Zb, mean, var)


def log_Kp(pot, T, **kw):
    return bound_state(pot, T, **kw).log_Kp


def log_Kp_curve(pot, Ts, **kw):
    return np.array([bound_state(pot, T, **kw).log_Kp for T in np.atleast_1d(Ts)])


# ---------------------------------------------------------------- composition
def dimer_fraction(K_V, rho0):
    """Fraction of monomer units in dimers, f_d = 2 rho_A2 / rho0, from
    f_d / (1 - f_d)^2 = 2 K_V rho0   (dilute-gas mass action; K_V and rho0 in
    consistent units, e.g. A^3 and A^-3)."""
    a = 2.0 * np.asarray(K_V) * np.asarray(rho0)
    a = np.maximum(a, 1e-300)
    return ((2 * a + 1) - np.sqrt(4 * a + 1)) / (2 * a)


def K_V_for_fraction(fd, rho0):
    """Inverse of :func:`dimer_fraction`."""
    return fd / ((1 - fd) ** 2 * 2.0 * rho0)


def number_density_ideal(p_Pa, T):
    """Monomer-unit number density (A^-3) of an ideal gas at p, T (counted as monomers)."""
    return p_Pa / (KB * T) * A3_TO_M3


def finite_box_two_particle(pot, T, L_box):
    """Exact dimer probability for two particles in a periodic cube of edge L_box (A).

    Assumes the potential vanishes beyond L_box/2 (minimum-image convention),
    which is required anyway for a valid two-particle simulation.

    Returns a dict with P_b, the exact relation P_b/(1-P_b) = Z_b/Z_u, the
    thermodynamic-limit K_V = Z_b/2, and the "naive" K_V that one obtains by
    inserting mean counts <N_A2> = P_b, <N_A> = 2(1 - P_b) into the
    macroscopic mass-action law (biased by the factor 2(1 - P_b)).
    """
    if pot.rcut > L_box / 2:
        raise ValueError("box too small: cut-off exceeds L/2")
    bs = bound_state(pot, T)
    rd = bs.r_div
    beta = 1.0 / (R_KJ * T)
    edges = np.linspace(rd, L_box / 2, 2001)
    r, w = gl_nodes(edges)
    integ = 4 * np.pi * np.sum(w * r * r * np.expm1(-beta * pot.U(r)))
    Z_u = L_box ** 3 - 4.0 / 3.0 * np.pi * rd ** 3 + integ
    Zb = np.exp(bs.log_Zb)
    Pb = Zb / (Zb + Z_u)
    V = L_box ** 3
    return dict(
        P_b=Pb,
        Z_b=Zb,
        Z_u=Z_u,
        K_V_exact=Zb / 2.0,
        K_V_from_Pb=Pb / (1 - Pb) * Z_u / 2.0,
        K_V_naive_counts=(Pb / V) / (2 * (1 - Pb) / V) ** 2,
        naive_bias_factor=((Pb / V) / (2 * (1 - Pb) / V) ** 2) / (Zb / 2.0),
    )
