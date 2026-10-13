"""Second virial coefficient of the complete pair potential and its decomposition.

    B2(T) = -2 pi N_A int_0^inf [exp(-U/RT) - 1] r^2 dr

Splitting the integral at the dividing surface r_div gives an *exact* identity

    B2 = B2_frozen - N_A K_V,
    B2_frozen = (2 pi N_A / 3) r_div^3  -  2 pi N_A int_{r_div}^inf [exp(-U/RT) - 1] r^2 dr

``B2_frozen`` is the virial coefficient of monomers that are forbidden to bond
(the bound region replaced by a hard core of radius r_div) and ``N_A K_V`` is
the association constant in molar units.  This is the classical result of
Woolley / Stogryn-Hirschfelder for an associating gas: the *apparent* second
virial coefficient of an equilibrium NO2/N2O4 mixture, expressed per NO2
formula unit, equals B2 of the full model, while its "physical" part is
B2_frozen.  It also shows why K_p and B2 are not independent targets: when
association is strong, B2 ~ -N_A K_V.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import A3_TO_CM3, NA, R_KJ
from .equilibrium_integrals import bound_state, gl_nodes


@dataclass
class VirialB2:
    T: float
    total: float      # cm^3/mol
    frozen: float     # cm^3/mol
    assoc: float      # -N_A K_V, cm^3/mol
    core: float       # (2 pi N_A/3) r_div^3, cm^3/mol
    outer: float      # Mayer integral beyond r_div, cm^3/mol


def _outer_mayer(pot, T, r_from, r_max=None, dr=0.01):
    """-2 pi int_{r_from}^{r_max} (exp(-U/RT) - 1) r^2 dr in A^3 (per particle pair)."""
    beta = 1.0 / (R_KJ * T)
    rc = pot.rcut
    if r_max is None:
        r_max = rc if np.isfinite(rc) else 60.0
    n = max(50, int(np.ceil((r_max - r_from) / dr)))
    edges = np.unique(np.r_[np.linspace(r_from, r_max, n + 1), [b for b in pot.breakpoints() if r_from < b < r_max]])
    r, w = gl_nodes(edges)
    return -2.0 * np.pi * np.sum(w * r * r * np.expm1(-beta * pot.U(r)))


def B2(pot, T, r_div=None) -> VirialB2:
    bs = bound_state(pot, T, r_div=r_div)
    rd = bs.r_div
    core = 2.0 * np.pi / 3.0 * rd ** 3
    outer = _outer_mayer(pot, T, rd)
    assoc = -np.exp(bs.log_KV)
    f = NA * A3_TO_CM3
    return VirialB2(T, (core + outer + assoc) * f, (core + outer) * f, assoc * f, core * f, outer * f)


def B2_plain(pot, T, r_min=1e-3):
    """Direct Mayer integral without the bound/unbound split (for potentials without a bond well)."""
    beta = 1.0 / (R_KJ * T)
    rc = pot.rcut
    r_max = rc if np.isfinite(rc) else 60.0
    edges = np.linspace(r_min, r_max, 6001)
    r, w = gl_nodes(edges)
    with np.errstate(over="ignore"):
        f = np.expm1(-beta * pot.U(r))
    return -2.0 * np.pi * np.sum(w * r * r * f) * NA * A3_TO_CM3


def B2_lennard_jones(eps_K, sigma_A, T):
    """B2 of a full (uncut) 12-6 Lennard-Jones fluid, cm^3/mol (eps given as eps/k_B in K)."""
    Ts = np.atleast_1d(np.asarray(T, float))
    x = np.linspace(1e-3, 12.0, 120001)  # r/sigma
    out = []
    for t in Ts:
        u = 4.0 * (x ** -12 - x ** -6) * eps_K / t
        f = np.expm1(-np.minimum(u, 700.0))
        integ = np.trapezoid(f * x * x, x)
        # analytic tail beyond 12 sigma: exp(-u)-1 ~ +4 eps/T x^-6
        tail = 4.0 * eps_K / t / (3.0 * 12.0 ** 3)
        out.append(-2.0 * np.pi * (integ + tail) * sigma_A ** 3 * NA * A3_TO_CM3)
    out = np.array(out)
    return out if np.ndim(T) else float(out[0])
