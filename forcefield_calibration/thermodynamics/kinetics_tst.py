"""Pre-MD kinetic *estimates* from the radial transition-state theory (TST).

These numbers are screening estimates, not predictions of MD rates:

* one-dimensional TST on the pair distance, dividing surface at the barrier;
* no recrossing correction (TST is an upper bound to the classical rate);
* "high-pressure limit": the bound basin is assumed thermally equilibrated.
  In a dilute gas with a purely pairwise conservative potential, two colliding
  particles cannot stay bound without a third body removing energy, so real
  association/dissociation at low density is collision-limited (fall-off).
  In the dense fluid of the transport simulations this limit is approached.

TST dissociation rate (s^-1):

    k_d = sqrt(k_B T / (2 pi mu)) * 4 pi r_div^2 exp(-U(r_div)/RT) / Z_b

association rate constant k_a = k_d K_V (A^3 s^-1), and the linearised
chemical relaxation time of 2A <-> A2 at monomer density rho_A:

    1/tau_r = 4 k_a rho_A + k_d
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import AMU, KB, R_KJ
from .equilibrium_integrals import bound_state, dimer_fraction


@dataclass
class TSTRates:
    T: float
    log_kd: float        # ln(k_d / s^-1)
    log_KV: float        # ln(K_V / A^3)
    dF_diss: float       # radial free-energy barrier for dissociation, kJ/mol
    dF_assoc: float      # radial free-energy barrier for association (from the outer basin), kJ/mol

    @property
    def tau_d(self):
        """Mean dimer lifetime 1/k_d in seconds."""
        return float(np.exp(-self.log_kd))

    @property
    def log_ka(self):
        return self.log_kd + self.log_KV

    def tau_relax(self, rho0):
        """Chemical relaxation time (s) at monomer-unit density rho0 (A^-3)."""
        fd = dimer_fraction(np.exp(self.log_KV), rho0)
        rhoA = rho0 * (1 - fd)
        return 1.0 / (4.0 * np.exp(self.log_ka) * rhoA + np.exp(self.log_kd))


def radial_free_energy(pot, r, T, r_ref=1.0):
    """F(r) = U(r) - 2 RT ln(r / r_ref): free energy along r including the r^2 Jacobian."""
    return pot.U(r) - 2.0 * R_KJ * T * np.log(np.asarray(r) / r_ref)


def tst_rates(pot, T, reduced_mass_gmol) -> TSTRates:
    bs = bound_state(pot, T)
    rd = bs.r_div
    beta = 1.0 / (R_KJ * T)
    mu = reduced_mass_gmol * AMU
    v = np.sqrt(KB * T / (2 * np.pi * mu)) * 1e10  # A/s
    log_kd = np.log(v * 4 * np.pi * rd ** 2) - beta * float(pot.U(rd)) - bs.log_Zb
    F = lambda x: float(radial_free_energy(pot, x, T))
    L = pot.landscape()
    dF_d = F(rd) - F(L.bond.r)
    dF_a = F(rd) - F(L.disp.r) if L.disp is not None else np.nan
    return TSTRates(T, float(log_kd), bs.log_KV, dF_d, dF_a)
