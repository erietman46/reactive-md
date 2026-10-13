import numpy as np
import pytest
from scipy.integrate import quad

from forcefield_calibration.constants import KB, R_KJ
from forcefield_calibration.potentials.candidates import candidate_families
from forcefield_calibration.reference_data import build_reference_table, janaf_anchors, nasa7_reaction
from forcefield_calibration.thermodynamics.equilibrium_integrals import (bound_state, dimer_fraction,
                                                                        finite_box_two_particle, K_V_for_fraction)
from forcefield_calibration.thermodynamics.reaction_enthalpy import deltaH_vant_hoff_local
from forcefield_calibration.thermodynamics.virial_coefficients import B2, B2_lennard_jones, B2_plain


@pytest.fixture(scope="module")
def pot():
    return candidate_families()["A"]


def test_bound_integral_against_quad(pot):
    T = 298.15
    b = 1 / (R_KJ * T)
    um = pot.u_bond
    v, _ = quad(lambda r: 4 * np.pi * r * r * np.exp(-b * (pot.U(r) - um)), 0.6, pot.r_barrier,
                points=[pot.r_bond], limit=500, epsrel=1e-12)
    assert bound_state(pot, T).log_Zb == pytest.approx(np.log(v) - b * um, abs=1e-9)


def test_very_stiff_well_does_not_overflow():
    # additive family: the Morse well stays intact at any stiffness
    p = candidate_families()["C"].with_bond(eb=56.0, kb=1e12)
    bs = bound_state(p, 298.15)
    assert np.isfinite(bs.log_Zb)
    # harmonic (Laplace) estimate: 4 pi r_b^2 sqrt(2 pi RT/k) exp(eps/RT)
    w = np.sqrt(R_KJ * 298.15 / 1e12)
    est = np.log(4 * np.pi * p.r_bond ** 2 * np.sqrt(2 * np.pi) * w) - p.u_bond / (R_KJ * 298.15)
    # agreement up to the Morse anharmonic correction, which is O(RT/eps_b) ~ 0.02 here
    assert bs.log_Zb == pytest.approx(est, abs=0.05)


def test_vant_hoff_equals_mean_energy(pot):
    for T in (300.0, 400.0):
        assert deltaH_vant_hoff_local(pot, T) == pytest.approx(bound_state(pot, T).deltaH, abs=1e-4)


def test_B2_split_identity(pot):
    v = B2(pot, 350.0)
    assert v.total == pytest.approx(v.frozen + v.assoc, rel=1e-12)
    # B2_plain starts at r_min: add the (exp(-U/RT) - 1 = -1) core below it
    core = 2 * np.pi / 3 * 0.6 ** 3 * 6.02214076e23 * 1e-24
    assert v.total == pytest.approx(B2_plain(pot, 350.0, r_min=0.6) + core, rel=1e-8)


def test_B2_lennard_jones_reduced_value():
    # reduced B2* = B2 / (2 pi sigma^3 N_A / 3) of LJ at T* = 1 is -2.538 (standard tables)
    sigma = 3.0
    b0 = 2 * np.pi / 3 * sigma ** 3 * 6.02214076e23 * 1e-24
    assert B2_lennard_jones(100.0, sigma, 100.0) / b0 == pytest.approx(-2.538, abs=2e-3)


def test_dimer_fraction_roundtrip():
    for fd in (0.05, 0.5, 0.95):
        assert dimer_fraction(K_V_for_fraction(fd, 0.002), 0.002) == pytest.approx(fd, rel=1e-10)


def test_finite_box_relations(pot):
    res = finite_box_two_particle(pot, 400.0, 30.0)
    assert res["K_V_from_Pb"] == pytest.approx(res["K_V_exact"], rel=1e-10)
    assert res["naive_bias_factor"] == pytest.approx(
        30.0 ** 3 / (2 * res["Z_u"] * (1 - res["P_b"])), rel=1e-10)


def test_reference_data_consistency():
    T, lnK, dH = janaf_anchors()
    ln, h, _ = nasa7_reaction(T)
    assert np.max(np.abs(ln - lnK)) < 0.05
    assert np.max(np.abs(h - dH)) < 0.2
    ref = build_reference_table(T)
    assert ref["sigma_lnK"][0] >= 0.05
    # JANAF Delta_r G at 298.15 K = -4.729 kJ/mol -> K = 6.737
    assert np.exp(4.729 / (R_KJ * 298.15)) == pytest.approx(6.737, rel=1e-3)


def test_Kp_definition(pot):
    bs = bound_state(pot, 300.0)
    KV_m3 = np.exp(bs.log_KV) * 1e-30
    assert bs.log_Kp == pytest.approx(np.log(KV_m3 * 1e5 / (KB * 300.0)), abs=1e-12)
