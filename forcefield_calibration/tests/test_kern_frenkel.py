import numpy as np
import pytest
from scipy.integrate import dblquad, quad

from forcefield_calibration.constants import R_KJ
from forcefield_calibration.potentials.candidates import (ALL_LABELS, KERN_FRENKEL, candidate_families,
                                                          isotropic_families)
from forcefield_calibration.potentials.kern_frenkel import angular_moments, patch_factor
from forcefield_calibration.thermodynamics.equilibrium_integrals import bound_state
from forcefield_calibration.thermodynamics.reaction_enthalpy import deltaH_vant_hoff_local
from forcefield_calibration.thermodynamics.virial_coefficients import B2


@pytest.fixture(scope="module")
def kf():
    return candidate_families()["KF-A"].with_bond(eb=60.0).replace(w=0.4)


def brute_average(b, w, power=0):
    """<(1-S)^k exp(-b (1-S))> over c_i, c_j uniform on [-1, 1] by 2-D quadrature."""
    def f(ci, cj):
        y = 1.0 - patch_factor(ci, w)[0] * patch_factor(cj, w)[0]
        return y ** power * np.exp(-b * y) / 4.0

    v = 0.0
    edges = np.unique(np.r_[-1.0, 1.0 - np.geomspace(1e-7, 2.0, 12)])
    for a0, a1 in zip(edges[:-1], edges[1:]):
        for c0, c1 in zip(edges[:-1], edges[1:]):
            v += dblquad(f, c0, c1, a0, a1, epsabs=0, epsrel=1e-9)[0]
    return v


@pytest.mark.parametrize("b", [0.3, 5.0])
def test_angular_moments_match_direct_quadrature(b):
    """The 1-D density of s = u_i + u_j against the direct 2-D average over (c_i, c_j)."""
    w = 0.4
    lm0, m1, m2 = angular_moments(np.array([b]), w)
    m0 = brute_average(b, w)
    assert lm0[0] == pytest.approx(np.log(m0), abs=1e-6)
    assert m1[0] == pytest.approx(brute_average(b, w, 1) / m0, rel=1e-6)
    assert m2[0] == pytest.approx(brute_average(b, w, 2) / m0, rel=1e-6)


@pytest.mark.parametrize("b", [1e-3, 0.3, 30.0, 3e3, 3e7])
def test_angular_table_accuracy(b):
    """Spline table against adaptive quadrature of the 1-D integral (deep wells: b up to 1e7)."""
    w = 0.4
    L = 2 / w ** 2
    edges = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, L, 2 * L]

    def mom(k):
        f = lambda s: 0.25 * w ** 4 * min(s, 2 * L - s) * (-np.expm1(-s)) ** k * np.exp(b * np.expm1(-s))
        return sum(quad(f, a, c, epsabs=0, epsrel=1e-13, limit=500)[0] for a, c in zip(edges[:-1], edges[1:]))

    lm0, m1, m2 = angular_moments(np.array([b]), w)
    m0 = mom(0)
    assert lm0[0] == pytest.approx(np.log(m0), abs=1e-7)
    assert m1[0] == pytest.approx(mom(1) / m0, rel=1e-6)
    assert m2[0] == pytest.approx(mom(2) / m0, rel=1e-6)


def test_angular_moments_limits():
    w = 0.3
    lm0, m1, m2 = angular_moments(np.array([1e12, 0.0]), w)
    assert lm0[0] == pytest.approx(np.log(w ** 4 / 4) - 2 * np.log(1e12), abs=1e-6)
    assert m1[0] * 1e12 == pytest.approx(2.0, rel=1e-6)   # two 2-D librations: <E_ang> = 2 RT
    assert lm0[1] == 0.0


def test_full_energy_limits_and_gradients(kf):
    rng = np.random.default_rng(3)
    rb = kf.r_bond
    x = np.array([[rb, 0, 0]])
    U, *_ = kf.energy(x, np.array([[1.0, 0, 0]]), np.array([[-1.0, 0, 0]]))
    assert U[0] == pytest.approx(kf.U(rb))                      # aligned: the radial reactive curve
    U, *_ = kf.energy(x, np.array([[-1.0, 0, 0]]), np.array([[1.0, 0, 0]]))
    assert U[0] == pytest.approx(kf.U_off(rb), rel=1e-6)        # misaligned: repulsive wall
    far = np.array([[kf.r_s + 0.3, 0, 0]])
    for _ in range(5):
        ni, nj = rng.normal(size=(2, 1, 3))
        ni /= np.linalg.norm(ni)
        nj /= np.linalg.norm(nj)
        assert kf.energy(far, ni, nj)[0][0] == pytest.approx(kf.U(kf.r_s + 0.3), abs=1e-12)  # isotropic outside r_s
    # analytic gradients against central differences
    h = 1e-6
    for _ in range(10):
        r = rng.normal(size=3)
        r *= rng.uniform(1.6, 3.2) / np.linalg.norm(r)
        ni, nj = rng.normal(size=(2, 3))
        ni /= np.linalg.norm(ni)
        nj /= np.linalg.norm(nj)
        U, gr, gi, gj = kf.energy(r[None], ni[None], nj[None])
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            num_r = (kf.energy((r + e)[None], ni[None], nj[None])[0] - kf.energy((r - e)[None], ni[None], nj[None])[0]) / (2 * h)
            num_i = (kf.energy(r[None], (ni + e)[None], nj[None])[0] - kf.energy(r[None], (ni - e)[None], nj[None])[0]) / (2 * h)
            num_j = (kf.energy(r[None], ni[None], (nj + e)[None])[0] - kf.energy(r[None], ni[None], (nj - e)[None])[0]) / (2 * h)
            scale = 1 + abs(gr[0, k])
            assert num_r[0] == pytest.approx(gr[0, k], abs=1e-5 * scale)
            assert num_i[0] == pytest.approx(gi[0, k], abs=1e-5 * (1 + abs(gi[0, k])))
            assert num_j[0] == pytest.approx(gj[0, k], abs=1e-5 * (1 + abs(gj[0, k])))


def test_misaligned_curve_is_repulsive(kf):
    assert kf.feasible
    r = np.linspace(0.8, kf.r_s - 1e-4, 2000)
    assert np.all(kf.dU_off(r) < 0)
    # C2 at r_s
    for d in (kf.wall,):
        u, du = d(np.array([kf.r_s - 1e-7]))
        assert abs(u[0]) < 1e-15 and abs(du[0]) < 1e-10


def test_bound_state_against_monte_carlo(kf):
    """Z_b from the tabulated orientation average vs direct 3-D (r, c_i, c_j) quadrature."""
    T = 400.0
    beta = 1 / (R_KJ * T)
    bs = bound_state(kf, T)
    r = np.linspace(kf.r_bond - 0.35, kf.r_barrier, 1401)
    c = 1.0 - np.geomspace(1e-9, 2.0, 1500)
    c = np.unique(np.r_[c, 1.0])
    f = patch_factor(c, kf.p["w"])[0]
    S = f[:, None] * f[None, :]
    um = kf.u_bond
    vals = []
    for ri in r:
        W = kf.wall(ri)[0]
        g = np.exp(-beta * (1 - S) * W) / 4.0
        inner = np.trapezoid(g, c, axis=1)
        vals.append(np.trapezoid(inner, c))
    vals = np.array(vals) * np.exp(-beta * (kf.U(r) - um)) * 4 * np.pi * r * r
    log_Zb = np.log(np.trapezoid(vals, r)) - beta * um
    assert bs.log_Zb == pytest.approx(log_Zb, abs=2e-3)


def test_vant_hoff_equals_mean_energy_kf(kf):
    for T in (300.0, 450.0):
        assert deltaH_vant_hoff_local(kf, T) == pytest.approx(bound_state(kf, T).deltaH, abs=2e-4)


def test_wide_patch_recovers_isotropic():
    iso = isotropic_families()["A"]
    # (1 - S) <= 4/w^2: the residual angular energy at the bond is ~W_b 4/w^2 ~ 1e-5 kJ/mol here
    kf = candidate_families()["KF-A"].replace(w=3000.0)
    a, b = bound_state(iso, 350.0), bound_state(kf, 350.0)
    assert b.log_Kp == pytest.approx(a.log_Kp, abs=2e-5)
    assert b.deltaH == pytest.approx(a.deltaH, abs=2e-4)
    assert B2(kf, 350.0).frozen == pytest.approx(B2(iso, 350.0).frozen, rel=1e-3)


def test_orientational_entropy_is_the_missing_entropy(kf):
    """Narrowing the patch lowers K at fixed well, mostly through the entropy."""
    T = 298.15
    wide, narrow = bound_state(kf.replace(w=0.8), T), bound_state(kf.replace(w=0.3), T)
    assert narrow.log_Kp < wide.log_Kp - 1.0
    assert abs(narrow.deltaH - wide.deltaH) < 0.05
    assert narrow.deltaS < wide.deltaS - 10.0


def test_all_candidates_clean():
    fams = candidate_families()
    assert set(fams) == set(ALL_LABELS)
    for lab in KERN_FRENKEL:
        p = fams[lab].with_bond(eb=60.0)
        assert p.landscape().clean and p.feasible


def test_single_patch_limits_valence(kf):
    """With one patch, every two-bond geometry costs energy relative to dimer + free monomer, and the trimer
    content (rigorous upper bound) is orders of magnitude below that of the same curve without orientation."""
    from forcefield_calibration.manybody.patchy_valence import (_ln_h, relaxed_trimer, second_bond_map,
                                                                trimer_equilibrium_kf)
    from forcefield_calibration.manybody.third_virial import trimer_equilibrium
    from forcefield_calibration.thermodynamics.equilibrium_integrals import number_density_ideal

    # one-neighbour limit of the two-neighbour table against the exact 1-D average
    w = kf.p["w"]
    c = np.linspace(-1, 1, 200001)
    for x in (1.0, 30.0):
        exact = np.log(np.trapezoid(np.exp(-x * (1 - patch_factor(c, w)[0])), c) / 2)
        assert _ln_h(w, np.array([x]), 0.0, -0.3)[0] == pytest.approx(exact, abs=0.03)
    b = np.linspace(0.8 * kf.r_bond, kf.r_barrier, 40)
    th = np.deg2rad(np.linspace(20, 180, 33))
    B_, T_ = np.meshgrid(b, th, indexing="ij")
    assert np.min(second_bond_map(kf, B_, T_, n_phi=91)) > 0.0
    assert relaxed_trimer(kf, n_starts=8).dE > 0.0
    rho = number_density_ideal(1e5, 298.15)
    kf3 = trimer_equilibrium_kf(kf, 298.15).fractions(rho)["trimer"]
    iso3 = trimer_equilibrium(kf.radial, 298.15).fractions(rho)["trimer"]
    assert kf3 < 1e-3 and iso3 > 0.5
