import numpy as np
import pytest

from forcefield_calibration.export.lammps_export import python_energy, verify_exports
from forcefield_calibration.manybody.coordination_penalties import hinge, make_penalty
from forcefield_calibration.manybody.third_virial import trimer_equilibrium
from forcefield_calibration.manybody.trimer_energy_surface import required_strength, vdw_reference, worst_trimer
from forcefield_calibration.potentials.candidates import candidate_families


@pytest.fixture(scope="module")
def pot():
    return candidate_families()["A"]


def test_exact_hinges_vanish_below_one():
    x = np.linspace(-1, 0, 101)
    assert np.all(hinge(x, "cubic", 0.05) == 0)
    assert np.all(hinge(x, "glue", 0.05) == 0)
    # the explorer's tanh-squared hinge is small but not zero for 0 < n < 1
    assert np.max(np.abs(hinge(x, "tanh_sq", 0.05))) > 0


@pytest.mark.parametrize("kind", ["coord_hermite_cubic", "coord_quintic_cubic", "glue", "kts", "sw"])
def test_isolated_pair_untouched(pot, kind):
    mb = make_penalty(pot, kind, lam=50.0)
    r = np.linspace(0.8, 10, 5000)
    assert np.max(np.abs(mb.pair_energy(r))) == 0.0


@pytest.mark.parametrize("kind", ["coord_hermite_cubic", "coord_quintic_cubic", "kts"])
def test_required_strength_suppresses_trimers(pot, kind):
    mb = make_penalty(pot, kind)
    for ref in ("free", "vdw"):
        sa = required_strength(pot, mb, reference=ref)
        e_ref = vdw_reference(pot)[0] if ref == "vdw" else 0.0
        assert sa.lam_min > 0 and sa.unfixable == 0.0
        assert worst_trimer(pot, mb.with_strength(1.01 * sa.lam_min))[0] >= e_ref - 1e-6
        assert worst_trimer(pot, mb.with_strength(0.9 * sa.lam_min))[0] < e_ref


def test_vdw_reference_is_two_dispersion_contacts(pot):
    e, (b, c, _) = vdw_reference(pot)
    assert e == pytest.approx(2 * pot.u_disp, rel=0.02)
    assert b >= pot.r_barrier and c >= pot.r_barrier


def test_penalty_reduces_trimer_constant(pot):
    mb = make_penalty(pot, "coord_hermite_cubic")
    mb = mb.with_strength(1.15 * required_strength(pot, mb).lam_min)
    t0 = trimer_equilibrium(pot, 298.15)
    t1 = trimer_equilibrium(pot, 298.15, mb)
    assert t1.log_K3 < t0.log_K3 - 10
    assert t1.log_K2 == pytest.approx(t0.log_K2, abs=1e-12)


def test_eam_export_matches_python(pot):
    pytest.importorskip("lammps")
    mb = make_penalty(pot, "coord_hermite_cubic", lam=40.0)
    rb = pot.r_bond
    cfgs = [np.array([[0, 0, 0], [rb, 0, 0], [-1.9, 0.3, 0]]), np.array([[0, 0, 0], [rb, 0, 0], [0.9, 1.6, 0.0]])]
    rows = verify_exports(pot, mb, cfgs)
    for r in rows:
        assert r["lammps_eam"] == pytest.approx(r["python_total"], abs=1e-6)
        assert r["lammps_table"] == pytest.approx(r["python_pair"], abs=1e-4)
        assert r["python_total"] == pytest.approx(python_energy(pot, mb, cfgs[r["config"]]))
