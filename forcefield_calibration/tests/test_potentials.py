import json
from pathlib import Path

import numpy as np
import pytest

from forcefield_calibration.potentials.additive import AdditivePotential
from forcefield_calibration.potentials.branches import bond_k, shape_for_stiffness
from forcefield_calibration.potentials.candidates import candidate_families
from forcefield_calibration.potentials.piecewise import PiecewisePotential

REF = json.loads((Path(__file__).parent / "explorer_reference.json").read_text())


@pytest.mark.parametrize("i", range(4))
def test_piecewise_matches_explorer(i):
    s = REF["p1"][i]
    p = {k: v for k, v in s["p"].items() if k in PiecewisePotential.defaults}
    p["kh"] = None
    pot = PiecewisePotential(**p)
    rs = np.array(REF["rs"])
    assert pot.kh == pytest.approx(s["kh"], rel=1e-12)
    assert np.allclose(pot.U(rs), s["U"], atol=1e-9)
    assert np.allclose(pot.dU(rs), s["dU"], atol=1e-8)


@pytest.mark.parametrize("i", range(4))
def test_additive_matches_explorer(i):
    s = REF["p2"][i]
    p = {k: v for k, v in s["p"].items() if k in AdditivePotential.defaults}
    pot = AdditivePotential(**p)
    rs = np.array(REF["rs"])
    assert np.allclose(pot.U(rs), s["U"], atol=1e-9)
    L = pot.landscape()
    if s["cl"]["bond"]:
        assert L.bond.r == pytest.approx(s["cl"]["bond"]["r"], abs=1e-6)
    if s["cl"]["bar"]:
        assert L.barrier.r == pytest.approx(s["cl"]["bar"]["r"], abs=1e-6)


@pytest.mark.parametrize("label", list("ABCDEF"))
def test_candidates_clean_and_smooth(label):
    pot = candidate_families()[label]
    assert pot.landscape().clean
    r = np.linspace(1.2, pot.rcut - 0.01, 20001)
    # force = -dU/dr: compare analytic derivative with finite differences
    h = 1e-6
    fd = (pot.U(r + h) - pot.U(r - h)) / (2 * h)
    assert np.max(np.abs(fd - pot.dU(r)) / (1 + np.abs(pot.dU(r)))) < 1e-5
    # C2 continuity across joins
    for b in pot.breakpoints():
        # window must be small: the quintic switches are C2 with a finite third derivative
        d2l = pot.d2U(np.array([b - 3e-6]), h=1e-6)[0]
        d2r = pot.d2U(np.array([b + 3e-6]), h=1e-6)[0]
        assert abs(d2l - d2r) < 1e-2 * (1 + abs(d2l))


@pytest.mark.parametrize("form", ["morse", "mie", "wf", "kts"])
def test_shape_for_stiffness(form):
    p = dict(PiecewisePotential.defaults, formB=form, rb=1.78, eb=30.0, rcb=3.2)
    p.update(shape_for_stiffness(p, 777.0))
    assert bond_k(p) == pytest.approx(777.0, rel=1e-8)


def test_finite_cutoff_reaches_zero():
    for pot in candidate_families().values():
        assert pot.U(np.array([pot.rcut]))[0] == pytest.approx(0.0, abs=1e-12)
        assert pot.dU(np.array([pot.rcut - 1e-9]))[0] == pytest.approx(0.0, abs=1e-6)
