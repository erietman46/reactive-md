"""Candidate pair-potential families compared in Stage 1.

All candidates start from the same physical guesses so that differences come
from the functional form, not from the starting point:

* bond: r_b = 1.78 A (N-N in N2O4), eps_b = 30 kJ/mol, k_b = 580 kJ/mol/A^2
  (N-N stretch ~266 cm^-1 with mu = 23 g/mol);
* fluid: the Polat et al. NO2 Lennard-Jones well, r_min = 2^(1/6) sigma = 4.18 A,
  eps = 262.17 K = 2.18 kJ/mol;
* barrier: a modest design barrier of +5 kJ/mol near 2.6 A (a kinetic design
  parameter - real NO2 association is essentially barrierless);
* all Mie/Morse/Gaussian/TT tails are switched to zero between 8 and 10 A
  (the same MD cut-off treatment for every family).
"""
from __future__ import annotations

import numpy as np

from ..constants import R_KJ
from .additive import AdditivePotential
from .piecewise import PiecewisePotential

RB = 1.78
EB0 = 30.0
KB0 = 580.0
LJ_EPS_K, LJ_SIGMA = 262.17, 3.727
RF0 = 2 ** (1 / 6) * LJ_SIGMA
EF0 = LJ_EPS_K * R_KJ
SWITCH = dict(rin=8.0, rout=10.0)


def _pw(name, **kw):
    base = dict(rb=RB, eb=EB0, rh=2.6, Eh=5.0, rf=RF0, ef=EF0, nf=12.0, mf=6.0, **SWITCH)
    base.update(kw)
    return PiecewisePotential(name=name, **base).with_bond(kb=KB0)


def _ad(name, **kw):
    base = dict(formB="morse", rb=RB, eb=EB0, rbon=2.3, rboff=3.2, H=5.0, rx=2.6, sx=0.25, **SWITCH)
    base.update(kw)
    return AdditivePotential(name=name, **base).with_bond(kb=KB0)


def candidate_families():
    """Initial (unfitted) potentials, keyed by candidate label."""
    return {
        "A": _pw("A: piecewise Morse | quintic barrier | Mie 12-6", formB="morse", formF="mie"),
        "B": _pw("B: piecewise Mie(2m-m) | quintic barrier | Mie 12-6", formB="mie", formF="mie"),
        "C": _ad("C: additive Morse + Gaussian barrier + Gaussian dispersion",
                 formF="gauss", ef=EF0, rf=RF0, sf=0.7),
        "D": _ad("D: additive Morse + Gaussian barrier + Tang-Toennies C6",
                 formF="tt", C6=4 * EF0 * LJ_SIGMA ** 6, bt=1.0),
        "E": _pw("E: piecewise Wang-Frenkel | quintic barrier | Wang-Frenkel (supplementary)",
                 formB="wf", formF="wf", xf=2.5),
        "F": _pw("F: piecewise KTS | quintic barrier | Mie 12-6 (supplementary)",
                 formB="kts", formF="mie", rcb=3.2),
    }


# Free parameters of the non-bonded (fluid) branch fitted in Step 2, with bounds.
# The outer-minimum position r_f is held at the LJ minimum 2^(1/6) sigma: B2(T) alone
# cannot separate well position from well depth (a free (r_f, eps_f) fit is degenerate
# and wanders to either bound), so the structural length is fixed and the energy/shape
# parameters are fitted.  For the Tang-Toennies form the position is not a parameter.
FLUID_PARAMS = {
    "A": [("ef", 0.5, 6.0)],
    "B": [("ef", 0.5, 6.0)],
    "C": [("ef", 0.5, 6.0), ("sf", 0.3, 1.5)],
    "D": [("C6", 5e3, 8e4), ("bt", 0.5, 3.0)],
    "E": [("ef", 0.5, 6.0), ("xf", 1.3, 3.0)],
    "F": [("ef", 0.5, 6.0)],
}

BARRIER_KEY = {"A": "Eh", "B": "Eh", "C": "H", "D": "H", "E": "Eh", "F": "Eh"}

PRIMARY = ("A", "B", "C", "D")
SUPPLEMENTARY = ("E", "F")


def label_of(pot):
    return pot.name.split(":")[0]


def finite(x):
    return x is not None and np.isfinite(x)
