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

Families A-I are isotropic (U = U(r)); KF-A ... KF-D are the smooth
Kern-Frenkel versions U(r, Omega_i, Omega_j) of four radial constructions
(``potentials/kern_frenkel.py``), started from a patch width of 0.5 rad.
"""
from __future__ import annotations

import numpy as np

from ..constants import R_KJ
from .additive import AdditivePotential
from .kern_frenkel import KernFrenkelPotential
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


def _kf(name, radial, w=0.5):
    q = {k: v for k, v in radial.p.items()}
    return KernFrenkelPotential(name=name, radial_class=type(radial).__name__, w=w, **q)


def candidate_families():
    """Initial (unfitted) potentials, keyed by candidate label."""
    iso = isotropic_families()
    out = dict(iso)
    out.update({
        "KF-A": _kf("KF-A: smooth Kern-Frenkel | Morse | quintic Hermite barrier | LJ 12-6", iso["A"]),
        "KF-B": _kf("KF-B: smooth Kern-Frenkel | Morse | Gaussian barrier | LJ 12-6", iso["I"]),
        "KF-C": _kf("KF-C: smooth Kern-Frenkel | Morse | quintic Hermite barrier | Wang-Frenkel", iso["G"]),
        "KF-D": _kf("KF-D: smooth Kern-Frenkel | Mie | quintic Hermite barrier | LJ 12-6", iso["B"]),
    })
    return out


def isotropic_families():
    """Isotropic radial families A-I."""
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
        "G": _pw("G: piecewise Morse | quintic barrier | Wang-Frenkel", formB="morse", formF="wf", xf=2.5),
        "H": _ad("H: additive Morse + Gaussian barrier + KTS", formF="kts", ef=EF0, rf=RF0, pkf=4.0, g2f=20.0,
                 rcf=8.0),
        "I": _ad("I: additive Morse + Gaussian barrier + LJ 12-6", formF="mie", ef=EF0, rf=RF0, nf=12.0, mf=6.0),
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
    "G": [("ef", 0.5, 6.0), ("xf", 1.3, 3.0)],
    "H": [("ef", 0.5, 6.0), ("g2f", 2.0, 40.0)],
    "I": [("ef", 0.5, 6.0)],
    "KF-A": [("ef", 0.5, 6.0)],
    "KF-B": [("ef", 0.5, 6.0)],
    "KF-C": [("ef", 0.5, 6.0), ("xf", 1.3, 3.0)],
    "KF-D": [("ef", 0.5, 6.0)],
}

BARRIER_KEY = {"A": "Eh", "B": "Eh", "C": "H", "D": "H", "E": "Eh", "F": "Eh", "G": "Eh", "H": "H", "I": "H",
               "KF-A": "Eh", "KF-B": "H", "KF-C": "Eh", "KF-D": "Eh"}
BARRIER_POSITION = {"Eh": "rh", "H": "rx"}
# The bond length r_b is well defined (N-N in N2O4); the barrier position is a design choice, so the
# bond-well fits may move it by up to RH_DEV around its starting value (2.6 A).  For the additive
# construction only outwards: a Gaussian barrier moved inwards is swallowed by the Morse well (at
# r_x = 2.45 A the barrier disappears for eps_b > ~28 kJ/mol), which breaks the anchoring.
RH_DEV = 0.15
BARRIER_POSITION_RANGE = {"rh": (-RH_DEV, RH_DEV), "rx": (0.0, RH_DEV)}
# patch width of the Kern-Frenkel families (rad): bounds of the fits
W_BOUNDS = (0.05, 1.5)

ISOTROPIC = ("A", "B", "C", "D", "E", "F", "G", "H", "I")
KERN_FRENKEL = ("KF-A", "KF-B", "KF-C", "KF-D")
ALL_LABELS = ISOTROPIC + KERN_FRENKEL
PRIMARY = ("A", "B", "C", "D", "KF-A")
SUPPLEMENTARY = ("E", "F", "G", "H", "I", "KF-B", "KF-C", "KF-D")
# radial construction under each Kern-Frenkel family (same colour in the figures)
KF_PARENT = {"KF-A": "A", "KF-B": "I", "KF-C": "G", "KF-D": "B"}


def label_of(pot):
    return pot.name.split(":")[0]


def is_kf(label):
    return label in KERN_FRENKEL


def finite(x):
    return x is not None and np.isfinite(x)
