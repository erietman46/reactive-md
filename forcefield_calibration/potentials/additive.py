"""Potential 2 of the explorer: additive bond + Gaussian barrier + dispersion.

    U(r) = [ U_b(r) C_b(r) + H exp(-((r - r_x)/s_x)^2) + U_disp(r) ] C(r)

* ``U_b``: bond-well function (morse | mie | wf | kts) with minimum -eps_b at r_b,
  switched off by the quintic switch ``C_b`` between ``rbon`` and ``rboff``.
* barrier: Gaussian of height ``H`` at ``rx`` with width ``sx``.
* ``U_disp``: ``gauss`` (-eps_f Gaussian at r_f, width s_f), ``tt``
  (Tang-Toennies-damped -C6/r^6) or ``bj`` (Becke-Johnson-damped C6/C8).
* ``C``: overall quintic switch between ``rin`` and ``rout`` (finite cut-off).

The input parameters are *nominal*: because the terms are added, the actual
extrema of U are shifted. All thermodynamics uses the actual extrema (found
numerically by :meth:`PairPotential.landscape`).
"""
from __future__ import annotations

import numpy as np

from . import branches as B
from .base import PairPotential


class AdditivePotential(PairPotential):
    family = "additive"
    defaults = dict(
        formB="morse", eb=60.0, rb=1.78, xb=2.0, n=12.0, m=6.0, pk=4.0, g2=3.902767, rcb=3.2, bb=2.5,
        rbon=2.3, rboff=3.2,
        H=10.0, rx=2.6, sx=0.3,
        formF="gauss", ef=1.8, rf=3.8, sf=0.7,
        C6=6000.0, bt=1.5, C8=54000.0, a1=0.4289, a2=2.35, s6=1.0, s8=0.7875,
        rin=7.0, rout=9.0,
    )

    def _setup(self):
        p = self.p
        if not (p["rbon"] < p["rboff"]) or not (p["rin"] < p["rout"]):
            raise ValueError("switch ranges must be increasing")

    @property
    def rcut(self):
        return float(self.p["rout"])

    def breakpoints(self):
        p = self.p
        return [p["rbon"], p["rboff"], p["rin"]]

    def nominal_bond_radius(self):
        return self.p["rb"]

    # ------------------------------------------------------------- terms
    def bond_term(self, r):
        p = self.p
        u, du = B.bond_branch(r, p)
        c, dc = B.sw5(r, p["rbon"], p["rboff"])
        return u * c, du * c + u * dc

    def barrier_term(self, r):
        g, dg = B.gauss(r, self.p["rx"], self.p["sx"])
        return self.p["H"] * g, self.p["H"] * dg

    def disp_term(self, r):
        p = self.p
        f = p["formF"]
        if f == "tt":
            x = p["bt"] * r
            f6 = B.tt6(x)
            r6 = r ** 6
            return -p["C6"] * f6 / r6, -p["C6"] * (p["bt"] * B.dtt6(x) / r6 - 6 * f6 / (r6 * r))
        if f == "bj":
            R0 = np.sqrt(p["C8"] / p["C6"]) if p["C6"] > 0 else 0.0
            R = p["a1"] * R0 + p["a2"]
            r6, r8 = r ** 6, r ** 8
            R6, R8 = R ** 6, R ** 8
            u = -p["s6"] * p["C6"] / (r6 + R6) - p["s8"] * p["C8"] / (r8 + R8)
            du = p["s6"] * p["C6"] * 6 * r ** 5 / (r6 + R6) ** 2 + p["s8"] * p["C8"] * 8 * r ** 7 / (r8 + R8) ** 2
            return u, du
        if f == "gauss":
            g, dg = B.gauss(r, p["rf"], p["sf"])
            return -p["ef"] * g, -p["ef"] * dg
        raise ValueError(f"unknown dispersion form {f!r}")

    def _ud(self, r):
        p = self.p
        ub, dub = self.bond_term(r)
        ux, dux = self.barrier_term(r)
        uf, duf = self.disp_term(r)
        C, dC = B.sw5(r, p["rin"], p["rout"])
        s, ds = ub + ux + uf, dub + dux + duf
        inside = r < p["rout"]
        return np.where(inside, s * C, 0.0), np.where(inside, ds * C + s * dC, 0.0)

    def with_bond(self, eb=None, kb=None, rb=None, name=None):
        """Copy with new nominal bond depth / branch curvature (curvature of U_b at r_b)."""
        q = dict(self.p)
        if rb is not None:
            q["rb"] = rb
        if eb is not None:
            q["eb"] = eb
        if kb is not None:
            q.update(B.shape_for_stiffness(q, kb))
        return type(self)(name=name or self.name, **q)

    def barrier_controls(self):
        return {"height": "H", "position": "rx"}
