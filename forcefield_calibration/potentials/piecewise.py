"""Potential 1 of the explorer: bond branch + quintic barrier bridge + fluid branch.

    U(r) = u_bond(r)                       r <= r_b
         = quintic(r; -eps_b -> E_h)       r_b < r < r_h
         = quintic(r; E_h -> -eps_f)       r_h <= r < r_f
         = u_fluid(r) * S(r)               r >= r_f

Both quintic segments have zero slope at their ends and match the second
derivative of the neighbouring branch (c1 = k_b at r_b, -k_h at r_h, k_f at
r_f), so U is C2 everywhere.  ``kh=None`` reproduces the explorer's
"auto k_h": the midpoint of the range of barrier curvatures for which both
segments are monotonic (one bond minimum, one barrier, one dispersion
minimum).  ``rin/rout`` add an optional quintic switch on the fluid branch so
that Mie/Morse tails reach zero at a finite MD cut-off (the explorer has no
cut-off for those forms).
"""
from __future__ import annotations

import numpy as np

from . import branches as B
from .base import PairPotential


def _quint(t, E1, E2, c1, c2, L):
    dE, L2 = E2 - E1, L * L
    t2 = t * t
    t3, t4, t5 = t2 * t, t2 * t2, t2 * t2 * t
    u = E1 + dE * (10 * t3 - 15 * t4 + 6 * t5) + L2 * (
        0.5 * c1 * (t2 - 3 * t3 + 3 * t4 - t5) + 0.5 * c2 * (t3 - 2 * t4 + t5)
    )
    du = (
        dE * (30 * t2 - 60 * t3 + 30 * t4)
        + L2 * (0.5 * c1 * (2 * t - 9 * t2 + 12 * t3 - 5 * t4) + 0.5 * c2 * (3 * t2 - 8 * t3 + 5 * t4))
    ) / L
    return u, du


class PiecewisePotential(PairPotential):
    family = "piecewise"
    defaults = dict(
        formB="morse", rb=1.7, eb=100.0, n=12.0, m=6.0, xb=2.0, pk=4.0, g2=3.902767, rcb=2.8, bb=2.5,
        rh=2.6, Eh=10.0, kh=None,
        formF="wf", rf=3.5, ef=1.0, nf=12.0, mf=6.0, xf=2.0, bf=1.5, pkf=4.0, g2f=3.0, rcf=8.0,
        rin=None, rout=None,
    )

    # ------------------------------------------------------------------ setup
    def _setup(self):
        p = self.p
        if not (p["rb"] < p["rh"] < p["rf"]):
            raise ValueError("need r_b < r_h < r_f")
        self.kb = float(B.bond_k(p))
        self.kf = float(B.fluid_k(p))
        self.kh_range = self._kh_range()
        if p["kh"] is None:
            self.kh = 0.5 * (self.kh_range[0] + self.kh_range[1]) if self.kh_range else 1.0
        else:
            self.kh = float(p["kh"])
        self.feasible = self.kh_range is not None and self._monotonic(np.array([self.kh]))[0]

    def _segs(self, kh):
        p = self.p
        return (
            (p["rb"], p["rh"], -p["eb"], p["Eh"], self.kb, -kh),
            (p["rh"], p["rf"], p["Eh"], -p["ef"], -kh, self.kf),
        )

    def _monotonic(self, khs, N=400):
        """Vectorised explorer check: segment 0 rising, segment 1 falling, for each k_h."""
        khs = np.atleast_1d(khs)[:, None]
        t = (np.arange(1, N) / N)[None, :]
        ok = np.ones(khs.shape[0], bool)
        for j, (r1, r2, E1, E2, _c1, _c2) in enumerate(self._segs(0.0)):
            L = r2 - r1
            # segment 0: c1 = k_b, c2 = -k_h ; segment 1: c1 = -k_h, c2 = k_f
            c1v = self.kb if j == 0 else -khs
            c2v = -khs if j == 0 else self.kf
            _, d = _quint(t, E1, E2, c1v, c2v, L)
            sign = 1.0 if j == 0 else -1.0
            scale = abs(E2 - E1) / L
            ok &= np.all(d * sign >= -1e-9 * scale, axis=1)
        return ok

    def _kh_range(self):
        ks = 10.0 ** np.linspace(-3, 7, 301)
        ok = self._monotonic(ks)
        if not ok.any():
            return None
        i0 = int(np.argmax(ok))
        i1 = i0
        while i1 + 1 < ok.size and ok[i1 + 1]:
            i1 += 1
        return (float(ks[i0]), float(ks[i1]))

    # --------------------------------------------------------------- energy
    @property
    def rcut(self):
        p = self.p
        rc = B.fluid_cutoff(p)
        return float(rc if p["rout"] is None else min(rc, p["rout"]))

    def breakpoints(self):
        p = self.p
        out = [p["rb"], p["rh"], p["rf"]]
        if p["rin"] is not None:
            out += [p["rin"], p["rout"]]
        return out

    def _ud(self, r):
        p = self.p
        u = np.zeros_like(r)
        du = np.zeros_like(r)
        m0 = r <= p["rb"]
        if m0.any():
            u[m0], du[m0] = B.bond_branch(r[m0], p)
        (s0, s1) = self._segs(self.kh)
        m1 = (r > p["rb"]) & (r < p["rh"])
        if m1.any():
            r1, r2, E1, E2, c1, c2 = s0
            u[m1], du[m1] = _quint((r[m1] - r1) / (r2 - r1), E1, E2, c1, c2, r2 - r1)
        m2 = (r >= p["rh"]) & (r < p["rf"])
        if m2.any():
            r1, r2, E1, E2, c1, c2 = s1
            u[m2], du[m2] = _quint((r[m2] - r1) / (r2 - r1), E1, E2, c1, c2, r2 - r1)
        m3 = r >= p["rf"]
        if m3.any():
            uf, duf = B.fluid_branch(r[m3], p)
            if p["rin"] is not None:
                C, dC = B.sw5(r[m3], p["rin"], p["rout"])
                uf, duf = uf * C, duf * C + uf * dC
            u[m3], du[m3] = uf, duf
        return u, du

    def nominal_bond_radius(self):
        return self.p["rb"]

    @property
    def r_end(self):
        rc = self.rcut
        return rc if np.isfinite(rc) else max(2.2 * self.p["rf"], self.p["rf"] + 3.0)

    # convenience --------------------------------------------------------
    def with_bond(self, eb=None, kb=None, rb=None, name=None):
        """Copy with new bond depth/stiffness, keeping the branch form."""
        q = dict(self.p)
        if rb is not None:
            q["rb"] = rb
        if eb is not None:
            q["eb"] = eb
        if kb is not None:
            q.update(B.shape_for_stiffness(q, kb))
        return type(self)(name=name or self.name, **q)

    def barrier_controls(self):
        return {"height": "Eh", "position": "rh"}
