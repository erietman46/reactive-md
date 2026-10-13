"""Valence-limiting many-body terms (ports of the explorer's three-body module).

All terms act on top of a fixed pair potential ``U2``.  The coordination-type
terms (``coord`` and ``glue``) have embedded-atom form

    E_mb = sum_i F(n_i),   n_i = sum_{j != i} s(r_ij)

and can therefore be run in LAMMPS with ``pair_style eam/fs`` (see
``export/eam_fs.py``).  The angular three-body forms (KTS, Stillinger-Weber)
are included for comparison.

Counters ``s(r)`` (all equal ~1 inside the bond and 0 beyond their reach):

    gauss        exp[-((r - r0)/sigma_s)^2]                 (misses r < r0, never 0)
    tanh         (1 - tanh((r - r_s)/w))/2                   (never exactly 0)
    tanh_damped  tanh step x C2 damping ending at r_cut
    hermite      quintic smoothstep centred at r1, width W   (C2, compact)
    cycloidal    1 - t + sin(2 pi t)/(2 pi)                  (C2, compact)
    septic       septic smoothstep                           (C3, compact)
    quintic      quintic step with initial slope -m          (falls from its start)

Hinges ``h(x)``, x = n_i - 1:

    tanh_sq      x^2 (1 + tanh(x/delta))/2, minus the free-atom value
                 (explorer form; small but NON-ZERO for 0 < n < 1)
    cubic        x^3/(x + delta) for x > 0, else 0
                 (C2, EXACTLY zero for n <= 1 -> isolated pairs untouched)
    glue         C2 hinge of the explorer's glue model, linear for x > d
"""
from __future__ import annotations

import numpy as np

from ..potentials.branches import sw5


# ------------------------------------------------------------- counters
def _smooth5(t):
    t = np.clip(t, 0.0, 1.0)
    return 1.0 - t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t)


def _qS(t, m):
    return (1 - 10 * t ** 3 + 15 * t ** 4 - 6 * t ** 5) - m * (t - 6 * t ** 3 + 8 * t ** 4 - 3 * t ** 5)


def _qT(m):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _qS(mid, m) > 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class Counter:
    """Smooth bond counter s(r)."""

    def __init__(self, kind="hermite", **kw):
        self.kind = kind
        self.kw = kw

    def __call__(self, r):
        r = np.asarray(r, float)
        k, q = self.kind, self.kw
        if k == "gauss":
            z = (r - q["r0"]) / q["sigma"]
            return np.exp(-z * z)
        if k == "tanh":
            return 0.5 * (1 - np.tanh((r - q["rs"]) / q["w"]))
        if k == "tanh_damped":
            return 0.5 * (1 - np.tanh((r - q["rs"]) / q["w"])) * _smooth5((r - (q["rcut"] - q["dw"])) / q["dw"])
        if k == "hermite":
            return _smooth5((r - q["r1"]) / q["W"] + 0.5)
        if k == "cycloidal":
            t = np.clip((r - q["r1"]) / q["W"] + 0.5, 0, 1)
            return 1 - t + np.sin(2 * np.pi * t) / (2 * np.pi)
        if k == "septic":
            t = np.clip((r - q["r1"]) / q["W"] + 0.5, 0, 1)
            return 1 - t ** 4 * (35 - 84 * t + 70 * t * t - 20 * t ** 3)
        if k == "quintic":
            m = q.get("m", 0.5)
            t = (r - q["r1"]) / q["W"] + _qT(m)
            return np.where(t <= 0, 1.0, np.where(t >= 1, 0.0, _qS(np.clip(t, 0, 1), m)))
        raise ValueError(k)

    @property
    def reach(self):
        k, q = self.kind, self.kw
        if k == "gauss":
            return q["r0"] + 5 * q["sigma"]
        if k == "tanh":
            return q["rs"] + 10 * q["w"]
        if k == "tanh_damped":
            return q["rcut"]
        if k in ("hermite", "cycloidal", "septic"):
            return q["r1"] + 0.5 * q["W"]
        if k == "quintic":
            return q["r1"] + (1 - _qT(q.get("m", 0.5))) * q["W"]
        raise ValueError(k)

    @property
    def compact(self):
        return self.kind not in ("gauss", "tanh")


# ------------------------------------------------------------- hinges
def hinge(x, kind, delta):
    x = np.asarray(x, float)
    if kind == "tanh_sq":
        return x * x * 0.5 * (1 + np.tanh(x / delta)) - 1.0 * 0.5 * (1 + np.tanh(-1.0 / delta))
    if kind == "cubic":
        xp = np.maximum(x, 0.0)
        return xp ** 3 / (xp + delta)
    if kind == "glue":
        xp = np.maximum(x, 0.0)
        d = delta
        return np.where(xp < d, xp ** 3 / (d * d) - xp ** 4 / (2 * d ** 3), xp - d / 2)
    raise ValueError(kind)


# ------------------------------------------------------------- terms
class ManyBody:
    """Interface: energies of isolated pairs and of three-particle clusters."""

    name = "none"
    eam_compatible = False

    def __init__(self, lam=0.0):
        self.lam = float(lam)

    def with_strength(self, lam):
        import copy

        c = copy.copy(self)
        c.lam = float(lam)
        return c

    def pair_energy(self, r):
        return np.zeros_like(np.asarray(r, float))

    def triplet_energy(self, r12, r13, r23):
        raise NotImplementedError

    @property
    def reach(self):
        raise NotImplementedError


class CoordinationPenalty(ManyBody):
    eam_compatible = True

    def __init__(self, counter: Counter, lam=0.0, hinge_kind="cubic", delta=0.05, name=None):
        super().__init__(lam)
        self.counter = counter
        self.hinge_kind = hinge_kind
        self.delta = delta
        self.name = name or f"coord[{counter.kind},{hinge_kind}]"

    def embed(self, n):
        """F(n) per atom (zero for a free atom)."""
        return self.lam * hinge(np.asarray(n) - 1.0, self.hinge_kind, self.delta)

    def density(self, r):
        return self.counter(r)

    def pair_energy(self, r):
        s = self.counter(r)
        return 2.0 * self.embed(s)

    def triplet_energy(self, r12, r13, r23):
        a, b, c = self.counter(r12), self.counter(r13), self.counter(r23)
        return self.embed(a + b) + self.embed(a + c) + self.embed(b + c)

    @property
    def reach(self):
        return self.counter.reach


class GluePenalty(ManyBody):
    """Explorer's glue (valence-one embedding): rho counts bond *strength*.

    rho(r) = 1 for r <= r_b; 1 - (1 - y)^p with y = -U2(r)/eps_b between r_b
    and r_0 (first zero of U2 past r_b), smoothly switched on from y = 0 over
    d_w; F(n) = lambda_g * h_glue(n - 1, d).  An isolated pair has n <= 1, so
    the pair potential is exactly unchanged.  Default lambda_g = 1.2 eps_b.
    """

    eam_compatible = True

    def __init__(self, pair, lam=None, p=3.0, d=0.05, dw=0.02, name="glue"):
        rb, eb = pair.r_bond, -pair.u_bond
        edge = pair.r_barrier
        r = np.arange(rb, edge + 0.6, 0.002)
        u = pair.U(r)
        idx = np.nonzero(u >= 0)[0]
        r0 = float(r[idx[0]]) if idx.size else float(edge)
        self.pair, self.rb, self.eb, self.r0 = pair, float(rb), float(max(eb, 1e-9)), r0
        self.p, self.d, self.dw = p, d, dw
        super().__init__(1.2 * self.eb if lam is None else lam)
        self.name = name

    def density(self, r):
        r = np.asarray(r, float)
        y = np.clip(-self.pair.U(r) / self.eb, 0.0, 1.0)
        rho = (1 - (1 - y) ** self.p) * (1 - _smooth5(y / self.dw))
        rho = np.where(y <= 0, 0.0, rho)
        rho = np.where(r <= self.rb, 1.0, rho)
        return np.where(r >= self.r0, 0.0, rho)

    def embed(self, n):
        return self.lam * hinge(np.asarray(n) - 1.0, "glue", self.d)

    def pair_energy(self, r):
        return 2.0 * self.embed(self.density(r))

    def triplet_energy(self, r12, r13, r23):
        a, b, c = self.density(r12), self.density(r13), self.density(r23)
        return self.embed(a + b) + self.embed(a + c) + self.embed(b + c)

    @property
    def reach(self):
        return self.r0


def _cosines(r12, r13, r23):
    """Cosines of the angles at particles 1, 2, 3."""
    c1 = (r12 ** 2 + r13 ** 2 - r23 ** 2) / (2 * r12 * r13)
    c2 = (r12 ** 2 + r23 ** 2 - r13 ** 2) / (2 * r12 * r23)
    c3 = (r13 ** 2 + r23 ** 2 - r12 ** 2) / (2 * r13 * r23)
    return [np.clip(c, -1, 1) for c in (c1, c2, c3)]


class AngularThreeBody(ManyBody):
    """KTS (Kohen-Tully-Stillinger / Skorpa) or Stillinger-Weber vertex terms.

    KTS: h = lam (1 + mu cos + nu cos^2) exp[g/(r1 - rc) + g/(r2 - rc)]
    SW:  h = lam (cos - cos0)^2       exp[g/(r1 - rc) + g/(r2 - rc)]
    """

    def __init__(self, kind="kts", lam=0.0, g=1.5, rc=2.8, mu=0.132587, nu=-0.2997, cos0=1.0, name=None):
        super().__init__(lam)
        self.kind, self.g, self.rc, self.mu, self.nu, self.cos0 = kind, g, rc, mu, nu, cos0
        self.name = name or kind

    def _h(self, r1, r2, c):
        inside = (r1 < self.rc) & (r2 < self.rc)
        d1 = np.where(inside, r1 - self.rc, -1.0)
        d2 = np.where(inside, r2 - self.rc, -1.0)
        e = np.exp(self.g / d1 + self.g / d2)
        ang = (c - self.cos0) ** 2 if self.kind == "sw" else 1 + self.mu * c + self.nu * c * c
        return np.where(inside, self.lam * ang * e, 0.0)

    def triplet_energy(self, r12, r13, r23):
        c1, c2, c3 = _cosines(r12, r13, r23)
        return self._h(r12, r13, c1) + self._h(r12, r23, c2) + self._h(r13, r23, c3)

    @property
    def reach(self):
        return self.rc


# ------------------------------------------------------------- defaults
def explorer_counter_defaults(pair, kind):
    """Counter geometry chosen exactly as the explorer's tbAutoInit()."""
    rb, rd, edge = pair.r_bond, pair.r_disp, pair.r_barrier
    if kind == "gauss":
        return Counter("gauss", r0=round(rb, 4), sigma=0.6)
    if kind == "tanh":
        return Counter("tanh", rs=round(edge, 3), w=0.1)
    if kind == "tanh_damped":
        return Counter("tanh_damped", rs=round(edge, 3), w=0.1, rcut=round(min(edge + 0.5, rd - 0.25), 3), dw=0.3)
    h0, h1 = rb - 0.1, min(edge + 0.2, rd - 0.3)
    W = max(0.2, h1 - h0)
    if kind in ("hermite", "cycloidal", "septic"):
        return Counter(kind, r1=h0 + 0.5 * W, W=W)
    if kind == "quintic":
        m = 0.5
        q0, q1 = rb, min(edge + 0.3, rd - 0.3)
        Wq = max(0.2, q1 - q0)
        return Counter("quintic", r1=q0 + _qT(m) * Wq, W=Wq, m=m)
    raise ValueError(kind)


def make_penalty(pair, kind, lam=None, **kw):
    """Factory used by the Stage-2 scripts.

    kind: 'coord_<counter>_<hinge>' (e.g. coord_hermite_cubic, coord_hermite_tanh_sq),
          'glue', 'kts', 'sw'.  ``lam=None`` -> strength left at 0 (set later).
    """
    if kind == "glue":
        return GluePenalty(pair, lam=lam, **kw)
    if kind in ("kts", "sw"):
        rc = kw.pop("rc", round(pair.r_barrier + 0.2, 3))
        return AngularThreeBody(kind, lam=lam or 0.0, rc=rc, **kw)
    if kind.startswith("coord_"):
        _, rest = kind.split("_", 1)
        for h in ("tanh_sq", "cubic"):
            if rest.endswith("_" + h):
                ck = rest[: -len(h) - 1]
                return CoordinationPenalty(explorer_counter_defaults(pair, ck), lam=lam or 0.0,
                                           hinge_kind=h, delta=kw.get("delta", 0.05), name=kind)
    raise ValueError(kind)
