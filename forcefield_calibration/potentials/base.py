"""Common interface of all pair potentials used in the calibration."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Extremum:
    r: float
    u: float
    kind: str  # "min" or "max"


@dataclass
class Landscape:
    """Classified stationary points of U(r)."""

    bond: Extremum | None
    barrier: Extremum | None
    disp: Extremum | None
    all: list = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """Exactly bond minimum < barrier maximum < dispersion minimum and nothing else."""
        kinds = [e.kind for e in self.all]
        return (
            kinds == ["min", "max", "min"]
            and self.bond is not None
            and self.barrier is not None
            and self.disp is not None
            and self.bond.r < self.barrier.r < self.disp.r
            and self.bond.u < self.disp.u < self.barrier.u
        )


class PairPotential:
    """Base class. Subclasses implement ``_ud(r)`` returning (U, dU/dr) in kJ/mol, kJ/mol/A."""

    family = "base"
    defaults: dict = {}
    anisotropic = False

    def __init__(self, name: str | None = None, **params):
        p = copy.deepcopy(self.defaults)
        unknown = set(params) - set(p)
        if unknown:
            raise KeyError(f"unknown parameters for {type(self).__name__}: {sorted(unknown)}")
        p.update(params)
        self.p = p
        self.name = name or self.family
        self._landscape = None
        self._setup()

    # -- to be overridden -------------------------------------------------
    def _setup(self):
        pass

    def _ud(self, r):
        raise NotImplementedError

    @property
    def rcut(self) -> float:
        return np.inf

    def breakpoints(self) -> list[float]:
        """Radii where the functional form changes (used as quadrature panel edges)."""
        return []

    def nominal_bond_radius(self) -> float:
        return self.p.get("rb", 1.0)

    def angular_terms(self, r, T):
        """Orientation average at fixed r: (ln <exp(-beta E_ang)>, <E_ang>, <E_ang^2>).

        Zero for isotropic potentials; overridden by the Kern-Frenkel wrapper,
        whose U(r) is then the aligned curve and E_ang the orientation-dependent rest."""
        z = np.zeros_like(np.asarray(r, float))
        return z, z, z

    # -- public API -------------------------------------------------------
    def replace(self, name: str | None = None, **kw) -> "PairPotential":
        p = dict(self.p)
        p.update(kw)
        return type(self)(name=name or self.name, **p)

    def U(self, r):
        return self._ud(np.asarray(r, dtype=float))[0]

    def dU(self, r):
        return self._ud(np.asarray(r, dtype=float))[1]

    def force(self, r):
        return -self.dU(r)

    def d2U(self, r, h=1e-5):
        r = np.asarray(r, dtype=float)
        return (self.dU(r + h) - self.dU(r - h)) / (2.0 * h)

    @property
    def r_end(self) -> float:
        """Upper end of the range searched for extrema."""
        rc = self.rcut
        return rc if np.isfinite(rc) else 4.0 * self.nominal_bond_radius() + 10.0

    def extrema(self, n: int = 8000) -> list[Extremum]:
        r0 = max(0.3, 0.45 * self.nominal_bond_radius())
        r1 = self.r_end - 1e-6
        r = np.linspace(r0, r1, n + 1)
        d = self.dU(r)
        idx = np.nonzero((d[:-1] < 0) & (d[1:] >= 0) | (d[:-1] > 0) & (d[1:] <= 0))[0]
        if idx.size == 0:
            return []
        lo, hi = r[idx].copy(), r[idx + 1].copy()
        dlo = d[idx].copy()
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            dm = self.dU(mid)
            same = np.sign(dm) == np.sign(dlo)
            lo = np.where(same, mid, lo)
            dlo = np.where(same, dm, dlo)
            hi = np.where(same, hi, mid)
        rr = 0.5 * (lo + hi)
        uu = self.U(rr)
        scale = max(1.0, abs(float(np.min(uu))) if uu.size else 1.0)
        out = []
        for ri, ui, di in zip(rr, uu, d[idx]):
            if abs(ui) <= 1e-9 * scale:  # spurious stationary points in the far tail
                continue
            out.append(Extremum(float(ri), float(ui), "min" if di < 0 else "max"))
        return out

    def landscape(self) -> Landscape:
        if self._landscape is None:
            ex = self.extrema()
            mins = [e for e in ex if e.kind == "min"]
            maxs = [e for e in ex if e.kind == "max"]
            bond = min(mins, key=lambda e: e.u) if mins else None
            barrier = None
            if bond is not None:
                after = [e for e in maxs if e.r > bond.r]
                barrier = max(after, key=lambda e: e.u) if after else None
            disp = None
            if barrier is not None:
                outer = [e for e in mins if e.r > barrier.r]
                disp = min(outer, key=lambda e: e.u) if outer else None
            self._landscape = Landscape(bond, barrier, disp, ex)
        return self._landscape

    # convenient shortcuts
    @property
    def r_bond(self):
        L = self.landscape()
        return L.bond.r if L.bond else np.nan

    @property
    def u_bond(self):
        L = self.landscape()
        return L.bond.u if L.bond else np.nan

    @property
    def r_barrier(self):
        L = self.landscape()
        return L.barrier.r if L.barrier else np.nan

    @property
    def u_barrier(self):
        L = self.landscape()
        return L.barrier.u if L.barrier else np.nan

    @property
    def r_disp(self):
        L = self.landscape()
        return L.disp.r if L.disp else np.nan

    @property
    def u_disp(self):
        L = self.landscape()
        return L.disp.u if L.disp else np.nan

    def bond_stiffness(self) -> float:
        """Actual curvature U''(r) at the actual bond minimum (kJ/mol/A^2).

        The finite-difference step is shrunk until it is small compared with
        the harmonic length 1/sqrt(k) (matters for extremely stiff wells)."""
        rb = self.r_bond
        if not np.isfinite(rb):
            return np.nan
        h = 1e-5
        k = float(self.d2U(rb, h=h))
        for _ in range(6):
            h_new = min(1e-5, 1e-3 / np.sqrt(max(abs(k), 1e-12)))
            if h_new >= 0.5 * h:
                break
            h = h_new
            k = float(self.d2U(rb, h=h))
        return k

    def bond_wavenumber(self, reduced_mass_gmol: float) -> float:
        """Harmonic bond-stretch wavenumber (cm^-1) for a reduced mass in g/mol."""
        from ..constants import C_CM

        k = self.bond_stiffness() * 1e3 * 1e20  # J/mol/m^2
        mu = reduced_mass_gmol * 1e-3            # kg/mol
        return float(np.sqrt(k / mu) / (2 * np.pi * C_CM))

    def describe(self) -> dict:
        L = self.landscape()
        return dict(
            name=self.name,
            family=self.family,
            clean=L.clean,
            r_bond=self.r_bond,
            u_bond=self.u_bond,
            r_barrier=self.r_barrier,
            u_barrier=self.u_barrier,
            r_disp=self.r_disp,
            u_disp=self.u_disp,
            k_bond=self.bond_stiffness(),
            rcut=self.rcut,
        )

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r}, {self.p})"
