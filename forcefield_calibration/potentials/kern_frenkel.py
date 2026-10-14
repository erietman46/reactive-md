"""Smooth Kern-Frenkel (patchy) modulation of a reactive pair potential.

Each bead carries one bonding patch, a unit vector n (for NO2: the C2 axis,
along which the unpaired electron of N points).  With r_hat the unit vector
from i to j,

    c_i = n_i . r_hat,   c_j = -n_j . r_hat,   f(c) = exp(-(1 - c) / w^2),
    S(Omega_i, Omega_j) = f(c_i) f(c_j)                   (0 < S <= 1)

    U(r, Omega_i, Omega_j) = S U_on(r) + (1 - S) U_off(r)
                           = U_off(r) - S W(r),           W = U_off - U_on >= 0

* ``U_on`` (patches facing each other) is the calibrated isotropic reactive
  potential of one of the radial families (bond well, barrier, non-bonded
  branch), unchanged;
* ``U_off`` (patches turned away) is the same curve plus the repulsive
  "misalignment" wall W(r) = A (r_s - r)^3 for r < r_s = r_h + ds, zero
  beyond.  W, W' and W'' vanish at r_s, so U is C2 in r and the non-bonded
  region r >= r_s stays exactly isotropic.  A is set by W(r_b) = c_W |U_on(r_b)|
  (default c_W = 2: a misaligned pair at the bond length sits as far above
  zero as the aligned one sits below it), raised if necessary to 1.25 times
  the smallest A for which U_off decreases monotonically for all r < r_s
  (misaligned pairs feel no well and no bump).

This is the Kern-Frenkel decomposition U = U_iso(r) + U_patch(r) S(Omega) with
U_iso = U_off and U_patch = -W, but with the hard cone replaced by the
Gaussian f (Wilber-Doye-Louis type), which keeps forces and torques
continuous.  For small angles f ~ exp(-theta^2 / 2 w^2): ``w`` is the angular
width of the patch (rad).

Thermodynamics.  The momenta (translational and rotational) integrate out and
cancel between monomers and dimers, so every dilute-gas quantity only needs
the orientation average at fixed r.  With c_i, c_j independent and uniform on
[-1, 1] and u = (1 - c)/w^2, the angular energy E_ang = (1 - S) W(r) depends
only on s = u_i + u_j, whose density is (w^4/4) min(s, 2L - s) on [0, 2L],
L = 2/w^2.  Hence

    <exp(-beta E_ang)>(r) = M0(beta W(r)),   M0(b) = int rho(s) exp(-b (1 - e^-s)) ds

is a one-dimensional function of b = beta W, tabulated once per w (with the
first two energy moments).  For a deep, aligned bond (b >> 1) M0 -> (w^4/4)/b^2:
each patch keeps a fraction ~ RT w^2 / (2 W) of its orientations - the
orientational entropy that an isotropic bead lacks - and each of the four
librational coordinates adds RT/2 to the bound-state energy.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline

from ..constants import AMU, NA, R_KJ
from .additive import AdditivePotential
from .base import PairPotential
from .piecewise import PiecewisePotential

RADIAL_CLASSES = {"PiecewisePotential": PiecewisePotential, "AdditivePotential": AdditivePotential}
ANGULAR_DEFAULTS = dict(w=0.5, cW=2.0, ds=0.5)
MONO_MARGIN = 1.25

# ------------------------------------------------------------ orientation averages
_GLX, _GLW = np.polynomial.legendre.leggauss(8)
_LNB = np.linspace(np.log(1e-6), np.log(1e9), 721)


def _s_nodes(w):
    L = 2.0 / (w * w)
    edges = np.unique(np.r_[0.0, np.geomspace(1e-12, 2 * L, 110), L])
    a, b = edges[:-1, None], edges[1:, None]
    h = 0.5 * (b - a)
    s = (0.5 * (a + b) + h * _GLX).ravel()
    ws = (h * _GLW).ravel()
    rho = 0.25 * w ** 4 * np.minimum(s, 2 * L - s)
    return s, ws * rho


@lru_cache(maxsize=64)
def _tables(w):
    """Splines in ln b of ln M0, M1 = <(1-S) e^{-b(1-S)}>/M0 and M2 = <(1-S)^2 e^{-b(1-S)}>/M0."""
    s, wr = _s_nodes(w)
    y = -np.expm1(-s)  # 1 - S
    b = np.exp(_LNB)[:, None]
    e = np.exp(-b * y[None, :]) * wr[None, :]
    m0 = e.sum(axis=1)
    m1 = (e * y).sum(axis=1) / m0
    m2 = (e * y * y).sum(axis=1) / m0
    mean0 = float((wr * y).sum())          # b -> 0 limits
    mean0_2 = float((wr * y * y).sum())
    return (CubicSpline(_LNB, np.log(m0)), CubicSpline(_LNB, np.log(m1)), CubicSpline(_LNB, np.log(m2)),
            mean0, mean0_2)


def angular_moments(b, w):
    """(ln M0, M1, M2) at b = beta W >= 0 (vectorised), see module docstring."""
    b = np.asarray(b, float)
    lm0_s, lm1_s, lm2_s, y1, y2 = _tables(float(w))
    lo, hi = np.exp(_LNB[0]), np.exp(_LNB[-1])
    lb = np.log(np.clip(b, lo, hi))
    lm0, m1, m2 = lm0_s(lb), np.exp(lm1_s(lb)), np.exp(lm2_s(lb))
    small = b < lo                       # first order in b
    lm0 = np.where(small, -b * y1, lm0)
    m1 = np.where(small, y1, m1)
    m2 = np.where(small, y2, m2)
    big = b > hi                         # Laplace limit: rho ~ (w^4/4) s
    with np.errstate(divide="ignore"):
        lbig = np.log(0.25 * w ** 4) - 2.0 * np.log(np.maximum(b, 1e-300))
    lm0 = np.where(big, lbig, lm0)
    m1 = np.where(big, 2.0 / np.maximum(b, 1e-150), m1)
    with np.errstate(divide="ignore", over="ignore"):
        m2 = np.where(big, 6.0 / np.maximum(b, 1e-150) ** 2, m2)
    return lm0, m1, m2


def patch_factor(c, w):
    """f(c) = exp(-(1 - c)/w^2) and df/dc."""
    f = np.exp(-(1.0 - np.asarray(c, float)) / (w * w))
    return f, f / (w * w)


# ------------------------------------------------------------ potential
class KernFrenkelPotential(PairPotential):
    """Smooth Kern-Frenkel wrapper around a radial reactive potential.

    The scalar interface (``U``, ``dU``, ``landscape``, ``with_bond``, ...) is
    that of the *aligned* radial curve U_on, so every Stage-1 tool works on it;
    the orientation dependence enters the thermodynamics through
    :meth:`angular_terms` and the full anisotropic energy through
    :meth:`energy`.
    """

    family = "kern_frenkel"
    anisotropic = True

    def __init__(self, name: str | None = None, radial_class: str = "PiecewisePotential", **params):
        ang = dict(ANGULAR_DEFAULTS)
        for k in list(params):
            if k in ang:
                ang[k] = float(params.pop(k))
        cls = RADIAL_CLASSES[radial_class]
        self.radial = cls(name=name, **params)
        self.radial_class = radial_class
        self.ang = ang
        self.name = name or f"KF[{self.radial.name}]"
        self.p = dict(self.radial.p, radial_class=radial_class, **ang)
        self._landscape = None
        self._setup()

    def _setup(self):
        rad = self.radial
        p = rad.p
        rh = p["rh"] if "rh" in p else p["rx"]
        self.r_s = float(rh + self.ang["ds"])
        rb = p["rb"]
        A_target = self.ang["cW"] * abs(float(rad.U(np.array([rb]))[0])) / (self.r_s - rb) ** 3
        r = np.linspace(0.6 * rb, self.r_s - 1e-6, 3000)
        self.A_mono = float(np.max(np.maximum(rad.dU(r), 0.0) / (3.0 * (self.r_s - r) ** 2)))
        self.A = max(A_target, MONO_MARGIN * self.A_mono)
        self.W_b = self.A * (self.r_s - rb) ** 3
        self.kh = getattr(rad, "kh", None)
        self.kb = getattr(rad, "kb", None)
        self.misaligned_monotonic = self._off_monotonic()
        self.feasible = bool(getattr(rad, "feasible", True) and self.misaligned_monotonic)

    # ---------------------------------------------------- radial pieces
    def wall(self, r):
        """W(r) = A (r_s - r)^3 for r < r_s and its derivative."""
        r = np.asarray(r, float)
        x = np.maximum(self.r_s - r, 0.0)
        return self.A * x ** 3, -3.0 * self.A * x * x

    def _ud(self, r):
        return self.radial._ud(r)

    def U_off(self, r):
        return self.radial.U(r) + self.wall(r)[0]

    def dU_off(self, r):
        return self.radial.dU(r) + self.wall(r)[1]

    def _off_monotonic(self, n=3000):
        """The misaligned curve must be purely repulsive inside r_s (no well, no bump)."""
        r = np.linspace(0.6 * self.radial.p["rb"], self.r_s - 1e-6, n)
        return bool(np.all(self.dU_off(r) < 0.0))

    # ---------------------------------------------------- delegation
    @property
    def rcut(self):
        return self.radial.rcut

    @property
    def r_end(self):
        return self.radial.r_end

    def breakpoints(self):
        return list(self.radial.breakpoints()) + [self.r_s]

    def nominal_bond_radius(self):
        return self.radial.nominal_bond_radius()

    def barrier_controls(self):
        return self.radial.barrier_controls()

    def with_bond(self, eb=None, kb=None, rb=None, name=None):
        rad = self.radial.with_bond(eb=eb, kb=kb, rb=rb)
        return type(self)(name=name or self.name, radial_class=self.radial_class, **rad.p, **self.ang)

    def replace(self, name=None, **kw):
        p = {k: v for k, v in self.p.items() if k != "radial_class"}
        p.update(kw)
        return type(self)(name=name or self.name, radial_class=self.radial_class, **p)

    # ---------------------------------------------------- thermodynamics
    def angular_terms(self, r, T):
        """(ln <e^{-beta E_ang}>, <E_ang>, <E_ang^2>) over both orientations at fixed r.

        The averages are weighted by e^{-beta E_ang} (conditional moments), so
        that bound-state averages follow by one more radial quadrature."""
        W = self.wall(r)[0]
        b = W / (R_KJ * T)
        lm0, m1, m2 = angular_moments(b, self.ang["w"])
        return lm0, W * m1, W * W * m2

    def angular_stiffness(self):
        """kappa = W(r_b)/w^2 (kJ/mol/rad^2): U ~ U_on + kappa (theta_i^2 + theta_j^2)/2 near alignment."""
        return self.W_b / self.ang["w"] ** 2

    def libration(self, inertia_amuA2):
        """Harmonic libration wavenumber (cm^-1) and period (fs) of one patch about an axis with
        moment of inertia I (amu A^2) when the partner is held fixed."""
        from ..constants import C_CM

        k = self.angular_stiffness() * 1e3 / NA            # J/rad^2 per molecule
        I = inertia_amuA2 * AMU * 1e-20                    # kg m^2
        omega = np.sqrt(k / I)
        return float(omega / (2 * np.pi * C_CM)), float(2 * np.pi / omega * 1e15)

    def equivalent_cone(self, T):
        """Half-angle (deg) of a hard Kern-Frenkel cone with the same per-patch bonded fraction at r_b."""
        lm0, _, _ = self.angular_terms(np.array([self.r_bond]), T)
        chi = np.exp(0.5 * lm0[0])
        return float(np.rad2deg(np.arccos(1.0 - 2.0 * chi)))

    # ---------------------------------------------------- full anisotropic energy
    def energy(self, rij, ni, nj):
        """U and its gradients for vectors r_ij = r_j - r_i (..., 3) and unit patch vectors.

        Returns (U, dU/dr_ij, dU/dn_i, dU/dn_j); the force on j is -dU/dr_ij,
        the torque on i is n_i x (-dU/dn_i)."""
        rij, ni, nj = (np.asarray(a, float) for a in (rij, ni, nj))
        r = np.linalg.norm(rij, axis=-1)
        e = rij / r[..., None]
        ci = np.sum(ni * e, axis=-1)
        cj = -np.sum(nj * e, axis=-1)
        w = self.ang["w"]
        fi, dfi = patch_factor(ci, w)
        fj, dfj = patch_factor(cj, w)
        S = fi * fj
        uon, duon = self.radial._ud(np.atleast_1d(r))
        Wr, dW = self.wall(r)
        uon, duon = uon.reshape(r.shape), duon.reshape(r.shape)
        U = uon + (1.0 - S) * Wr
        dUdr = duon + (1.0 - S) * dW
        dci = (ni - ci[..., None] * e) / r[..., None]
        dcj = (-nj - cj[..., None] * e) / r[..., None]
        gS = (dfi * fj)[..., None] * dci + (fi * dfj)[..., None] * dcj
        grad_r = dUdr[..., None] * e - Wr[..., None] * gS
        grad_ni = -(Wr * dfi * fj)[..., None] * e
        grad_nj = (Wr * fi * dfj)[..., None] * e
        return U, grad_r, grad_ni, grad_nj

    def describe(self) -> dict:
        d = super().describe()
        d.update(w=self.ang["w"], w_deg=float(np.rad2deg(self.ang["w"])), cW=self.ang["cW"], r_s=self.r_s,
                 W_b=self.W_b, kappa=self.angular_stiffness(), misaligned_monotonic=self.misaligned_monotonic)
        return d
