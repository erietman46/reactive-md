"""Analytic building blocks shared by both pair-potential constructions.

Every function is vectorised over ``r`` and returns ``(u, du/dr)``.  The forms
and parameterisations are direct ports of ``Pair_potential_explorer_2_1.html``
so that a parameter set copied from the explorer gives the same curve here.

Each bond-well function is parameterised by its minimum position ``r0`` and its
depth ``eps`` (minimum value ``-eps``); the remaining shape parameter sets the
curvature ``k = u''(r0)``.  ``shape_for_stiffness`` inverts that relation so
that all bond forms can be fitted with the same physical pair ``(eps_b, k_b)``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def _arr(r):
    return np.asarray(r, dtype=float)


# --------------------------------------------------------------------- Morse
def morse(r, re, D, b):
    """Shifted Morse, minimum -D at re, zero at infinity."""
    r = _arr(r)
    e = np.exp(np.minimum(-b * (r - re), 300.0))
    return D * (e * e - 2 * e), 2 * D * b * (e - e * e)


def morse_k(re, D, b):
    return 2.0 * D * b * b


# ----------------------------------------------------------------------- Mie
def mie(r, r0, e, n, m):
    """Mie n-m with minimum -e at r0 (12-6 is Lennard-Jones)."""
    r = _arr(r)
    with np.errstate(over="ignore", divide="ignore"):
        x = r0 / r
        xn, xm = x ** n, x ** m
        u = e / (n - m) * (m * xn - n * xm)
        du = e / (n - m) * (-m * n * xn + n * m * xm) / r
    # extreme exponents overflow (inf - inf) deep inside the repulsive wall
    bad = ~np.isfinite(u)
    if np.any(bad):
        u = np.where(bad, 1e300, u)
        du = np.where(bad, -1e300, du)
    return u, du


def mie_k(r0, e, n, m):
    return e * n * m / r0 ** 2


# ------------------------------------------------------------- Wang-Frenkel
def wf_par(rmin, eps, x):
    """Wang-Frenkel (mu = nu = 1) parameterised by its minimum and x = rc/sigma."""
    sigma = rmin / (x * np.sqrt(3.0 / (1.0 + 2.0 * x * x)))
    rc = x * sigma
    alpha = 2.0 * x * x * (3.0 / (2.0 * (x * x - 1.0))) ** 3
    return dict(sigma=sigma, rc=rc, alpha=alpha, eps=eps)


def wf(r, q):
    r = _arr(r)
    a, b = q["sigma"] ** 2, q["rc"] ** 2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        f = a / (r * r) - 1.0
        g = b / (r * r) - 1.0
        fp = -2.0 * a / r ** 3
        gp = -2.0 * b / r ** 3
        u = q["eps"] * q["alpha"] * f * g * g
        du = q["eps"] * q["alpha"] * (fp * g * g + 2.0 * f * g * gp)
    inside = r < q["rc"]
    return np.where(inside, u, 0.0), np.where(inside, du, 0.0)


def wf_k(q, rmin):
    a, b, r = q["sigma"] ** 2, q["rc"] ** 2, rmin
    r2 = r * r
    f, g = a / r2 - 1, b / r2 - 1
    fp, gp = -2 * a / (r2 * r), -2 * b / (r2 * r)
    fpp, gpp = 6 * a / (r2 * r2), 6 * b / (r2 * r2)
    return q["eps"] * q["alpha"] * (fpp * g * g + 4 * fp * g * gp + 2 * f * (gp * gp + g * gpp))


# ------------------------------------------------- Kohen-Tully-Stillinger
def kts_par(rb, eb, p, g2, rc):
    """alpha, beta fixed so that the KTS pair term has minimum -eb at rb."""
    D = g2 / (rb - rc) ** 2
    beta = D * rb ** p / (D + p / rb)
    alpha = eb / ((1.0 - beta * rb ** (-p)) * np.exp(g2 / (rb - rc)))
    return dict(alpha=alpha, beta=beta, p=p, g=g2, rc=rc)


def kts(r, q):
    r = _arr(r)
    d = np.where(r < q["rc"], r - q["rc"], -1.0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        e = np.exp(q["g"] / d)
        g = q["beta"] * r ** (-q["p"]) - 1.0
        gp = -q["p"] * q["beta"] * r ** (-q["p"] - 1.0)
        u = q["alpha"] * g * e
        du = q["alpha"] * e * (gp - g * q["g"] / (d * d))
    inside = r < q["rc"]
    return np.where(inside, u, 0.0), np.where(inside, du, 0.0)


def kts_k(q, r):
    d = r - q["rc"]
    e = np.exp(q["g"] / d)
    g = q["beta"] * r ** (-q["p"]) - 1
    gp = -q["p"] * q["beta"] * r ** (-q["p"] - 1)
    gpp = q["p"] * (q["p"] + 1) * q["beta"] * r ** (-q["p"] - 2)
    ep = -q["g"] / (d * d) * e
    epp = e * (q["g"] ** 2 / d ** 4 + 2 * q["g"] / d ** 3)
    return q["alpha"] * (gpp * e + 2 * gp * ep + g * epp)


# ------------------------------------------------------------ dispatch layer
BOND_FORMS = ("morse", "mie", "wf", "kts")
FLUID_FORMS = ("morse", "mie", "wf", "kts")


def bond_branch(r, p):
    """Bond-well function of the explorer for parameter dict ``p`` (keys rb, eb, ...)."""
    f = p["formB"]
    if f == "morse":
        return morse(r, p["rb"], p["eb"], p["bb"])
    if f == "mie":
        return mie(r, p["rb"], p["eb"], p["n"], p["m"])
    if f == "wf":
        return wf(r, wf_par(p["rb"], p["eb"], p["xb"]))
    if f == "kts":
        return kts(r, kts_par(p["rb"], p["eb"], p["pk"], p["g2"], p["rcb"]))
    raise ValueError(f"unknown bond form {f!r}")


def bond_k(p):
    f = p["formB"]
    if f == "morse":
        return morse_k(p["rb"], p["eb"], p["bb"])
    if f == "mie":
        return mie_k(p["rb"], p["eb"], p["n"], p["m"])
    if f == "wf":
        return wf_k(wf_par(p["rb"], p["eb"], p["xb"]), p["rb"])
    if f == "kts":
        return kts_k(kts_par(p["rb"], p["eb"], p["pk"], p["g2"], p["rcb"]), p["rb"])
    raise ValueError(f)


def fluid_branch(r, p):
    f = p["formF"]
    if f == "morse":
        return morse(r, p["rf"], p["ef"], p["bf"])
    if f == "mie":
        return mie(r, p["rf"], p["ef"], p["nf"], p["mf"])
    if f == "wf":
        return wf(r, wf_par(p["rf"], p["ef"], p["xf"]))
    if f == "kts":
        return kts(r, kts_par(p["rf"], p["ef"], p["pkf"], p["g2f"], p["rcf"]))
    raise ValueError(f"unknown fluid form {f!r}")


def fluid_k(p):
    f = p["formF"]
    if f == "morse":
        return morse_k(p["rf"], p["ef"], p["bf"])
    if f == "mie":
        return mie_k(p["rf"], p["ef"], p["nf"], p["mf"])
    if f == "wf":
        return wf_k(wf_par(p["rf"], p["ef"], p["xf"]), p["rf"])
    if f == "kts":
        return kts_k(kts_par(p["rf"], p["ef"], p["pkf"], p["g2f"], p["rcf"]), p["rf"])
    raise ValueError(f)


def fluid_cutoff(p):
    """Intrinsic cut-off of the non-bonded branch (inf for Morse/Mie)."""
    f = p["formF"]
    if f == "wf":
        return float(wf_par(p["rf"], p["ef"], p["xf"])["rc"])
    if f == "kts":
        return float(p["rcf"])
    return np.inf


def shape_for_stiffness(p, kb):
    """Return the shape-parameter update that gives bond curvature ``kb``.

    morse: beta_b = sqrt(kb / 2 eb)
    mie:   n = 2m (LJ-like ratio), n m = kb rb^2 / eb
    wf:    x = rc/sigma solved numerically
    kts:   gamma_2 solved numerically (r_c,b and p fixed)
    """
    f, rb, eb = p["formB"], p["rb"], p["eb"]
    if f == "morse":
        return {"bb": float(np.sqrt(kb / (2.0 * eb)))}
    if f == "mie":
        m = float(np.sqrt(kb * rb * rb / (2.0 * eb)))
        return {"n": 2.0 * m, "m": m}
    if f == "wf":
        g = lambda x: wf_k(wf_par(rb, eb, x), rb) - kb
        lo, hi = 1.0005, 50.0
        if g(lo) * g(hi) > 0:
            raise ValueError("requested stiffness outside the Wang-Frenkel range")
        return {"xb": brentq(g, lo, hi, xtol=1e-12)}
    if f == "kts":
        def g(g2):
            q = kts_par(rb, eb, p["pk"], g2, p["rcb"])
            return kts_k(q, rb) - kb
        lo, hi = 1e-4, 200.0
        if g(lo) * g(hi) > 0:
            raise ValueError("requested stiffness outside the KTS range")
        return {"g2": brentq(g, lo, hi, xtol=1e-12)}
    raise ValueError(f)


# ------------------------------------------------- additive-model pieces
def gauss(r, c, s):
    r = _arr(r)
    z = (r - c) / s
    g = np.exp(-z * z)
    return g, -2.0 * (r - c) / (s * s) * g


def sw5(r, a, b):
    """Quintic switch: C = 1 for r <= a, 0 for r >= b, C2 continuous. Returns (C, dC/dr)."""
    r = _arr(r)
    L = b - a
    t = np.clip((r - a) / L, 0.0, 1.0)
    C = 1.0 - t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t)
    dC = -30.0 * t * t * (1.0 - t) ** 2 / L
    return C, dC


def tt6(x):
    s = np.ones_like(x)
    t = np.ones_like(x)
    for k in range(1, 7):
        t = t * x / k
        s = s + t
    return 1.0 - np.exp(-x) * s


def dtt6(x):
    return np.exp(-x) * x ** 6 / 720.0
