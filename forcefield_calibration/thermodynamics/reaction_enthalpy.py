"""Van't Hoff analysis of model (or reference) equilibrium constants."""
from __future__ import annotations

import numpy as np

from ..constants import R_KJ
from .equilibrium_integrals import bound_state


def vant_hoff_fit(T, lnK):
    """Least-squares line ln K = a - dH/(R T). Returns (dH kJ/mol, dS J/mol/K, residual rms)."""
    T, lnK = np.asarray(T, float), np.asarray(lnK, float)
    A = np.vstack([np.ones_like(T), 1.0 / T]).T
    (a, b), *_ = np.linalg.lstsq(A, lnK, rcond=None)
    res = lnK - A @ np.array([a, b])
    return -b * R_KJ, a * R_KJ * 1e3, float(np.sqrt(np.mean(res ** 2)))


def deltaH_vant_hoff_local(pot, T, dT=0.5):
    """Numerical derivative -R d lnK / d(1/T) at T (central difference)."""
    l1 = bound_state(pot, T - dT).log_Kp
    l2 = bound_state(pot, T + dT).log_Kp
    return -R_KJ * (l2 - l1) / (1.0 / (T + dT) - 1.0 / (T - dT))


def deltaH_analytic(pot, T):
    """<U>_b - RT (exact for a T-independent potential and dividing surface)."""
    return bound_state(pot, T).deltaH
