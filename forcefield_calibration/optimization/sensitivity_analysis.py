"""Local sensitivity and identifiability of fitted parameters.

S_ij = d O_i / d ln(theta_j), with every observable O_i divided by its
uncertainty, is computed by central differences.  Its singular values show how
many parameter combinations the data actually determine: a singular value much
smaller than the largest one (large condition number) marks a combination of
parameters that the calibration data cannot distinguish, e.g. a deeper but
narrower bond well with the same Z_b.  The approximate parameter correlation
matrix follows from (S^T S)^-1.
"""
from __future__ import annotations

import numpy as np

from ..thermodynamics.equilibrium_integrals import bound_state
from ..thermodynamics.kinetics_tst import tst_rates
from ..thermodynamics.virial_coefficients import B2


def observables(pot, T_K, T_H, T_B, mu_gmol):
    """Dict of named observables used in the sensitivity analysis."""
    out = {}
    for t in T_K:
        out[f"lnK({t:g})"] = bound_state(pot, t).log_Kp
    for t in T_H:
        out[f"dH({t:g})"] = bound_state(pot, t).deltaH
    for t in T_B:
        out[f"B2frozen({t:g})"] = B2(pot, t).frozen
    out[f"ln_kTST({T_K[0]:g})"] = tst_rates(pot, T_K[0], mu_gmol).log_kd
    return out


def sensitivity(builder, names, values, obs_fn, scales, rel=1e-3):
    """Normalised Jacobian d(O/scale)/d ln(theta) and its SVD.

    builder(dict) -> potential; obs_fn(pot) -> dict of observables;
    scales: dict observable -> uncertainty used for normalisation.
    """
    base = obs_fn(builder(dict(zip(names, values))))
    keys = list(base)
    J = np.zeros((len(keys), len(names)))
    for j, n in enumerate(names):
        vp = dict(zip(names, values))
        vm = dict(zip(names, values))
        vp[n] = values[j] * (1 + rel)
        vm[n] = values[j] * (1 - rel)
        op, om = obs_fn(builder(vp)), obs_fn(builder(vm))
        for i, k in enumerate(keys):
            J[i, j] = (op[k] - om[k]) / (np.log1p(rel) - np.log1p(-rel)) / scales.get(k, 1.0)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    cov = None
    corr = None
    try:
        cov = np.linalg.inv(J.T @ J)
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
    except np.linalg.LinAlgError:
        pass
    return dict(observables=keys, params=list(names), J=J, singular_values=s, Vt=Vt,
                condition_number=float(s[0] / s[-1]) if s[-1] > 0 else np.inf,
                correlation=corr, base=base)
