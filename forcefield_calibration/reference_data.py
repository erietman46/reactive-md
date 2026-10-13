"""Reference (experimental / thermochemical) calibration data.

Primary target: the gas-phase association 2 NO2 -> N2O4 at p0 = 1 bar,

    ln K_p(T) = -[G(N2O4) - 2 G(NO2)] / RT,   dH(T) = H(N2O4) - 2 H(NO2).

Sources, in order of preference:

1. Raw NIST-JANAF text tables placed in ``data/janaf/NO2.txt`` and
   ``data/janaf/N2O4.txt`` (download from janaf.nist.gov; the build container
   could not reach that host).  Parsed by :func:`read_janaf_table`.
2. NASA-7 polynomials (``data/nasa7_NO2_N2O4.json``) - used to generate a
   smooth curve on any temperature grid.
3. JANAF anchor values quoted in the project notes
   (``data/janaf_anchor_values.csv``) - used to validate source 2 and to set
   the data uncertainty from the inter-compilation spread.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .constants import R_J

DATA = Path(__file__).resolve().parent / "data"


def _read_csv(path):
    with open(path) as f:
        rows = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    return list(csv.DictReader(rows))


# ---------------------------------------------------------------- NASA-7
def load_nasa7(path=DATA / "nasa7_NO2_N2O4.json"):
    return json.loads(Path(path).read_text())


def _coeffs(sp, T):
    lo, mid, hi = sp["T_ranges"]
    if np.any(np.asarray(T) < lo) or np.any(np.asarray(T) > hi):
        raise ValueError("temperature outside NASA-7 validity range")
    return np.where(np.asarray(T)[..., None] <= mid, np.array(sp["low"]), np.array(sp["high"]))


def nasa7_H(sp, T):
    """Enthalpy (J/mol) including formation enthalpy."""
    T = np.asarray(T, float)
    a = _coeffs(sp, T)
    a = np.moveaxis(a, -1, 0)
    return R_J * T * (a[0] + a[1] * T / 2 + a[2] * T ** 2 / 3 + a[3] * T ** 3 / 4 + a[4] * T ** 4 / 5 + a[5] / T)


def nasa7_S(sp, T):
    """Absolute entropy (J/mol/K) at p0."""
    T = np.asarray(T, float)
    a = np.moveaxis(_coeffs(sp, T), -1, 0)
    return R_J * (a[0] * np.log(T) + a[1] * T + a[2] * T ** 2 / 2 + a[3] * T ** 3 / 3 + a[4] * T ** 4 / 4 + a[6])


def nasa7_reaction(T, nasa=None):
    """(ln K_p, dH kJ/mol, dS J/mol/K) of 2 NO2 -> N2O4 from NASA-7 polynomials."""
    nasa = nasa or load_nasa7()
    T = np.asarray(T, float)
    dH = nasa7_H(nasa["N2O4"], T) - 2 * nasa7_H(nasa["NO2"], T)
    dS = nasa7_S(nasa["N2O4"], T) - 2 * nasa7_S(nasa["NO2"], T)
    dG = dH - T * dS
    return -dG / (R_J * T), dH / 1e3, dS


# ---------------------------------------------------------------- JANAF
def read_janaf_table(path):
    """Parse a NIST-JANAF text table. Returns dict of arrays T, dfH (kJ/mol), dfG (kJ/mol)."""
    T, dfH, dfG = [], [], []
    for line in Path(path).read_text().splitlines():
        parts = line.replace("\t", " ").split()
        if len(parts) < 7:
            continue
        try:
            t = float(parts[0])
            h = float(parts[5])
            g = float(parts[6])
        except ValueError:
            continue
        T.append(t)
        dfH.append(h)
        dfG.append(g)
    return dict(T=np.array(T), dfH=np.array(dfH), dfG=np.array(dfG))


def janaf_reaction(no2_path, n2o4_path):
    """(T, ln K_p, dH) of 2 NO2 -> N2O4 on the common JANAF temperature grid."""
    a, b = read_janaf_table(no2_path), read_janaf_table(n2o4_path)
    T = np.intersect1d(a["T"], b["T"])
    T = T[T > 0]
    ia = np.searchsorted(a["T"], T)
    ib = np.searchsorted(b["T"], T)
    dG = b["dfG"][ib] - 2 * a["dfG"][ia]
    dH = b["dfH"][ib] - 2 * a["dfH"][ia]
    return T, -dG * 1e3 / (R_J * T), dH


def janaf_anchors(path=DATA / "janaf_anchor_values.csv"):
    rows = _read_csv(path)
    T = np.array([float(r["T_K"]) for r in rows])
    K = np.array([float(r["Kp_assoc"]) for r in rows])
    H = np.array([float(r["deltaH_kJmol"]) for r in rows])
    return T, np.log(K), H


# ------------------------------------------------------- calibration table
def build_reference_table(T_grid, sigma_lnK=None, sigma_dH=None):
    """Reference ln K_p and dH on ``T_grid`` with uncertainties.

    If raw JANAF tables are present in data/janaf/ they are interpolated
    (ln K linearly in 1/T); otherwise the NASA-7 curve is used.  The default
    uncertainties are the max(|NASA-7 - JANAF anchor|) spread, floored at
    0.05 in ln K and 0.2 kJ/mol in dH.
    """
    T_grid = np.asarray(T_grid, float)
    jdir = DATA / "janaf"
    if (jdir / "NO2.txt").exists() and (jdir / "N2O4.txt").exists():
        Tj, lnKj, dHj = janaf_reaction(jdir / "NO2.txt", jdir / "N2O4.txt")
        lnK = np.interp(1 / T_grid, (1 / Tj)[::-1], lnKj[::-1])
        dH = np.interp(T_grid, Tj, dHj)
        source = "NIST-JANAF tables (data/janaf)"
    else:
        lnK, dH, _ = nasa7_reaction(T_grid)
        source = "NASA-7 polynomials (Cantera nasa_gas.yaml: NO2 L7/88, N2O4 TPIS89)"
    Ta, lnKa, dHa = janaf_anchors()
    lnK_n, dH_n, _ = nasa7_reaction(Ta)
    spread_lnK = float(np.max(np.abs(lnK_n - lnKa)))
    spread_dH = float(np.max(np.abs(dH_n - dHa)))
    sK = sigma_lnK if sigma_lnK is not None else max(0.05, spread_lnK)
    sH = sigma_dH if sigma_dH is not None else max(0.2, spread_dH)
    return dict(
        T=T_grid, lnK=np.asarray(lnK), dH=np.asarray(dH),
        sigma_lnK=np.full_like(T_grid, sK), sigma_dH=np.full_like(T_grid, sH),
        source=source, spread_lnK=spread_lnK, spread_dH=spread_dH,
    )


def load_reference_csv(path=DATA / "reference_Kp.csv"):
    rows = _read_csv(path)
    f = lambda k: np.array([float(r[k]) for r in rows])
    return dict(
        T=f("T_K"), lnK=f("lnKp_assoc"), dH=f("deltaH_kJmol"),
        sigma_lnK=f("sigma_lnK"), sigma_dH=f("sigma_dH_kJmol"),
        role=[r["role"] for r in rows], source=rows[0]["source"],
    )


def polat_lj(species="NO2"):
    for r in _read_csv(DATA / "polat_lj_parameters.csv"):
        if r["species"] == species:
            return float(r["eps_over_kB_K"]), float(r["sigma_A"])
    raise KeyError(species)


def constraints():
    return json.loads((DATA / "coarse_grain_constraints.json").read_text())


def k_bond_md_max(c=None):
    """Largest bond curvature (kJ/mol/A^2) whose harmonic period still spans
    ``md_min_steps_per_bond_period`` time steps of ``md_timestep_fs``."""
    from .constants import NA

    c = c or constraints()
    mu = c["reduced_mass_gmol"] * 1e-3 / NA          # kg
    period = c["md_timestep_fs"] * 1e-15 * c["md_min_steps_per_bond_period"]
    omega = 2 * np.pi / period
    k_SI = mu * omega ** 2                            # J/m^2 per pair
    return k_SI * NA * 1e-3 * 1e-20                   # kJ/mol/A^2
