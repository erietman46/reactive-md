"""Export calibrated potentials to LAMMPS and verify them against the Python model.

* Pair potential only -> ``pair_style table`` file (units real or metal).
* Pair potential + coordination/glue penalty -> ``pair_style eam/fs`` setfl
  file (units metal).  The coordination penalty has exactly the embedded-atom
  form E = sum_i F(n_i) + 1/2 sum_{i != j} U2(r_ij) with n_i = sum_j s(r_ij),
  so F is the penalty hinge and the "density" is the bond counter.  This keeps
  the many-body term a native, energy-conserving LAMMPS style with correct
  per-atom energies and virials (useful later for heat-flux work).

Below ``r_cap`` (default 0.6 r_b, never visited in MD) the pair energy is
continued linearly from its value and slope at r_cap, so tables stay finite.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from ..constants import KJ_PER_KCAL, KJMOL_PER_EV, M_NO2

UNIT_SCALE = {"real": 1.0 / KJ_PER_KCAL, "metal": 1.0 / KJMOL_PER_EV}


def capped_pair(pot, r, r_cap):
    r = np.asarray(r, float)
    u, du = pot.U(r), pot.dU(r)
    uc, duc = float(pot.U(r_cap)), float(pot.dU(r_cap))
    inner = r < r_cap
    return np.where(inner, uc + duc * (r - r_cap), u), np.where(inner, duc, du)


def write_pair_table(pot, path, keyword="REACTIVE", units="real", n=3000, r_lo=None, r_cap=None):
    """Write a LAMMPS pair_style table file; returns the pair_coeff line to use."""
    rc = pot.rcut
    if not np.isfinite(rc):
        raise ValueError("potential needs a finite cut-off for MD")
    r_lo = r_lo if r_lo is not None else 0.5 * pot.r_bond
    r_cap = r_cap if r_cap is not None else 0.6 * pot.r_bond
    r = np.linspace(r_lo, rc, n)
    u, du = capped_pair(pot, r, r_cap)
    s = UNIT_SCALE[units]
    lines = [f"# {pot.name}", f"# exported by forcefield_calibration; units {units}; params {pot.p}", "",
             keyword, f"N {n} R {r_lo:.10g} {rc:.10g}", ""]
    lines += [f"{i + 1} {ri:.8f} {ui * s:.10e} {-di * s:.10e}" for i, (ri, ui, di) in enumerate(zip(r, u, du))]
    Path(path).write_text("\n".join(lines) + "\n")
    return f"pair_coeff 1 1 {Path(path).name} {keyword} {rc:.6g}"


def write_eam_fs(pot, mb, path, element="NO2", mass=M_NO2, nr=5001, nrho=4001, rho_max=10.0, r_cap=None):
    """Write an eam/fs setfl file for pair potential + coordination/glue penalty (metal units)."""
    if not getattr(mb, "eam_compatible", False):
        raise ValueError(f"{mb.name} is not of embedded-atom form")
    rc = pot.rcut
    r_cap = r_cap if r_cap is not None else 0.6 * pot.r_bond
    dr = rc / (nr - 1)
    drho = rho_max / (nrho - 1)
    r = np.arange(nr) * dr
    rho = np.arange(nrho) * drho
    s = UNIT_SCALE["metal"]
    F = mb.embed(rho) * s
    dens = np.where(r > 0, mb.density(np.maximum(r, 1e-12)), mb.density(np.array([1e-6]))[0])
    dens = np.where(r >= rc, 0.0, dens)
    u, _ = capped_pair(pot, np.maximum(r, 1e-12), r_cap)
    rphi = r * u * s
    rphi[-1] = 0.0

    def block(a):
        return "\n".join(" ".join(f"{v:.12e}" for v in a[i:i + 5]) for i in range(0, a.size, 5))

    hdr = [f"# {pot.name}", f"# many-body: {mb.name}, lambda = {mb.lam:.8g} kJ/mol", "# forcefield_calibration eam/fs export (metal units)",
           f"1 {element}", f"{nrho} {drho:.16e} {nr} {dr:.16e} {rc:.16e}", f"1 {mass:.6f} 0.0 none"]
    Path(path).write_text("\n".join(hdr) + "\n" + block(F) + "\n" + block(dens) + "\n" + block(rphi) + "\n")
    return f"pair_coeff * * {Path(path).name} {element}"


# --------------------------------------------------------------- verification
def python_energy(pot, mb, X):
    """Total energy (kJ/mol) of a cluster of particles at positions X (n x 3, A)."""
    X = np.asarray(X, float)
    n = len(X)
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    iu = np.triu_indices(n, 1)
    e = float(np.sum(pot.U(d[iu])))
    if mb is not None and getattr(mb, "eam_compatible", False):
        s = mb.density(np.where(d > 0, d, 1e9))
        np.fill_diagonal(s, 0.0)
        e += float(np.sum(mb.embed(s.sum(axis=1))))
    elif mb is not None and n == 3:
        e += float(mb.triplet_energy(d[0, 1], d[0, 2], d[1, 2]))
    return e


def lammps_energy(style_lines, X, units, box=40.0, mass=M_NO2):
    """Potential energy of a configuration evaluated by LAMMPS (needs the lammps Python module)."""
    from lammps import lammps

    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_list([
            f"units {units}", "atom_style atomic", "boundary f f f",
            f"region box block {-box} {box} {-box} {box} {-box} {box}",
            "create_box 1 box",
        ])
        for x in X:
            lmp.command(f"create_atoms 1 single {x[0]:.10f} {x[1]:.10f} {x[2]:.10f}")
        lmp.command(f"mass 1 {mass}")
        lmp.commands_list(style_lines)
        lmp.commands_list(["neighbor 1.0 bin", "thermo_style custom pe", "run 0"])
        return float(lmp.get_thermo("pe"))
    finally:
        lmp.close()


def verify_exports(pot, mb, configs, workdir=None):
    """Compare Python and LAMMPS energies (kJ/mol) for a list of configurations.

    Returns a list of dicts (config index, python, table, eam).  Skips LAMMPS
    if the module is not available.
    """
    try:
        import lammps  # noqa: F401
    except Exception:  # noqa: BLE001
        return [dict(note="lammps Python module not available")]
    workdir = Path(workdir or tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)
    tab = workdir / "pair.table"
    coeff_t = write_pair_table(pot, tab, units="real")
    out = []
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        eam_lines = None
        if mb is not None and getattr(mb, "eam_compatible", False):
            coeff_e = write_eam_fs(pot, mb, workdir / "manybody.eam.fs")
            eam_lines = ["pair_style eam/fs", coeff_e]
        for i, X in enumerate(configs):
            row = dict(config=i, python_pair=python_energy(pot, None, X), python_total=python_energy(pot, mb, X))
            row["lammps_table"] = lammps_energy(["pair_style table spline 3000", coeff_t], X, "real") * KJ_PER_KCAL
            if eam_lines:
                row["lammps_eam"] = lammps_energy(eam_lines, X, "metal") * KJMOL_PER_EV
            out.append(row)
    finally:
        os.chdir(cwd)
    return out
