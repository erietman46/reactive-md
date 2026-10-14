"""Export frozen parameter sets and LAMMPS input files for the later MD validation.

For the two short-listed pair families (A, C) and the selected penalty (coord_hermite_cubic):

  export_lammps/real_<A|C>_anchored/      pair.table (units real), manybody.eam.fs (units metal), params.json
  export_lammps/generic_<A|C>/<state>/    same for each of the nine E x K design states

Every eam/fs file is checked against the Python energies with the LAMMPS
Python module (if installed); the comparison is stored in params.json.

The Kern-Frenkel models (export_lammps/real_KF-<X>_anchored/) have no native
LAMMPS pair style.  They are exported as two radial tables, ALIGNED (U_on) and
MISALIGNED (U_off), plus the angular parameters: U = S U_on + (1 - S) U_off
with S = f(n_i . r_hat) f(-n_j . r_hat), f(c) = exp(-(1 - c)/w^2), which is
what a custom pair style (or a plugin) has to evaluate.
"""
from common import ROOT, load_json, potential_record, rebuild

import json
from pathlib import Path

import numpy as np

from forcefield_calibration.export.lammps_export import verify_exports, write_eam_fs, write_pair_table
from forcefield_calibration.manybody.coordination_penalties import make_penalty
from forcefield_calibration.manybody.trimer_energy_surface import required_strength

OUT = ROOT / "export_lammps"
PENALTY = "coord_hermite_cubic"
# 2 x lambda_min (not the explorer's 1.15): the Stage-2 strength scans show that doubling the
# minimum strength lowers both the trimer content and the escape barrier of the residual trimer
# minima for family A (real: 1e-4 -> 3e-5, generic E2K2: 1.2 % -> 0.4 %; trap barrier 8.5 -> 2.5 kJ/mol).
LAMBDA_FACTOR = 2.0


def export_one(pot, folder: Path, meta):
    folder.mkdir(parents=True, exist_ok=True)
    mb = make_penalty(pot, PENALTY)
    sa = required_strength(pot, mb)
    mb = mb.with_strength(LAMBDA_FACTOR * sa.lam_min)
    coeff_t = write_pair_table(pot, folder / "pair.table", units="real")
    coeff_e = write_eam_fs(pot, mb, folder / "manybody.eam.fs")
    c = mb.counter
    rec = dict(meta, pair=potential_record(pot),
               manybody=dict(kind=PENALTY, lam_kJmol=mb.lam, lam_min_kJmol=sa.lam_min, lam_factor=LAMBDA_FACTOR,
                             strength_reference="dimer + monomer van der Waals complex",
                             hinge=mb.hinge_kind, delta=mb.delta,
                             counter=dict(kind=c.kind, **c.kw), reach_A=c.reach),
               lammps=dict(
                   pair_only=["units real", "pair_style table spline 3000", coeff_t],
                   with_penalty=["units metal", "pair_style eam/fs", coeff_e],
                   note="eam/fs carries both the pair term and the coordination penalty; use it alone"),
               )
    rb = pot.r_bond
    cfgs = [np.array([[0, 0, 0], [rb, 0, 0]]), np.array([[0, 0, 0], [rb, 0, 0], [-rb, 0.1, 0]]),
            np.array([[0, 0, 0], [rb, 0, 0], [0.5 * rb, 0.866 * rb, 0]])]
    rec["verification"] = verify_exports(pot, mb, cfgs, workdir=folder / ".verify")
    for f in (folder / ".verify").glob("*"):
        f.unlink()
    (folder / ".verify").rmdir()
    (folder / "params.json").write_text(json.dumps(rec, indent=2, default=float))
    return rec


class _Curve:
    """Minimal scalar potential (U, dU, r_bond, rcut, name, p) for the table writer."""

    def __init__(self, base, u, du, name):
        self.U, self.dU, self.name, self.p = u, du, name, base.p
        self.r_bond, self.rcut = base.r_bond, base.rcut


def export_kf(label, folder: Path):
    from forcefield_calibration.reference_data import no2_libration_inertia

    folder.mkdir(parents=True, exist_ok=True)
    s1 = load_json(f"stage1_{label}.json")
    pot = rebuild(s1["variants"]["anchored"]["potential"])
    on = _Curve(pot, pot.U, pot.dU, pot.name + " [aligned]")
    off = _Curve(pot, pot.U_off, pot.dU_off, pot.name + " [misaligned]")
    c_on = write_pair_table(on, folder / "kf_radial.table", keyword="ALIGNED", units="real")
    path_off = folder / "kf_radial_off.table"
    c_off = write_pair_table(off, path_off, keyword="MISALIGNED", units="real")
    # one file with both sections
    (folder / "kf_radial.table").write_text((folder / "kf_radial.table").read_text() + "\n"
                                            + path_off.read_text().split("\n", 2)[2])
    path_off.unlink()
    rec = dict(track="real", model="smooth Kern-Frenkel", T_K=298.15,
               calibration="K_p(298.15 K) and dH(298.15 K) exact, k_b = 580 kJ/mol/A^2, patch width fitted",
               pair=potential_record(pot), angular=s1["variants"]["anchored"]["angular"],
               energy="U = S U_on(r) + (1 - S) U_off(r), S = f(n_i.r_hat) f(-n_j.r_hat), f(c) = exp(-(1 - c)/w^2)",
               tables=dict(file="kf_radial.table", sections=["ALIGNED", "MISALIGNED"], units="real",
                           aligned=c_on.replace("kf_radial.table", "kf_radial.table"), misaligned=c_off.replace(
                               "kf_radial_off.table", "kf_radial.table")),
               rotational=dict(patch_axis="C2 axis of NO2", inertia_amuA2_IA_IC=no2_libration_inertia()),
               lammps="no native pair style: needs a custom anisotropic pair style (e.g. a plugin) evaluating the "
                      "expression above; the coordination penalty is not needed (see results/stage2_kf.json)")
    (folder / "params.json").write_text(json.dumps(rec, indent=2, default=float))
    return rec


def main():
    for lab in ("KF-A", "KF-B"):
        try:
            export_kf(lab, OUT / f"real_{lab}_anchored")
            print(f"Kern-Frenkel {lab}: tables + params written")
        except FileNotFoundError:
            pass
    g = load_json("generic_design.json")
    for fam in ("A", "C"):
        s1 = load_json(f"stage1_{fam}.json")
        pot = rebuild(s1["variants"]["anchored"]["potential"])
        r = export_one(pot, OUT / f"real_{fam}_anchored",
                       dict(track="real", T_K=298.15, calibration="K_p(298.15 K) exact, k_b = 580 kJ/mol/A^2"))
        print(f"real {fam}:", max(abs(c["lammps_eam"] - c["python_total"]) for c in r["verification"])
              if "lammps_eam" in r["verification"][0] else r["verification"])
        for name, v in g["states"][fam].items():
            p = rebuild(v["potential"])
            r = export_one(p, OUT / f"generic_{fam}" / name,
                           dict(track="generic", state=name, T_K=g["T"], rho0=g["rho0"], design=v["row"]))
            err = max(abs(c["lammps_eam"] - c["python_total"]) for c in r["verification"] if "lammps_eam" in c) \
                if r["verification"] and "lammps_eam" in r["verification"][0] else None
            print(f"generic {fam} {name}: eam/fs max |E_lammps - E_python| = {err}")


if __name__ == "__main__":
    main()
