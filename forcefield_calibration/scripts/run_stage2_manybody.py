"""Stage 2 - valence-limiting penalty on top of the calibrated pair potential (Milestone 5).

The pair potential is held FIXED (Stage 1 result) and only the many-body term
is varied, so that differences come from the penalty alone.  Two pair
potentials are screened:

  real       family A anchored to JANAF/NASA K_p at 298.15 K (deep well)
  generic    family A, state E2K2 of the generic design (shallow well)
  generic_C  family C, state E2K2 (same f_d and time scale, additive construction)

For every penalty form:
  * strength: 1.15 x the smallest value that puts every two-bond geometry above
    the dimer-monomer van der Waals complex (glue: the explorer's 1.2 eps_b rule)
  * pair invariance: max energy the term adds to an isolated pair and the
    resulting shift of ln K_p  (must be ~0)
  * worst remaining trimer energy and geometries the counter cannot see
  * trapped trimers: local minima and their lowest escape barriers
  * trimer content from the 3-particle cluster integral (K3) at a reference
    density, compared with the pair-only model; Delta B3
  * LAMMPS check of the eam/fs export (coordination and glue forms)

A separate track checks the intrinsic valence of the smooth Kern-Frenkel
families (anchored at 298.15 K, 1 bar): one patch per bead, no penalty, then
the selected penalty on top (manybody/patchy_valence.py).

Outputs results/stage2_<track>.json, results/stage2_kf.json, results/stage2_summary.csv.
"""
from common import MU, T_GEN, T_REF, load_json, rebuild, save_csv, save_json

import sys

import numpy as np

from forcefield_calibration.export.lammps_export import verify_exports
from forcefield_calibration.manybody.coordination_penalties import make_penalty
from forcefield_calibration.manybody.patchy_valence import (pair_bound_log_KV, relaxed_trimer, second_bond_map,
                                                            trimer_equilibrium_kf)
from forcefield_calibration.manybody.third_virial import delta_B3, trimer_equilibrium
from forcefield_calibration.manybody.trimer_energy_surface import (required_strength, trap_check, vdw_reference,
                                                                   worst_trimer)
from forcefield_calibration.potentials.candidates import KERN_FRENKEL
from forcefield_calibration.thermodynamics.equilibrium_integrals import bound_state, number_density_ideal

PENALTIES = [
    # (kind, number of penalty parameters besides lambda, description)
    ("coord_hermite_tanh_sq", 3, "explorer coordination form: Hermite counter, tanh-squared hinge"),
    ("coord_hermite_cubic", 3, "Hermite counter, exact cubic hinge (zero for n <= 1)"),
    ("coord_quintic_cubic", 4, "quintic counter falling from r_b, exact cubic hinge"),
    ("coord_tanh_damped_cubic", 5, "damped tanh counter, exact cubic hinge"),
    ("coord_gauss_tanh_sq", 3, "Gaussian counter, tanh-squared hinge (explorer default)"),
    ("glue", 3, "glue / valence-one embedding, density = bond strength"),
    ("kts", 4, "Kohen-Tully-Stillinger angular three-body (Skorpa et al.)"),
    ("sw", 3, "Stillinger-Weber angular three-body, cos(theta0) = 1"),
]
LAMBDA_SCAN = (0.5, 0.8, 1.0, 1.15, 1.5, 2.0, 4.0)


def configs(pot):
    rb = pot.r_bond
    return [np.array([[0, 0, 0], [rb, 0, 0]]),
            np.array([[0, 0, 0], [rb, 0, 0], [-rb, 0, 0]]),
            np.array([[0, 0, 0], [rb, 0, 0], [0.5 * rb, 0.866 * rb, 0]]),
            np.array([[0, 0, 0], [rb, 0, 0], [-2.2, 0.4, 0.3]]),
            np.array([[0, 0, 0], [rb, 0, 0], [5.0, 0.5, 0], [5.0 + rb, 0.5, 0.2]])]


def screen(pot, T, rho, track, verify=True):
    te0 = trimer_equilibrium(pot, T)
    frac0 = te0.fractions(rho)
    bs0 = bound_state(pot, T)
    E_vdw, vdw_geom = vdw_reference(pot)
    rows = []
    for kind, npar, desc in PENALTIES:
        mb = make_penalty(pot, kind)
        sa = required_strength(pot, mb)
        if kind != "glue":
            mb = mb.with_strength(1.15 * sa.lam_min)
        lam_min = sa.lam_min if kind != "glue" else np.nan
        r = np.linspace(0.6 * pot.r_bond, pot.rcut, 4000)
        pe = mb.pair_energy(r)
        dlnK = bound_state(pot, T, extra_energy=mb.pair_energy).log_Kp - bs0.log_Kp
        wt = worst_trimer(pot, mb)
        traps = trap_check(pot, mb, T)
        te = trimer_equilibrium(pot, T, mb)
        fr = te.fractions(rho)
        row = dict(
            track=track, penalty=kind, description=desc, n_params=npar + 1,
            lam=mb.lam, lam_min=lam_min, lam_over_eb=mb.lam / -pot.u_bond,
            unfixable_dE=sa.unfixable, unfixable_at=sa.unfixable_at,
            max_pair_energy=float(np.max(np.abs(pe))), dlnK_pair=float(dlnK),
            worst_trimer_dE=wt[0], worst_trimer_geom=wt[1:], E_vdw=E_vdw, vdw_geom=vdw_geom,
            worst_trimer_vs_vdw=wt[0] - E_vdw,
            n_traps=len(traps),
            worst_trap_barrier=max((t.barrier for t in traps), default=0.0),
            worst_trap_lifetime_ps=max((t.lifetime_s for t in traps), default=0.0) * 1e12,
            worst_trap_dE=min((t.dE for t in traps), default=np.nan),
            traps=[t.__dict__ for t in traps],
            log_K3=te.log_K3, log_K3_suppression=te.log_K3 - te0.log_K3,
            trimer_fraction=fr["trimer"], dimer_fraction=fr["dimer"],
            trimer_fraction_pair_only=frac0["trimer"], dimer_fraction_pair_only=frac0["dimer"],
            delta_B3=delta_B3(pot, mb, T),
            eam_compatible=bool(getattr(mb, "eam_compatible", False)),
            exact_pair_invariance=bool(np.max(np.abs(pe)) == 0.0),
        )
        if verify and row["eam_compatible"]:
            chk = verify_exports(pot, mb, configs(pot))
            if chk and "python_total" in chk[0]:
                row["lammps_eam_max_abs_err"] = max(abs(c["lammps_eam"] - c["python_total"]) for c in chk)
                row["lammps_table_max_abs_err"] = max(abs(c["lammps_table"] - c["python_pair"]) for c in chk)
        rows.append(row)
        print(f"[stage2:{track}] {kind}: lam={mb.lam:.3g} dlnK={dlnK:.2e} worst={wt[0]:.3g} "
              f"traps={len(traps)} (max barrier {row['worst_trap_barrier']:.3g}) "
              f"trimer {fr['trimer']:.3g} (pair-only {frac0['trimer']:.3g})", flush=True)
    return rows, dict(log_K3_pair_only=te0.log_K3, fractions_pair_only=frac0, E_vdw=E_vdw, vdw_geom=vdw_geom)


def lambda_scan(pot, T, rho, kind):
    mb0 = make_penalty(pot, kind)
    if kind == "glue":
        ref = mb0.lam
    else:
        ref = required_strength(pot, mb0).lam_min
    out = []
    for f in LAMBDA_SCAN:
        mb = mb0.with_strength(f * ref)
        traps = trap_check(pot, mb, T)
        fr = trimer_equilibrium(pot, T, mb).fractions(rho)
        out.append(dict(factor=f, lam=f * ref, trimer_fraction=fr["trimer"], dimer_fraction=fr["dimer"],
                        worst_trimer_dE=worst_trimer(pot, mb)[0],
                        worst_trap_barrier=max((t.barrier for t in traps), default=0.0),
                        n_traps=len(traps)))
    return out


def kf_valence(rho, T=T_REF, factor=2.0):
    """Intrinsic valence of every Kern-Frenkel family (anchored), without and with the selected penalty."""
    rows = []
    for lab in KERN_FRENKEL:
        try:
            pot = rebuild(load_json(f"stage1_{lab}.json")["variants"]["anchored"]["potential"])
        except FileNotFoundError:
            continue
        E_vdw, vdw_geom = vdw_reference(pot)
        iso0 = trimer_equilibrium(pot.radial, T).fractions(rho)  # same aligned curve, isotropic
        b = np.linspace(0.8 * pot.r_bond, pot.r_barrier, 120)
        th = np.deg2rad(np.linspace(20, 180, 81))
        Bm, Th = np.meshgrid(b, th, indexing="ij")
        for pen in (None, "coord_hermite_cubic"):
            mb = None
            if pen is not None:
                mb = make_penalty(pot, pen)
                mb = mb.with_strength(factor * required_strength(pot, mb).lam_min)
            E = second_bond_map(pot, Bm, Th, mb=mb, n_phi=181)
            i = np.unravel_index(np.argmin(E), E.shape)
            rt = relaxed_trimer(pot, mb=mb, n_starts=40)
            te = trimer_equilibrium_kf(pot, T, mb=mb)
            fr = te.fractions(rho)
            row = dict(family=lab, penalty=pen or "none (patch only)", lam=None if mb is None else mb.lam,
                       w_deg=float(np.rad2deg(pot.p["w"])), E_vdw=E_vdw,
                       worst_second_bond_dE=float(E[i]), worst_second_bond_at=(float(Bm[i]), float(np.rad2deg(Th[i]))),
                       relaxed_trimer_dE=rt.dE, relaxed_trimer_vs_vdw=rt.dE - E_vdw, relaxed_trimer=rt.__dict__,
                       log_K3_bound=te.log_K3, log_K2=te.log_K2,
                       pair_bound_excess=pair_bound_log_KV(pot, T) - te.log_K2,
                       trimer_fraction_bound=fr["trimer"], dimer_fraction=fr["dimer"],
                       trimer_fraction_isotropic_same_curve=iso0["trimer"])
            rows.append(row)
            print(f"[stage2:kf] {lab} {row['penalty']}: worst 2nd bond {E[i]:.3g}, relaxed {rt.dE:.3g} kJ/mol, "
                  f"trimer <= {fr['trimer']:.3g} (isotropic same curve {iso0['trimer']:.3g})", flush=True)
    save_json(dict(T=T, rho0=rho, rows=rows,
                   note="trimer fractions are rigorous upper bounds (patchy_valence.trimer_equilibrium_kf)"),
              "stage2_kf.json")
    return rows


def main():
    real = rebuild(load_json("stage1_A.json")["variants"]["anchored"]["potential"])
    gen = rebuild(load_json("generic_design.json")["states"]["A"]["E2K2"]["potential"])
    genC = rebuild(load_json("generic_design.json")["states"]["C"]["E2K2"]["potential"])
    from run_stage1_generic import RHO_GEN

    tracks = {
        "real": (real, T_REF, number_density_ideal(1e5, T_REF)),
        "generic": (gen, T_GEN, RHO_GEN),
        "generic_C": (genC, T_GEN, RHO_GEN),
    }
    summary = []
    for track, (pot, T, rho) in tracks.items():
        rows, base = screen(pot, T, rho, track)
        scans = {k: lambda_scan(pot, T, rho, k) for k in ("coord_hermite_cubic", "coord_quintic_cubic", "glue")}
        save_json(dict(track=track, T=T, rho0=rho, pair=pot.describe(), base=base, rows=rows, lambda_scans=scans),
                  f"stage2_{track}.json")
        summary += [{k: v for k, v in r.items() if k not in ("traps",)} for r in rows]
    save_csv(summary, "stage2_summary.csv",
             header=["track", "penalty", "n_params", "lam", "lam_min", "lam_over_eb", "exact_pair_invariance",
                     "max_pair_energy", "dlnK_pair", "unfixable_dE", "E_vdw", "worst_trimer_dE",
                     "worst_trimer_vs_vdw", "n_traps",
                     "worst_trap_barrier", "worst_trap_lifetime_ps", "trimer_fraction", "trimer_fraction_pair_only",
                     "dimer_fraction", "log_K3_suppression", "delta_B3", "eam_compatible",
                     "lammps_eam_max_abs_err", "lammps_table_max_abs_err", "description"])
    kf_valence(number_density_ideal(1e5, T_REF))


if __name__ == "__main__":
    if "--kf-only" in sys.argv:
        from forcefield_calibration.thermodynamics.equilibrium_integrals import number_density_ideal as _nd

        kf_valence(_nd(1e5, T_REF))
    else:
        main()
