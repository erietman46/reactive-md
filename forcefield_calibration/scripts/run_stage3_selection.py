"""Stage 3 - compare and select (Milestone 6).

Two separate selections, as argued in the strategy document:

1. pair potential family (Stage 1 data, many-body term off), real-fluid
   track and generic-model requirements;
2. many-body penalty with the pair potential held fixed (Stage 2 data).

Writes results/model_selection.md and results/model_selection_pairs.csv.
"""
from common import RESULTS, load_json, rebuild, save_csv

import numpy as np

from forcefield_calibration.potentials.candidates import ALL_LABELS, FLUID_PARAMS, PRIMARY, is_kf
from forcefield_calibration.reference_data import k_bond_md_max
from forcefield_calibration.selection.model_selection import SCORED, rank, smoothness


def pair_rows():
    kmax = k_bond_md_max()
    rows = []
    for lab in ALL_LABELS:
        try:
            d = load_json(f"stage1_{lab}.json")
        except FileNotFoundError:
            continue
        a = d["variants"]["anchored"]
        p = d["variants"]["physical"]
        bs = d["barrier_scan"]
        pa = rebuild(a["potential"])
        sm = smoothness(pa)
        try:  # can this family hold a shallow (generic, E1-like) well at the physical stiffness?
            q = pa.with_bond(eb=6.0, kb=580.0)
            shallow_ok = bool(q.landscape().clean and getattr(q, "feasible", True))
        except ValueError:
            shallow_ok = False
        sv = np.array(d["sensitivity"]["singular_values"])
        n_free = 2 + 2 + len(FLUID_PARAMS[lab]) + is_kf(lab)  # eps_b, k_b, barrier height+position, fluid, w
        ang = a.get("angular") or {}
        pt = d["per_T_anchoring"]["summary"]
        clean_scan = [r for r in bs["rows"] if r.get("clean_comp")]
        rows.append(dict(
            label=lab, name=d["name"], primary=lab in PRIMARY,
            clean=a["potential"]["landscape"]["clean"],
            C1=sm["U"] < 1e-6 and sm["dU"] < 1e-4, C2=sm["d2U"] < 0.05,
            jump_dU=sm["dU"], jump_d2U=sm["d2U"], shallow_well_at_kb580=shallow_ok,
            stiffness_md_ok=a["k_bond_actual"] < kmax and ang.get("libration_md_ok", True),
            barrier_tunable=len(clean_scan) >= 3 and (bs.get("tau_span_decades") or 0) >= 2.0,
            rms_lnK_fit=p["rms_lnK_fit"], rms_lnK_test=p["rms_lnK_test"],
            abs_dH_err=abs(a["dH_err_298"]), rms_B2frozen_sigma=a["rms_B2frozen_sigma"],
            orthogonality=bs["orthogonality"], B2frozen_shift_per_decade=bs.get("B2frozen_shift_per_decade", np.inf),
            tau_span_decades=bs.get("tau_span_decades"), n_free=n_free,
            log10_condition=float(np.log10(sv[0] / sv[-1])),
            fluid_chi2=d["fluid_fit"]["chi2"],
            exact_kb=d["exact_at_298"]["kb"] if d["exact_at_298"].get("converged") else np.nan,
            dS_missing=d["entropy_anchored"]["dS_missing"],
            kern_frenkel=is_kf(lab), phys_rms_dH=p["rms_dH"], phys_dH_err=p["dH_err_298"],
            anch_rms_lnK_fit=a["rms_lnK_fit"], anch_rms_lnK_test=a["rms_lnK_test"], anch_rms_dH=a["rms_dH"],
            anch_dS=a["dS_298"], anch_dCp=a["dCp_298"], anch_eb=a["eb"], anch_tau_d=a["tau_d_TST_298"],
            phys_barrier_pos=p["barrier_position"], phys_eb=p["eb"],
            w_deg=ang.get("w_deg", np.nan), kappa=ang.get("kappa", np.nan),
            libration_fs=ang.get("libration_period_fs_IA", np.nan), cone_deg=ang.get("equivalent_hard_cone_deg_298", np.nan),
            perT_eb_min=pt["eb_min"], perT_eb_max=pt["eb_max"], perT_rms_dH_err=pt["rms_dH_err"],
            perT_worst_transfer=float(np.nanmax(np.array(pt["rms_lnK_transfer"], float))),
            perT_best_transfer=float(np.nanmin(np.array(pt["rms_lnK_transfer"], float))),
            perT=d["per_T_anchoring"],
        ))
    return rank(rows)


def per_T_table(prs):
    """eps_b(T) anchored separately at each temperature (k_b = 580; KF: w of the 298 K anchor kept)."""
    T = prs[0]["perT"]["T"]
    head = "| family | " + " | ".join(f"{t:g} K" for t in T) + " | span | rms dH err (kJ/mol) | worst transfer rms lnK |\n"
    head += "|" + "---|" * (len(T) + 4) + "\n"
    out = head
    for r in sorted(prs, key=lambda r: ALL_LABELS.index(r["label"])):
        ebs = [row.get("eb", np.nan) for row in r["perT"]["rows"]]
        out += (f"| {r['label']} | " + " | ".join(fmt(e, 4) for e in ebs)
                + f" | {fmt(r['perT_eb_max'] - r['perT_eb_min'])} | {fmt(r['perT_rms_dH_err'])} | {fmt(r['perT_worst_transfer'])} |\n")
    return out


def kf_valence_rows():
    try:
        return load_json("stage2_kf.json")["rows"]
    except FileNotFoundError:
        return []


def penalty_rows(track):
    d = load_json(f"stage2_{track}.json")
    rows = []
    for r in d["rows"]:
        long_trap = r["worst_trap_lifetime_ps"] > 1.0
        rows.append(dict(
            penalty=r["penalty"], exact_pair=r["exact_pair_invariance"], dlnK_pair=r["dlnK_pair"],
            suppresses=r["worst_trimer_vs_vdw"] >= -0.5 and r["unfixable_dE"] >= -0.5,
            worst_trimer_vs_vdw=r["worst_trimer_vs_vdw"],
            no_long_traps=not long_trap, eam=r["eam_compatible"],
            trimer_fraction=r["trimer_fraction"], worst_trimer_dE=r["worst_trimer_dE"],
            worst_trap_lifetime_ps=r["worst_trap_lifetime_ps"], n_params=r["n_params"],
            lam_over_eb=r["lam_over_eb"],
        ))
    for r in rows:
        r["hard_pass"] = bool(r["suppresses"] and abs(r["dlnK_pair"]) < 0.01)
        r["score"] = (not r["hard_pass"], not r["exact_pair"], not r["no_long_traps"], not r["eam"],
                      r["trimer_fraction"], r["n_params"])
    return sorted(rows, key=lambda r: r["score"])


def fmt(v, n=3):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, float):
        if not np.isfinite(v):
            return "-"
        return f"{v:.{n}g}"
    return str(v)


def table(rows, cols):
    head = "| " + " | ".join(c for c, _ in cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(fmt(r.get(k)) for _, k in cols) + " |\n" for r in rows)


def main():
    prs = pair_rows()
    save_csv([{k: v for k, v in r.items() if k != "perT"} for r in prs], "model_selection_pairs.csv")
    md = ["# Model selection (auto-generated by scripts/run_stage3_selection.py)\n",
          "## 1. Pair-potential families (many-body term off)\n",
          "Hard criteria: clean landscape, C1 force, bond stiffness below the MD limit, barrier tunable over "
          ">= 2 decades of TST lifetime at fixed K. Scored (lower is better): " + ", ".join(SCORED) + ".\n",
          table(prs, [("family", "label"), ("hard pass", "hard_pass"), ("Pareto", "pareto"), ("C2", "C2"),
                      ("rms lnK fit", "rms_lnK_fit"), ("rms lnK test", "rms_lnK_test"),
                      ("|dH err| 298 K (kJ/mol)", "abs_dH_err"), ("B2frozen rms/sigma", "rms_B2frozen_sigma"),
                      ("|dlnK/dln tau|", "orthogonality"), ("dB2frozen per decade tau", "B2frozen_shift_per_decade"),
                      ("tau span (decades)", "tau_span_decades"), ("free params", "n_free"),
                      ("log10 cond", "log10_condition"), ("k_b needed for K and dH", "exact_kb"),
                      ("eps_b = 6 at k_b = 580 valid", "shallow_well_at_kb580")]),
          "\n",
          "## 1b. Equilibrium mapping: isotropic versus Kern-Frenkel\n",
          "physical: k_b = 580, eps_b (+ barrier position, + patch width w for KF) fitted to ln K at the fit "
          "temperatures; anchored: k_b = 580, ln K exact at 298.15 K (KF: ln K and dH exact via w). rms dH over all "
          "nine temperatures. Libration period of the patch about the NO2 a axis (MD limit 20 fs).\n",
          table(sorted(prs, key=lambda r: ALL_LABELS.index(r["label"])),
                [("family", "label"), ("phys eps_b", "phys_eb"), ("phys r_barrier nominal", "phys_barrier_pos"),
                 ("phys rms lnK fit", "rms_lnK_fit"), ("phys rms lnK test", "rms_lnK_test"),
                 ("phys dH err 298", "phys_dH_err"), ("phys rms dH", "phys_rms_dH"),
                 ("anch eps_b", "anch_eb"), ("anch rms lnK fit", "anch_rms_lnK_fit"),
                 ("anch rms lnK test", "anch_rms_lnK_test"), ("anch dH err 298", "abs_dH_err"),
                 ("anch rms dH", "anch_rms_dH"), ("anch dS 298 (J/mol/K)", "anch_dS"),
                 ("anch dCp 298", "anch_dCp"), ("w (deg)", "w_deg"), ("kappa (kJ/mol/rad^2)", "kappa"),
                 ("libration (fs)", "libration_fs"), ("tau_d TST 298 (s)", "anch_tau_d")]),
          "\nReference at 298.15 K: dS = -175.9 J/mol/K; dCp = d(dH)/dT ~ +8 J/mol/K (298-450 K).\n",
          "\n## 1c. ln K anchored separately at every temperature\n",
          "eps_b (kJ/mol) solving ln K_p(T) exactly at each temperature (k_b = 580; KF families keep w of the "
          "298 K anchor). A temperature-independent potential that describes the reaction needs one eps_b for all "
          "T. 'worst transfer': rms ln K error over all nine temperatures of the potential anchored at the worst "
          "single temperature. '-': no eps_b reaches ln K at that T (family D: its Tang-Toennies attraction alone "
          "binds too strongly above 350 K).\n",
          per_T_table(prs), "\n"]
    kv = kf_valence_rows()
    if kv:
        md += ["## 1d. Intrinsic valence of the Kern-Frenkel families (real track, 298.15 K, 1 bar)\n",
               "No many-body term: one patch per bead. dE relative to the aligned dimer + free monomer; trimer "
               "fraction is a rigorous upper bound. 'isotropic same curve': the aligned radial curve without "
               "orientation dependence.\n",
               table(kv, [("family", "family"), ("penalty", "penalty"), ("w (deg)", "w_deg"),
                          ("worst 2nd bond dE (kJ/mol)", "worst_second_bond_dE"),
                          ("relaxed trimer dE", "relaxed_trimer_dE"), ("relaxed vs vdW", "relaxed_trimer_vs_vdw"),
                          ("trimer fraction <=", "trimer_fraction_bound"),
                          ("isotropic same curve", "trimer_fraction_isotropic_same_curve")]),
               "\n"]
    for track in ("real", "generic", "generic_C"):
        try:
            pen = penalty_rows(track)
        except FileNotFoundError:
            continue
        md += [f"## 2. Many-body penalty, {track} track (pair potential fixed)\n",
               "Hard criteria: every two-bond geometry lies above the dimer-monomer van der Waals complex "
               "(worst dE_vdw >= -0.5 kJ/mol, nothing invisible to the counter) and |d lnK| < 0.01 from the term's "
               "action on isolated pairs. Then preferred: exact pair "
               "invariance, no trimer trap living > 1 ps, native LAMMPS form (eam/fs), lowest trimer content, "
               "fewest parameters.\n",
               table(pen, [("penalty", "penalty"), ("hard pass", "hard_pass"), ("exact pair", "exact_pair"),
                           ("d lnK (pair)", "dlnK_pair"), ("worst trimer dE (vs free)", "worst_trimer_dE"),
                           ("worst trimer dE (vs vdW)", "worst_trimer_vs_vdw"),
                           ("longest trap (ps)", "worst_trap_lifetime_ps"), ("trimer fraction", "trimer_fraction"),
                           ("lambda/eps_b", "lam_over_eb"), ("eam/fs", "eam"), ("params", "n_params")]),
               "\n"]
    (RESULTS / "model_selection.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
