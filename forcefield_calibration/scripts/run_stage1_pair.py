"""Stage 1 - two-body calibration, no many-body term, no MD (Milestones 2-4).

For every candidate family (A-D primary, E-F supplementary):

  Step 2  fluid branch: fit the outer (non-bonded) parameters so that the
          frozen-chemistry B2 of the model matches the B2 of the Polat et al.
          NO2 Lennard-Jones monomer (coarse-graining reference, 250-500 K).
  Step 3  bond well, three modes (r_b fixed at 1.78 A):
            physical  k_b = 580 kJ/mol/A^2 fixed, eps_b fitted to ln K_p(T_fit)
            anchored  k_b = 580 fixed, eps_b solved so ln K_p(298.15 K) is exact
                      (honours dG at the MD temperature; carried to Stage 2)
            bounded   eps_b and k_b free, k_b <= MD limit, fitted to ln K_p AND dH
            exact     (eps_b, k_b) matching ln K_p and dH at 298.15 K, no bounds
                      (diagnostic: what an isotropic model would need)
  Step 4  attainability scan over k_b, sensitivity/identifiability, and an
          equal-equilibrium barrier scan with TST kinetic estimates.

Outputs results/stage1_<label>.json, results/stage1_summary.csv.
"""
from common import (KB_PHYS, KB_RANGE, MU, T_B2, T_FIT, T_REF, T_TEST, potential_record,
                    save_csv, save_json)

import sys
import time

import numpy as np

from forcefield_calibration.optimization.attainability import entropy_budget, solve_both, stiffness_scan
from forcefield_calibration.optimization.barrier_scan import barrier_scan, compensate_eb, orthogonality
from forcefield_calibration.optimization.fit_pair_potentials import (b2_frozen_residuals, fit, lnK_residuals,
                                                                     nominal_kb, predictions)
from forcefield_calibration.optimization.sensitivity_analysis import observables, sensitivity
from forcefield_calibration.potentials.candidates import BARRIER_KEY, FLUID_PARAMS, candidate_families
from forcefield_calibration.reference_data import k_bond_md_max, load_reference_csv, polat_lj
from forcefield_calibration.selection.model_selection import smoothness
from forcefield_calibration.thermodynamics.equilibrium_integrals import bound_state
from forcefield_calibration.thermodynamics.kinetics_tst import tst_rates
from forcefield_calibration.thermodynamics.virial_coefficients import B2_lennard_jones

BARRIER_HEIGHTS = {"Eh": [1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0],
                   "H": [1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]}


def subset(ref, mask):
    return {k: (np.asarray(v)[mask] if isinstance(v, (np.ndarray, list)) and len(v) == len(mask) else v)
            for k, v in ref.items()}


def metrics(pot, ref, T_fit, T_test, B2_target_T, B2_target, sB):
    pred = predictions(pot, np.r_[T_fit, T_test])
    lnK_ref = np.interp(pred["T"], ref["T"], ref["lnK"])
    dH_ref = np.interp(pred["T"], ref["T"], ref["dH"])
    nf = len(T_fit)
    pf = predictions(pot, B2_target_T)
    i298 = int(np.argmin(np.abs(pred["T"] - T_REF)))
    return dict(
        rms_lnK_fit=float(np.sqrt(np.mean((pred["lnK"][:nf] - lnK_ref[:nf]) ** 2))),
        rms_lnK_test=float(np.sqrt(np.mean((pred["lnK"][nf:] - lnK_ref[nf:]) ** 2))),
        dH_err_298=float(pred["dH"][i298] - dH_ref[i298]),
        rms_dH=float(np.sqrt(np.mean((pred["dH"] - dH_ref) ** 2))),
        rms_B2frozen_sigma=float(np.sqrt(np.mean(((pf["B2_frozen"] - B2_target) / sB) ** 2))),
        predictions=pred, predictions_B2=pf,
    )


def run_family(label, base, ref, quick=False):
    t0 = time.time()
    rec = dict(label=label, name=base.name)
    fit_mask = np.array([r == "fit" for r in ref["role"]])
    ref_fit = subset(ref, fit_mask)
    eps, sig = polat_lj("NO2")
    B2t = B2_lennard_jones(eps, sig, T_B2)
    sB = np.maximum(5.0, 0.05 * np.abs(B2t))
    rec["B2_target"] = dict(T=T_B2, B2_LJ=B2t, sigma=sB, source="Polat et al. NO2 LJ (eps/k=262.17 K, sigma=3.727 A)")
    kmax = k_bond_md_max()

    # ---------------- Step 2: fluid branch
    fl = fit(base, FLUID_PARAMS[label], b2_frozen_residuals(T_B2, B2t, sB), maxiter=20 if quick else 40)
    pot_f = fl.pot
    rec["fluid_fit"] = dict(values=fl.values, chi2=fl.chi2, n_points=len(T_B2),
                            at_bound=[p[0] for p in FLUID_PARAMS[label]
                                      if min(abs(fl.values[p[0]] - p[1]), abs(fl.values[p[0]] - p[2]))
                                      < 1e-3 * (p[2] - p[1])])

    # ---------------- Step 3: bond well
    variants = {}
    fp = fit(pot_f, [("eb", 2.0, 150.0)], lnK_residuals(ref_fit), maxiter=20 if quick else 30)
    variants["physical"] = fp.pot
    i_ref = int(np.argmin(np.abs(ref["T"] - T_REF)))
    variants["anchored"] = compensate_eb(fp.pot, T_REF, ref["lnK"][i_ref], kb=KB_PHYS, span=(0.3, 3.0))
    fb = fit(pot_f, [("eb", 2.0, 150.0), ("kb", KB_RANGE[0], kmax, True)], lnK_residuals(ref_fit, w_H=1.0),
             maxiter=20 if quick else 40)
    variants["bounded"] = fb.pot
    rec["bounded_fit"] = dict(values=fb.values, chi2=fb.chi2, kb_at_upper_bound=bool(fb.values["kb"] > 0.99 * kmax),
                              kb_md_max=kmax)
    ex = solve_both(pot_f, T_REF, ref["lnK"][i_ref], ref["dH"][i_ref], MU)
    rec["exact_at_298"] = ex

    rec["variants"] = {}
    for name, pot in variants.items():
        m = metrics(pot, ref, T_FIT, T_TEST, T_B2, B2t, sB)
        bs = bound_state(pot, T_REF)
        tst = tst_rates(pot, T_REF, MU)
        rec["variants"][name] = dict(
            potential=potential_record(pot), eb=pot.p["eb"], kb_branch=nominal_kb(pot),
            k_bond_actual=pot.bond_stiffness(), wavenumber=pot.bond_wavenumber(MU), **m,
            dS_298=bs.deltaS, dCp_298=bs.deltaCp, tau_d_TST_298=tst.tau_d, dF_diss_298=tst.dF_diss,
        )
    # entropy budget of the anchored model versus experiment
    from forcefield_calibration.reference_data import nasa7_reaction
    dS_ref = float(nasa7_reaction(np.array([T_REF]))[2][0])
    rec["entropy_anchored"] = entropy_budget(variants["anchored"], T_REF, dS_ref)

    # ---------------- Step 4a: attainability scan over k_b
    kbs = np.logspace(2, 12, 11 if quick else 21)
    rec["stiffness_scan"] = stiffness_scan(pot_f, ref_fit, kbs, MU)

    # ---------------- Step 4b: sensitivity / identifiability (anchored potential)
    anc = variants["anchored"]
    bkey = BARRIER_KEY[label]
    pos = {"Eh": "rh", "H": "rx"}[bkey]
    names = ["eb", "kb", bkey, pos] + [p[0] for p in FLUID_PARAMS[label]] + (["rf"] if label != "D" else [])
    vals = [anc.p["eb"], nominal_kb(anc)] + [anc.p[n] for n in names[2:]]

    def builder(d):
        raw = {k: v for k, v in d.items() if k not in ("eb", "kb")}
        return anc.replace(**raw).with_bond(eb=d["eb"], kb=d["kb"])

    scales = {}
    obs_fn = lambda p: observables(p, T_FIT, [T_REF], T_B2[::2], MU)
    base_obs = obs_fn(anc)
    for k in base_obs:
        scales[k] = 0.05 if k.startswith("lnK") else 0.2 if k.startswith("dH") else 5.0 if k.startswith("B2") else 1.0
    sens = sensitivity(builder, names, vals, obs_fn, scales)
    rec["sensitivity"] = dict(params=names, values=vals, observables=sens["observables"], J=sens["J"],
                              singular_values=sens["singular_values"], Vt=sens["Vt"],
                              condition_number=sens["condition_number"], correlation=sens["correlation"])

    # ---------------- Step 4c: equal-equilibrium barrier scan
    rows = barrier_scan(anc, bkey, BARRIER_HEIGHTS[bkey], T_REF, MU)
    rec["barrier_scan"] = dict(key=bkey, rows=rows, orthogonality=orthogonality(rows))
    good = [r for r in rows if r.get("clean_comp")]
    if len(good) >= 2:
        dl = np.array([r["dln_tau"] for r in good])
        db = np.array([r["B2_frozen"] for r in good])
        span_decades = float(np.ptp(dl) / np.log(10))
        rec["barrier_scan"]["tau_span_decades"] = span_decades
        rec["barrier_scan"]["B2frozen_shift_per_decade"] = float(np.ptp(db) / max(span_decades, 1e-9))
    rec["smoothness"] = smoothness(anc)
    rec["runtime_s"] = time.time() - t0
    return rec


def main(labels=None, quick=False):
    ref = load_reference_csv()
    fams = candidate_families()
    labels = labels or list(fams)
    summary = []
    for lab in labels:
        print(f"[stage1] family {lab} ...", flush=True)
        rec = run_family(lab, fams[lab], ref, quick=quick)
        save_json(rec, f"stage1_{lab}.json")
        v = rec["variants"]
        summary.append(dict(
            label=lab, name=rec["name"],
            fluid_chi2=rec["fluid_fit"]["chi2"], fluid_at_bound=";".join(rec["fluid_fit"]["at_bound"]),
            phys_eb=v["physical"]["eb"], phys_rms_lnK_fit=v["physical"]["rms_lnK_fit"],
            phys_rms_lnK_test=v["physical"]["rms_lnK_test"], phys_dH_err_298=v["physical"]["dH_err_298"],
            anch_eb=v["anchored"]["eb"], anch_dH_err_298=v["anchored"]["dH_err_298"],
            anch_rms_lnK_test=v["anchored"]["rms_lnK_test"], anch_tau_d_298=v["anchored"]["tau_d_TST_298"],
            bnd_eb=v["bounded"]["eb"], bnd_kb=rec["bounded_fit"]["values"]["kb"],
            bnd_rms_lnK_fit=v["bounded"]["rms_lnK_fit"], bnd_dH_err_298=v["bounded"]["dH_err_298"],
            exact_kb=rec["exact_at_298"]["kb"], exact_width_A=rec["exact_at_298"]["thermal_width_A"],
            exact_period_fs=rec["exact_at_298"]["period_fs"],
            dS_missing=rec["entropy_anchored"]["dS_missing"],
            cond=rec["sensitivity"]["condition_number"],
            orthogonality=rec["barrier_scan"]["orthogonality"],
            tau_span_decades=rec["barrier_scan"].get("tau_span_decades"),
            B2frozen_shift_per_decade=rec["barrier_scan"].get("B2frozen_shift_per_decade"),
            runtime_s=rec["runtime_s"],
        ))
        print(f"[stage1] {lab} done in {rec['runtime_s']:.0f} s", flush=True)
    if set(labels) == set(fams):
        save_csv(summary, "stage1_summary.csv", header=list(summary[0]))
    return summary


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args or None, quick="--quick" in sys.argv)
