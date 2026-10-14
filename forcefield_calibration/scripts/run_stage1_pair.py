"""Stage 1 - two-body calibration, no many-body term, no MD (Milestones 2-4).

For every candidate family (A-I isotropic, KF-A ... KF-D smooth Kern-Frenkel):

  Step 2  fluid branch: fit the outer (non-bonded) parameters so that the
          frozen-chemistry B2 of the model matches the B2 of the Polat et al.
          NO2 Lennard-Jones monomer (coarse-graining reference, 250-500 K).
  Step 3  bond well (r_b fixed at 1.78 A; the barrier position r_h may move by
          +-0.15 A in the fits - r_x of the additive Gaussian only outwards, by up to
          0.15 A -, the patch width w is free for KF):
            physical  k_b = 580 kJ/mol/A^2 fixed, eps_b (+ r_h, w) fitted to ln K_p(T_fit)
            anchored  k_b = 580 fixed; isotropic: eps_b solved so ln K_p(298.15 K) is exact;
                      KF: (eps_b, w) solved so ln K_p AND dH at 298.15 K are exact
                      (honours the MD temperature; carried to Stage 2)
            bounded   eps_b, k_b (+ r_h, w) free, k_b <= MD limit, fitted to ln K_p AND dH
            exact     isotropic: (eps_b, k_b) matching ln K_p and dH at 298.15 K, no bounds
                      (diagnostic: what an isotropic model would need); KF: the anchored
                      solution, at the physical k_b
          per-T    the anchoring repeated at every reference temperature separately
  Step 4  attainability scan over k_b (KF: over the patch width w), sensitivity /
          identifiability, and an equal-equilibrium barrier scan with TST estimates.

Outputs results/stage1_<label>.json, results/stage1_summary.csv.
Usage: python run_stage1_pair.py [labels ...] [--quick] [--workers N]
"""
from common import (KB_PHYS, KB_RANGE, MU, T_B2, T_FIT, T_REF, T_TEST, load_json, potential_record,
                    save_csv, save_json)

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from forcefield_calibration.optimization.attainability import (anchor_orientational, entropy_budget,
                                                               per_temperature_anchoring, solve_both,
                                                               stiffness_scan, width_scan)
from forcefield_calibration.optimization.barrier_scan import barrier_scan, compensate_eb, orthogonality
from forcefield_calibration.optimization.fit_pair_potentials import (b2_frozen_residuals, fit, lnK_residuals,
                                                                     nominal_kb, predictions)
from forcefield_calibration.optimization.sensitivity_analysis import observables, sensitivity
from forcefield_calibration.potentials.candidates import (ALL_LABELS, BARRIER_KEY, BARRIER_POSITION,
                                                          BARRIER_POSITION_RANGE, FLUID_PARAMS, W_BOUNDS,
                                                          candidate_families, is_kf)
from forcefield_calibration.reference_data import (k_bond_md_max, libration_period_md_min, load_reference_csv,
                                                   no2_libration_inertia, polat_lj)
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


def angular_record(pot):
    """Patch width, angular stiffness and libration periods of a Kern-Frenkel potential."""
    I_A, I_C = no2_libration_inertia()
    nuA, tA = pot.libration(I_A)
    nuC, tC = pot.libration(I_C)
    return dict(w=pot.p["w"], w_deg=float(np.rad2deg(pot.p["w"])), kappa=pot.angular_stiffness(), W_b=pot.W_b,
                r_s=pot.r_s, wall_A=pot.A, libration_wavenumber_IA=nuA, libration_period_fs_IA=tA,
                libration_wavenumber_IC=nuC, libration_period_fs_IC=tC,
                equivalent_hard_cone_deg_298=pot.equivalent_cone(T_REF),
                libration_md_ok=bool(tA >= libration_period_md_min()))


def run_family(label, base, ref, quick=False):
    t0 = time.time()
    kf = is_kf(label)
    rec = dict(label=label, name=base.name, kern_frenkel=kf)
    fit_mask = np.array([r == "fit" for r in ref["role"]])
    ref_fit = subset(ref, fit_mask)
    eps, sig = polat_lj("NO2")
    B2t = B2_lennard_jones(eps, sig, T_B2)
    sB = np.maximum(5.0, 0.05 * np.abs(B2t))
    rec["B2_target"] = dict(T=T_B2, B2_LJ=B2t, sigma=sB, source="Polat et al. NO2 LJ (eps/k=262.17 K, sigma=3.727 A)")
    kmax = k_bond_md_max()
    i_ref = int(np.argmin(np.abs(ref["T"] - T_REF)))

    # ---------------- Step 2: fluid branch
    fl = fit(base, FLUID_PARAMS[label], b2_frozen_residuals(T_B2, B2t, sB), maxiter=20 if quick else 40)
    pot_f = fl.pot
    rec["fluid_fit"] = dict(values=fl.values, chi2=fl.chi2, n_points=len(T_B2),
                            at_bound=[p[0] for p in FLUID_PARAMS[label]
                                      if min(abs(fl.values[p[0]] - p[1]), abs(fl.values[p[0]] - p[2]))
                                      < 1e-3 * (p[2] - p[1])])

    # ---------------- Step 3: bond well
    bkey = BARRIER_KEY[label]
    pos = BARRIER_POSITION[bkey]
    rh0 = pot_f.p[pos]
    lo, hi = BARRIER_POSITION_RANGE[pos]
    extra = [(pos, rh0 + lo, rh0 + hi)] + ([("w", W_BOUNDS[0], W_BOUNDS[1], True)] if kf else [])
    variants = {}
    fp = fit(pot_f, [("eb", 2.0, 150.0)] + extra, lnK_residuals(ref_fit), maxiter=20 if quick else 30)
    variants["physical"] = fp.pot
    rec["physical_fit"] = dict(values=fp.values, chi2=fp.chi2, barrier_position_key=pos, barrier_position_start=rh0)
    if kf:
        variants["anchored"] = anchor_orientational(fp.pot, T_REF, ref["lnK"][i_ref], ref["dH"][i_ref], kb=KB_PHYS)
    else:
        variants["anchored"] = compensate_eb(fp.pot, T_REF, ref["lnK"][i_ref], kb=KB_PHYS, span=(0.3, 3.0))
    fb = fit(pot_f, [("eb", 2.0, 150.0), ("kb", KB_RANGE[0], kmax, True)] + extra, lnK_residuals(ref_fit, w_H=1.0),
             maxiter=20 if quick else 40)
    variants["bounded"] = fb.pot
    rec["bounded_fit"] = dict(values=fb.values, chi2=fb.chi2, kb_at_upper_bound=bool(fb.values["kb"] > 0.99 * kmax),
                              kb_md_max=kmax)
    if kf:
        anc = variants["anchored"]
        w_th = float(np.sqrt(8.314462618e-3 * T_REF / KB_PHYS))
        from forcefield_calibration.optimization.attainability import period_fs, wavenumber

        rec["exact_at_298"] = dict(eb=anc.p["eb"], kb=KB_PHYS, w=anc.p["w"], converged=True, shape_valid=anc.feasible,
                                   thermal_width_A=w_th, wavenumber=wavenumber(KB_PHYS, MU),
                                   period_fs=period_fs(KB_PHYS, MU), md_timestep_fs_20pts=period_fs(KB_PHYS, MU) / 20,
                                   r_bond=anc.r_bond, handle="patch width w (orientational entropy)")
    else:
        rec["exact_at_298"] = solve_both(pot_f, T_REF, ref["lnK"][i_ref], ref["dH"][i_ref], MU)

    rec["variants"] = {}
    for name, pot in variants.items():
        m = metrics(pot, ref, T_FIT, T_TEST, T_B2, B2t, sB)
        bs = bound_state(pot, T_REF)
        tst = tst_rates(pot, T_REF, MU)
        rec["variants"][name] = dict(
            potential=potential_record(pot), eb=pot.p["eb"], kb_branch=nominal_kb(pot),
            k_bond_actual=pot.bond_stiffness(), wavenumber=pot.bond_wavenumber(MU),
            barrier_position=pot.p[pos], **m,
            dS_298=bs.deltaS, dCp_298=bs.deltaCp, tau_d_TST_298=tst.tau_d, dF_diss_298=tst.dF_diss,
            angular=angular_record(pot) if kf else None,
        )
    # entropy budget of the anchored model versus experiment
    from forcefield_calibration.reference_data import nasa7_reaction
    dS_ref = float(nasa7_reaction(np.array([T_REF]))[2][0])
    rec["entropy_anchored"] = entropy_budget(variants["anchored"], T_REF, dS_ref)

    # ---------------- Step 3b: anchoring at every temperature separately
    rec["per_T_anchoring"] = per_temperature_anchoring(variants["anchored"], ref, kb=KB_PHYS, orientational=kf)

    # ---------------- Step 4a: attainability scan over k_b (KF: over the patch width)
    if kf:
        rec["width_scan"] = width_scan(variants["anchored"], ref_fit, np.geomspace(0.1, 1.2, 8 if quick else 15),
                                       kb=KB_PHYS)
        rec["stiffness_scan"] = []
    else:
        kbs = np.logspace(2, 12, 11 if quick else 21)
        rec["stiffness_scan"] = stiffness_scan(pot_f, ref_fit, kbs, MU)

    # ---------------- Step 4b: sensitivity / identifiability (anchored potential)
    anc = variants["anchored"]
    names = ["eb", "kb", bkey, pos] + [p[0] for p in FLUID_PARAMS[label]] + (["rf"] if label != "D" else []) \
        + (["w"] if kf else [])
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


def _run_and_save(args):
    lab, quick = args
    ref = load_reference_csv()
    fams = candidate_families()
    print(f"[stage1] family {lab} ...", flush=True)
    rec = run_family(lab, fams[lab], ref, quick=quick)
    save_json(rec, f"stage1_{lab}.json")
    print(f"[stage1] {lab} done in {rec['runtime_s']:.0f} s", flush=True)
    return lab


def summary_row(rec):
    v = rec["variants"]
    pt = rec["per_T_anchoring"]["summary"]
    ang = v["anchored"].get("angular") or {}
    return dict(
        label=rec["label"], name=rec["name"],
        fluid_chi2=rec["fluid_fit"]["chi2"], fluid_at_bound=";".join(rec["fluid_fit"]["at_bound"]),
        phys_eb=v["physical"]["eb"], phys_barrier_pos=v["physical"]["barrier_position"],
        phys_rms_lnK_fit=v["physical"]["rms_lnK_fit"],
        phys_rms_lnK_test=v["physical"]["rms_lnK_test"], phys_dH_err_298=v["physical"]["dH_err_298"],
        phys_rms_dH=v["physical"]["rms_dH"],
        anch_eb=v["anchored"]["eb"], anch_dH_err_298=v["anchored"]["dH_err_298"],
        anch_rms_lnK_fit=v["anchored"]["rms_lnK_fit"], anch_rms_lnK_test=v["anchored"]["rms_lnK_test"],
        anch_rms_dH=v["anchored"]["rms_dH"], anch_dS_298=v["anchored"]["dS_298"], anch_dCp_298=v["anchored"]["dCp_298"],
        anch_tau_d_298=v["anchored"]["tau_d_TST_298"],
        anch_w_deg=ang.get("w_deg"), anch_kappa=ang.get("kappa"), anch_libration_fs_IA=ang.get("libration_period_fs_IA"),
        bnd_eb=v["bounded"]["eb"], bnd_kb=rec["bounded_fit"]["values"]["kb"],
        bnd_rms_lnK_fit=v["bounded"]["rms_lnK_fit"], bnd_dH_err_298=v["bounded"]["dH_err_298"],
        exact_kb=rec["exact_at_298"]["kb"], exact_width_A=rec["exact_at_298"]["thermal_width_A"],
        exact_period_fs=rec["exact_at_298"]["period_fs"],
        perT_eb_min=pt["eb_min"], perT_eb_max=pt["eb_max"], perT_rms_dH_err=pt["rms_dH_err"],
        perT_worst_transfer_rms_lnK=float(np.nanmax(np.array(pt["rms_lnK_transfer"], float))),
        perT_n_failed=sum("error" in r for r in rec["per_T_anchoring"]["rows"]),
        dS_missing=rec["entropy_anchored"]["dS_missing"],
        cond=rec["sensitivity"]["condition_number"],
        orthogonality=rec["barrier_scan"]["orthogonality"],
        tau_span_decades=rec["barrier_scan"].get("tau_span_decades"),
        B2frozen_shift_per_decade=rec["barrier_scan"].get("B2frozen_shift_per_decade"),
        runtime_s=rec["runtime_s"],
    )


def write_summary():
    rows = []
    for lab in ALL_LABELS:
        try:
            rec = load_json(f"stage1_{lab}.json")
        except FileNotFoundError:
            continue
        if "per_T_anchoring" in rec:  # skip results written by an older version of this script
            rows.append(summary_row(rec))
    if rows:
        save_csv(rows, "stage1_summary.csv", header=list(rows[0]))
    return rows


def main(labels=None, quick=False, workers=None):
    labels = labels or list(ALL_LABELS)
    workers = workers or min(len(labels), os.cpu_count() or 1)
    if workers > 1:
        # slowest (Kern-Frenkel) families first
        order = sorted(labels, key=lambda l: not is_kf(l))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_run_and_save, [(l, quick) for l in order]))
    else:
        for lab in labels:
            _run_and_save((lab, quick))
    return write_summary()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    nw = None
    if "--workers" in sys.argv:
        nw = int(sys.argv[sys.argv.index("--workers") + 1])
        args = [a for a in args if a != str(nw)]
    main(args or None, quick="--quick" in sys.argv, workers=nw)
