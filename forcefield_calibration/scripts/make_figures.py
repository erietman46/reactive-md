"""Regenerate all calibration figures from the JSON results (no recomputation)."""
from common import (FAMILY_COLOR, FIGURES, GRID, INK2, MU, MUTED, PALETTE, T_REF, family_style, load_json,
                    rebuild, setup_mpl)

import numpy as np

from forcefield_calibration.potentials.candidates import ALL_LABELS, ISOTROPIC, KERN_FRENKEL
from forcefield_calibration.reference_data import k_bond_md_max, libration_period_md_min, load_reference_csv
from forcefield_calibration.thermodynamics.kinetics_tst import radial_free_energy

plt = setup_mpl()
LABELS = ALL_LABELS
KF_NOTE = "dashed: smooth Kern-Frenkel family, in the colour of its radial construction"


def stage1():
    out = {}
    for lab in LABELS:
        try:
            out[lab] = load_json(f"stage1_{lab}.json")
        except FileNotFoundError:
            pass
    return out


def fig_potentials(S):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    r = np.linspace(1.2, 10.0, 3000)
    for lab, d in S.items():
        if lab not in ISOTROPIC:
            continue
        p = rebuild(d["variants"]["anchored"]["potential"])
        c = FAMILY_COLOR[lab]
        ax[0].plot(r, p.U(r), color=c, label=lab)
        ax[1].plot(r, p.U(r), color=c, label=lab)
        ax[2].plot(r, -p.dU(r), color=c, label=lab)
    ax[0].set(xlim=(1.3, 6), ylim=(-32, 12), xlabel="r (A)", ylabel="U2 (kJ/mol)",
              title="Anchored pair potentials (K_p exact at 298 K)")
    ax[1].set(xlim=(2.4, 10), ylim=(-4, 6), xlabel="r (A)", ylabel="U2 (kJ/mol)", title="Barrier and dispersion region")
    ax[2].set(xlim=(1.5, 6), ylim=(-150, 150), xlabel="r (A)", ylabel="F = -dU/dr (kJ/mol/A)", title="Force")
    ax[0].legend(title="family", ncol=2, loc="lower right")
    fig.text(0.5, -0.07, "Isotropic families. A, B, E, F (and G inside 4.2 A) coincide outside the inner wall; "
             "the Kern-Frenkel families are shown in fig11.", ha="center", color=MUTED, fontsize=9)
    fig.savefig(FIGURES / "fig1_pair_potentials.png")
    plt.close(fig)


def fig_vanthoff(S, ref):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for i, mode in enumerate(("physical", "anchored", "bounded")):
        a = ax[i]
        a.plot(1000 / ref["T"], ref["lnK"], "o", ms=8, mfc="none", mew=2, color=INK2, label="reference (NASA-7)")
        for lab, d in S.items():
            v = d["variants"][mode]["predictions"]
            o = np.argsort(v["T"])
            a.plot(1000 / np.array(v["T"])[o], np.array(v["lnK"])[o], label=lab, lw=1.6, **family_style(lab))
        a.axvline(1000 / 412.5, color=MUTED, lw=1, ls=":")
        a.text(1000 / 412.5, 0.98, "held-out | fit ", color=MUTED, fontsize=8, va="top", ha="center",
               transform=a.get_xaxis_transform())
        a.set(xlabel="1000 / T (1/K)", title={"physical": "k_b = 580, eps_b (+ r_h, w) fitted to ln K(T)",
                                             "anchored": "k_b = 580, ln K exact at 298 K (KF: + dH)",
                                             "bounded": "eps_b, k_b free (MD-bounded), ln K + dH"}[mode])
    ax[0].set_ylabel("ln K_p (association)")
    ax[0].legend(ncol=3, fontsize=7, loc="lower right")
    fig.text(0.5, -0.07, "Open circles: reference; solid: isotropic families (A, B, E, F, G coincide); "
             + KF_NOTE + " (all on the reference).", ha="center", color=MUTED, fontsize=9)
    fig.savefig(FIGURES / "fig2_vanthoff_fits.png")
    plt.close(fig)


def fig_attainability(S):
    kmax = k_bond_md_max()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for lab, d in S.items():
        if lab not in ("A", "B", "C", "D", "G", "H", "I"):  # E/F bond forms cannot represent very stiff wells;
            continue                                             # KF families are scanned over w (fig13)
        rows = d["stiffness_scan"]
        kb = np.array([r["kb"] for r in rows])
        valid = np.array([r.get("shape_valid", True) for r in rows])
        dH = np.where(valid, np.array([r.get("dH_err", np.nan) for r in rows], float), np.nan)
        lk = np.where(valid, np.array([r.get("lnK_err_at_T", np.nan) for r in rows], float), np.nan)
        ax[0].plot(kb, dH, "-o", ms=4, color=FAMILY_COLOR[lab], label=lab)
        ax[1].plot(kb, lk, "-o", ms=4, color=FAMILY_COLOR[lab], label=lab)
    for a in ax:
        a.set_xscale("log")
        a.axvline(580, color=INK2, lw=1, ls="--")
        a.axvline(kmax, color=MUTED, lw=1, ls="-.")
        a.text(580, 0.98, " physical N-N", color=INK2, fontsize=8, va="top", transform=a.get_xaxis_transform())
        a.text(kmax, 0.90, " MD limit (1 fs, 20 steps/period)", color=MUTED, fontsize=8, va="top",
               transform=a.get_xaxis_transform())
        a.set_xlabel("bond curvature k_b (kJ/mol/A^2)")
    ax[0].set(ylabel="dH(model) - dH(ref) at 298 K (kJ/mol)", title="eps_b fitted to ln K(T): enthalpy error")
    ax[1].set(ylabel="ln K(model) - ln K(ref) at 298 K", title="eps_b fitted to dH: equilibrium error")
    ax[0].axhline(0, color=MUTED, lw=1)
    ax[1].axhline(0, color=MUTED, lw=1)
    ax[0].legend(ncol=4, loc="lower left")
    fig.text(0.5, -0.07, "Piecewise families (A, B, G) stop where no monotonic quintic bridge exists for the stiff well; "
             "additive families (C, D, H, I) continue to 1e12.", ha="center", color=MUTED, fontsize=9)
    fig.savefig(FIGURES / "fig3_attainability.png")
    plt.close(fig)


def fig_virial(S):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    first = next(iter(S.values()))
    bt = first["B2_target"]
    ax[0].plot(bt["T"], bt["B2_LJ"], "o", ms=8, mfc="none", mew=2, color=INK2, label="Polat LJ monomer (target)")
    for lab, d in S.items():
        v = d["variants"]["anchored"]["predictions_B2"]
        ax[0].plot(v["T"], v["B2_frozen"], label=lab, lw=1.6, **family_style(lab))
    ax[0].set(xlabel="T (K)", ylabel="B2 frozen (cm^3/mol)", title="Non-bonded (frozen-chemistry) B2")
    ax[0].legend(ncol=3, fontsize=7)
    d = S.get("A", first)
    v = d["variants"]["anchored"]["predictions"]
    T = np.array(v["T"])
    o = np.argsort(T)
    ax[1].plot(T[o], -np.array(v["B2"])[o], color=PALETTE[0], label="-B2 total")
    ax[1].plot(T[o], np.array(v["Kc_L_mol"])[o] * 1e3, "--", color=PALETTE[1], label="N_A K_V (association)")
    ax[1].plot(T[o], -np.array(v["B2_frozen"])[o], color=PALETTE[2], label="-B2 frozen")
    ax[1].set(yscale="log", xlabel="T (K)", ylabel="cm^3/mol", title="Family A: B2 = B2_frozen - N_A K_V")
    ax[1].legend()
    fig.savefig(FIGURES / "fig4_virial.png")
    plt.close(fig)


def fig_barrier(S):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for lab, d in S.items():
        rows = [r for r in d["barrier_scan"]["rows"] if r.get("clean_comp")]
        if not rows:
            continue
        h = np.array([r["U_barrier"] for r in rows])
        st = dict(marker="o", ms=3, lw=1.4, **family_style(lab))
        ax[0].plot(h, [r["dlnK_uncomp"] for r in rows], label=lab, **st)
        ax[1].plot(h, np.array([r["tau_d"] for r in rows]) * 1e9, label=lab, **st)
        ax[2].plot(np.array([r["dln_tau"] for r in rows]) / np.log(10), [r["B2_frozen"] for r in rows],
                   label=lab, **st)
    ax[0].set(xlabel="barrier energy U(r_barrier) (kJ/mol)", ylabel="d ln K_p (no compensation)",
              title="Barrier leakage into equilibrium (298 K)")
    ax[1].set(yscale="log", xlabel="barrier energy U(r_barrier) (kJ/mol)", ylabel="TST dimer lifetime (ns)",
              title="Kinetic control at fixed K (eps_b re-tuned)")
    ax[2].set(xlabel="decades of lifetime change", ylabel="B2 frozen (cm^3/mol)",
              title="Side effect on non-bonded structure")
    ax[0].legend(ncol=3, fontsize=7)
    fig.text(0.5, -0.07, KF_NOTE + ".", ha="center", color=MUTED, fontsize=9)
    fig.savefig(FIGURES / "fig5_barrier_scan.png")
    plt.close(fig)


def fig_free_energy(S):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    r = np.linspace(1.5, 8, 2000)
    for ax, lab in zip(axs, ("A", "KF-A")):
        if lab not in S:
            continue
        p = rebuild(S[lab]["variants"]["anchored"]["potential"])
        ax.plot(r, p.U(r), color=PALETTE[0], label="U2(r)" if lab == "A" else "U_on(r) (patches aligned)")
        F = radial_free_energy(p, r, T_REF, r_ref=p.r_disp)
        ax.plot(r, F - F[-1] + p.U(r)[-1], color=PALETTE[1],
                label="F(r) = U - 2RT ln r (298 K)" if lab == "A" else "F(r), incl. -RT ln <exp(-E_ang/RT)>")
        ax.axvline(p.r_barrier, color=MUTED, lw=1, ls="--")
        ax.text(p.r_barrier, 6, " dividing surface", color=MUTED, fontsize=8)
        lo = min(-32, p.u_bond - 3)
        ax.set(ylim=(lo, 12), xlabel="r (A)", ylabel="kJ/mol",
               title=f"Family {lab} (anchored): energy vs radial free energy")
        ax.legend(loc="lower right")
    fig.savefig(FIGURES / "fig6_radial_free_energy.png")
    plt.close(fig)


def fig_sensitivity(S):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for lab, d in S.items():
        s = np.array(d["sensitivity"]["singular_values"])
        ax.semilogy(np.arange(1, s.size + 1), s / s[0], marker="o", ms=4, lw=1.4, label=lab, **family_style(lab))
    ax.axhline(1e-2, color=MUTED, lw=1, ls="--")
    ax.set(xlabel="singular value index", ylabel="s_i / s_1",
           title="Identifiability of the anchored parameters")
    ax.legend(ncol=3, fontsize=7)
    fig.savefig(FIGURES / "fig7_identifiability.png")
    plt.close(fig)


def fig_equilibrium_mapping(S, ref):
    """Isotropic vs Kern-Frenkel: ln K error, dH and dS against T (anchored potentials)."""
    from forcefield_calibration.reference_data import nasa7_reaction

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    Tg = np.linspace(290, 510, 200)
    _, hg, sg = nasa7_reaction(Tg)
    ax[1].plot(Tg, hg, color=INK2, lw=3, alpha=0.35, label="reference (NASA-7)")
    ax[2].plot(Tg, sg, color=INK2, lw=3, alpha=0.35, label="reference (NASA-7)")
    from forcefield_calibration.thermodynamics.equilibrium_integrals import bound_state

    for lab, d in S.items():
        v = d["variants"]["anchored"]["predictions"]
        o = np.argsort(v["T"])
        T = np.array(v["T"])[o]
        lk = np.array(v["lnK"])[o] - np.interp(T, ref["T"], ref["lnK"])
        st = dict(lw=1.6, **family_style(lab))
        ax[0].plot(T, lk, label=lab, **st)
        ax[1].plot(T, np.array(v["dH"])[o], label=lab, **st)
        ax[2].plot(T, np.array(v["dS"])[o], label=lab, **st)
    ax[0].axhline(0, color=MUTED, lw=1)
    ax[0].axvline(412.5, color=MUTED, lw=1, ls=":")
    ax[0].set(xlabel="T (K)", ylabel="ln K_p(model) - ln K_p(ref)",
              title="Equilibrium constant, anchored at 298 K")
    ax[1].set(xlabel="T (K)", ylabel="dH (kJ/mol)", title="Reaction enthalpy")
    ax[2].set(xlabel="T (K)", ylabel="dS (J/mol/K)", title="Reaction entropy")
    ax[0].legend(ncol=3, fontsize=7, loc="lower left")
    ax[1].legend(handles=ax[1].get_lines()[:1], loc="center right")
    fig.text(0.5, -0.07, "Solid: isotropic families (all nine overlap within their bundle); " + KF_NOTE
             + ". The patch restores the missing ~94 J/mol/K of orientational entropy at the physical k_b.",
             ha="center", color=MUTED, fontsize=9)
    fig.savefig(FIGURES / "fig11_equilibrium_mapping.png")
    plt.close(fig)


def fig_per_T(S):
    """Anchoring at every temperature separately."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for lab, d in S.items():
        pt = d["per_T_anchoring"]
        T = np.array(pt["T"])
        eb = np.array([r.get("eb", np.nan) for r in pt["rows"]], float)
        dh = np.array([r.get("dH_err", np.nan) for r in pt["rows"]], float)
        st = dict(marker="o", ms=3, lw=1.4, **family_style(lab))
        ax[0].plot(T, eb - eb[0], label=lab, **st)
        ax[1].plot(T, dh, label=lab, **st)
        ax[2].plot(T, pt["summary"]["rms_lnK_transfer"], label=lab, **st)
    ax[0].set(xlabel="anchoring temperature T (K)", ylabel="eps_b(T) - eps_b(298 K) (kJ/mol)",
              title="Bond depth needed to hit ln K at each T")
    ax[1].set(xlabel="anchoring temperature T (K)", ylabel="dH(model) - dH(ref) at T (kJ/mol)",
              title="Enthalpy error of each single-T anchor")
    ax[2].set(yscale="log", xlabel="anchoring temperature T (K)", ylabel="rms ln K error over 298-500 K",
              title="Transfer of a single-T anchor to all T")
    ax[0].legend(ncol=3, fontsize=7)
    for a in ax[:2]:
        a.axhline(0, color=MUTED, lw=1)
    fig.text(0.5, -0.07, "k_b = 580 kJ/mol/A^2 throughout; the Kern-Frenkel families keep the patch width of their "
             "298 K anchor. " + KF_NOTE + ".", ha="center", color=MUTED, fontsize=9)
    fig.savefig(FIGURES / "fig12_per_temperature_anchoring.png")
    plt.close(fig)


def fig_kern_frenkel(S):
    kf = {lab: S[lab] for lab in KERN_FRENKEL if lab in S}
    if not kf:
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for lab, d in kf.items():
        rows = [r for r in d["width_scan"] if "dH_err" in r]
        w = np.array([r["w_deg"] for r in rows])
        st = dict(marker="o", ms=4, lw=1.6, **family_style(lab))
        ax[0].plot(w, [r["dH_err"] for r in rows], label=lab, **st)
        ax[1].plot(w, [r["rms_lnK"] for r in rows], label=lab, **st)
    ax[0].axhline(0, color=MUTED, lw=1)
    ax[0].set(xscale="log", xlabel="patch width w (deg)", ylabel="dH(model) - dH(ref) at 298 K (kJ/mol)",
              title="eps_b fitted to ln K(T) at each patch width")
    ax[1].set(xscale="log", yscale="log", xlabel="patch width w (deg)", ylabel="rms ln K error (fit temperatures)",
              title="Quality of that ln K(T) fit")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

    for a in ax[:2]:
        a.xaxis.set_major_locator(FixedLocator([5, 10, 20, 40, 70]))
        a.xaxis.set_minor_locator(NullLocator())
        a.xaxis.set_major_formatter(ScalarFormatter())
    ax[0].legend()
    d = kf.get("KF-A", next(iter(kf.values())))
    p = rebuild(d["variants"]["anchored"]["potential"])
    r = np.linspace(1.45, 6.0, 1500)
    ax[2].plot(r, p.U(r), color=PALETTE[0], label="aligned, S = 1 (U_on)")
    for Sv, c in ((0.5, PALETTE[2]), (0.1, PALETTE[3])):
        ax[2].plot(r, Sv * p.U(r) + (1 - Sv) * p.U_off(r), color=c, label=f"S = {Sv:g}")
    ax[2].plot(r, p.U_off(r), color=PALETTE[1], label="misaligned, S = 0 (U_off)")
    ax[2].axvline(p.r_s, color=MUTED, lw=1, ls=":")
    ax[2].text(p.r_s, 0.55, " r_s: isotropic beyond", color=MUTED, fontsize=8, va="top",
               transform=ax[2].get_xaxis_transform())
    ax[2].set(ylim=(p.u_bond - 5, 60), xlabel="r (A)", ylabel="U (kJ/mol)",
              title=f"{d['label']} anchored: U = S U_on + (1 - S) U_off, w = {np.rad2deg(p.p['w']):.1f} deg")
    ax[2].legend(loc="lower right", fontsize=8)
    fig.text(0.5, -0.07, f"k_b = 580 kJ/mol/A^2 fixed. S = f(c_i) f(c_j), f(c) = exp(-(1 - c)/w^2). Libration period "
             f"of the anchored KF-A patch: {d['variants']['anchored']['angular']['libration_period_fs_IA']:.0f} fs "
             f"(MD limit {libration_period_md_min():g} fs).", ha="center", color=MUTED, fontsize=9)
    fig.savefig(FIGURES / "fig13_kern_frenkel.png")
    plt.close(fig)


def fig_kf_valence(S):
    if "KF-A" not in S:
        return
    from forcefield_calibration.manybody.patchy_valence import second_bond_map
    from forcefield_calibration.manybody.trimer_energy_surface import addition_energy

    p = rebuild(S["KF-A"]["variants"]["anchored"]["potential"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    b = np.linspace(1.3, 4.5, 200)
    th = np.deg2rad(np.linspace(0, 180, 121))
    B, Th = np.meshgrid(b, th, indexing="ij")
    lim = abs(p.u_bond)
    E0 = addition_energy(p.radial, None, B, Th, with_mb=False)
    E1 = second_bond_map(p, B, Th, n_phi=121)
    for a, E, title in ((ax[0], E0, "same curve, no orientation"), (ax[1], E1, "smooth Kern-Frenkel, one patch")):
        im = a.pcolormesh(np.rad2deg(Th), B, np.clip(E, -lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim,
                          shading="auto")
        a.contour(np.rad2deg(Th), B, E, levels=[0.0], colors=INK2, linewidths=1)
        a.set(xlabel="angle at particle 1 (deg)", ylabel="r13 (A)", title=f"Adding a 3rd particle: {title}")
    fig.colorbar(im, ax=ax, label="dE relative to dimer + monomer (kJ/mol)")
    fig.text(0.5, -0.05, "KF-A anchored, no many-body term; right: partner patches point at particle 1, the patch of "
             "particle 1 rotated to its best direction.", ha="center", color=MUTED, fontsize=9)
    fig.savefig(FIGURES / "fig14_kf_valence.png")
    plt.close(fig)


def fig_generic():
    try:
        g = load_json("generic_design.json")
    except FileNotFoundError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    rows = [v["row"] for v in g["states"]["A"].values()]
    markers = {"K1": "o", "K2": "s", "K3": "^"}
    cols = {"E1": PALETTE[0], "E2": PALETTE[1], "E3": PALETTE[2]}
    for r in rows:
        e, k = r["state"][:2], r["state"][2:]
        ax[0].plot(r["fd_pred"], r["tau_d_ps"], markers[k], ms=9, color=cols[e])
        ax[0].annotate(r["state"], (r["fd_pred"], r["tau_d_ps"]), textcoords="offset points", xytext=(6, 2),
                       fontsize=8, color=INK2)
    ax[0].set(yscale="log", xlabel="predicted dimer fraction f_d", ylabel="TST dimer lifetime (ps)",
              title=f"Generic 3x3 design, family A (T = {g['T']:g} K, rho0 = {g['rho0']:g} 1/A^3)")
    p = {n: rebuild(v["potential"]) for n, v in g["states"]["A"].items()}
    r = np.linspace(1.4, 6, 1500)
    for n in ("E1K2", "E2K2", "E3K2", "E2K1", "E2K3"):
        ls = {"K1": ":", "K2": "-", "K3": "--"}[n[2:]]
        ax[1].plot(r, p[n].U(r), ls, color=cols[n[:2]] if n.endswith("K2") else MUTED, label=n)
    ax[1].set(ylim=(-20, 16), xlabel="r (A)", ylabel="U2 (kJ/mol)",
              title="Design potentials (E rows: depth; K columns: barrier)")
    ax[1].legend(ncol=2, fontsize=8)
    fig.savefig(FIGURES / "fig8_generic_design.png")
    plt.close(fig)


def fig_stage2():
    for track in ("real", "generic"):
        try:
            d = load_json(f"stage2_{track}.json")
        except FileNotFoundError:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        rows = d["rows"]
        names = [r["penalty"] for r in rows]
        y = np.arange(len(rows))
        ax[0].barh(y, [max(r["trimer_fraction"], 1e-12) for r in rows], color=PALETTE[0], height=0.6)
        ax[0].axvline(rows[0]["trimer_fraction_pair_only"], color=PALETTE[7], ls="--", lw=1)
        ax[0].text(rows[0]["trimer_fraction_pair_only"], len(rows) - 0.4, " pair only", color=PALETTE[7],
                   fontsize=8, ha="right")
        ax[0].set(xscale="log", yticks=y, yticklabels=names, xlabel="fraction of monomer units in trimers",
                  title=f"Trimer content ({track}, T = {d['T']:g} K)")
        for name, scan in d["lambda_scans"].items():
            ax[1].semilogy([s["factor"] for s in scan], [max(s["trimer_fraction"], 1e-14) for s in scan], "-o",
                           ms=5, label=name, color=PALETTE[["coord_hermite_cubic", "coord_quintic_cubic", "glue"].index(name)])
        ax[1].set(xlabel="lambda / lambda_min (glue: / 1.2 eps_b)", ylabel="trimer fraction",
                  title="Penalty strength scan")
        ax[1].legend()
        fig.savefig(FIGURES / f"fig9_stage2_{track}.png")
        plt.close(fig)
        # addition-energy map for the selected penalty
        from forcefield_calibration.manybody.coordination_penalties import make_penalty
        from forcefield_calibration.manybody.trimer_energy_surface import addition_energy, required_strength

        if track == "real":
            pot = rebuild(load_json("stage1_A.json")["variants"]["anchored"]["potential"])
        else:
            pot = rebuild(load_json("generic_design.json")["states"]["A"]["E2K2"]["potential"])
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        b = np.linspace(1.3, 4.5, 260)
        th = np.deg2rad(np.linspace(0, 180, 181))
        B, Th = np.meshgrid(b, th, indexing="ij")
        mb = make_penalty(pot, "coord_hermite_cubic")
        mb = mb.with_strength(1.15 * required_strength(pot, mb).lam_min)
        lim = abs(pot.u_bond)
        for a, (with_mb, title) in zip(ax, ((False, "pair potential only"), (True, "with coord_hermite_cubic"))):
            E = addition_energy(pot, mb, B, Th, with_mb=with_mb)
            im = a.pcolormesh(np.rad2deg(Th), B, np.clip(E, -lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim,
                              shading="auto")
            a.contour(np.rad2deg(Th), B, E, levels=[0.0], colors=INK2, linewidths=1)
            a.set(xlabel="angle at particle 1 (deg)", ylabel="r13 (A)", title=f"Adding a 3rd particle: {title}")
        fig.colorbar(im, ax=ax, label="dE relative to dimer + monomer (kJ/mol)")
        fig.savefig(FIGURES / f"fig10_addition_energy_{track}.png")
        plt.close(fig)


def main():
    S = stage1()
    ref = load_reference_csv()
    if S:
        fig_potentials(S)
        fig_vanthoff(S, ref)
        fig_attainability(S)
        fig_virial(S)
        fig_barrier(S)
        fig_free_energy(S)
        fig_sensitivity(S)
        fig_equilibrium_mapping(S, ref)
        fig_per_T(S)
        fig_kern_frenkel(S)
        fig_kf_valence(S)
    fig_generic()
    fig_stage2()
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
