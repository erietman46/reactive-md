"""Regenerate all calibration figures from the JSON results (no recomputation)."""
from common import (FAMILY_COLOR, FIGURES, GRID, INK2, MU, MUTED, PALETTE, T_REF, load_json, rebuild,
                    setup_mpl)

import numpy as np

from forcefield_calibration.reference_data import k_bond_md_max, load_reference_csv
from forcefield_calibration.thermodynamics.kinetics_tst import radial_free_energy

plt = setup_mpl()
LABELS = "ABCDEF"


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
    fig.text(0.5, -0.07, "A, B, E and F coincide outside the inner wall (they differ only for r < r_b).",
             ha="center", color=MUTED, fontsize=9)
    fig.savefig(FIGURES / "fig1_pair_potentials.png")
    plt.close(fig)


def fig_vanthoff(S, ref):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for i, mode in enumerate(("physical", "anchored", "bounded")):
        a = ax[i]
        a.plot(1000 / ref["T"], ref["lnK"], "o", ms=8, mfc="none", mew=2, color=INK2, label="reference (NASA-7)")
        for lab, d in S.items():
            v = d["variants"][mode]["predictions"]
            a.plot(1000 / np.array(v["T"]), v["lnK"], color=FAMILY_COLOR[lab], label=lab)
        a.axvline(1000 / 412.5, color=MUTED, lw=1, ls=":")
        a.text(1000 / 412.5, 0.98, "held-out | fit ", color=MUTED, fontsize=8, va="top", ha="center",
               transform=a.get_xaxis_transform())
        a.set(xlabel="1000 / T (1/K)", title={"physical": "k_b = 580, eps_b fitted to ln K(T)",
                                             "anchored": "k_b = 580, ln K exact at 298 K",
                                             "bounded": "eps_b, k_b free (MD-bounded), ln K + dH"}[mode])
    ax[0].set_ylabel("ln K_p (association)")
    ax[0].legend(ncol=2, fontsize=8, loc="lower right")
    fig.text(0.5, -0.07, "A, B, E and F coincide (the piecewise bond branch only shapes the inner wall); "
             "open circles: reference, lines: models", ha="center", color=MUTED, fontsize=9)
    fig.savefig(FIGURES / "fig2_vanthoff_fits.png")
    plt.close(fig)


def fig_attainability(S):
    kmax = k_bond_md_max()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for lab, d in S.items():
        if lab not in "ABCD":  # E/F bond forms cannot represent the very stiff wells of the scan
            continue
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
        a.axvline(kmax, color=PALETTE[7], lw=1, ls="--")
        a.text(580, 0.98, " physical N-N", color=INK2, fontsize=8, va="top", transform=a.get_xaxis_transform())
        a.text(kmax, 0.90, " MD limit (1 fs, 20 steps/period)", color=PALETTE[7], fontsize=8, va="top",
               transform=a.get_xaxis_transform())
        a.set_xlabel("bond curvature k_b (kJ/mol/A^2)")
    ax[0].set(ylabel="dH(model) - dH(ref) at 298 K (kJ/mol)", title="eps_b fitted to ln K(T): enthalpy error")
    ax[1].set(ylabel="ln K(model) - ln K(ref) at 298 K", title="eps_b fitted to dH: equilibrium error")
    ax[0].axhline(0, color=MUTED, lw=1)
    ax[1].axhline(0, color=MUTED, lw=1)
    ax[0].legend(ncol=4, loc="lower left")
    fig.text(0.5, -0.07, "Piecewise families (A, B) stop where no monotonic quintic bridge exists for the stiff well; "
             "additive families (C, D) continue to 1e12.", ha="center", color=MUTED, fontsize=9)
    fig.savefig(FIGURES / "fig3_attainability.png")
    plt.close(fig)


def fig_virial(S):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    first = next(iter(S.values()))
    bt = first["B2_target"]
    ax[0].plot(bt["T"], bt["B2_LJ"], "o", ms=8, mfc="none", mew=2, color=INK2, label="Polat LJ monomer (target)")
    for lab, d in S.items():
        v = d["variants"]["anchored"]["predictions_B2"]
        ax[0].plot(v["T"], v["B2_frozen"], color=FAMILY_COLOR[lab], label=lab)
    ax[0].set(xlabel="T (K)", ylabel="B2 frozen (cm^3/mol)", title="Non-bonded (frozen-chemistry) B2")
    ax[0].legend(ncol=2, fontsize=8)
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
        c = FAMILY_COLOR[lab]
        ax[0].plot(h, [r["dlnK_uncomp"] for r in rows], "-o", ms=4, color=c, label=lab)
        ax[1].plot(h, np.array([r["tau_d"] for r in rows]) * 1e9, "-o", ms=4, color=c, label=lab)
        ax[2].plot(np.array([r["dln_tau"] for r in rows]) / np.log(10), [r["B2_frozen"] for r in rows],
                   "-o", ms=4, color=c, label=lab)
    ax[0].set(xlabel="barrier energy U(r_barrier) (kJ/mol)", ylabel="d ln K_p (no compensation)",
              title="Barrier leakage into equilibrium (298 K)")
    ax[1].set(yscale="log", xlabel="barrier energy U(r_barrier) (kJ/mol)", ylabel="TST dimer lifetime (ns)",
              title="Kinetic control at fixed K (eps_b re-tuned)")
    ax[2].set(xlabel="decades of lifetime change", ylabel="B2 frozen (cm^3/mol)",
              title="Side effect on non-bonded structure")
    ax[0].legend(ncol=3)
    fig.savefig(FIGURES / "fig5_barrier_scan.png")
    plt.close(fig)


def fig_free_energy(S):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    d = S["A"]
    p = rebuild(d["variants"]["anchored"]["potential"])
    r = np.linspace(1.5, 8, 2000)
    ax.plot(r, p.U(r), color=PALETTE[0], label="U2(r)")
    F = radial_free_energy(p, r, T_REF, r_ref=p.r_disp)
    ax.plot(r, F - F[-1] + p.U(r)[-1], color=PALETTE[1], label="F(r) = U - 2RT ln r (298 K)")
    ax.axvline(p.r_barrier, color=MUTED, lw=1, ls="--")
    ax.text(p.r_barrier, 6, " dividing surface", color=MUTED, fontsize=8)
    ax.set(ylim=(-32, 10), xlabel="r (A)", ylabel="kJ/mol", title="Family A (anchored): energy vs radial free energy")
    ax.legend()
    fig.savefig(FIGURES / "fig6_radial_free_energy.png")
    plt.close(fig)


def fig_sensitivity(S):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for lab, d in S.items():
        s = np.array(d["sensitivity"]["singular_values"])
        ax.semilogy(np.arange(1, s.size + 1), s / s[0], "-o", ms=5, color=FAMILY_COLOR[lab], label=lab)
    ax.axhline(1e-2, color=MUTED, lw=1, ls="--")
    ax.set(xlabel="singular value index", ylabel="s_i / s_1",
           title="Identifiability of the anchored parameters")
    ax.legend(ncol=3)
    fig.savefig(FIGURES / "fig7_identifiability.png")
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
    fig_generic()
    fig_stage2()
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
