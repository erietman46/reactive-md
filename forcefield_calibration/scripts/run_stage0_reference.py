"""Stage 0 - build the reference calibration dataset (Milestone 1).

Outputs
  data/reference_Kp.csv                    ln K_p, dH, dS with uncertainties and fit/test role
  results/reference_check.csv              NASA-7 vs JANAF anchor comparison
  figures/fig0_reference_vanthoff.png
"""
from common import FIGURES, ROOT, T_FIT, T_TEST, save_csv, setup_mpl, PALETTE, INK2

import numpy as np

from forcefield_calibration.reference_data import build_reference_table, janaf_anchors, nasa7_reaction


def main():
    T_all = np.sort(np.r_[T_FIT, T_TEST])
    ref = build_reference_table(T_all)
    _, _, dS = nasa7_reaction(T_all)
    role = ["fit" if t in T_FIT else "test" for t in T_all]
    lines = [
        "# Reference calibration data for 2 NO2 -> N2O4 (association), p0 = 1 bar.",
        f"# source: {ref['source']}",
        f"# uncertainty = max(|NASA-7 - JANAF anchor|) floored: spread lnK = {ref['spread_lnK']:.4f}, "
        f"spread dH = {ref['spread_dH']:.4f} kJ/mol",
        "# dH values come from the same thermochemical evaluation as ln K: they are NOT independent data.",
        "T_K,lnKp_assoc,Kp_assoc,deltaH_kJmol,deltaS_JmolK,sigma_lnK,sigma_dH_kJmol,role,source",
    ]
    for i, t in enumerate(T_all):
        lines.append(f"{t:g},{ref['lnK'][i]:.6f},{np.exp(ref['lnK'][i]):.6g},{ref['dH'][i]:.4f},{dS[i]:.3f},"
                     f"{ref['sigma_lnK'][i]:.4f},{ref['sigma_dH'][i]:.4f},{role[i]},NASA7")
    (ROOT / "data" / "reference_Kp.csv").write_text("\n".join(lines) + "\n")

    Ta, lnKa, dHa = janaf_anchors()
    lnKn, dHn, dSn = nasa7_reaction(Ta)
    save_csv([dict(T_K=t, lnK_janaf=a, lnK_nasa7=b, dlnK=b - a, dH_janaf=c, dH_nasa7=d, ddH=d - c, dS_nasa7=s)
              for t, a, b, c, d, s in zip(Ta, lnKa, lnKn, dHa, dHn, dSn)], "reference_check.csv",
             header=["T_K", "lnK_janaf", "lnK_nasa7", "dlnK", "dH_janaf", "dH_nasa7", "ddH", "dS_nasa7"])

    plt = setup_mpl()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    Tg = np.linspace(280, 510, 200)
    lg, hg, _ = nasa7_reaction(Tg)
    ax[0].plot(1000 / Tg, lg, color=PALETTE[0], label="NASA-7 polynomials")
    ax[0].plot(1000 / Ta, lnKa, "o", ms=8, mfc="none", mew=2, color=PALETTE[1], label="JANAF anchors")
    for t, k, r in zip(T_all, ref["lnK"], role):
        ax[0].plot(1000 / t, k, "s" if r == "fit" else "^", ms=6, color=INK2)
    ax[0].plot([], [], "s", color=INK2, label="fit temperatures")
    ax[0].plot([], [], "^", color=INK2, label="held-out temperatures")
    ax[0].set(xlabel="1000 / T  (1/K)", ylabel="ln K_p (association)", title="2 NO2 -> N2O4: van 't Hoff plot")
    ax[0].legend()
    ax[1].plot(Tg, hg, color=PALETTE[0], label="NASA-7")
    ax[1].plot(Ta, dHa, "o", ms=8, mfc="none", mew=2, color=PALETTE[1], label="JANAF anchors")
    ax[1].set(xlabel="T (K)", ylabel="dH_r (kJ/mol)", title="Reaction enthalpy")
    ax[1].legend()
    fig.savefig(FIGURES / "fig0_reference_vanthoff.png")
    print("reference data written;", ref["source"])


if __name__ == "__main__":
    main()
