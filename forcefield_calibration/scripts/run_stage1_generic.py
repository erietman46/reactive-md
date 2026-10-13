"""Stage 1G - pre-MD design of the generic 3 x 3 (equilibrium x kinetics) matrix.

The core project does not need real NO2/N2O4 numbers; it needs potentials
whose equilibrium dimer fraction f_d and reaction time scale can be set
independently (research plan, Steps 5-7).  Using the dilute-gas theory this
script produces *starting* parameter sets for that matrix, to be refined by
the MD equilibrium calibration:

  * equilibrium regimes E1/E2/E3: eps_b such that the predicted f_d at
    (T_GEN, RHO_GEN) is 0.2 / 0.5 / 0.8.  The Morse range parameter beta_b of
    the anchored NO2-like potential is kept (one bond length scale), so k_b =
    2 eps_b beta_b^2 scales with the depth.  Keeping k_b = 580 instead is not
    possible for the piecewise family: with such a stiff but shallow well no
    barrier curvature keeps the quintic bridge monotonic (eps_b < ~12 kJ/mol);
  * kinetic regimes K1/K2/K3: barrier heights giving TST dimer lifetimes in
    the ratio 1 : 10 : 100, with eps_b re-tuned at every barrier so that K
    (hence f_d) is unchanged.

The dilute-gas prediction is not quantitative at liquid-like densities; it
fixes the ordering and the approximate parameter values that the MD
calibration starts from.  Outputs results/generic_design.json/.csv.
"""
from common import (FD_TARGETS, KB_PHYS, MU, T_GEN, TAU_RATIOS, load_json, potential_record, rebuild,
                    save_csv, save_json)

import numpy as np
from scipy.optimize import brentq

from forcefield_calibration.optimization.barrier_scan import heights_for_tau_ratios
from forcefield_calibration.potentials.candidates import BARRIER_KEY
from forcefield_calibration.thermodynamics.equilibrium_integrals import (K_V_for_fraction, bound_state,
                                                                        dimer_fraction, number_density_ideal)
from forcefield_calibration.thermodynamics.kinetics_tst import tst_rates
from forcefield_calibration.thermodynamics.virial_coefficients import B2

# Reference density: rho0 sigma^3 ~ 0.1 with sigma = 3.727 A (a dense gas, ~80 bar at 300 K
# if ideal); still close enough to the dilute limit for the mass-action estimate to be a
# useful starting point.
RHO_GEN = 0.002            # monomer units per A^3
H_LOW = {"Eh": 2.0, "H": 2.0}
H_HIGH = {"Eh": 40.0, "H": 60.0}


def eb_for_KV(pot, T, log_KV, kb=None):
    """eps_b giving ln K_V = log_KV; bracket found on a grid of valid (clean) shapes."""
    def g(eb):
        try:
            p = pot.with_bond(eb=eb, kb=kb)
            if not (p.landscape().clean and getattr(p, "feasible", True)):
                return np.nan
            return bound_state(p, T).log_KV - log_KV
        except ValueError:
            return np.nan

    grid = np.linspace(0.5, 60.0, 120)
    vals = np.array([g(e) for e in grid])
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] <= 0:
            return pot.with_bond(eb=brentq(g, grid[i], grid[i + 1], xtol=1e-10), kb=kb)
    raise ValueError("no valid eps_b reaches the requested K_V")


def design(label):
    rec = load_json(f"stage1_{label}.json")
    base = rebuild(rec["variants"]["anchored"]["potential"])
    key = BARRIER_KEY[label]
    base = base.replace(**{key: H_LOW[key]})
    rows, out = [], {}
    for ename, fd in FD_TARGETS.items():
        KV = K_V_for_fraction(fd, RHO_GEN)
        pe = eb_for_KV(base, T_GEN, np.log(KV))
        fam = heights_for_tau_ratios(pe, key, T_GEN, MU, list(TAU_RATIOS.values()), H_LOW[key], H_HIGH[key],
                                     keep_shape=True)
        for (kname, ratio), (h, p) in zip(TAU_RATIOS.items(), fam):
            bs = bound_state(p, T_GEN)
            r = tst_rates(p, T_GEN, MU)
            name = f"{ename.split()[0]}{kname.split()[0]}"
            row = dict(state=name, regime_eq=ename, regime_kin=kname, label=label,
                       eb=p.p["eb"], k_bond=p.bond_stiffness(), barrier_param=key, barrier_value=h,
                       U_bond=p.u_bond, U_barrier=p.u_barrier, r_barrier=p.r_barrier,
                       fd_pred=float(dimer_fraction(np.exp(bs.log_KV), RHO_GEN)), log_KV=bs.log_KV,
                       dH=bs.deltaH, dS=bs.deltaS, dF_diss=r.dF_diss,
                       tau_d_ps=r.tau_d * 1e12, tau_relax_ps=r.tau_relax(RHO_GEN) * 1e12,
                       B2_frozen=B2(p, T_GEN).frozen,
                       fd_at_1bar=float(dimer_fraction(np.exp(bs.log_KV), number_density_ideal(1e5, T_GEN))))
            rows.append(row)
            out[name] = dict(row=row, potential=potential_record(p))
    return rows, out


def main():
    all_rows, all_out = [], {}
    for lab in ("A", "C"):
        rows, out = design(lab)
        all_rows += rows
        all_out[lab] = out
        for r in rows:
            print(f"[generic] {lab} {r['state']}: eb={r['eb']:.3f} {r['barrier_param']}={r['barrier_value']:.3f} "
                  f"fd={r['fd_pred']:.3f} tau_d={r['tau_d_ps']:.3g} ps tau_r={r['tau_relax_ps']:.3g} ps")
    save_json(dict(T=T_GEN, rho0=RHO_GEN, bond="Morse range beta_b of the anchored potential kept fixed",
                   states=all_out), "generic_design.json")
    save_csv(all_rows, "generic_design.csv", header=list(all_rows[0]))


if __name__ == "__main__":
    main()
