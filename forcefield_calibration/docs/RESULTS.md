# Pre-MD calibration: results and recommendations

All numbers below are produced by `scripts/run_all.py`. The raw values are in
`results/*.json|csv`, the selection tables in `results/model_selection.md`
and the figures in `figures/`. The method is explained in
`CALIBRATION_STRATEGY.md`.

## Summary

1. **An isotropic single-site pair potential cannot reproduce K_p(T) and ΔH of
   2 NO₂ ⇌ N₂O₄ together with a bond that can be integrated in MD.** Matching
   both at 298 K needs a bond curvature of 3×10¹²–1×10¹³ kJ mol⁻¹ Å⁻²
   (thermal bond width ~10⁻⁶ Å, vibrational period ~0.002 fs). The reason is
   entropy. Experiment loses 176 J mol⁻¹ K⁻¹ on dimerisation. The model with
   a physical N–N bond loses only ~82, and the missing ~94 J mol⁻¹ K⁻¹ is
   mostly rotational entropy that a spherical bead does not have. This holds
   for **all six families**: it is a property of the isotropic single-site
   model, not of a functional form.
2. With the physical bond (k_b = 580 kJ mol⁻¹ Å⁻²) you choose what to honour.
   Anchoring K_p at the MD temperature gives correct composition with
   |ΔH| ≈ 29 instead of 57 kJ/mol. Fitting ε_b to the whole K_p(T) curve
   misses everywhere (rms 1.15 in ln K on the fit set, 2.85 on held-out
   temperatures) because its slope (ΔH) is wrong. Honouring ΔH makes K ~e¹¹ too large.
3. **The equilibrium data cannot tell the bond forms apart.** Morse, Mie,
   Wang–Frenkel and KTS bond branches give the same K, ΔH and B₂ to within
   0.03 in ln K. In the piecewise construction the bond branch only shapes the
   inner wall. The data fix two numbers of the well (depth and width). The
   non-bonded B₂ fixes one combination of outer-well depth and position.
4. **What separates the families is how they behave as kinetic controls.**
   Every family can change the TST dimer lifetime by 5–7 decades at fixed K.
   Barrier leakage into equilibrium is |d ln K/d ln τ| = 0.005 for the
   piecewise families and 0.0004–0.0005 for the additive ones. The side effect
   on non-bonded structure (B₂_frozen) is 11.6 cm³/mol per decade of lifetime
   for A/B against 1.3 for C. All families are C², none is rejected, and all
   six are Pareto-optimal. **Shortlist: A (primary: simplest,
   best-conditioned) and C (alternative: cleanest kinetic control, can hold
   shallow wells at physical stiffness).**
5. **Without a penalty the pair potential alone is unusable.** In the dilute
   limit 98% of monomer units end up in trimers at 1 bar and 298 K (real
   track), and 48% at the generic design density. Every exact-hinge
   coordination penalty leaves K exactly unchanged and reduces trimers to
   ~10⁻⁴ (real track) or 0.4–1% (generic, A). **Selected: Hermite counter +
   exact cubic hinge at 2 × λ_min.** It is EAM-form, so it runs as LAMMPS
   `pair_style eam/fs`, and the exported files reproduce the Python energies
   to 4×10⁻⁸ kJ/mol. The damped-tanh counter and Stillinger–Weber are rejected
   because they create long-lived trimer traps.
6. A generic 3 × 3 (f_d × time scale) design for the core project is ready as
   LAMMPS input. ε_b = 5.4 / 11.2 / 17.6 kJ/mol gives f_d = 0.2 / 0.5 / 0.8 at
   300 K and ρ₀ = 0.002 Å⁻³. Barriers of 2 / 7.7 / 13.5 kJ/mol give TST
   lifetimes in the ratio 1 : 10 : 100 (3 ps to 20 ns) at constant K. This is a
   starting point for the MD equilibrium and kinetic calibration (research
   plan Steps 5–6), not a replacement for it.

## 1. Reference data (Stage 0)

`data/reference_Kp.csv`, `results/reference_check.csv`, `figures/fig0_reference_vanthoff.png`.

| T (K) | ln K_p (NASA-7) | ln K_p (JANAF) | ΔH NASA-7 (kJ/mol) | ΔH JANAF | role |
|---|---|---|---|---|---|
| 298.15 | 1.950 | 1.908 | −57.27 | −57.11 | fit |
| 350 | −1.465 | −1.499 | −56.95 | −56.86 | fit |
| 400 | −3.902 | −3.932 | −56.51 | −56.50 | fit |
| 450 | −5.782 | −5.809 | −56.00 | −56.05 | held out |

The two compilations differ by ≤ 0.042 in ln K and ≤ 0.17 kJ/mol in ΔH. That
spread is the data uncertainty used in the fits (σ_lnK = 0.05,
σ_ΔH = 0.2 kJ/mol). ΔS° ≈ −176 J mol⁻¹ K⁻¹.

## 2. Two-body calibration (Stage 1)

`results/stage1_*.json`, `results/stage1_summary.csv`, `figures/fig1`–`fig7`.

### 2.1 Non-bonded branch (B₂_frozen vs. the Polat LJ monomer, 250–500 K)

| Family | fitted | χ² (6 T) |
|---|---|---|
| A, B, F | ε_f = 1.98 kJ/mol (r_f = 4.18 Å fixed) | 8.1 |
| C | ε_f = 3.20 kJ/mol, s_f = 0.62 Å | 0.65 |
| D | C₆ = 6.5×10³ kJ mol⁻¹ Å⁶, β_TT = 2.2 Å⁻¹ | 0.73 |
| E | ε_f = 2.99 kJ/mol, x_f = 1.58 | 0.58 |

The 12-6 outer branch with only ε_f free fits worst (rms 1.2σ). When r_f was
also free, the fit ran to whichever bound it started near (3.4 Å or 5.0 Å)
with similar χ². The sensitivity analysis confirms it: the weakest singular
direction of family A (s/s₁ = 7×10⁻⁴) is almost purely ε_f against r_f. The
monomer contact region of the model is softer than an LJ core (the barrier
replaces the repulsive wall between 2.6 and ~3.7 Å). Matching an LJ B₂ is a
coarse-graining choice, not experimental evidence.

### 2.2 Bond well, four modes (`figures/fig2_vanthoff_fits.png`)

| Family | `physical` ε_b | rms ln K fit / held-out | ΔH err (kJ/mol) | `anchored` ε_b (U_bond) | anchored ΔH err | `bounded` k_b | bounded rms ln K / ΔH err |
|---|---|---|---|---|---|---|---|
| A | 24.1 | 1.15 / 2.85 | +32.0 | 28.2 | +27.9 | 1618 | 3.79 / +20.1 |
| B | 24.2 | 1.15 / 2.85 | +32.0 | 28.2 | +27.9 | 1618 | 3.78 / +20.1 |
| C | 23.9 | 1.17 / 2.90 | +32.4 | 28.0 | +28.2 | 22 703 (MD cap) | 3.28 / +17.9 |
| D | 3.3 (−24.7)* | 1.16 / 2.92 | +32.1 | 7.3 (−28.7)* | +27.7 | 22 703 (MD cap) | 3.48 / +17.5 |
| E | 24.1 | 1.15 / 2.85 | +32.0 | 28.2 | +27.9 | 1663 | 3.79 / +20.1 |
| F | 24.1 | 1.15 / 2.85 | +32.1 | 28.2 | +27.9 | 1617 | 3.79 / +20.1 |

\*D: the Tang–Toennies attraction adds most of the well depth, so the nominal
ε_b is small. The actual well depth U_bond is given in brackets.

* The `bounded` fit for the piecewise families stops at k_b ≈ 1.6×10³, below
  the MD limit (2.3×10⁴). Beyond that, for ε_b ≈ 25–36 kJ/mol, no barrier
  curvature keeps the quintic bridge monotonic. The construction itself caps
  the stiffness.
* Anchored model at 298 K: ΔS = −82 J mol⁻¹ K⁻¹ (exp. −176), TST dimer
  lifetime 1.4×10⁻⁸ (C) to 3.9×10⁻⁸ s (A); radial free-energy barrier for
  dissociation 29–31 kJ/mol.

### 2.3 Attainability (`figures/fig3_attainability.png`)

Family C is shown because its Morse well stays valid at any stiffness (the
piecewise bridge stops being monotonic above k_b ≈ 2×10³). With ε_b fitted to
ln K_p(T), the ΔH error falls by only ~1.7 kJ/mol per decade of k_b. With ε_b
fitted to ΔH, the ln K error falls by ~0.6 per decade:

| k_b (kJ mol⁻¹ Å⁻²) | wavenumber (cm⁻¹) | period (fs) | ΔH err if K fitted | ln K err if ΔH fitted |
|---|---|---|---|---|
| 100 | 111 | 301 | +34.9 | +12.2 |
| 580 (physical) | 266 | 125 | ≈ +32.4 | ≈ +11.3 |
| 1 000 | 350 | 95 | +31.6 | +11.0 |
| 2.3×10⁴ (MD limit) | 1 670 | 20 | ≈ +26.9 | ≈ +9.4 |
| 1×10⁸ | 1.1×10⁵ | 0.30 | +14.9 | +5.2 |
| 3.5×10¹² (exact) | 2×10⁷ | 0.0016 | 0 | 0 |

Exact simultaneous solution at 298 K: k_b = 2.7×10¹² (A, B), 3.5×10¹² (C),
1.3×10¹³ (D). The Wang–Frenkel and KTS shapes (E, F) cannot represent wells
that stiff at all.

### 2.4 Identifiability (`figures/fig7_identifiability.png`)

Family A (parameters ε_b, k_b, E_h, r_h, ε_f, r_f; observables ln K at the fit
temperatures, ΔH(298), B₂_frozen, ln k_TST): normalised singular values
1, 0.41, 0.027, 0.0099, 0.0048, 0.0007, condition number 1.5×10³. The two
weakest directions are (i) outer-well depth against position and
(ii) E_h almost alone. The barrier height is invisible to the thermodynamics,
which is what a kinetic control should be. Condition numbers: A/B/F 1.5×10³,
D 3.9×10³, E 3.1×10⁴, C 1.3×10⁵ (C's Gaussian dispersion has three
correlated parameters).

### 2.5 Equal-equilibrium barrier families (`figures/fig5_barrier_scan.png`)

Barrier heights 1–30 (E_h) or 1–40 kJ/mol (H) at 298 K, with ε_b re-tuned at
each height so that K stays exact:

| Family | uncompensated Δln K over the scan | \|d ln K / d ln τ\| | τ span (decades) | ΔB₂_frozen per decade τ (cm³/mol) | ε_b compensation |
|---|---|---|---|---|---|
| A, B, F | +0.010 … −0.048 | 0.005 | 5.1 | 11.6 | ≤ 0.12 kJ/mol |
| E | same | 0.005 | 5.1 | 13.9 | ≤ 0.12 |
| C | +0.001 … −0.006 | 0.0005 | 6.6 | 1.3 | ≤ 0.02 |
| D | ~0 … −0.005 | 0.0004 | 6.5 | 8.4 | ≤ 0.02 |

Every family can tune kinetics at fixed equilibrium (RQ3). The piecewise
bridge, however, reshapes the whole region from r_b to r_f when E_h changes,
including the monomer contact flank. B₂_frozen then moves from −149 to
−90 cm³/mol over the scan, so the static structure changes along with the
kinetics. That is the confounder the research plan warns about in Step 6 /
Gate 3. The localised Gaussian barrier of C changes B₂_frozen by only a few
cm³/mol.

### 2.6 Selection (`results/model_selection.md`)

All six families pass the hard criteria (clean landscape, C¹ and in fact C²
at every join, stiffness below the MD limit, ≥ 2 decades of kinetic control).
All six are Pareto-optimal: each wins on at least one criterion. The data
therefore do not single out one "best" potential, and the honest outcome is a
shortlist:

* **A (piecewise Morse | quintic barrier | Mie 12-6), primary.** Fewest
  parameters (5), best conditioned, closest to the published PEL-II
  construction. B, E and F are the same model with a different inner wall and
  add nothing.
* **C (additive Morse + Gaussian barrier + Gaussian dispersion),
  alternative.** Ten times better barrier/equilibrium orthogonality, nine
  times smaller non-bonded side effect per decade of kinetic change, and it
  can hold shallow wells at the physical stiffness. Cost: one extra parameter
  and poor conditioning of its dispersion parameters.

If Gate 3 of the research plan ("static structure unchanged between kinetic
states") turns out to be decisive in MD, C should become the primary model.
Both are exported.

## 3. Generic 3 × 3 design (Stage 1G)

`results/generic_design.csv`, `figures/fig8_generic_design.png`,
`export_lammps/generic_{A,C}/`. T = 300 K, ρ₀ = 0.002 Å⁻³ (ρ₀σ³ ≈ 0.1),
Morse range parameter of the anchored potential kept.

| Family A | ε_b (kJ/mol) | E_h (kJ/mol) | f_d (dilute) | ΔH (kJ/mol) | τ_d TST | τ_r TST |
|---|---|---|---|---|---|---|
| E1K1 / K2 / K3 | 5.41 / 6.05 / 6.32 | 2.0 / 7.7 / 13.5 | 0.20 | −6.1 … −7.5 | 3.1 / 31 / 312 ps | 2.1 / 21 / 208 ps |
| E2K1 / K2 / K3 | 11.19 / 11.41 / 11.54 | 2.0 / 7.7 / 13.5 | 0.50 | −12.1 … −12.7 | 20 / 200 / 2000 ps | 6.7 / 67 / 666 ps |
| E3K1 / K2 / K3 | 17.61 / 17.70 / 17.77 | 2.0 / 7.7 / 13.5 | 0.80 | −18.7 … −19.0 | 0.2 / 2 / 20 ns | 22 / 222 / 2220 ps |

Family C gives the same pattern (ε_b 4.7–17.4 kJ/mol, Gaussian heights
2 / 7.8 / 13.6 kJ/mol), with B₂_frozen nearly constant along each kinetic row
(−131 → −124.5 cm³/mol, against −141 → −104 for A).

Things to keep in mind for the MD calibration:

* ΔH changes with the equilibrium regime (−6 → −19 kJ/mol), which is the
  K_eq/ΔH confounding of the research plan. Along a kinetic row it is
  constant to within ~1 kJ/mol.
* The dilute mass-action law is only approximate at ρ₀σ³ ≈ 0.1. MD will shift
  ε_b, but the ordering and the constant-K construction carry over.
* E3K3 (τ_d ≈ 20 ns) is at the upper edge of what plain MD will sample.
* With k_b = 580 fixed, the piecewise family cannot produce E1/E2 wells (no
  monotonic bridge below ε_b ≈ 12 kJ/mol). The additive family can.

## 4. Coordination penalty (Stage 2)

`results/stage2_*.json`, `results/stage2_summary.csv`, `figures/fig9_*`, `figures/fig10_*`.

Strength = 1.15 × the smallest value that puts every two-bond geometry above
the dimer–monomer van der Waals complex (glue: 1.2 ε_b). Trimer fractions are
dilute-limit values from the three-particle cluster integral K₃: real track
at 1 bar and 298 K, generic track at ρ₀ = 0.002 Å⁻³ and 300 K.

| Penalty | exact pair invariance | Δln K (pair) | trimers, real (pair only: 0.98) | trimers, generic A (pair only: 0.48) | longest trap, real / generic | LAMMPS |
|---|---|---|---|---|---|---|
| Hermite counter, cubic hinge | yes | 0 | 1.0×10⁻⁴ | 1.25% | 0.4 / 3 ps | eam/fs |
| quintic counter, cubic hinge | yes | 0 | 6.8×10⁻⁵ | 1.09% | 0.7 / 3 ps | eam/fs |
| Hermite counter, tanh² hinge (explorer) | no | −2.0×10⁻³ | 9.5×10⁻⁵ | 1.22% | 0.4 / 3 ps | eam/fs |
| Gaussian counter, tanh² hinge | no | −1.4×10⁻³ | 6.8×10⁻⁵ | 1.17% | 0.7 / 3 ps | eam/fs |
| damped-tanh counter, cubic hinge | yes | 0 | 9.8×10⁻⁶ | 0.46% | **12 ns** / 68 ps | eam/fs |
| glue (valence-one embedding) | yes | 0 | 8.0×10⁻⁵ | 0.72% | 1 / 2 ps | eam/fs |
| KTS angular (Skorpa) | yes | 0 | 2.8×10⁻⁵ | 0.61% | 1 / 3 ps | custom |
| Stillinger–Weber, cos θ₀ = 1 | yes | 0 | 6.4×10⁻⁶ | 0.10% | **16 / 100 ps** | sw + table |

Findings:

* **Sizing rule.** The explorer's rule (every two-bond geometry costs energy
  relative to dimer + *free* monomer) fails for low barriers. For a generic C
  state with a 0.65 kJ/mol barrier it demanded λ ≈ 2×10⁶ kJ/mol, because a
  particle at the barrier top of one partner sits in the dispersion well of
  the other. Referencing the van der Waals complex fixes this.
* **Pair invariance.** The tanh² hinge used by the explorer shifts ln K by
  ~10⁻³: negligible, but not zero. The cubic hinge, glue and the angular forms
  are exactly zero on isolated pairs, so no Stage-1 number changes.
* **Traps.** A counter that is flat across the bond (damped tanh) holds a
  trimer behind a 29 kJ/mol escape barrier (lifetime ~10 ns at 298 K).
  Thermodynamically suppressed but kinetically trapped trimers would corrupt
  the transport runs, so this form is rejected. Stillinger–Weber traps
  trimers for 16–100 ps.
* **Strength scan.** Contrary to the expectation in the explorer notes, for
  family A a stronger penalty makes the residual traps *shallower*. Generic
  E2K2: 1.15 λ_min gives 1.2% trimers and an 8.5 kJ/mol trap barrier; 2 λ_min
  gives 0.39% and 2.5 kJ/mol. Real: 1×10⁻⁴ → 3×10⁻⁵ and 3.4 → 1.1 kJ/mol.
  The exports therefore use **2 × λ_min**.
* **Family C is harder to protect.** Its fitted Gaussian dispersion well is
  deeper (van der Waals complex −6.4 against −4.0 kJ/mol for A). With the
  coordination penalties the generic C state keeps 8–9% trimers at
  1.15 λ_min and still 2% at 4 λ_min. Glue does best there (1.1–1.8%). If C
  becomes the primary model, use glue or re-fit its dispersion depth.
* ΔB₃ is ≈ 2 K₃(pair only) for every penalty (1.6×10¹⁴ (cm³/mol)² real,
  8.8×10⁵ generic). It measures how much trimer weight was removed and does
  not discriminate between penalties.

**Selected:** Hermite counter (switch from r_b − 0.1 Å to r‡ + 0.2 Å) with the
exact cubic hinge h(x) = x³/(x + 0.05) at 2 λ_min. The quintic counter is
equivalent within the noise (slightly fewer trimers, one more parameter).
Glue is the fallback for deep dispersion wells.

## 5. LAMMPS export

`export_lammps/` contains `real_{A,C}_anchored/` and `generic_{A,C}/<state>/`.
Each folder has `params.json` (all parameters, units, LAMMPS commands, design
values, verification). The `pair.table` (units real) and `manybody.eam.fs`
(units metal) files are committed for the real models and the E2K2 states;
the rest are regenerated by `python scripts/run_export.py`. Each `eam/fs`
file is evaluated by LAMMPS (2025-07-22 Python module) on dimer, linear and
triangular trimer, and dimer + monomer configurations. Agreement with Python
is ≤ 4.2×10⁻⁸ kJ/mol. The pair tables agree to ~3×10⁻⁶ kJ/mol.

## 6. What remains for MD (next steps)

1. NVE energy conservation of the exported `eam/fs` potentials (time step 1 fs
   for the real track; the generic wells, k_b = 110–365, allow more).
2. Equilibrium MD of the 3 × 3 generic states: measure f_d and re-tune ε_b
   (Step 5). Use the hysteresis/commitment state definition.
3. Measure τ_r and the event rates (Step 6) and compare them with the TST
   ordering. TST is an upper bound; recrossing will lengthen the lifetimes.
4. Measure trimer content at the production density. The K₃ numbers here are
   dilute-limit values.
5. Real-fluid track: the anchored potentials reproduce K_p(298 K) only. If a
   real NO₂ comparison is pursued, the entropy deficit (Section 2.2) needs an
   orientation-dependent model (research-plan Extension J).
6. Enter a verified experimental apparent-B₂ dataset in
   `data/reference_virial.csv` (Dymond et al. compilation) and add it as a
   residual. The model side (`B2(...).total`) is already in place.
