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
   a physical N–N bond loses only ~82, and the missing ~93 J mol⁻¹ K⁻¹ is
   mostly rotational entropy that a spherical bead does not have. This holds
   for **all nine isotropic families (A–I)**, including the added
   Morse + Hermite + WF (G), Morse + Gaussian + KTS (H) and
   Morse + Gaussian + LJ (I) combinations. It is a property of the isotropic
   single-site model, not of a functional form.
2. **A smooth Kern–Frenkel (patchy) bead removes the problem.** One patch per
   bead, U(r, Ω_i, Ω_j) = S U_on(r) + (1 − S) U_off(r), adds one parameter, the
   patch width w. At the physical k_b = 580 kJ mol⁻¹ Å⁻², KF-A/C/D fit ln K_p
   on the fit temperatures to rms 0.004 and on the held-out 425–500 K to 0.031,
   against 1.14 and 2.83 for the best isotropic fit. ΔH(298 K) is off by
   −0.4 kJ/mol instead of +32. Anchored on ln K and ΔH at 298 K (w = 24°,
   ε_b = 61.2 kJ/mol), they give ΔS = −175.9 J mol⁻¹ K⁻¹ and stay within
   0.10 in ln K and 1.4 kJ/mol in ΔH up to 500 K. The patch librates with a
   29 fs period (MD limit 20 fs at 1 fs). The cost: rigid-rotor beads and a
   custom LAMMPS pair style. KF-B (Gaussian barrier) works as well on ln K but
   is about 3 times worse on ΔH(T).
3. **Anchoring ln K at one temperature or at each temperature separately makes
   a large difference for the isotropic models and none for Kern–Frenkel.**
   To hit ln K_p(T) exactly at each T, the isotropic ε_b has to fall from
   28.3 kJ/mol at 298 K to 9.4 kJ/mol at 500 K (family A; 7.7 for C/H/I; D
   cannot be anchored above 350 K at all). Each single-T anchor misses the other
   temperatures by rms 1.8–4.2 in ln K. The Kern–Frenkel ε_b(T) stays within
   60.7–61.3 kJ/mol, and each single-T anchor transfers to all nine
   temperatures within rms 0.04–0.11.
4. **Freeing the barrier position (r_h ± 0.15 Å) changes nothing that
   matters.** The fits run to r_h = 2.45 Å, but the rms ln K error of the
   isotropic physical fit only drops from 1.149 to 1.144. The barrier is
   invisible to the equilibrium data, as the identifiability analysis already
   showed.
5. **The equilibrium data cannot tell the bond or non-bonded forms apart.**
   Within the isotropic set, and within the KF set, all families give the
   same K, ΔH and B₂ to within 0.03 in ln K. What separates them is how they
   behave as kinetic controls. The barrier leakage |d ln K/d ln τ| is 0.007 for
   the piecewise families, 0.0004–0.0005 for the additive ones, and 0.066–0.098
   for KF (the misalignment wall is tied to the barrier shape). The B₂_frozen
   shift per decade of lifetime is 12–14 cm³/mol for A/B/E/F/G, 1.3 for C/H/I,
   and 0.7 for KF-B.
6. **Without a penalty the isotropic pair potential is unusable.** In the
   dilute limit, 98% of monomer units end up in trimers at 1 bar and 298 K
   (real track), and 48% at the generic design density. Every exact-hinge
   coordination penalty leaves K exactly unchanged and reduces trimers to
   ~10⁻⁴ (real track) or ~1% (generic, A). **Selected: Hermite counter + exact
   cubic hinge at 2 × λ_min**, which runs as LAMMPS `pair_style eam/fs`. The
   exported files reproduce the Python energies to 7×10⁻⁸ kJ/mol. **The
   Kern–Frenkel bead needs no penalty.** A single patch cannot hold two bonds.
   The lowest two-bond configuration, minimised over all positions and
   orientations, is 16.5 kJ/mol above dimer + free monomer (it is the dimer
   with the third particle parked on the barrier top), and trimers are
   ≤ 3×10⁻⁷ at 1 bar (rigorous upper bound).
7. A generic 3 × 3 (f_d × time scale) design for the core project is ready as
   LAMMPS input. ε_b = 5.8 / 11.6 / 17.8 kJ/mol gives f_d = 0.2 / 0.5 / 0.8 at
   300 K and ρ₀ = 0.002 Å⁻³. Barriers of 2 / 7.7 / 13.5 kJ/mol give TST
   lifetimes in the ratio 1 : 10 : 100 (3.5 ps to 22 ns) at constant K. The
   generic model does not need NO₂ thermodynamics, so it stays isotropic.

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
σ_ΔH = 0.2 kJ/mol). ΔS° ≈ −176 J mol⁻¹ K⁻¹. ΔH rises by 1.85 kJ/mol between
298 and 500 K, i.e. ΔC_p ≈ +9 J mol⁻¹ K⁻¹.

## 2. Two-body calibration (Stage 1)

`results/stage1_*.json`, `results/stage1_summary.csv`, `figures/fig1`–`fig7`, `fig11`–`fig13`.

### 2.1 Non-bonded branch (B₂_frozen vs. the Polat LJ monomer, 250–500 K)

| Family | fitted | χ² (6 T) |
|---|---|---|
| A, B, F | ε_f = 1.98 kJ/mol (r_f = 4.18 Å fixed) | 8.1 |
| C | ε_f = 3.20 kJ/mol, s_f = 0.62 Å | 0.65 |
| D | C₆ = 6.5×10³ kJ mol⁻¹ Å⁶, β_TT = 2.2 Å⁻¹ | 0.73 |
| E, G | ε_f = 2.99 kJ/mol, x_f = 1.58 | 0.58 |
| H (KTS) | ε_f = 3.32 kJ/mol, γ₂ = 31.4 | 0.66 |
| I (LJ, additive) | ε_f = 1.82 kJ/mol | 12.7 |
| KF-A, KF-D | ε_f = 1.98 kJ/mol | 7.4 |
| KF-B | ε_f = 1.84 kJ/mol | 11.8 |
| KF-C | ε_f = 2.92 kJ/mol, x_f = 1.62 | 0.52 |

The 12-6 outer branch with only ε_f free fits worst (rms 1.2–1.5σ). Forms with
a second shape parameter (Gaussian width, WF x, KTS γ₂) fit to 0.3σ. When r_f
was also free, the fit ran to whichever bound it started near (3.4 Å or
5.0 Å) with similar χ². The sensitivity analysis confirms it: the weakest
singular direction of family A (s/s₁ = 6×10⁻⁴) is almost purely ε_f against
r_f. The monomer contact region of the model is softer than an LJ core (the
barrier replaces the repulsive wall between r_h and ~3.7 Å). Matching an LJ B₂
is a coarse-graining choice, not experimental evidence.

### 2.2 Bond well, four modes (`figures/fig2_vanthoff_fits.png`)

| Family | `physical` ε_b (r_h) | rms ln K fit / held-out | ΔH err (kJ/mol) | `anchored` ε_b | anchored ΔH err | anchored ΔS (J/mol/K) | `bounded` k_b | bounded rms ln K / ΔH err |
|---|---|---|---|---|---|---|---|---|
| A | 24.2 (2.45) | 1.14 / 2.83 | +31.9 | 28.3 | +27.8 | −82.8 | 2 299 | 3.71 / +19.8 |
| B | 24.3 (2.45) | 1.14 / 2.83 | +31.8 | 28.3 | +27.7 | −82.9 | 2 302 | 3.73 / +19.7 |
| C | 23.9 (2.60) | 1.17 / 2.90 | +32.4 | 28.0 | +28.2 | −81.3 | 22 703 (MD cap) | 3.28 / +17.9 |
| D | 3.3 (−24.7)* | 1.16 / 2.92 | +32.1 | 7.3 (−28.7)* | +27.7 | −82.9 | 22 703 (MD cap) | 3.48 / +17.5 |
| E | 24.2 (2.45) | 1.15 / 2.84 | +31.9 | 28.3 | +27.8 | −82.7 | 2 345 | 3.69 / +19.9 |
| F | 24.2 (2.45) | 1.15 / 2.84 | +31.9 | 28.3 | +27.8 | −82.6 | 2 296 | 3.71 / +19.9 |
| G | 24.2 (2.45) | 1.14 / 2.83 | +31.9 | 28.3 | +27.8 | −82.8 | 2 354 | 3.71 / +19.8 |
| H | 23.9 (2.60) | 1.17 / 2.90 | +32.4 | 28.0 | +28.2 | −81.3 | 22 703 (MD cap) | 3.28 / +17.9 |
| I | 23.9 (2.60) | 1.17 / 2.90 | +32.4 | 28.0 | +28.2 | −81.3 | 22 703 (MD cap) | 3.28 / +17.9 |
| **KF-A** | 61.6 (2.45), w = 23.2° | **0.004 / 0.031** | **−0.38** | 61.2, w = 24.0° | **0** | **−175.9** | 150 (lower bound) | 0.004 / −0.37 |
| KF-B | 62.2 (2.75), w = 25.9° | 0.007 / 0.089 | −0.56 | 61.6, w = 27.2° | 0 | −175.9 | 22 703 (MD cap) | 0.004 / −0.36 |
| KF-C | 61.6 (2.46), w = 23.2° | 0.004 / 0.032 | −0.38 | 61.3, w = 24.0° | 0 | −175.9 | 150 (lower bound) | 0.004 / −0.37 |
| KF-D | 61.6 (2.45), w = 23.3° | 0.004 / 0.031 | −0.38 | 61.2, w = 24.1° | 0 | −175.9 | 150 (lower bound) | 0.004 / −0.37 |

\*D: the Tang–Toennies attraction adds most of the well depth, so the nominal
ε_b is small. The actual well depth U_bond is given in brackets.
Anchored: isotropic families ln K exact at 298.15 K; Kern–Frenkel families
ln K and ΔH exact at 298.15 K (through w).

* The isotropic families are interchangeable: A, B, E, F, G agree to 0.01 in
  ln K and 0.1 kJ/mol in ΔH; C, H, I to better than that. The non-bonded
  branch only changes B₂.
* The `bounded` fit of the piecewise families stops at k_b ≈ 2.3×10³, below
  the MD limit (2.3×10⁴). Beyond that no barrier curvature keeps the quintic
  bridge monotonic for the ~36 kJ/mol well. The construction itself caps the
  stiffness. With k_b free, the KF families go to the *lower* bound
  (150 kJ mol⁻¹ Å⁻²): a softer bond adds a little heat capacity, but the gain
  is negligible (rms 0.004 either way). k_b stays at the physical 580.
* Anchored isotropic model at 298 K: ΔS = −82 J mol⁻¹ K⁻¹ (exp. −176), TST
  dimer lifetime 1.4×10⁻⁸ (C, H, I) to 4.4×10⁻⁸ s (A); radial free-energy
  barrier for dissociation 25–32 kJ/mol. The anchored KF-A dimer is far more
  stable: ε_b = 61 kJ/mol, radial free-energy barrier 45 kJ/mol, TST lifetime
  9×10⁻⁶ s. The real-fluid MD at 298 K will sample no dissociation at all.

### 2.3 Attainability (`figures/fig3_attainability.png`, `figures/fig13_kern_frenkel.png`)

Isotropic: family C is shown because its Morse well stays valid at any
stiffness (the piecewise bridge stops being monotonic above k_b ≈ 2×10³).
With ε_b fitted to ln K_p(T), the ΔH error falls by only ~1.7 kJ/mol per decade
of k_b. With ε_b fitted to ΔH, the ln K error falls by ~0.6 per decade:

| k_b (kJ mol⁻¹ Å⁻²) | wavenumber (cm⁻¹) | period (fs) | ΔH err if K fitted | ln K err if ΔH fitted |
|---|---|---|---|---|
| 100 | 111 | 301 | +34.9 | +12.2 |
| 580 (physical) | 266 | 125 | ≈ +32.4 | ≈ +11.3 |
| 1 000 | 350 | 95 | +31.6 | +11.0 |
| 2.3×10⁴ (MD limit) | 1 670 | 20 | ≈ +26.9 | ≈ +9.4 |
| 1×10⁸ | 1.1×10⁵ | 0.30 | +14.9 | +5.2 |
| 3.5×10¹² (exact) | 2×10⁷ | 0.0016 | 0 | 0 |

Exact simultaneous solution at 298 K: k_b = 2.7×10¹² (A, B, G), 3.5×10¹² (C,
H, I), 1.3×10¹³ (D). The Wang–Frenkel and KTS bond shapes (E, F) cannot
represent wells that stiff at all.

Kern–Frenkel, k_b = 580 fixed, patch width scanned (ε_b fitted to ln K_p(T)
at each width): the ΔH error crosses zero at w ≈ 23° (KF-A/C/D) and 25° (KF-B)
and changes by ~16 kJ/mol per decade of w. The rms ln K error has a sharp
minimum (0.01) there. The patch width is the second handle that the isotropic
model lacks, and it is reached at a physical bond. In the deep-well limit
only κ = W(r_b)/w² is identifiable (W(r_b) and w are degenerate). Anchored
KF-A: κ = 981 kJ mol⁻¹ rad⁻², libration period 29 fs about the NO₂ a axis
(I_A = 2.1 amu Å²) and 129 fs about the c axis. Both are above the 20 fs MD
limit. The per-patch bonded fraction equals a hard Kern–Frenkel cone of
half-angle 4.1°, which shows why the smooth form is needed: a hard 4° cone
would be almost impossible to sample.

### 2.4 Anchoring at every temperature (`figures/fig12_per_temperature_anchoring.png`, `results/model_selection.md` §1c)

ε_b (kJ/mol) solving ln K_p(T) exactly at each temperature, k_b = 580 (KF:
w of the 298 K anchor kept):

| Family | 298 K | 350 K | 400 K | 450 K | 500 K | rms ΔH err at the anchor T | rms ln K over 298–500 K of a single-T anchor |
|---|---|---|---|---|---|---|---|
| A (B, E, F, G within 0.15) | 28.3 | 23.5 | 18.8 | 14.2 | 9.4 | 36.8 | 1.8 (375 K) – 4.0 (500 K) |
| C, H, I | 28.0 | 23.1 | 18.3 | 13.4 | 7.7 | 38.0 | 1.9 – 4.2 |
| D | 7.3 | 2.4 | – | – | – | 30.3 | 2.0 – 2.9 |
| KF-A, KF-C, KF-D | 61.25 | 61.21 | 61.10 | 60.94 | 60.74 | 1.1 | 0.04 – 0.11 |
| KF-B | 61.61 | 61.55 | 61.38 | 61.06 | 60.38 | 3.3 | 0.09 – 0.28 |

* For the isotropic models it matters a lot at which temperature K is
  anchored. ε_b(T) would have to fall by 0.093 kJ mol⁻¹ K⁻¹. Such a
  temperature-dependent "potential" would just mimic the missing entropy
  (Δε_b ≈ −T ΔS_missing). The best single anchor is at 375 K (centre of the
  range), with rms 1.8 in ln K, and the ΔH error grows from +28 to +45 kJ/mol
  across the range. Family D cannot be anchored above 350 K: its
  Tang–Toennies attraction alone binds too strongly even with ε_b → 0.
* For the KF models the anchoring temperature hardly matters. ε_b drifts by
  0.5 kJ/mol over 200 K and ΔH by 1.9 kJ/mol. Re-anchoring both ln K and ΔH at
  each T gives w = 22–24° and ε_b = 61.2–62.6 kJ/mol (KF-A). The residual drift
  is a too-large heat capacity: the model has ΔC_p = +15 J mol⁻¹ K⁻¹ (one radial
  plus four librational quadratic coordinates), experiment ~+9.

### 2.5 Identifiability (`figures/fig7_identifiability.png`)

Family A (parameters ε_b, k_b, E_h, r_h, ε_f, r_f; observables ln K at the fit
temperatures, ΔH(298), B₂_frozen, ln k_TST): normalised singular values 1,
0.40, 0.040, 0.0072, 0.0041, 0.0006, condition number 1.7×10³. The two
weakest directions are (i) outer-well depth against position and (ii) E_h
and r_h almost alone. The barrier is invisible to the thermodynamics, which is
what a kinetic control should be, and why freeing r_h does not improve the
fits. Condition numbers: A/B/F 1.7×10³, I 2.7×10³, D 3.9×10³, E/G 1.9×10⁴,
KF-B 1.1×10⁴, KF-A/D 2.5×10⁴, C/H 1.2–1.3×10⁵ (three correlated dispersion
parameters), KF-C 1.3×10⁷ (the WF shape parameter plus the patch). With the
patch width added, ε_b and w are both well determined (ln K fixes their
combination, ΔH fixes ε_b); the weakest KF-A direction is k_b almost alone,
because the orientational volume now dominates the bound-state entropy.

### 2.6 Equal-equilibrium barrier families (`figures/fig5_barrier_scan.png`)

Barrier heights 1–30 (E_h) or 1–40 kJ/mol (H) at 298 K, with ε_b re-tuned at
each height so that K stays exact:

| Family | \|d ln K / d ln τ\| | τ span (decades) | ΔB₂_frozen per decade τ (cm³/mol) |
|---|---|---|---|
| A, B, F | 0.007 | 5.1 | 11.9 |
| E, G | 0.007 | 5.1 | 13.9 |
| C, H, I | 0.0005 | 6.6 | 1.3 |
| D | 0.0004 | 6.5 | 8.4 |
| KF-A, KF-D | 0.066 | 6.2 | 9.1 |
| KF-C | 0.066 | 6.2 | 10.6 |
| KF-B | 0.098 | 9.0 | 0.7 |

Every family can tune kinetics at fixed equilibrium (RQ3). The piecewise
bridge reshapes the whole region from r_b to r_f when E_h changes, including
the monomer contact flank, so B₂_frozen moves from −155 to −94 cm³/mol over the
scan (family A). The static structure then changes along with the kinetics. That is the
confounder the research plan warns about in Step 6 / Gate 3. The localised
Gaussian barrier of C/H/I changes B₂_frozen by only a few cm³/mol. In the KF
families the barrier leaks ten times more into K (still < 0.1 in ln K per
decade of lifetime, and fully compensated by ε_b), because the misalignment
wall is sized to the steepest slope of the barrier flank and changes with it.

### 2.7 Equilibrium mapping: isotropic vs Kern–Frenkel (`figures/fig11_equilibrium_mapping.png`)

Anchored potentials, 298–500 K:

| | isotropic (A) | KF-A | reference |
|---|---|---|---|
| ln K error at 400 / 500 K | +2.80 / +4.37 | +0.04 / +0.10 | – |
| ΔH at 298 / 500 K (kJ/mol) | −29.5 / −30.2 | −57.27 / −54.06 | −57.27 / −55.43 |
| ΔS at 298 K (J mol⁻¹ K⁻¹) | −82.8 | −175.9 | −175.9 |
| ΔC_p at 298 K (J mol⁻¹ K⁻¹) | −3.8 | +14.6 | ≈ +9 |
| rms ΔH over 9 T (kJ/mol) | 26.6 | 0.86 | – |

The isotropic families form one bundle, and so do KF-A/C/D. The patch
supplies the ~93 J mol⁻¹ K⁻¹ of orientational entropy: −2R ln of the per-patch
bonded fraction (≈ 1.3×10⁻³ at 298 K). It does so inside the force field, so
an MD simulation will obey it.

### 2.8 Selection (`results/model_selection.md`)

All thirteen families pass the hard criteria (clean landscape, C² at every
join, bond stiffness and – for KF – libration period within the MD limit,
≥ 2 decades of kinetic control). All are Pareto-optimal on the eight scored
criteria: KF wins every thermodynamic criterion, the additive families win
barrier orthogonality, and A/B/F win simplicity and conditioning. The
recommendation depends on the track:

* **Real NO₂ fluid: KF-A (smooth Kern–Frenkel | Morse | quintic Hermite
  barrier | LJ 12-6), primary.** It is the only kind of model that reproduces
  K_p(T) and ΔH together, it needs no valence penalty, and it has one
  parameter more than A. KF-C and KF-D are numerically the same model with a
  different non-bonded or inner-wall shape. KF-B is the alternative if a
  localised barrier matters (smallest B₂ side effect, 0.7 cm³/mol per decade),
  at the price of a 3× larger ΔH(T) drift and a barrier that the deep well
  pushes out to 2.93 Å.
* **Generic model (core project) and an isotropic real-fluid fallback: A
  (primary) and C (alternative)**, as before. Isotropic MD needs no rotational
  integration and runs with native LAMMPS styles. G, H and I add nothing
  beyond A and C: G = A with the E outer branch, H and I = C with a different
  outer branch.

## 3. Generic 3 × 3 design (Stage 1G)

`results/generic_design.csv`, `figures/fig8_generic_design.png`,
`export_lammps/generic_{A,C}/`. T = 300 K, ρ₀ = 0.002 Å⁻³ (ρ₀σ³ ≈ 0.1),
Morse range parameter of the anchored potential (now with r_h = 2.45 Å) kept.

| Family A | ε_b (kJ/mol) | E_h (kJ/mol) | f_d (dilute) | ΔH (kJ/mol) | τ_d TST | τ_r TST |
|---|---|---|---|---|---|---|
| E1K1 / K2 / K3 | 5.80 / 6.36 / 6.60 | 2.0 / 7.7 / 13.5 | 0.20 | −6.7 … −7.9 | 3.5 / 35 / 352 ps | 2.3 / 23 / 234 ps |
| E2K1 / K2 / K3 | 11.37 / 11.60 / 11.74 | 2.0 / 7.7 / 13.5 | 0.50 | −12.4 … −13.0 | 22 / 225 / 2250 ps | 7.5 / 75 / 750 ps |
| E3K1 / K2 / K3 | 17.72 / 17.83 / 17.91 | 2.0 / 7.7 / 13.5 | 0.80 | −18.9 … −19.2 | 0.2 / 2.2 / 22 ns | 25 / 250 / 2500 ps |

Family C gives the same pattern (ε_b 4.7–17.4 kJ/mol, Gaussian heights
2 / 7.8 / 13.6 kJ/mol), with B₂_frozen nearly constant along each kinetic row
(−131 → −124.5 cm³/mol, against −147 → −109 for A).

Things to keep in mind for the MD calibration:

* ΔH changes with the equilibrium regime (−7 → −19 kJ/mol), which is the
  K_eq/ΔH confounding of the research plan. Along a kinetic row it is
  constant to within ~1 kJ/mol.
* The dilute mass-action law is only approximate at ρ₀σ³ ≈ 0.1. MD will shift
  ε_b, but the ordering and the constant-K construction carry over.
* E3K3 (τ_d ≈ 22 ns) is at the upper edge of what plain MD will sample.
* The generic design stays isotropic. It needs no NO₂ entropy, and a
  Kern–Frenkel version would add rotational dynamics to every transport
  quantity.

## 4. Coordination penalty (Stage 2)

`results/stage2_*.json`, `results/stage2_summary.csv`, `figures/fig9_*`, `figures/fig10_*`, `figures/fig14_kf_valence.png`.

### 4.1 Isotropic pair potentials

Strength = 1.15 × the smallest value that puts every two-bond geometry above
the dimer–monomer van der Waals complex (glue: 1.2 ε_b). Trimer fractions are
dilute-limit values from the three-particle cluster integral K₃: real track
(family A anchored) at 1 bar and 298 K, generic track (A, E2K2) at
ρ₀ = 0.002 Å⁻³ and 300 K.

| Penalty | exact pair invariance | Δln K (pair) | trimers, real (pair only: 0.98) | trimers, generic A (pair only: 0.48) | longest trap, real / generic | LAMMPS |
|---|---|---|---|---|---|---|
| Hermite counter, cubic hinge | yes | 0 | 1.1×10⁻⁴ | 1.08% | 0.5 / 3 ps | eam/fs |
| quintic counter, cubic hinge | yes | 0 | 5.6×10⁻⁵ | 0.88% | 0.9 / 3 ps | eam/fs |
| Hermite counter, tanh² hinge (explorer) | no | −2.6×10⁻³ | 1.0×10⁻⁴ | 1.05% | 0.5 / 3 ps | eam/fs |
| Gaussian counter, tanh² hinge | no | −1.3×10⁻³ | 3.4×10⁻⁵ | 0.80% | 2 / 5 ps | eam/fs |
| damped-tanh counter, cubic hinge | yes | 0 | 1.0×10⁻⁵ | 0.43% | **7 ns** / 54 ps | eam/fs |
| glue (valence-one embedding) | yes | 0 | 5.5×10⁻⁵ | 0.53% | 4 / 2 ps | eam/fs |
| KTS angular (Skorpa) | yes | 0 | 2.5×10⁻⁵ | 0.49% | 1 / 3 ps | custom |
| Stillinger–Weber, cos θ₀ = 1 | yes | 0 | 4.4×10⁻⁶ | 0.07% | **10 / 95 ps** | sw + table |

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
  trimer behind a 28 kJ/mol escape barrier (lifetime ~7 ns at 298 K).
  Thermodynamically suppressed but kinetically trapped trimers would corrupt
  the transport runs, so this form is rejected. Stillinger–Weber traps
  trimers for 10–95 ps.
* **Strength scan.** For family A a stronger penalty makes the residual
  traps *shallower*. Generic E2K2: 1.15 λ_min gives 1.1% trimers and an
  8.6 kJ/mol trap barrier; 2 λ_min gives 0.29% and 2.9 kJ/mol. Real:
  1.1×10⁻⁴ → 2.2×10⁻⁵ and 3.8 → 2.6 kJ/mol. The exports therefore use
  **2 × λ_min**.
* **Family C is harder to protect.** Its fitted Gaussian dispersion well is
  deeper (van der Waals complex −6.4 against −4.0 kJ/mol for A). With the
  coordination penalties the generic C state keeps 8–9% trimers at
  1.15 λ_min and still 2% at 4 λ_min. Glue does best there (1.1–1.8%). If C
  becomes the primary model, use glue or re-fit its dispersion depth.
* ΔB₃ is ≈ 2 K₃(pair only) for every penalty (1.6×10¹⁴ (cm³/mol)² real,
  9×10⁵ generic). It measures how much trimer weight was removed and does
  not discriminate between penalties.

**Selected:** Hermite counter (switch from r_b − 0.1 Å to r‡ + 0.2 Å) with the
exact cubic hinge h(x) = x³/(x + 0.05) at 2 λ_min. The quintic counter is
equivalent within the noise (slightly fewer trimers, one more parameter, and
its eam/fs tabulation agrees with Python only to ~10⁻³ kJ/mol). Glue is the
fallback for deep dispersion wells.

### 4.2 Kern–Frenkel: valence from the patch (`results/stage2_kf.json`, `figures/fig14_kf_valence.png`)

Anchored KF potentials at 298.15 K and 1 bar, no many-body term. dE is
relative to the aligned dimer + free monomer. The trimer fraction is a
rigorous upper bound: the same bound overestimates an isolated bond by
e^1.4 ≈ 4, so K₃ by roughly e^2.8.

| Family | lowest two-bond configuration (all positions and orientations) | vs. vdW complex | trimer fraction ≤ | same radial curve, no orientation |
|---|---|---|---|---|
| KF-A | +16.5 kJ/mol | +20.5 | 3.4×10⁻⁷ | 1.00 |
| KF-C | +15.3 | +21.1 | 4.8×10⁻⁷ | 1.00 |
| KF-D | +16.5 | +20.5 | 3.4×10⁻⁷ | 1.00 |
| KF-B | +2.1 | +5.8 | 1.7×10⁻⁴ | 1.00 |

* With one patch a second bond is impossible. The lowest "two-bond"
  configuration is the aligned dimer with the third particle on the barrier
  top of particle 1 (r₁₃ = r‡), not a second bond. Its energy is simply the
  barrier height plus the misalignment penalty. Without the orientation
  dependence the same radial curve would turn the gas completely into
  trimers.
* KF-A alone keeps trimers below 3.4×10⁻⁷, three orders of magnitude below
  the isotropic model with the selected penalty (1.1×10⁻⁴ at 1.15 λ_min,
  2×10⁻⁵ at 2 λ_min). Adding that penalty on top changes little (6×10⁻⁸). The
  valence penalty is not needed for the Kern–Frenkel model.
* KF-B is weaker because its deep well pushes the Gaussian barrier out to
  2.93 Å and down to 1.3 kJ/mol. A third particle can then sit just inside
  the dividing surface. (Sizing the isotropic penalty on KF-B also runs into
  the low-barrier divergence of Section 4.1, λ ≈ 4×10⁷ kJ/mol, so that row in
  `model_selection.md` is not meaningful.)

## 5. LAMMPS export

`export_lammps/` contains `real_{A,C}_anchored/`, `generic_{A,C}/<state>/` and
`real_{KF-A,KF-B}_anchored/`. Each folder has `params.json` (all parameters,
units, LAMMPS commands, design values, verification). The `pair.table`
(units real) and `manybody.eam.fs` (units metal) files are committed for the
real isotropic models and the E2K2 states; the rest are regenerated by
`python scripts/run_export.py`. Each `eam/fs` file is evaluated by LAMMPS
(2025-07-22 Python module) on dimer, linear and triangular trimer, and dimer
+ monomer configurations. Agreement with Python is ≤ 7×10⁻⁸ kJ/mol for every
exported real and generic state. The Kern–Frenkel folders contain
`kf_radial.table` with the `ALIGNED` (U_on) and `MISALIGNED` (U_off) curves and
the patch parameters. U = S U_on + (1 − S) U_off needs a custom anisotropic
pair style (e.g. a LAMMPS plugin) and rigid-rotor integration of the beads.
No native style evaluates it, so it is not cross-checked in LAMMPS.

## 6. What remains for MD (next steps)

1. NVE energy conservation of the exported `eam/fs` potentials (time step 1 fs
   for the real track; the generic wells, k_b = 97–367, allow more).
2. Equilibrium MD of the 3 × 3 generic states: measure f_d and re-tune ε_b
   (Step 5). Use the hysteresis/commitment state definition.
3. Measure τ_r and the event rates (Step 6) and compare them with the TST
   ordering. TST is an upper bound; recrossing will lengthen the lifetimes.
4. Measure trimer content at the production density. The K₃ numbers here are
   dilute-limit values.
5. Real-fluid track with the Kern–Frenkel model: implement U(r, Ω_i, Ω_j) as a
   LAMMPS plugin (energy, force and torque are in
   `KernFrenkelPotential.energy`, tested against finite differences),
   integrate the beads as rigid rotors (moment of inertia of the NO₂ C2-axis
   libration), check NVE conservation with the 29 fs libration, and verify
   K_p in MD. At 298 K the TST dimer lifetime is ~9 μs. Reactive sampling of
   the real track needs higher temperatures or enhanced sampling.
6. Enter a verified experimental apparent-B₂ dataset in
   `data/reference_virial.csv` (Dymond et al. compilation) and add it as a
   residual. The model side (`B2(...).total`) is already in place and works
   for the KF models too.
