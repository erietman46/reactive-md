# Pre-MD calibration of the reactive force field: strategy

This document explains **what** is calibrated before any reactive MD is run,
**why** it is worth doing, **how** every number in `results/` is obtained, and
**how** the "best" potential is chosen. It belongs to Step 3 ("construct the
reactive potential") of the research plan and covers the calibration of the
two-body potential `U2(r)` first and the coordination penalty second.

The companion document `RESULTS.md` discusses the numbers produced by the
pipeline; `../README.md` explains how to run it.

---

## 0. Why calibrate before MD at all?

For a dilute gas of particles interacting through `U2(r)` the equilibrium
thermodynamics of `2A <-> A2` is a *one-dimensional integral*. A full
calibration sweep (thousands of candidate potentials, every temperature)
therefore costs seconds, while a single reactive MD equilibrium point costs
hours. The pre-MD stage is used to:

1. reject potentials that cannot work (wrong shape, unstable force, bond too
   stiff for any sensible time step, barrier that leaks into equilibrium);
2. find out **which experimental targets an isotropic single-site model can
   reproduce at all** (Section 4.6 - this turns out to be the key result);
3. map the force-field parameters onto `K`, `dH`, `B2` and a first kinetic
   estimate, so that the MD campaign starts from parameter sets that are
   already close to the desired `f_d` and time-scale separation;
4. size and test the valence-limiting penalty on three-particle energy
   surfaces and cluster integrals, before it can create trimers or trapped
   clusters in a large simulation;
5. export smooth, verified LAMMPS input (pair table, `eam/fs` file).

What the pre-MD stage **cannot** do: give true association/dissociation rates
(only TST estimates), predict cluster populations at liquid densities
quantitatively, or predict transport. Those remain MD tasks (Steps 4-8 of the
research plan). See the table in Section 8.

---

## 1. Calibration targets and where they come from

| Quantity | Symbol | Source in this repository | Role |
|---|---|---|---|
| Association equilibrium constant, 2 NO2 -> N2O4, p0 = 1 bar | `ln K_p(T)` | `data/reference_Kp.csv` (NASA-7 polynomials, cross-checked against JANAF) | primary fit target |
| Reaction enthalpy | `dH(T)` | same file | second target / diagnostic |
| Reaction entropy | `dS(T)` | same file | diagnostic (explains the representability problem) |
| Non-bonded monomer `B2(T)` | `B2_frozen` | Polat et al. NO2 LJ (eps/k = 262.17 K, sigma = 3.727 A) | fluid-branch target (coarse-graining assumption) |
| Experimental apparent `B2` | `B2_app` | `data/reference_virial.csv` | **empty** until a verified dataset is entered |
| Bond length / stiffness | `r_b`, `k_b` | `data/coarse_grain_constraints.json` | constraints, not targets |

### 1.1 Equilibrium constant and enthalpy

```
ln K_p(T) = -[G(N2O4) - 2 G(NO2)] / RT          dH(T) = H(N2O4) - 2 H(NO2)
```

`janaf.nist.gov` is not reachable from the build container, so the
temperature-dependent curve is generated from the NASA-7 polynomials of NO2
(Burcat, "L 7/88") and N2O4 (Gurvich et al., TPIS89) in Cantera's
`nasa_gas.yaml`. They agree with the JANAF values quoted in the project notes
to |d ln K| <= 0.042 and |d dH| <= 0.17 kJ/mol (`results/reference_check.csv`);
that spread is used as the data uncertainty (sigma_lnK = 0.05,
sigma_dH = 0.2 kJ/mol). If the raw JANAF tables are placed in
`data/janaf/NO2.txt` and `data/janaf/N2O4.txt`, `reference_data.py` uses them
instead.

Fit temperatures: 298.15, 325, 350, 375, 400 K. Held-out temperatures:
425, 450, 475, 500 K. Because `dH` comes from the same thermochemical
evaluation as `ln K`, it is **not** independent information: it is reported
and used as a diagnostic, and only used as a second residual where this is
stated explicitly (the "bounded" fit).

### 1.2 Second virial coefficient: which one?

Splitting the Mayer integral at the dividing surface `r_div` (the barrier
top) gives an exact identity (`thermodynamics/virial_coefficients.py`):

```
B2 = B2_frozen - N_A K_V
B2_frozen = (2 pi N_A / 3) r_div^3 - 2 pi N_A int_{r_div}^inf [exp(-U/RT) - 1] r^2 dr
```

* `B2` (full Mayer integral) is the **apparent** virial coefficient of the
  reacting gas per NO2 formula unit, i.e. what a p-V-T measurement on
  equilibrium NO2/N2O4 gives when the composition is counted in NO2 units.
* `B2_frozen` is the virial coefficient of monomers that are not allowed to
  bond - the "physical" non-ideality of the classical Woolley /
  Stogryn-Hirschfelder analysis of associating gases.
* `N_A K_V` is the association constant in molar units.

Consequences:

1. When association is strong (298-400 K for NO2), `B2 ~ -N_A K_V`: a
   measured `B2_app` adds almost nothing beyond `K_p`. It becomes informative
   at high T, where `K_V` and `B2_frozen` are comparable.
2. An experimental `B2_app` can be compared directly with the model's full
   `B2` - no correction is needed if it is per formula unit. A `B2` "of NO2
   monomers" from the literature must first be checked for which of the three
   quantities it actually is.
3. In the absence of verified data, the non-bonded branch is calibrated to the
   `B2` of the Polat et al. LJ monomer, matched to the model's `B2_frozen`.
   This is a coarse-graining assumption, not an experimental fit, and is
   labelled as such everywhere.

---

## 2. The candidate potentials

Both constructions of `Pair_potential_explorer_2_1.html` are ported exactly to
Python (`potentials/piecewise.py`, `potentials/additive.py`) and checked
against the explorer's own JavaScript to 1e-12 (`tests/explorer_reference.json`).

| Label | Construction | Bond branch | Barrier | Non-bonded branch | Role |
|---|---|---|---|---|---|
| A | piecewise (Potential 1) | Morse | quintic C2 bridge, top `E_h` at `r_h` | Mie 12-6, switched off 8-10 A | primary |
| B | piecewise | Mie (n = 2m) | quintic bridge | Mie 12-6 | primary (different wall shape) |
| C | additive (Potential 2) | Morse x switch | Gaussian `H` at `r_x` | Gaussian | primary (flexible phenomenological) |
| D | additive | Morse x switch | Gaussian | Tang-Toennies-damped C6 | primary (realistic tail) |
| E | piecewise | Wang-Frenkel | quintic bridge | Wang-Frenkel | supplementary |
| F | piecewise | KTS (Skorpa) | quintic bridge | Mie 12-6 | supplementary |

Common starting point (`potentials/candidates.py`): `r_b = 1.78 A` (N-N in
N2O4; the NO2 centre-of-mass distance, ~2.43 A, is an alternative bead
definition and changes `ln K` by only `2 ln(2.43/1.78) = 0.62`),
`eps_b = 30 kJ/mol`, `k_b = 580 kJ/mol/A^2` (N-N stretch ~266 cm^-1 with
mu = 23 g/mol), a design barrier of +5 kJ/mol near 2.6 A, the Polat LJ well
for the non-bonded branch, and the same 8-10 A switching for every family so
that the cut-off treatment is identical.

**Bond coordinates.** Every bond form is fitted in the same physical
coordinates `(eps_b, k_b)`: the shape parameter of each form (Morse beta, Mie
exponents, Wang-Frenkel `x`, KTS `gamma_2`) is solved from the requested
curvature (`branches.shape_for_stiffness`). Differences between families are
then differences of functional form, not of parameterisation.

**Shape constraints (hard).** One bond minimum < one barrier < one dispersion
minimum, nothing else (`PairPotential.landscape().clean`); for the piecewise
form also a barrier curvature `k_h` that keeps both quintic segments monotonic
(the explorer's "auto k_h"). Every trial potential violating this is rejected
inside the optimiser.

---

## 3. Theory used for every number

All in `thermodynamics/`. Energies kJ/mol, distances A.

**Bound configurational volume and K.** With `beta = 1/RT`,

```
Z_b(T) = 4 pi int_0^{r_div} r^2 exp(-beta U2) dr               [A^3]
K_V    = rho_A2 / rho_A^2 = Z_b / 2                            (1/2: identical beads)
K_p    = K_V p0 / (k_B T)                                      (dimensionless, p0 = 1 bar)
```

`r_div` is the actual barrier maximum of the complete potential (not the
nominal input, which differs for the additive form). The integral is
evaluated in log form on Gauss-Legendre panels refined around the bond
minimum, so it stays exact for thermal widths down to 1e-6 A (tested against
`scipy.integrate.quad` and against the Laplace estimate).

**Enthalpy, entropy, heat capacity.** For a T-independent potential and
dividing surface (exact; checked against the numerical van 't Hoff slope):

```
dH = <U2>_b - RT,     dS = (dH + RT ln K_p)/T,     dCp = var_b(U2)/(R T^2) - R
```

**Dimer fraction at a density.** With `rho0 = rho_A + 2 rho_A2` and
`f_d = 2 rho_A2 / rho0`: `f_d / (1 - f_d)^2 = 2 K_V rho0` (dilute mass action).

**Finite two-particle box.** For two particles in a periodic cube of volume V
the exact relation is `P_b/(1 - P_b) = Z_b / Z_u` with
`Z_u = V - 4/3 pi r_div^3 + int (exp(-beta U) - 1) d^3r`; inserting the mean
counts of a two-particle box into the macroscopic mass-action law instead gives
`K_V` wrong by the factor `V/(2 Z_u (1 - P_b))`: x18 for the anchored family-A
potential at 298 K in a 25 A box, and still x0.5 (a factor of two) for the
generic E2K2 potential, where dimers are rare in such a box. Use
`finite_box_two_particle()` if a two-particle MD/MC check is made; never the
naive substitution.

**TST kinetic estimate (screening only).**

```
k_d = sqrt(k_B T / 2 pi mu) 4 pi r_div^2 exp(-beta U(r_div)) / Z_b,   k_a = k_d K_V,
1/tau_r = 4 k_a rho_A + k_d,       F(r) = U2(r) - 2RT ln r  (radial free energy)
```

This is the high-pressure-limit, no-recrossing rate: an upper bound for the
MD rate in a dense fluid and not valid in a dilute gas, where a conservative
pair potential cannot stabilise a collision complex without a third body.

**Three-particle cluster integrals** (`manybody/third_virial.py`). With
inclusion-exclusion over the vertex particle, the trimer association constant

```
K3 = rho_A3 / rho_A^3 = (1/3!) int_{connected} exp(-beta [U12 + U13 + U23 + E_mb]) d^3r2 d^3r3
```

is evaluated exactly (6-D integral reduced to `(r12, r13, cos theta)`), giving
the dilute-limit trimer content at any density. `Delta B3` is computed too,
but for a penalty that removes the trimer basin it reduces to `~2 K3(pair
only)`; it measures how much trimer weight was removed and does not
discriminate between penalties, so the trimer fraction is the reported
criterion.

---

## 4. Stage 1 - two-body calibration (no penalty, no MD)

Script: `scripts/run_stage1_pair.py` (per family; results in
`results/stage1_<label>.json`, summary in `results/stage1_summary.csv`).

### 4.1 Step 1 - geometry first

Construct `U2`, locate all extrema, require the clean three-extremum
landscape, check C1/C2 continuity at every join (`selection.smoothness`) and
compare analytic forces with finite differences (tests).

### 4.2 Step 2 - non-bonded branch

Fit the outer-branch parameters so that `B2_frozen(T)` of the model matches
`B2` of the Polat LJ monomer at 250-500 K (sigma = max(5 cm^3/mol, 5 %)).
The outer-minimum position is fixed at the LJ minimum `2^(1/6) sigma = 4.18 A`
and only the depth (and width/shape where the form has one) is fitted:
`B2(T)` alone cannot separate well position from depth, and a free
`(r_f, eps_f)` fit wanders to whichever bound it starts nearest (we observed
both `r_f -> 3.4 A` and `r_f -> 5.0 A` with similar chi^2). That is an
identifiability result in its own right.

### 4.3 Step 3 - bond well, in four modes

| Mode | Free | Objective | Purpose |
|---|---|---|---|
| `physical` | `eps_b` (k_b = 580) | `ln K_p` at the fit temperatures | best compromise with a physical bond |
| `anchored` | `eps_b` (k_b = 580) | `ln K_p(298.15 K)` exactly | honours `dG` at the MD temperature; carried to Stage 2 |
| `bounded` | `eps_b, k_b <= k_MD` | `ln K_p` and `dH` | best a timestep-compatible bond can do on both |
| `exact` | `eps_b, k_b` unbounded | `ln K_p` and `dH` at 298.15 K | what an isotropic model would *need* |

Optimisation: bounded `differential_evolution` (seeded, Sobol init) followed
by `least_squares` (trust region reflective) in the same bounds
(`optimization/fit_pair_potentials.py`).

### 4.4 Step 4 - identifiability

`optimization/sensitivity_analysis.py` computes the normalised Jacobian
`S_ij = d(O_i/sigma_i)/d ln theta_j` of the observables (ln K at the fit
temperatures, dH, B2_frozen, TST rate) with respect to all parameters of the
anchored potential, and its singular values. Singular values more than ~100x
below the largest mark parameter combinations that the data cannot determine.

### 4.5 Step 5 - equal-equilibrium barrier families

`optimization/barrier_scan.py`: vary the barrier height (`E_h` or `H`), record
the *uncompensated* change of `ln K` (leakage), then re-tune `eps_b` so that
`ln K(T_ref)` is restored and record the TST lifetime, radial free-energy
barrier and side effects (`dH`, `B2_frozen`). Two scalar metrics:

* **orthogonality** `|d ln K / d ln tau|` - 0 means the barrier controls
  kinetics without touching equilibrium;
* **non-bonded side effect** - change of `B2_frozen` per decade of lifetime
  (the barrier also reshapes the monomer-monomer contact region, which the
  research plan flags as a confounder of transport).

### 4.6 The representability test (most important Stage-1 result)

For a near-harmonic bound basin

```
Z_b ~ 4 pi r_b^2 sqrt(2 pi RT / k_b) exp(eps_b / RT),   <U>_b ~ -eps_b + RT/2.
```

`dH = -57.3 kJ/mol` fixes `eps_b ~ 56 kJ/mol`; `K_p(298 K) = 7.0` then fixes
the bond width `sqrt(RT/k_b)`. `optimization/attainability.py` solves both
equations exactly and scans `k_b`. The answer (see `RESULTS.md`): the width
must be ~1e-6 A, i.e. `k_b ~ 3e12 kJ/mol/A^2`, a bond period of ~0.002 fs.
**No timestep-compatible isotropic pair potential reproduces both `K_p` and
`dH` of NO2/N2O4.** The physical reason is the reaction entropy: experiment
loses `dS = -176 J/mol/K` on dimerisation, the isotropic model with a
physical bond only ~-82 J/mol/K. The missing ~94 J/mol/K is mostly the
rotational entropy that two NO2 molecules lose when they lock into a planar
N2O4 - degrees of freedom a spherical bead does not have.

What to do with this (in order of preference for this project):

1. **Generic model (the core project).** Do not demand NO2 numbers: design the
   `f_d` and time-scale matrix directly (Section 6). Nothing in RQ1-RQ6
   requires `dH` to be that of NO2.
2. **Real-fluid track, `dG`-honouring.** Anchor `K_p` at the MD temperature
   (mode `anchored`): correct composition at that state point, `|dH|` about
   half the experimental value. State this explicitly; it matters for heat
   transport (reaction enthalpy carried by diffusing species).
3. **`dH`-honouring** variant (fit `eps_b` to `dH`): correct reaction heat but
   K too large by ~e^11 - fully dimerised at any useful density. Not useful.
4. **Fix the model**, not the fit: add orientational entropy (patchy /
   Kern-Frenkel-type bonding sites or a two-site NO2) - research-plan
   Extension J. A temperature-dependent degeneracy factor added in
   post-processing would *not* make the MD system obey it and is rejected.

---

## 5. Stage 2 - adding the coordination penalty (still no MD)

Script: `scripts/run_stage2_manybody.py`. The pair potential is **frozen** at
its Stage-1 values; only the many-body term changes, so differences are due
to the penalty alone. Two pair potentials are screened because penalty
behaviour depends on well depth: the anchored real-fluid potential (deep
well) and the generic design state E2K2 (shallow well).

### 5.1 Design rule

`E_mb = 0` for every isolated monomer or dimer configuration. Then all
Stage-1 integrals remain valid and nothing needs re-fitting. Penalties that
violate this slightly (the explorer's tanh-squared hinge, which is small but
non-zero for `0 < n < 1`) are quantified by the shift of `ln K` they cause.

### 5.2 Forms compared (`manybody/coordination_penalties.py`)

| Penalty | Exact pair invariance | LAMMPS |
|---|---|---|
| coordination, Hermite counter, tanh^2 hinge (explorer) | no (tiny) | `eam/fs` |
| coordination, Hermite counter, cubic hinge `x^3/(x+delta)` | **yes** | `eam/fs` |
| coordination, quintic counter (falls from r_b), cubic hinge | **yes** | `eam/fs` |
| coordination, damped-tanh counter, cubic hinge | **yes** | `eam/fs` |
| coordination, Gaussian counter, tanh^2 hinge | no | `eam/fs` |
| glue / valence-one embedding | **yes** | `eam/fs` |
| KTS angular three-body (Skorpa) | **yes** | custom / `sw` + overlay |
| Stillinger-Weber, cos theta0 = 1 | **yes** | `sw` + `table` overlay |

Every coordination-type penalty is of embedded-atom form
`E = sum_i F(n_i) + 1/2 sum U2`, `n_i = sum_j s(r_ij)`, so it runs natively in
LAMMPS `pair_style eam/fs` with exact per-atom energies and virials (relevant
for the later heat-flux work). `export/lammps_export.py` writes these files
and the pipeline checks them against the Python energies with the LAMMPS
Python module.

### 5.3 Procedure for every penalty

1. **Strength.** Scan all two-bond geometries (both bonds inside the bonded
   region, 20-180 degrees) and find the smallest `lambda` that puts every one
   of them above the **dimer-monomer van der Waals complex** (third particle
   outside the bonded region of both partners, `vdw_reference`;
   `trimer_energy_surface.required_strength`). The explorer's tbAnalyse uses
   dimer + *free* monomer instead; that reference makes `lambda` diverge when
   the barrier is low, because a particle at the barrier top of one partner
   sits in the dispersion well of the other (we hit lambda ~ 2e6 kJ/mol for a
   generic state with a 0.65 kJ/mol barrier). Screening uses 1.15 x lambda_min
   (glue: the explorer's 1.2 eps_b rule); the strength scan then picks the
   production value - 2 x lambda_min for the exported models, which lowers
   both the trimer content and the escape barrier of residual trimer minima.
   Report geometries the counter cannot see (`unfixable`) - no lambda can fix
   those.
2. **Pair invariance.** Maximum energy added to an isolated pair and the
   resulting `d ln K_p`.
3. **Worst remaining trimer** energy.
4. **Traps.** Local trimer minima on the `(r12, r13, theta)` grid and the
   lowest escape barrier to a state where one particle has left (threshold
   flooding with 26-connectivity); rough lifetime `0.1 ps exp(barrier/RT)`.
   A thermodynamically unfavourable trimer that is kinetically trapped is as
   bad as a stable one for the transport study.
5. **Trimer content** from `K3` at a reference density, compared with the
   pair-only model, and a strength scan `lambda/lambda_min`.
6. **LAMMPS check** of the `eam/fs` export.

---

## 6. Generic 3 x 3 design (the core project)

Script: `scripts/run_stage1_generic.py`. For family A (and C for comparison),
at `T = 300 K`, `rho0 = 0.002 A^-3` (rho0 sigma^3 ~ 0.1):

* **E1/E2/E3:** `eps_b` such that the dilute-limit `f_d` = 0.2 / 0.5 / 0.8,
  with the Morse range parameter of the anchored potential kept (one bond
  length scale, so `k_b = 2 eps_b beta_b^2` scales with the depth). Keeping
  `k_b = 580` is impossible for the piecewise family below
  `eps_b ~ 12 kJ/mol`: no barrier curvature keeps the quintic bridge
  monotonic for a stiff but shallow well (the additive family has no such
  limit);
* **K1/K2/K3:** barrier heights giving TST lifetimes 1 : 10 : 100, with
  `eps_b` re-tuned at each barrier so that `K` (hence `f_d`) is unchanged.

The output (`results/generic_design.csv`, LAMMPS files in
`export_lammps/generic_A/`) is the *starting point* of research-plan Steps
5-6: the MD equilibrium calibration will shift `eps_b` because the dilute
mass-action law is only approximate at rho0 sigma^3 ~ 0.1, and the real
relaxation times must be measured. What the design guarantees is the
ordering, the near-constancy of `K` across a kinetic row, and that the
parameter values are in a range where reactions happen on MD time scales.

---

## 7. How the best potential is chosen

A single least-squares number is not used, because the families have
different numbers of parameters, the targets are correlated, and a good
thermodynamic fit says nothing about kinetics or MD stability.
`selection/model_selection.py` and `scripts/run_stage3_selection.py`:

**Pass 1 - hard rejection.** Clean landscape; C1-continuous force; bond
stiffness below the MD limit (20 steps per period at 1 fs, i.e.
`k_b < 2.3e4 kJ/mol/A^2`); barrier tunable over >= 2 decades of lifetime at
fixed K.

**Pass 2 - Pareto comparison** of the survivors on (lower is better): rms
`ln K` error on fit and held-out temperatures, `|dH|` error, `B2_frozen`
error, barrier leakage `|d ln K/d ln tau|`, `B2_frozen` shift per decade of
lifetime, number of free parameters, `log10` condition number. Dominated
candidates are dropped; among the non-dominated ones the simplest wins.

**Is it possible to determine a best potential from these data?** Partly.

* The equilibrium data constrain only `Z_b(T)`, i.e. essentially two numbers
  of the bond well (depth and width). All bond forms with the same
  `(eps_b, k_b)` give almost the same `K(T)` (differences are anharmonic
  corrections of order `RT/eps_b`): **thermodynamics cannot discriminate
  between Morse, Mie, Wang-Frenkel or KTS bond branches.**
* The outer branch is constrained by `B2_frozen` only up to a
  depth-position degeneracy.
* What *does* discriminate the families is the behaviour that matters for the
  project: barrier/equilibrium orthogonality, the side effect of the barrier
  on non-bonded structure, smoothness and monotonic feasibility, compatibility
  with the valence penalty, and simplicity. The selection therefore ends with
  a small shortlist (a primary model and a structurally different
  alternative), not a single "true" potential - which is what the MD
  validation needs.
* The penalty is selected separately with the pair potential fixed:
  pass = every two-bond geometry lies above the van der Waals complex and
  `|d ln K| < 0.01`; then exact pair invariance, no trimer trap living
  > 1 ps, native LAMMPS form, lowest trimer content, fewest parameters.

---

## 8. What is and is not established before MD

| Question | Pre-MD answer possible? |
|---|---|
| Does `U2` reproduce dilute-gas `K_p(T)`? | yes (exactly, within the model) |
| Does it reproduce the van 't Hoff `dH`? | yes - and the answer is "not together with K" for an isotropic bead |
| What is its `B2`, and how is it split? | yes |
| Is the bond basin numerically feasible for MD? | yes |
| Can the barrier change without shifting K? | yes, to numerical precision, with compensation |
| Are trimers energetically suppressed / trapped? | yes (3-particle surfaces) |
| Dilute-limit trimer content and Delta B3 | yes |
| Trimer content at liquid density | no - many-particle MD needed |
| True association/dissociation rates, tau_r | no - TST is only an upper bound / ordering |
| NVE energy conservation of the exported potential | no - first MD validation step |
| Transport coefficients | no |

---

## 9. Remarks on the calibration notes this pipeline was built from

The notes that motivated this pipeline are correct on the main points (K_V =
Z_b/2, `dH = <U>_b - RT`, the representability warning, the finite-box
caveat, `Delta B3 >= 0` for a non-negative penalty). Points that are
sharpened or corrected here:

1. The apparent `B2` of the reacting gas per formula unit is exactly the
   model's full Mayer `B2`; no "reduction" is needed, but it is dominated by
   `-N_A K_V` at 298-400 K, so it adds little information there.
2. The naive two-particle mean-count estimate of `K` is wrong by
   `V/(2 Z_u (1 - P_b))` (x18 or x0.5 in the examples above), not merely
   "finite-size combinatorics".
3. `Delta B3` for an effective penalty is ~`2 K3(pair only)` and does not
   distinguish penalties; the trimer content from `K3` does.
4. The TST "lifetime" of a real-NO2-anchored dimer at 298 K is ~1e-8 s
   (2e-9 to 4e-8 s across the families): reactive MD of the real system at
   room temperature will barely sample dissociation. The generic model must
   use shallower wells (the design gives eps_b = 5-18 kJ/mol at the chosen
   density, TST lifetimes 3 ps - 20 ns).
5. The outer-branch fit to `B2` is degenerate in well position and depth;
   the position must be fixed by another argument (here: the LJ minimum).
6. Sizing the penalty against dimer + free monomer (the explorer's rule) is
   unsafe for low barriers; size it against the van der Waals complex and
   choose the final strength from the trimer content, not from the energy
   criterion alone.
