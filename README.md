# reactive-md

MD study of a simple dimerizing fluid, 2A ⇌ A₂. The project asks how chemical
equilibrium and reaction time scale independently affect transport
(diffusion, viscosity, thermal conductivity).

| Folder | Content |
|---|---|
| `lammps/01_lj_basics/` | baseline Lennard-Jones NVT → NVE simulation |
| `analysis/` | post-processing (MSD / diffusion) |
| `forcefield_calibration/` | **pre-MD calibration of the reactive pair potential and the valence-limiting penalty**: reference data, fitting, attainability, barrier families, trimer suppression, model selection, LAMMPS export |
| `notes/`, `results/` | notes and raw output |

Start with
[`forcefield_calibration/docs/CALIBRATION_STRATEGY.md`](forcefield_calibration/docs/CALIBRATION_STRATEGY.md)
(method) and
[`forcefield_calibration/docs/RESULTS.md`](forcefield_calibration/docs/RESULTS.md)
(results and recommendations).

```bash
pip install -r requirements.txt
cd forcefield_calibration/scripts && python run_all.py       # ~10 min, regenerates results/ and figures/
cd ../.. && python -m pytest forcefield_calibration/tests -q
```
