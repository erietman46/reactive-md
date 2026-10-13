"""Shared paths, settings, I/O and plot style for the calibration scripts."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# ---------------------------------------------------------------- settings
T_FIT = np.array([298.15, 325.0, 350.0, 375.0, 400.0])
T_TEST = np.array([425.0, 450.0, 475.0, 500.0])
T_B2 = np.arange(250.0, 501.0, 50.0)
T_REF = 298.15                     # anchor temperature of the real-fluid track
MU = 23.00275                      # reduced mass of an NO2 pair, g/mol
KB_PHYS = 580.0                    # physical N-N curvature, kJ/mol/A^2
KB_RANGE = (150.0, 3000.0)

# Generic (core-project) design state
T_GEN = 300.0
FD_TARGETS = {"E1 monomer-rich": 0.2, "E2 mixed": 0.5, "E3 dimer-rich": 0.8}
TAU_RATIOS = {"K1 fast": 1.0, "K2 medium": 10.0, "K3 slow": 100.0}

# ---------------------------------------------------------------- plot style
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAMILY_COLOR = {k: PALETTE[i] for i, k in enumerate("ABCDEF")}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 6, "font.size": 10,
        "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "bold",
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })
    return plt


# ---------------------------------------------------------------- I/O
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def save_json(obj, name):
    (RESULTS / name).write_text(json.dumps(_clean(obj), indent=2))


def load_json(name):
    return json.loads((RESULTS / name).read_text())


def save_csv(rows, name, header=None):
    rows = [_clean(r) for r in rows]
    keys = header or sorted({k for r in rows for k in r}, key=lambda k: (k != "label", k))
    with open(RESULTS / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (f"{r[k]:.6g}" if isinstance(r.get(k), float) else r[k]))
                        for k in keys})


def potential_record(pot):
    """Machine-readable description of a potential (functional form, parameters, units)."""
    d = pot.describe()
    return dict(name=pot.name, family=pot.family, class_name=type(pot).__name__, params=dict(pot.p),
                kh=getattr(pot, "kh", None), landscape=d,
                units=dict(energy="kJ/mol", distance="Angstrom"))


def rebuild(rec):
    from forcefield_calibration.potentials.additive import AdditivePotential
    from forcefield_calibration.potentials.piecewise import PiecewisePotential

    cls = {"PiecewisePotential": PiecewisePotential, "AdditivePotential": AdditivePotential}[rec["class_name"]]
    return cls(name=rec["name"], **rec["params"])
