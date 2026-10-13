"""Physical constants (CODATA 2018 exact SI values where applicable)."""

R_J = 8.314462618            # J mol^-1 K^-1
R_KJ = R_J * 1e-3            # kJ mol^-1 K^-1
KB = 1.380649e-23            # J K^-1
NA = 6.02214076e23           # mol^-1
P0 = 1.0e5                   # standard pressure, Pa (1 bar)
AMU = 1.66053906660e-27      # kg
KJMOL_PER_EV = 96.48533212   # kJ/mol per eV
KJ_PER_KCAL = 4.184
A3_TO_M3 = 1e-30
A3_TO_CM3 = 1e-24
C_CM = 2.99792458e10         # speed of light, cm/s

M_NO2 = 46.0055              # g/mol


def beta(T):
    """1/(RT) in (kJ/mol)^-1."""
    return 1.0 / (R_KJ * T)
