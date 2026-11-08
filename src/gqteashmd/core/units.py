"""Physical constants and unit conversions (CODATA 2018).

gqteashmd works in Hartree atomic units internally (bohr, hartree, electron mass, e·a0,
hbar/E_h); conversions happen only when files are read or written.
"""

from __future__ import annotations

BOHR_IN_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_IN_ANGSTROM
HARTREE_IN_EV = 27.211386245988
SPEED_OF_LIGHT_AU = 137.035999084  # c in atomic units (1/alpha)
AVOGADRO = 6.02214076e23  # 1/mol
HARTREE_IN_CM1 = 219474.6313632
DALTON_IN_ME = 1822.888486209
AU_TIME_IN_FS = 0.024188843265857
BOLTZMANN_HARTREE_PER_K = 3.1668115634556e-06
