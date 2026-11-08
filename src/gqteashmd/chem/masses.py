"""Atomic masses of the most abundant isotope, in amu (dalton), for H..Rn (master plan O5).

Values: NIST atomic weights and isotopic compositions (as tabulated by PySCF's
COMMON_ISOTOPE_MASSES). Tc, Pm, Po, At and Rn have no stable isotope; the listed isotope is used.
"""

from __future__ import annotations

from gqteashmd.chem.elements import atomic_number

_MASSES = (
    1.007825, 4.002603, 7.016004, 9.012182, 11.009305, 12.0, 14.003074, 15.994915, 18.998403,
    19.99244, 22.98977, 23.985042, 26.981538, 27.976927, 30.973762, 31.972071, 34.968853,
    39.962383, 38.963707, 39.962591, 44.95591, 47.947947, 50.943964, 51.940512, 54.93805,
    55.934942, 58.9332, 57.935348, 62.929601, 63.929147, 68.925581, 73.921178, 74.921596,
    79.916522, 78.918338, 83.911507, 84.911789, 87.905614, 88.905848, 89.904704, 92.906378,
    97.905408, 98.907216, 101.90435, 102.905504, 105.903483, 106.905093, 113.903358, 114.903878,
    119.902197, 120.903818, 129.906223, 126.904468, 131.904154, 132.905447, 137.905241,
    138.906348, 139.905435, 140.907648, 141.907719, 144.912744, 151.919729, 152.921227,
    157.924101, 158.925343, 163.929171, 164.930319, 165.93029, 168.934211, 173.938858,
    174.940768, 179.946549, 180.947996, 183.950933, 186.955751, 191.961479, 192.962924,
    194.964774, 196.966552, 201.970626, 204.974412, 207.976636, 208.980383, 208.982416,
    209.987131, 222.01757,
)  # fmt: skip


def isotope_mass(symbol: str) -> float:
    """Mass (amu) of the most abundant isotope of `symbol`; ValueError for unknown symbols."""
    z = atomic_number(symbol)  # raises ValueError for unknown symbols
    return _MASSES[z - 1]
