"""Generator of the synthetic LVC fixtures (own invented model, not taken from any program).

Run ``python tests/data/lvc/make_fixtures.py`` to (re)write ``V0.txt`` and the three templates next
to this file. Tests import the functions below and must not regenerate the committed files.

The molecule is an invented bent triatomic (centre atom first): masses 12.0, 15.99491462 and
15.99491462 amu, two bonds of 2.4 bohr and an angle of 120 degrees, in the y-z plane. The normal
modes come from a valence force field (two stretches, k = 0.45 Eh/bohr^2, one bend,
k = 0.18 Eh/rad^2), H = B^T F B with an analytic Wilson B matrix. Modes 1-6 are the exactly zero
translations and rotations (built by Gram-Schmidt, so they do not depend on the LAPACK
eigenvector basis of a degenerate eigenspace); modes 7, 8, 9 are the bend, the symmetric stretch
and the asymmetric stretch (ascending frequency), with the largest component made positive.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from gqteashmd.core.units import DALTON_IN_ME
from gqteashmd.dynamics.lvc_format import (
    LVCParameters,
    V0Data,
    format_lvc_template,
    format_v0,
)

HERE = Path(__file__).parent
SYMBOLS = ("C", "O", "O")
CHARGES = (6.0, 8.0, 8.0)
MASSES_AMU = (12.0, 15.99491462, 15.99491462)
BOND_BOHR = 2.4
ANGLE_DEG = 120.0
K_STRETCH = 0.45  # Eh/bohr^2
K_BEND = 0.18  # Eh/rad^2


def _r12(x: float) -> float:
    """Round to the 12-digit precision of the template files."""
    return float(f"{x:.12e}")


def geometry_bohr() -> np.ndarray:
    """(3, 3) coordinates: centre atom at the origin, the O atoms in the y-z plane."""
    half = math.radians(ANGLE_DEG) / 2
    y, z = BOND_BOHR * math.sin(half), BOND_BOHR * math.cos(half)
    return np.array([[0.0, 0.0, 0.0], [0.0, -y, z], [0.0, y, z]])


def wilson_b(coords: np.ndarray) -> np.ndarray:
    """(3, 9) B matrix of (r1, r2, theta) with respect to Cartesian coordinates."""
    c, o1, o2 = coords
    d1, d2 = o1 - c, o2 - c
    r1, r2 = np.linalg.norm(d1), np.linalg.norm(d2)
    e1, e2 = d1 / r1, d2 / r2
    cos_t = float(e1 @ e2)
    sin_t = math.sqrt(1.0 - cos_t**2)
    b = np.zeros((3, 3, 3))  # (internal, atom, xyz)
    b[0, 1], b[0, 0] = e1, -e1
    b[1, 2], b[1, 0] = e2, -e2
    t1 = -(e2 - cos_t * e1) / (r1 * sin_t)
    t2 = -(e1 - cos_t * e2) / (r2 * sin_t)
    b[2, 1], b[2, 2], b[2, 0] = t1, t2, -(t1 + t2)
    return b.reshape(3, 9)


def force_field_hessian() -> np.ndarray:
    """Cartesian Hessian (9, 9) in Eh/bohr^2 of the invented valence force field."""
    b = wilson_b(geometry_bohr())
    force = np.diag([K_STRETCH, K_STRETCH, K_BEND])
    hessian = b.T @ force @ b
    return 0.5 * (hessian + hessian.T)


def _zero_modes(masses_au: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Orthonormal mass-weighted translations and rotations, (9, 6), deterministic."""
    sqrt_m = np.sqrt(masses_au)
    vectors = []
    for k in range(3):  # translations
        v = np.zeros((3, 3))
        v[:, k] = 1.0
        vectors.append((v * sqrt_m[:, None]).ravel())
    centre = (masses_au[:, None] * coords).sum(axis=0) / masses_au.sum()
    for k in range(3):  # rotations about the centre of mass
        axis = np.zeros(3)
        axis[k] = 1.0
        v = np.cross(axis, coords - centre)
        vectors.append((v * sqrt_m[:, None]).ravel())
    basis: list[np.ndarray] = []
    for v in vectors:  # modified Gram-Schmidt
        for u in basis:
            v = v - (u @ v) * u
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            raise ValueError("a translation/rotation vector is linearly dependent")
        basis.append(v / norm)
    return np.array(basis).T


def v0_data() -> V0Data:
    """Reference geometry, frequencies (Eh) and mass-weighted modes of the fixture molecule."""
    coords = geometry_bohr()
    masses_amu = np.array(MASSES_AMU)
    masses_au = masses_amu * DALTON_IN_ME
    inv_sqrt = 1.0 / np.sqrt(np.repeat(masses_au, 3))
    mw_hessian = inv_sqrt[:, None] * force_field_hessian() * inv_sqrt[None, :]
    evals, evecs = np.linalg.eigh(mw_hessian)
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    zero = np.abs(evals) < 1e-10
    if int(zero.sum()) != 6:
        raise ValueError(f"expected 6 zero eigenvalues, found {int(zero.sum())}")
    vib = evecs[:, ~zero].copy()
    for i in range(vib.shape[1]):
        if vib[np.argmax(np.abs(vib[:, i])), i] < 0:
            vib[:, i] = -vib[:, i]
    modes = np.hstack([_zero_modes(masses_au, coords), vib])
    freqs = np.concatenate([np.zeros(6), np.sqrt(evals[~zero])])
    return V0Data(
        symbols=SYMBOLS,
        nuclear_charges=np.array(CHARGES),
        coords_bohr=coords,
        masses_amu=masses_amu,
        frequencies_Eh=freqs,
        modes=modes,
    )


# ------------------------------------------------------------------ parameters (invented)
# Mode indices are 0-based here: 6 = bend, 7 = symmetric stretch, 8 = asymmetric stretch.


def _clean(d: dict) -> dict:
    return {k: _r12(v) for k, v in d.items()}


def _soc_matrix(n_exp: int, blocks: dict[tuple[int, int], complex]) -> np.ndarray:
    soc = np.zeros((n_exp, n_exp), dtype=complex)
    for (a, b), value in blocks.items():
        value = complex(_r12(value.real), _r12(value.imag))
        soc[a, b] = value
        soc[b, a] = value.conjugate()
    return soc


def singlets_params() -> LVCParameters:
    return LVCParameters(
        v0=v0_data(),
        counts=(3,),
        epsilon=_clean({(1, 0): 0.0, (1, 1): 0.16, (1, 2): 0.19}),
        kappa=_clean(
            {
                (1, 0, 6): -0.004,
                (1, 0, 7): 0.006,
                (1, 0, 8): 0.002,
                (1, 1, 6): 0.009,
                (1, 1, 7): -0.007,
                (1, 1, 8): 0.003,
                (1, 2, 6): -0.005,
                (1, 2, 7): 0.008,
                (1, 2, 8): -0.002,
            }
        ),
        lam=_clean({(1, 1, 2, 8): 0.008, (1, 0, 1, 8): 0.003}),
        soc=None,
        dipoles=None,
    )


def soc_params() -> LVCParameters:
    """2 singlets + 2 triplets (N_exp = 8); order S0 S1 T1(-1) T2(-1) T1(0) T2(0) T1(+1) T2(+1)."""
    n = 8
    s0, s1 = 0, 1
    t1 = {-1: 2, 0: 4, 1: 6}
    t2 = {-1: 3, 0: 5, 1: 7}
    soc = _soc_matrix(
        n,
        {
            (s0, t1[-1]): 2.0e-4 + 1.5e-4j,
            (s1, t1[-1]): 3.0e-4 - 1.0e-4j,
            (s1, t2[-1]): 1.5e-4 + 2.5e-4j,
            (s0, t1[0]): 3.0e-4j,
            (s1, t1[0]): 4.0e-4 + 0.0j,
            (s1, t2[0]): -2.0e-4 + 0.5e-4j,
            (s0, t1[1]): -2.0e-4 + 1.5e-4j,
            (s1, t1[1]): -3.0e-4 - 1.0e-4j,
            (s1, t2[1]): -1.5e-4 + 2.5e-4j,
        },
    )
    dipoles = np.zeros((3, n, n), dtype=complex)
    for axis, (d00, d01, d11) in enumerate([(0.1, 0.35, 0.2), (0.0, 0.1, 0.05), (-0.1, 0.0, 0.15)]):
        dipoles[axis, 0, 0], dipoles[axis, 0, 1], dipoles[axis, 1, 0] = d00, d01, d01
        dipoles[axis, 1, 1] = d11
        for ms in (-1, 0, 1):
            a, b = t1[ms], t2[ms]
            dipoles[axis, a, a], dipoles[axis, b, b] = 0.15 + 0.01 * axis, 0.25
            dipoles[axis, a, b] = dipoles[axis, b, a] = 0.05 * (axis + 1)
    dipoles = np.vectorize(_r12)(dipoles.real) + 0j
    return LVCParameters(
        v0=v0_data(),
        counts=(2, 0, 2),
        epsilon=_clean({(1, 0): 0.0, (1, 1): 0.16, (3, 0): 0.14, (3, 1): 0.17}),
        kappa=_clean(
            {
                (1, 0, 6): -0.004,
                (1, 0, 7): 0.005,
                (1, 1, 6): 0.008,
                (1, 1, 7): -0.006,
                (1, 1, 8): 0.003,
                (3, 0, 6): 0.006,
                (3, 0, 7): -0.004,
                (3, 0, 8): 0.002,
                (3, 1, 6): -0.003,
                (3, 1, 7): 0.007,
                (3, 1, 8): -0.002,
            }
        ),
        lam=_clean({(1, 0, 1, 8): 0.003, (3, 0, 1, 8): 0.005}),
        soc=soc,
        dipoles=dipoles,
    )


def twomode_params() -> LVCParameters:
    """2 singlets + 1 triplet (N_exp = 5); only modes 7 (tuning) and 9 (coupling)."""
    soc = _soc_matrix(
        5,
        {(1, 2): 2.0e-4 + 0.0j, (1, 3): 2.0e-4j, (1, 4): -2.0e-4 + 0.0j},
    )
    return LVCParameters(
        v0=v0_data(),
        counts=(2, 0, 1),
        epsilon=_clean({(1, 0): 0.0, (1, 1): 0.15, (3, 0): 0.12}),
        kappa=_clean({(1, 0, 6): -0.006, (1, 1, 6): 0.009}),
        lam=_clean({(1, 0, 1, 8): 0.004}),
        soc=soc,
        dipoles=None,
    )


def fixture_texts() -> dict[str, str]:
    """File name -> exact committed contents."""
    return {
        "V0.txt": format_v0(v0_data()),
        "lvc_singlets.template": format_lvc_template(singlets_params()),
        "lvc_soc.template": format_lvc_template(soc_params()),
        "lvc_twomode.template": format_lvc_template(twomode_params()),
    }


def main() -> None:
    for name, text in fixture_texts().items():
        with open(HERE / name, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print(f"wrote {HERE / name}")


if __name__ == "__main__":
    main()
