"""Harmonic vibrational analysis of a Cartesian Hessian (Wilson-Decius-Cross; Eckart frame).

    H_mw = M^-1/2 H M^-1/2,   M = diag(m_1, m_1, m_1, m_2, ...)  (amu)
    D    = orthonormal mass-weighted translations (sqrt(m_i) e_a) and infinitesimal rotations
           (sqrt(m_i) a x (r_i - R_com)) about the principal axes a, orthonormalised by SVD; linear
           when I_min / I_max < 1e-3 (5 vectors, the axis rotation is dropped), 3 for an atom
    Q    = orthonormal basis of the complement of D;  H_int = Q^T H_mw Q;  H_int v = lambda v
    omega = sqrt(lambda / m_e-per-amu) (atomic units);  nu~ = omega * HARTREE_IN_CM1  (cm^-1)
    l = Q v (mass-weighted, orthonormal);  x = M^-1/2 l;  reduced mass mu = 1 / |x|^2 (amu)
Imaginary frequencies (lambda < 0) are reported as negative wavenumbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from gqteashmd.core.units import DALTON_IN_ME, HARTREE_IN_CM1

LINEAR_INERTIA_RATIO = 1e-3  # I_min / I_max below which a molecule is treated as linear
_ATOM_MOMENT_AMU_BOHR2 = 1e-8  # largest moment below which the system is a single point


@dataclass(frozen=True, eq=False)
class HarmonicAnalysis:
    frequencies_cm1: np.ndarray
    modes_mass_weighted: np.ndarray
    modes_cartesian: np.ndarray
    reduced_masses_amu: np.ndarray
    linear: bool
    n_external: int
    external_residual_cm1: float


def _wavenumbers(eigenvalues: np.ndarray) -> np.ndarray:
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues) / DALTON_IN_ME) * HARTREE_IN_CM1


def _external_basis(masses: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Translations plus the rotations that exist, decided from the principal moments of inertia.

    Linear (I_min / I_max < LINEAR_INERTIA_RATIO): the rotation about the near-molecular axis
    is dropped (5 vectors). An atom (all moments ~ 0) has the 3 translations only.
    """
    n = len(masses)
    root = np.sqrt(masses)
    centre = (masses[:, None] * coords).sum(axis=0) / masses.sum()
    x = coords - centre
    inertia = (masses * (x * x).sum(axis=1)).sum() * np.eye(3) - np.einsum(
        "i,ia,ib->ab", masses, x, x
    )
    moments, axes = np.linalg.eigh(inertia)  # ascending
    vectors = []
    for axis in range(3):
        v = np.zeros((n, 3))
        v[:, axis] = root
        vectors.append(v.ravel())
    if moments[-1] > _ATOM_MOMENT_AMU_BOHR2:
        keep = (1, 2) if moments[0] / moments[-1] < LINEAR_INERTIA_RATIO else (0, 1, 2)
        for k in keep:
            vectors.append((np.cross(axes[:, k], x) * root[:, None]).ravel())
    u, _, _ = np.linalg.svd(np.array(vectors).T, full_matrices=False)
    return u


def harmonic_analysis(
    masses_amu: ArrayLike, coords_bohr: ArrayLike, hessian: ArrayLike
) -> HarmonicAnalysis:
    masses = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords_bohr, dtype=float)
    h = np.asarray(hessian, dtype=float)
    n3 = 3 * len(masses)
    if coords.shape != (len(masses), 3) or h.shape != (n3, n3):
        raise ValueError(
            f"inconsistent shape: {len(masses)} masses, coords {coords.shape}, hessian {h.shape}"
        )
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(coords))):
        raise ValueError("the Hessian and coordinates must be finite")
    if np.any(masses <= 0):
        raise ValueError("masses must be positive")
    h = 0.5 * (h + h.T)
    root3 = np.sqrt(np.repeat(masses, 3))
    h_mw = h / np.outer(root3, root3)
    d = _external_basis(masses, coords)
    n_external = d.shape[1]
    q_full, _ = np.linalg.qr(np.hstack([d, np.eye(n3)]))
    q = q_full[:, n_external:n3]
    eigenvalues, vectors = np.linalg.eigh(q.T @ h_mw @ q)
    modes_mw = q @ vectors
    cartesian = modes_mw / root3[:, None]
    norms = np.linalg.norm(cartesian, axis=0)
    residual = _wavenumbers(np.linalg.eigvalsh(d.T @ h_mw @ d)) if n_external else np.zeros(0)
    return HarmonicAnalysis(
        frequencies_cm1=_wavenumbers(eigenvalues),
        modes_mass_weighted=modes_mw,
        modes_cartesian=cartesian / norms if norms.size else cartesian,
        reduced_masses_amu=1.0 / norms**2 if norms.size else norms,
        linear=n_external == 5,
        n_external=n_external,
        external_residual_cm1=float(np.abs(residual).max()) if residual.size else 0.0,
    )
