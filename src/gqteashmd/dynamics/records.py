"""Plain records exchanged by the dynamics driver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class ElectronicSnapshot:
    """Electronic quantities at one geometry (atomic units)."""

    energies: np.ndarray  # (n,) Eh
    gradients: np.ndarray  # (n, f) dE_k/dq
    nacs: np.ndarray  # (n, n, f) d_jk, antisymmetric in (j, k)
    hamiltonian: np.ndarray  # (n, n) complex MCH Hamiltonian
    vectors: np.ndarray | None = None  # model-internal eigenvectors (opaque, phase continuity)


@dataclass(eq=False)
class TrajectoryState:
    time_au: float
    step: int
    q: np.ndarray
    v: np.ndarray
    c: np.ndarray  # complex (n,)
    active: int
    snapshot: ElectronicSnapshot
    e_total_initial: float


@dataclass(frozen=True, eq=False)
class StepRecord:
    step: int
    time_au: float
    q: np.ndarray
    v: np.ndarray
    c: np.ndarray
    active: int
    energies: np.ndarray
    e_total: float
    e_kinetic: float
    probabilities: np.ndarray  # (n,) hop probabilities from the old active state
    random_number: float
    hop_from: int | None
    hop_to: int | None
    frustrated: bool
