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
    # Model-internal eigenvectors for phase continuity, opaque to the driver; shape (m, n)
    # with one column per state (DiabaticModel: the diabatic->adiabatic matrix, m = n). Not U.
    vectors: np.ndarray | None = None
    # (n, n) real, S[b, a] = <psi_b(previous)|psi_a(this)>; None when there is no previous snapshot
    overlaps: np.ndarray | None = None


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
    u: np.ndarray | None = None  # complex (n, n); None = identity in MCH order (c is c^diag)


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
    energies_diag: np.ndarray | None = None
    c_mch: np.ndarray | None = None
    active_mch: int | None = None
    u: np.ndarray | None = None  # copy of U' (None = identity)
