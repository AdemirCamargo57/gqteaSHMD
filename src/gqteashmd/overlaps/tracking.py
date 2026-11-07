"""State matching and phase alignment from S[i, j] = <Psi_i(previous)|Psi_j(current)>.

The current states are matched to the previous ones by the assignment that maximises
sum_j |S[a(j), j]| (Hungarian algorithm), and each current state gets the phase that makes its
matched overlap positive. A matched pair with a != j is a root flip (two adiabatic states swapped
their energy order between the geometries). A small singular value of S signals an intruder
state or too few computed states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, eq=False)
class TrackedOverlaps:
    overlaps: np.ndarray  # S with column j multiplied by phases[j]
    phases: np.ndarray  # +1.0 / -1.0 for each current state
    assignment: np.ndarray  # assignment[j] = previous state matched to current state j
    min_singular_value: float

    @property
    def root_flips(self) -> list[tuple[int, int]]:
        """(previous, current) pairs of matched states whose indices differ."""
        return [(int(a), j) for j, a in enumerate(self.assignment) if a != j]


def _square(s: ArrayLike) -> np.ndarray:
    m = np.asarray(s, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"the overlap matrix must be square, got shape {m.shape}")
    if m.size == 0:
        raise ValueError("the overlap matrix is empty")
    return m


def track(s: ArrayLike) -> TrackedOverlaps:
    m = _square(s)
    n = m.shape[0]
    rows, cols = linear_sum_assignment(-np.abs(m))
    assignment = np.empty(n, dtype=np.int64)
    assignment[cols] = rows
    matched = m[assignment, np.arange(n)]
    phases = np.where(matched < 0.0, -1.0, 1.0)
    singular_values = np.linalg.svd(m, compute_uv=False)
    return TrackedOverlaps(m * phases[None, :], phases, assignment, float(singular_values.min()))


def lowdin_orthonormalize(s: ArrayLike) -> np.ndarray:
    """The orthogonal matrix nearest to S (Frobenius norm): U V^T from S = U Sigma V^T."""
    u, _, vt = np.linalg.svd(_square(s))
    return u @ vt
