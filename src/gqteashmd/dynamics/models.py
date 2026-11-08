"""Analytic electronic models (Tully 1990) and the adiabatic quantities derived from them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from gqteashmd.dynamics.records import ElectronicSnapshot

DEGENERACY_THRESHOLD = 1e-12


class ElectronicModel(Protocol):
    n_states: int
    n_dof: int
    masses: np.ndarray  # (f,) electron masses

    def evaluate(
        self, q: np.ndarray, previous: ElectronicSnapshot | None
    ) -> ElectronicSnapshot: ...


class DiabaticModel(ABC):
    """An electronic model given by a real symmetric diabatic matrix V(q) and its derivatives."""

    n_states: int
    n_dof: int
    masses: np.ndarray

    @abstractmethod
    def diabatic(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return V (n, n) real symmetric and dV (f, n, n) = dV/dq_f."""

    def evaluate(self, q: np.ndarray, previous: ElectronicSnapshot | None) -> ElectronicSnapshot:
        q = np.asarray(q, dtype=float)
        v, dv = self.diabatic(q)
        energies, w = np.linalg.eigh(v)
        n = len(energies)
        if previous is not None and previous.vectors is not None:
            signs = np.sign(np.einsum("ik,ik->k", w, previous.vectors))
            signs[signs == 0] = 1.0
            w = w * signs
        else:
            # Deterministic frame: the largest-|component| of every eigenvector is positive.
            idx = np.argmax(np.abs(w), axis=0)
            w = w * np.where(w[idx, np.arange(n)] < 0, -1.0, 1.0)
        gaps = energies[None, :] - energies[:, None]  # E_k - E_j at [j, k]
        off = ~np.eye(n, dtype=bool)
        if np.any(np.abs(gaps[off]) < DEGENERACY_THRESHOLD):
            raise ValueError(
                f"degenerate states at q = {q.tolist()}: the non-adiabatic coupling is undefined"
            )
        g = np.einsum("ja,fab,bk->fjk", w.T, dv, w)  # G_f = W^T dV_f W
        gradients = np.einsum("fkk->kf", g)
        safe = np.where(off, gaps, 1.0)
        nacs = np.where(off[:, :, None], g.transpose(1, 2, 0) / safe[:, :, None], 0.0)
        return ElectronicSnapshot(
            energies=energies,
            gradients=gradients,
            nacs=nacs,
            hamiltonian=np.diag(energies).astype(complex),
            vectors=w,
        )


def _masses(mass: float, n_dof: int = 1) -> np.ndarray:
    return np.full(n_dof, float(mass))


def _x(q: np.ndarray) -> float:
    return float(np.asarray(q, dtype=float).reshape(-1)[0])


class TullySimpleAvoidedCrossing(DiabaticModel):
    n_states = 2
    n_dof = 1

    def __init__(
        self,
        A: float = 0.01,
        B: float = 1.6,
        C: float = 0.005,
        D: float = 1.0,
        mass: float = 2000.0,
    ) -> None:
        self.A, self.B, self.C, self.D = A, B, C, D
        self.masses = _masses(mass)

    def diabatic(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = _x(q)
        A, B, C, D = self.A, self.B, self.C, self.D
        if x >= 0:
            v11 = A * (1.0 - np.exp(-B * x))
            d11 = A * B * np.exp(-B * x)
        else:
            v11 = -A * (1.0 - np.exp(B * x))
            d11 = A * B * np.exp(B * x)
        v12 = C * np.exp(-D * x * x)
        d12 = -2.0 * D * x * v12
        v = np.array([[v11, v12], [v12, -v11]])
        dv = np.array([[[d11, d12], [d12, -d11]]])
        return v, dv


class TullyDualAvoidedCrossing(DiabaticModel):
    n_states = 2
    n_dof = 1

    def __init__(
        self,
        A: float = 0.1,
        B: float = 0.28,
        C: float = 0.015,
        D: float = 0.06,
        E0: float = 0.05,
        mass: float = 2000.0,
    ) -> None:
        self.A, self.B, self.C, self.D, self.E0 = A, B, C, D, E0
        self.masses = _masses(mass)

    def diabatic(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = _x(q)
        A, B, C, D, E0 = self.A, self.B, self.C, self.D, self.E0
        g = np.exp(-B * x * x)
        v22 = -A * g + E0
        d22 = 2.0 * A * B * x * g
        v12 = C * np.exp(-D * x * x)
        d12 = -2.0 * D * x * v12
        v = np.array([[0.0, v12], [v12, v22]])
        dv = np.array([[[0.0, d12], [d12, d22]]])
        return v, dv


class TullyExtendedCoupling(DiabaticModel):
    n_states = 2
    n_dof = 1

    def __init__(
        self,
        A: float = 6e-4,
        B: float = 0.1,
        C: float = 0.9,
        mass: float = 2000.0,
    ) -> None:
        self.A, self.B, self.C = A, B, C
        self.masses = _masses(mass)

    def diabatic(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = _x(q)
        A, B, C = self.A, self.B, self.C
        if x < 0:
            v12 = B * np.exp(C * x)
            d12 = B * C * np.exp(C * x)
        else:
            v12 = B * (2.0 - np.exp(-C * x))
            d12 = B * C * np.exp(-C * x)
        v = np.array([[A, v12], [v12, -A]])
        dv = np.array([[[0.0, d12], [d12, 0.0]]])
        return v, dv


class TwoDimensionalCrossing(DiabaticModel):
    """Two states, two coordinates (x, y), with a conical intersection near V11 = V22, y = 0."""

    n_states = 2
    n_dof = 2

    def __init__(
        self,
        k: float = 0.02,
        a: float = 1.0,
        delta: float = 0.01,
        c: float = 0.005,
        mass: float = 2000.0,
    ) -> None:
        self.k, self.a, self.delta, self.c = k, a, delta, c
        self.masses = _masses(mass, 2)

    def diabatic(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = (float(t) for t in np.asarray(q, dtype=float).reshape(-1)[:2])
        k, a, delta, c = self.k, self.a, self.delta, self.c
        v11 = 0.5 * k * ((x - a) ** 2 + y * y)
        v22 = 0.5 * k * ((x + a) ** 2 + y * y) + delta
        v = np.array([[v11, c * y], [c * y, v22]])
        dv = np.zeros((2, 2, 2))
        dv[0] = [[k * (x - a), 0.0], [0.0, k * (x + a)]]
        dv[1] = [[k * y, c], [c, k * y]]
        return v, dv
