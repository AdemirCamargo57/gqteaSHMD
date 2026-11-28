"""The core's spin-expanded (MCH) basis for spin-free backend data (dynamics with SOC, spec 3.1).

Spin-free states S0..S(n-1), T1..Tm; expanded order S0..S(n-1), then T1..Tm with Ms = -1, then
0, then +1 (`dynamics/spin.py`): index of S_I is I, of (T_J, M) n + M_index m + (J - 1).
"""

from __future__ import annotations

import numpy as np

M_COMPONENTS = 3


def expanded_size(n: int, m: int) -> int:
    return int(n) + M_COMPONENTS * int(m)


def expanded_index(n: int, m: int, spin_free: int, m_index: int = 0) -> int:
    if spin_free < n:
        return int(spin_free)
    return int(n + m_index * m + (spin_free - n))


def spin_free_rows(n: int, m: int) -> list[int]:
    """One expanded row per spin-free state (a triplet's Ms = -1 row), in spin-free order."""
    return [expanded_index(n, m, i) for i in range(int(n) + int(m))]


def spin_free_index(n: int, m: int, k: int) -> int:
    return int(k) if k < n else int(n + (k - n) % m)


def _rows(n: int, m: int) -> np.ndarray:
    """Spin-free row of every expanded index."""
    return np.array([spin_free_index(n, m, k) for k in range(expanded_size(n, m))], dtype=int)


def expand_energies(e: np.ndarray, n: int, m: int) -> np.ndarray:
    out: np.ndarray = np.asarray(e, dtype=float)[_rows(n, m)]
    return out


def expand_gradients(g: np.ndarray, n: int, m: int) -> np.ndarray:
    out: np.ndarray = np.asarray(g, dtype=float)[_rows(n, m)]
    return out


def expand_overlaps(s: np.ndarray, n: int, m: int) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    if m and (np.any(s[:n, n:] != 0) or np.any(s[n:, :n] != 0)):
        raise ValueError("overlaps have nonzero singlet-triplet blocks")
    out = np.zeros((expanded_size(n, m),) * 2)
    out[:n, :n] = s[:n, :n]
    if m:
        out[n:, n:] = np.kron(np.eye(M_COMPONENTS), s[n:, n:])
    return out


def spin_free_populations(p: np.ndarray, n: int, m: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.zeros(p.shape[:-1] + (n + m,))
    for k, row in enumerate(_rows(n, m)):
        out[..., row] += p[..., k]
    return out
