"""Energy drift and jump diagnostics (P8 spec section 7)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gqteashmd.core.units import HARTREE_IN_EV


def max_jump_eV(e_total_hartree: np.ndarray) -> float:
    e = np.asarray(e_total_hartree, dtype=float)
    return float(np.abs(np.diff(e)).max() * HARTREE_IN_EV) if e.size >= 2 else 0.0


def max_drift_eV(e_total_hartree: np.ndarray) -> float:
    e = np.asarray(e_total_hartree, dtype=float)
    return float(np.abs(e - e[0]).max() * HARTREE_IN_EV) if e.size else 0.0


def summary_eV(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None, "max": None}
    v = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
    }
