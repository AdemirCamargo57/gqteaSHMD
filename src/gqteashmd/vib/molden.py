"""Molden vibrational file ([FREQ], [FR-COORD], [FR-NORM-COORD]); coordinates in bohr."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from gqteashmd.chem.elements import atomic_number


def format_molden(
    symbols: Sequence[str],
    coords_bohr: ArrayLike,
    frequencies_cm1: ArrayLike,
    modes_cartesian: ArrayLike,
) -> str:
    coords = np.asarray(coords_bohr, dtype=float)
    freqs = np.asarray(frequencies_cm1, dtype=float)
    modes = np.asarray(modes_cartesian, dtype=float)
    lines = ["[Molden Format]", "[Atoms] AU"]
    for index, (symbol, (x, y, z)) in enumerate(zip(symbols, coords, strict=True), start=1):
        lines.append(
            f"{symbol:<2} {index:5d} {atomic_number(symbol):4d} {x:18.10f} {y:18.10f} {z:18.10f}"
        )
    lines.append("[FREQ]")
    lines += [f"{f:16.6f}" for f in freqs]
    lines.append("[FR-COORD]")
    for symbol, (x, y, z) in zip(symbols, coords, strict=True):
        lines.append(f"{symbol:<2} {x:18.10f} {y:18.10f} {z:18.10f}")
    lines.append("[FR-NORM-COORD]")
    for k in range(modes.shape[1]):
        lines.append(f"vibration {k + 1:6d}")
        for dx, dy, dz in modes[:, k].reshape(-1, 3):
            lines.append(f"{dx:18.10f} {dy:18.10f} {dz:18.10f}")
    return "\n".join(lines) + "\n"
