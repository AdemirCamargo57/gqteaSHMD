"""Electron count and its consistency with charge and spin multiplicity."""

from __future__ import annotations

from collections.abc import Sequence


def count_electrons(atomic_numbers: Sequence[int], charge: int, multiplicity: int) -> int:
    """N_e = sum(Z) - q; requires N_e > 0, M - 1 <= N_e and N_e ≡ M - 1 (mod 2)."""
    n_electrons = sum(atomic_numbers) - charge
    unpaired = multiplicity - 1
    if n_electrons <= 0 or unpaired > n_electrons or n_electrons % 2 != unpaired % 2:
        raise ValueError(
            f"Charge {charge} and multiplicity {multiplicity} are inconsistent with "
            f"{n_electrons} electrons."
        )
    return n_electrons
