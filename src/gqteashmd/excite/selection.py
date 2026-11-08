"""Initial-state selection from the vertical excitations of the Wigner samples.

A candidate is a pair (sample, excited state) that passes the optional energy window
``lo <= dE <= hi`` (eV). In ``bright`` mode (Barbatti et al., J. Photochem. Photobiol. A 190,
228 (2007)) a candidate with oscillator strength ``f`` is accepted when ``r < f / f_max``, where
``r`` is uniform in [0, 1) and ``f_max`` is the largest ``f`` among the candidates. Exactly one
random number is drawn per candidate, in candidate order (sample-major, then S1, S2, ...), so
the stream is a fixed function of the candidate list. The first ``n`` accepted pairs are kept.
``fixed`` mode takes the first ``n`` candidates of one state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gqteashmd.core.states import StateLabel


@dataclass(frozen=True)
class Candidate:
    sample: int  # 1-based
    state: str
    excitation_energy_eV: float
    oscillator_strength: float


@dataclass(frozen=True)
class Selection:
    selected: tuple[Candidate, ...]
    candidates: int
    f_max: float
    accepted: int  # accepted pairs before truncation to n


def candidates_from(
    excitation_energies_eV: NDArray[np.float64],
    oscillator_strengths: NDArray[np.float64],
    ok: NDArray[np.bool_],
    states: list[str],
    window: tuple[float, float] | None,
) -> list[Candidate]:
    """Candidates, sample-major then S1, S2, ...; columns of the arrays are S1, S2, ..."""
    columns = sorted({StateLabel.parse(s).index for s in states})
    result: list[Candidate] = []
    for i in range(excitation_energies_eV.shape[0]):
        if not ok[i]:
            continue
        for index in columns:
            energy = float(excitation_energies_eV[i, index - 1])
            if window is not None and not window[0] <= energy <= window[1]:
                continue
            result.append(
                Candidate(i + 1, f"S{index}", energy, float(oscillator_strengths[i, index - 1]))
            )
    return result


def select_bright(candidates: list[Candidate], n: int, rng: np.random.Generator) -> Selection:
    """Accept each candidate with probability f / f_max; keep the first n accepted."""
    f_max = max((c.oscillator_strength for c in candidates), default=0.0)
    draws = [float(rng.random()) for _ in candidates]
    if f_max <= 0.0:
        return Selection((), len(candidates), max(f_max, 0.0), 0)
    accepted = [
        c for c, r in zip(candidates, draws, strict=True) if r < c.oscillator_strength / f_max
    ]
    return Selection(tuple(accepted[:n]), len(candidates), f_max, len(accepted))


def select_fixed(candidates: list[Candidate], n: int) -> Selection:
    """The first n candidates (used when a single state is requested)."""
    f_max = max((c.oscillator_strength for c in candidates), default=0.0)
    return Selection(tuple(candidates[:n]), len(candidates), f_max, len(candidates))
