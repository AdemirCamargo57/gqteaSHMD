"""Initial-state selection from the vertical excitations of the Wigner samples.

A candidate is a pair (sample, excited state) that passes the optional energy window
``lo <= dE <= hi`` (eV). In ``bright`` mode a candidate with weight ``w`` is accepted when
``r < w / w_max``, where ``r`` is uniform in [0, 1) and ``w_max`` is the largest weight among
the candidates (the window is applied first, so the maximum is over the in-window pairs only).
Two weightings are offered:

* ``f_over_dE2`` (default): ``w = f / dE**2``, with ``f`` the oscillator strength and ``dE`` the
  vertical excitation energy of the pair (eV; the ratio ``w / w_max`` does not depend on the
  energy unit). This is SHARC 4.1 excite.py-compatible (determined by black-box comparison).
* ``f``: ``w = f`` (acceptance ``f / f_max``, the Newton-X-style rule; Barbatti et al.,
  J. Photochem. Photobiol. A 190, 228 (2007)).

``w_max <= 0`` selects nothing. Exactly one random number is drawn per candidate, in candidate
order (sample-major, then S1, S2, ...), so the stream is a fixed function of the candidate list.
The first ``n`` accepted pairs are kept. ``fixed`` mode takes the first ``n`` candidates of one
state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    f_max: float  # largest oscillator strength among the candidates (for reports)
    accepted: int  # accepted pairs before truncation to n
    weight_max: float = 0.0  # w_max of the weighting used (f_max for "fixed")


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


def _weight(candidate: Candidate, weighting: str) -> float:
    if weighting == "f":
        return candidate.oscillator_strength
    return candidate.oscillator_strength / candidate.excitation_energy_eV**2


def select_bright(
    candidates: list[Candidate],
    n: int,
    rng: np.random.Generator,
    *,
    weighting: Literal["f_over_dE2", "f"] = "f_over_dE2",
) -> Selection:
    """Accept each candidate with probability w / w_max; keep the first n accepted."""
    if weighting not in ("f_over_dE2", "f"):
        raise ValueError(f"unknown weighting {weighting!r}")
    f_max = max((c.oscillator_strength for c in candidates), default=0.0)
    draws = [float(rng.random()) for _ in candidates]
    weights = [_weight(c, weighting) for c in candidates]
    w_max = max(weights, default=0.0)
    if w_max <= 0.0:
        return Selection((), len(candidates), max(f_max, 0.0), 0, 0.0)
    accepted = [c for c, r, w in zip(candidates, draws, weights, strict=True) if r < w / w_max]
    return Selection(tuple(accepted[:n]), len(candidates), f_max, len(accepted), w_max)


def select_fixed(candidates: list[Candidate], n: int) -> Selection:
    """The first n candidates (used when a single state is requested)."""
    f_max = max((c.oscillator_strength for c in candidates), default=0.0)
    return Selection(tuple(candidates[:n]), len(candidates), f_max, len(candidates), f_max)
