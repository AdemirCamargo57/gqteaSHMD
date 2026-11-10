"""Random-number sources with a JSON-serialisable state (for restarts and replay in tests)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

BIT_GENERATORS = ("PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64")


class RandomSource(Protocol):
    def uniform(self) -> float: ...

    def state(self) -> dict[str, Any]: ...


def _plain(obj: Any) -> Any:
    """Convert NumPy scalars/arrays inside a nested structure to Python types."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class GeneratorSource:
    """Uniform draws from a ``numpy.random.Generator``."""

    def __init__(self, generator: np.random.Generator) -> None:
        self._rng = generator

    def uniform(self) -> float:
        return float(self._rng.random())

    def state(self) -> dict[str, Any]:
        return {"kind": "generator", "bit_generator": _plain(self._rng.bit_generator.state)}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> GeneratorSource:
        """Rebuild the generator. A restart file is untrusted input: only the bit generators
        in ``BIT_GENERATORS`` are accepted, checked before anything is called."""
        if state.get("kind") != "generator":
            raise ValueError(f"not a generator state: kind={state.get('kind')!r}")
        try:
            bg_state = state["bit_generator"]
            name = bg_state["bit_generator"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"incomplete random source state: missing {exc}") from None
        if not isinstance(name, str) or name not in BIT_GENERATORS:
            raise ValueError(
                f"unknown bit generator {name!r}; expected one of {', '.join(BIT_GENERATORS)}"
            )
        bg = getattr(np.random, name)()
        try:
            bg.state = bg_state
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed random source state for {name}: {exc}") from None
        return cls(np.random.Generator(bg))


class ReplaySource:
    """Returns the given values in order (tests, comparisons)."""

    def __init__(self, values: Sequence[float], position: int = 0) -> None:
        self._values = [float(x) for x in values]
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position <= len(self._values)
        ):
            raise ValueError(
                f"replay position must be an integer in 0..{len(self._values)}; got {position!r}"
            )
        self._pos = position

    def uniform(self) -> float:
        try:
            value = self._values[self._pos]
        except IndexError:
            raise ValueError(f"replay sequence exhausted at draw {self._pos}") from None
        self._pos += 1
        return value

    def state(self) -> dict[str, Any]:
        return {"kind": "replay", "position": self._pos, "values": list(self._values)}


def source_from_state(state: dict[str, Any]) -> RandomSource:
    kind = state.get("kind")
    if kind == "generator":
        return GeneratorSource.from_state(state)
    if kind == "replay":
        try:
            values, position = state["values"], state["position"]
        except KeyError as exc:
            raise ValueError(f"incomplete random source state: missing {exc}") from None
        return ReplaySource(values, position)
    raise ValueError(f"unknown random source kind {kind!r}")
