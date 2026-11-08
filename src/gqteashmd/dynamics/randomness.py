"""Random-number sources with a JSON-serialisable state (for restarts and replay in tests)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


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
        if state.get("kind") != "generator":
            raise ValueError(f"not a generator state: kind={state.get('kind')!r}")
        bg_state = state["bit_generator"]
        name = bg_state["bit_generator"]
        try:
            bg = getattr(np.random, name)()
        except (AttributeError, TypeError) as exc:
            raise ValueError(f"unknown bit generator {name!r}") from exc
        bg.state = bg_state
        return cls(np.random.Generator(bg))


class ReplaySource:
    """Returns the given values in order (tests, comparisons)."""

    def __init__(self, values: Sequence[float], position: int = 0) -> None:
        self._values = [float(x) for x in values]
        self._pos = int(position)

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
        return ReplaySource(state["values"], state["position"])
    raise ValueError(f"unknown random source kind {kind!r}")
