"""Options of the surface-hopping integrator."""

from __future__ import annotations

import math
from dataclasses import dataclass

from gqteashmd.core.units import HARTREE_IN_EV
from gqteashmd.dynamics import decoherence, propagators

HOPPING_SCHEMES = ("sharc", "fssh", "none")
EKIN_CORRECTIONS = ("parallel_vel", "parallel_nac")
FRUSTRATED_MODES = ("keep", "reflect")
DECOHERENCE_MODES = ("edc", "none")
EDC_CONVENTIONS = decoherence.EDC_CONVENTIONS
SUBSTEP_INTERPOLATIONS = propagators.INTERPOLATIONS


def _choice(name: str, value: object, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}; got {value!r}")


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class DynamicsOptions:
    dt_au: float
    nsubsteps: int = 25
    hopping: str = "sharc"
    ekin_correction: str = "parallel_vel"
    frustrated: str = "keep"
    decoherence: str = "edc"
    decoherence_param_Eh: float = 0.1
    energy_drift_limit_au: float = 0.5 / HARTREE_IN_EV
    edc_convention: str = "sharc"
    substep_interpolation: str = "midpoint"

    def __post_init__(self) -> None:
        if not _finite(self.dt_au) or not self.dt_au > 0:
            raise ValueError(f"dt_au must be a finite number > 0; got {self.dt_au!r}")
        if (
            isinstance(self.nsubsteps, bool)
            or not isinstance(self.nsubsteps, int)
            or self.nsubsteps < 1
        ):
            raise ValueError(f"nsubsteps must be an integer >= 1; got {self.nsubsteps!r}")
        _choice("hopping", self.hopping, HOPPING_SCHEMES)
        _choice("ekin_correction", self.ekin_correction, EKIN_CORRECTIONS)
        _choice("frustrated", self.frustrated, FRUSTRATED_MODES)
        _choice("decoherence", self.decoherence, DECOHERENCE_MODES)
        _choice("edc_convention", self.edc_convention, EDC_CONVENTIONS)
        _choice("substep_interpolation", self.substep_interpolation, SUBSTEP_INTERPOLATIONS)
        if not _finite(self.decoherence_param_Eh) or self.decoherence_param_Eh < 0:
            raise ValueError(
                f"decoherence_param_Eh must be a finite number >= 0; "
                f"got {self.decoherence_param_Eh!r}"
            )
        if not _finite(self.energy_drift_limit_au) or not self.energy_drift_limit_au > 0:
            raise ValueError(
                f"energy_drift_limit_au must be a finite number > 0; "
                f"got {self.energy_drift_limit_au!r}"
            )
