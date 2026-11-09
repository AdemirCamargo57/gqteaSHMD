"""Input file of ``gqteashmd model-dynamics``: 1D analytic model ensembles (no project)."""

from __future__ import annotations

import inspect
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from gqteashmd.dynamics.models import (
    ElectronicModel,
    TullyDualAvoidedCrossing,
    TullyExtendedCoupling,
    TullySimpleAvoidedCrossing,
)
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.errors import ConfigError

MODELS: dict[str, type] = {
    "tully_sac": TullySimpleAvoidedCrossing,
    "tully_dac": TullyDualAvoidedCrossing,
    "tully_ecr": TullyExtendedCoupling,
}


def _parameter_names(cls: type) -> list[str]:
    return [p for p in inspect.signature(cls).parameters if p != "mass"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ModelSection(_Section):
    name: Literal["tully_sac", "tully_dac", "tully_ecr"]
    mass: float = Field(default=2000.0, gt=0)
    parameters: dict[str, float] = Field(default_factory=dict)

    @field_validator("parameters")
    @classmethod
    def _check_parameters(cls, value: dict[str, float], info: ValidationInfo) -> dict[str, float]:
        name = info.data.get("name")
        if name is None:
            return value
        allowed = _parameter_names(MODELS[name])
        unknown = sorted(set(value) - set(allowed))
        if unknown:
            raise ValueError(
                f"unknown parameter(s) {', '.join(unknown)} for model '{name}'; "
                f"allowed: {', '.join(allowed)}"
            )
        return value


class InitialSection(_Section):
    x0: float = -10.0
    k: float
    sigma: float | None = Field(default=None, gt=0)
    state: int = Field(default=0, ge=0)
    sampling: Literal["wigner", "fixed"] = "wigner"

    @field_validator("k")
    @classmethod
    def _check_k(cls, value: float) -> float:
        if value == 0:
            raise ValueError("k must be non-zero")
        return value


class DynamicsSection(_Section):
    dt_au: float = Field(default=1.0, gt=0)
    nsubsteps: int = Field(default=25, ge=1)
    hopping: Literal["sharc", "fssh", "none"] = "sharc"
    ekin_correction: Literal["parallel_vel", "parallel_nac"] = "parallel_vel"
    frustrated: Literal["keep", "reflect"] = "keep"
    decoherence: Literal["edc", "none"] = "edc"
    decoherence_param_Eh: float = Field(default=0.1, ge=0)
    energy_drift_limit_au: float = Field(default=0.0184, gt=0)
    edc_convention: Literal["sharc", "granucci_persico"] = "sharc"
    substep_interpolation: Literal["midpoint", "endpoint"] = "midpoint"


class EnsembleSection(_Section):
    n_trajectories: int = Field(default=1000, ge=1)
    seed: int = Field(default=12345, ge=0)
    x_exit: float = Field(default=10.0, gt=0)
    max_steps: int = Field(default=200000, ge=1)
    workers: int = Field(default=1, ge=1)  # worker processes; the counts do not depend on it


class ModelDynamicsInput(_Section):
    model: ModelSection
    initial: InitialSection
    dynamics: DynamicsSection = DynamicsSection()
    ensemble: EnsembleSection = EnsembleSection()

    @model_validator(mode="after")
    def _check_state(self) -> ModelDynamicsInput:
        n_states = MODELS[self.model.name].n_states  # type: ignore[attr-defined]
        if self.initial.state >= n_states:
            raise ValueError(
                f"initial.state must be in 0..{n_states - 1} for model '{self.model.name}'; "
                f"got {self.initial.state}"
            )
        return self

    @property
    def sigma(self) -> float:
        return self.initial.sigma if self.initial.sigma is not None else 20.0 / abs(self.initial.k)

    def build_model(self) -> ElectronicModel:
        model: ElectronicModel = MODELS[self.model.name](
            **self.model.parameters, mass=self.model.mass
        )
        return model

    def build_options(self) -> DynamicsOptions:
        return DynamicsOptions(**self.dynamics.model_dump())


def load_model_dynamics(path: Path) -> ModelDynamicsInput:
    path = Path(path)
    try:
        raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Input file not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Input file {path} is not valid UTF-8 text.") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read input file {path}: {exc.strerror or exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Input file {path} is not valid TOML: {exc}") from exc
    try:
        return ModelDynamicsInput.model_validate(raw)
    except ValidationError as exc:
        lines = [f"Invalid input file {path}:"]
        for err in exc.errors():
            location = ".".join(str(part) for part in err["loc"]) or "<top level>"
            lines.append(f"  - {location}: {err['msg']}")
        raise ConfigError("\n".join(lines)) from exc
