"""Input file of ``gqteashmd model-dynamics``: 1D analytic model ensembles (no project)."""

from __future__ import annotations

import dataclasses
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

from gqteashmd.dynamics.lvc import LVCModel
from gqteashmd.dynamics.models import (
    ElectronicModel,
    TullyDualAvoidedCrossing,
    TullyExtendedCoupling,
    TullySimpleAvoidedCrossing,
)
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.spin import expanded_states
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


_OPTION_DEFAULTS: dict[str, Any] = {
    f.name: f.default for f in dataclasses.fields(DynamicsOptions) if f.name != "dt_au"
}


class DynamicsSection(_Section):
    """``[dynamics]``: the fields of ``DynamicsOptions``; every default except ``dt_au`` (1.0
    here, required in the library) is taken from it, so the two cannot drift apart."""

    dt_au: float = Field(default=1.0, gt=0)
    nsubsteps: int = Field(default=_OPTION_DEFAULTS["nsubsteps"], ge=1)
    hopping: Literal["sharc", "fssh", "none"] = _OPTION_DEFAULTS["hopping"]
    ekin_correction: Literal["parallel_vel", "parallel_nac"] = _OPTION_DEFAULTS["ekin_correction"]
    frustrated: Literal["keep", "reflect"] = _OPTION_DEFAULTS["frustrated"]
    decoherence: Literal["edc", "none"] = _OPTION_DEFAULTS["decoherence"]
    decoherence_param_Eh: float = Field(default=_OPTION_DEFAULTS["decoherence_param_Eh"], ge=0)
    energy_drift_limit_au: float = Field(default=_OPTION_DEFAULTS["energy_drift_limit_au"], gt=0)
    edc_convention: Literal["sharc", "granucci_persico"] = _OPTION_DEFAULTS["edc_convention"]
    substep_interpolation: Literal["midpoint", "endpoint"] = _OPTION_DEFAULTS[
        "substep_interpolation"
    ]
    coupling: Literal["nacdr", "overlap"] = _OPTION_DEFAULTS["coupling"]
    gradient_transformation: Literal["ngt", "none"] = _OPTION_DEFAULTS["gradient_transformation"]
    eselect_Eh: float | None = Field(default=_OPTION_DEFAULTS["eselect_Eh"], gt=0)


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


class LvcModelSection(_Section):
    name: Literal["lvc"]
    template: str
    spin_orbit: bool = True


class LvcInitialSection(_Section):
    state: str = Field(pattern=r"^[ST][0-9]+$")
    ms: Literal[-1, 0, 1] | None = None
    sampling: Literal["wigner", "fixed"] = "wigner"


class LvcEnsembleSection(_Section):
    n_trajectories: int = Field(default=100, ge=1)
    seed: int = Field(default=12345, ge=0)
    n_steps: int = Field(default=1000, ge=1)
    record_every: int = Field(default=10, ge=1)
    workers: int = Field(default=1, ge=1)


class LvcDynamicsInput(_Section):
    """``[model] name = "lvc"``: population ensembles on an LVC model (P7a-2)."""

    model: LvcModelSection
    initial: LvcInitialSection
    dynamics: DynamicsSection = DynamicsSection()
    ensemble: LvcEnsembleSection = LvcEnsembleSection()

    def build_model(self) -> LVCModel:
        path = Path(self.model.template)
        try:
            return LVCModel.from_template(path, spin_orbit=self.model.spin_orbit)
        except (OSError, ValueError) as exc:
            detail = exc.strerror if isinstance(exc, OSError) and exc.strerror else exc
            raise ConfigError(f"Cannot use the LVC template {path}: {detail}") from exc

    def build_options(self) -> DynamicsOptions:
        return DynamicsOptions(**self.dynamics.model_dump())

    def initial_index(self, model: LVCModel) -> int:
        """Expanded (canonical) index of ``initial.state`` with ``initial.ms``."""
        label = self.initial.state
        mult = 1 if label[0] == "S" else 3
        index = int(label[1:]) - (0 if mult == 1 else 1)
        if mult == 1 and self.initial.ms is not None:
            raise ValueError(f"initial.ms is only for triplet states; {label} is a singlet")
        ms = 0 if mult == 1 else (self.initial.ms if self.initial.ms is not None else 0)
        for i, st in enumerate(expanded_states(model.state_counts)):
            if (st.multiplicity, st.index, st.ms) == (mult, index, ms):
                return i
        raise ValueError(
            f"initial.state {label} does not exist in the model (counts {model.state_counts})"
        )


def load_model_dynamics(path: Path) -> ModelDynamicsInput | LvcDynamicsInput:
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
        model_table = raw.get("model")
        if isinstance(model_table, dict) and model_table.get("name") == "lvc":
            template = model_table.get("template")
            if isinstance(template, str):
                # resolved against the input file's directory (expanduser first)
                model_table["template"] = str(
                    (path.resolve().parent / Path(template).expanduser()).resolve()
                )
            return LvcDynamicsInput.model_validate(raw)
        return ModelDynamicsInput.model_validate(raw)
    except ValidationError as exc:
        lines = [f"Invalid input file {path}:"]
        for err in exc.errors():
            location = ".".join(str(part) for part in err["loc"]) or "<top level>"
            lines.append(f"  - {location}: {err['msg']}")
        raise ConfigError("\n".join(lines)) from exc
