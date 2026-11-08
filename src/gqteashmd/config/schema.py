"""Input file schema."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectSection(_Section):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    directory: Path
    seed: int = Field(default=12345, ge=0)


class MoleculeSection(_Section):
    geometry: Path
    charge: int = 0
    multiplicity: int = Field(default=1, ge=1)


class StageResources(_Section):
    """Per-stage override; unset keys fall back to the global [resources] values."""

    max_parallel_tasks: int | None = Field(default=None, ge=1)
    threads_per_task: int | None = Field(default=None, ge=1)
    memory_mb_per_task: int | None = Field(default=None, ge=100)


@dataclass(frozen=True)
class EffectiveResources:
    max_parallel_tasks: int
    threads_per_task: int
    memory_mb_per_task: int


class ResourcesSection(_Section):
    total_cores: int = Field(ge=1)
    total_memory_mb: int = Field(ge=100)
    max_parallel_tasks: int = Field(default=1, ge=1)
    threads_per_task: int = Field(default=1, ge=1)
    memory_mb_per_task: int = Field(default=2000, ge=100)
    stages: dict[str, StageResources] = Field(default_factory=dict)

    def for_stage(self, name: str) -> EffectiveResources:
        o = self.stages.get(name, StageResources())
        return EffectiveResources(
            max_parallel_tasks=(
                o.max_parallel_tasks
                if o.max_parallel_tasks is not None
                else self.max_parallel_tasks
            ),
            threads_per_task=(
                o.threads_per_task if o.threads_per_task is not None else self.threads_per_task
            ),
            memory_mb_per_task=(
                o.memory_mb_per_task
                if o.memory_mb_per_task is not None
                else self.memory_mb_per_task
            ),
        )

    @model_validator(mode="after")
    def _fits_the_workstation(self) -> ResourcesSection:
        checks = [
            (
                "[resources]",
                EffectiveResources(
                    self.max_parallel_tasks, self.threads_per_task, self.memory_mb_per_task
                ),
            )
        ]
        checks += [(f"[resources.stages.{name}]", self.for_stage(name)) for name in self.stages]
        problems: list[str] = []
        for label, eff in checks:
            p, t, m = eff.max_parallel_tasks, eff.threads_per_task, eff.memory_mb_per_task
            if p * t > self.total_cores:
                problems.append(
                    f"{label}: max_parallel_tasks × threads_per_task = {p} × {t} = {p * t} "
                    f"exceeds total_cores = {self.total_cores}"
                )
            if p * m > self.total_memory_mb:
                problems.append(
                    f"{label}: max_parallel_tasks × memory_mb_per_task = {p} × {m} = {p * m} MB "
                    f"exceeds total_memory_mb = {self.total_memory_mb}"
                )
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @field_validator("stages")
    @classmethod
    def _stage_names_are_safe(cls, value: dict[str, StageResources]) -> dict[str, StageResources]:
        from gqteashmd.stages.base import is_safe_name

        for name in value:
            if not is_safe_name(name):
                raise ValueError(f"{name!r} is not a valid stage name")
        return value


class ExecutionSection(_Section):
    max_attempts: int = Field(default=2, ge=1)
    poll_interval_s: float = Field(default=0.5, gt=0)
    stop_grace_period_s: float = Field(default=30.0, ge=0)
    min_free_disk_mb: int = Field(default=1000, ge=0)


class WorkflowSection(_Section):
    pause_after: list[str] = Field(default_factory=list)


def n_singlet_csfs(ncas: int, nelecas: int) -> int:
    """Number of singlet CSFs of a CAS(nelecas, ncas) space (Weyl formula, nelecas even)."""
    n, half = ncas, nelecas // 2
    return comb(n + 1, half) * comb(n + 1, half + 1) // (n + 1)


class CasscfSection(_Section):
    ncas: int = Field(ge=1)
    nelecas: int = Field(ge=1)
    conv_tol: float = Field(default=1e-10, gt=0)
    conv_tol_grad: float = Field(default=1e-6, gt=0)

    @model_validator(mode="after")
    def _singlet_active_space(self) -> CasscfSection:
        problems: list[str] = []
        if self.nelecas % 2:
            problems.append("electronic.casscf.nelecas must be even for singlet states")
        if self.nelecas > 2 * self.ncas:
            problems.append(
                f"electronic.casscf.nelecas = {self.nelecas} must be at most 2 × ncas = "
                f"{2 * self.ncas}"
            )
        if problems:
            raise ValueError("; ".join(problems))
        return self


class ElectronicSection(_Section):
    method: Literal["casscf", "tda", "tddft"]
    basis: str = Field(min_length=1)
    functional: str | None = None
    singlets: int = Field(ge=1)
    triplets: int = Field(default=0, ge=0)
    grid_level: int = Field(default=3, ge=0, le=9)
    scf_conv_tol: float = Field(default=1e-10, gt=0)
    scf_max_cycle: int = Field(default=100, ge=1)
    td_conv_tol: float = Field(default=1e-8, gt=0)
    overlap_norm_kept: float = Field(default=0.9999, gt=0.9, le=1.0)
    casscf: CasscfSection | None = None

    @model_validator(mode="after")
    def _method_options(self) -> ElectronicSection:
        problems: list[str] = []
        if self.triplets:
            problems.append(
                "electronic.triplets > 0 is not supported yet (planned together with "
                "intersystem crossing); set triplets = 0"
            )
        if self.method == "casscf":
            if self.casscf is None:
                problems.append(
                    "method = 'casscf' needs an [electronic.casscf] table with ncas and nelecas"
                )
            if self.functional is not None:
                problems.append("electronic.functional is not used by method = 'casscf'")
            if self.casscf is not None:
                available = n_singlet_csfs(self.casscf.ncas, self.casscf.nelecas)
                if self.singlets > available:
                    problems.append(
                        f"electronic.singlets = {self.singlets} exceeds the {available} singlet "
                        f"states of CAS({self.casscf.nelecas},{self.casscf.ncas}) "
                        f"(nelecas = {self.casscf.nelecas}, ncas = {self.casscf.ncas})"
                    )
        else:
            if self.functional is None:
                problems.append(
                    f"electronic.functional is required for method = '{self.method}' "
                    "(use 'hf' for CIS/TDHF)"
                )
            if self.casscf is not None:
                problems.append("[electronic.casscf] is only used by method = 'casscf'")
        if problems:
            raise ValueError("; ".join(problems))
        return self


class OptimizeSection(_Section):
    enabled: bool = True
    max_steps: int = Field(default=100, ge=1)
    convergence: Literal[
        "GAU",
        "GAU_LOOSE",
        "GAU_TIGHT",
        "GAU_VERYTIGHT",
        "NWCHEM_LOOSE",
        "TURBOMOLE",
        "INTERFRAG_TIGHT",
    ] = "GAU"


class FrequenciesSection(_Section):
    method: Literal["auto", "analytic", "finite_difference"] = "auto"
    fd_step_bohr: float = Field(default=0.005, gt=0, le=0.1)
    imaginary_cutoff_cm1: float = Field(default=20.0, ge=0)


class SamplingSection(_Section):
    n_samples: int = Field(default=200, ge=1)
    temperature_K: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    low_frequency_cutoff_cm1: float = Field(default=10.0, gt=0, allow_inf_nan=False)


class ShmdConfig(_Section):
    schema_version: Literal[1] = 1
    project: ProjectSection
    molecule: MoleculeSection
    resources: ResourcesSection
    execution: ExecutionSection = Field(default_factory=ExecutionSection)
    workflow: WorkflowSection = Field(default_factory=WorkflowSection)
    electronic: ElectronicSection | None = None
    optimize: OptimizeSection = Field(default_factory=OptimizeSection)
    frequencies: FrequenciesSection = Field(default_factory=FrequenciesSection)
    sampling: SamplingSection = Field(default_factory=SamplingSection)

    @model_validator(mode="after")
    def _consistent(self) -> ShmdConfig:
        problems: list[str] = []
        if self.electronic is not None and self.molecule.multiplicity != 1:
            problems.append(
                "the PySCF backend supports closed-shell singlet molecules only "
                "(molecule.multiplicity = 1)"
            )
        if (
            self.electronic is not None
            and self.electronic.method == "casscf"
            and self.frequencies.method == "analytic"
        ):
            problems.append(
                "frequencies.method = 'analytic': there is no analytic SA-CASSCF Hessian in "
                "PySCF; use 'auto' or 'finite_difference'"
            )
        if problems:
            raise ValueError("; ".join(problems))
        return self
