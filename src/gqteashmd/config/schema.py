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
        for label, eff in checks:
            p, t, m = eff.max_parallel_tasks, eff.threads_per_task, eff.memory_mb_per_task
            if p * t > self.total_cores:
                raise ValueError(
                    f"{label}: max_parallel_tasks × threads_per_task = {p} × {t} = {p * t} "
                    f"exceeds total_cores = {self.total_cores}"
                )
            if p * m > self.total_memory_mb:
                raise ValueError(
                    f"{label}: max_parallel_tasks × memory_mb_per_task = {p} × {m} = {p * m} MB "
                    f"exceeds total_memory_mb = {self.total_memory_mb}"
                )
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
        if self.nelecas % 2:
            raise ValueError("electronic.casscf.nelecas must be even for singlet states")
        if self.nelecas > 2 * self.ncas:
            raise ValueError(
                f"electronic.casscf.nelecas = {self.nelecas} must be at most 2 × ncas = "
                f"{2 * self.ncas}"
            )
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
    casscf: CasscfSection | None = None

    @model_validator(mode="after")
    def _method_options(self) -> ElectronicSection:
        if self.triplets:
            raise ValueError(
                "electronic.triplets > 0 is not supported yet (planned together with "
                "intersystem crossing); set triplets = 0"
            )
        if self.method == "casscf":
            if self.casscf is None:
                raise ValueError(
                    "method = 'casscf' needs an [electronic.casscf] table with ncas and nelecas"
                )
            if self.functional is not None:
                raise ValueError("electronic.functional is not used by method = 'casscf'")
            available = n_singlet_csfs(self.casscf.ncas, self.casscf.nelecas)
            if self.singlets > available:
                raise ValueError(
                    f"electronic.singlets = {self.singlets} exceeds the {available} singlet "
                    f"states of CAS({self.casscf.nelecas},{self.casscf.ncas}) "
                    f"(nelecas = {self.casscf.nelecas}, ncas = {self.casscf.ncas})"
                )
        else:
            if self.functional is None:
                raise ValueError(
                    f"electronic.functional is required for method = '{self.method}' "
                    "(use 'hf' for CIS/TDHF)"
                )
            if self.casscf is not None:
                raise ValueError("[electronic.casscf] is only used by method = 'casscf'")
        return self


class ShmdConfig(_Section):
    schema_version: Literal[1] = 1
    project: ProjectSection
    molecule: MoleculeSection
    resources: ResourcesSection
    execution: ExecutionSection = Field(default_factory=ExecutionSection)
    workflow: WorkflowSection = Field(default_factory=WorkflowSection)
    electronic: ElectronicSection | None = None

    @model_validator(mode="after")
    def _closed_shell_for_electronic_structure(self) -> ShmdConfig:
        if self.electronic is not None and self.molecule.multiplicity != 1:
            raise ValueError(
                "the PySCF backend supports closed-shell singlet molecules only "
                "(molecule.multiplicity = 1)"
            )
        return self
