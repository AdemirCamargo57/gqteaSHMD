"""Input file schema (core sections; P2 adds electronic, sampling, dynamics, ...)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
        checks = [("[resources]", self.for_stage(""))]
        checks += [(f"[resources.stages.{name}]", self.for_stage(name)) for name in self.stages]
        for label, eff in checks:
            cores = eff.max_parallel_tasks * eff.threads_per_task
            if cores > self.total_cores:
                raise ValueError(
                    f"{label}: max_parallel_tasks × threads_per_task = {cores} exceeds "
                    f"total_cores = {self.total_cores}"
                )
            memory = eff.max_parallel_tasks * eff.memory_mb_per_task
            if memory > self.total_memory_mb:
                raise ValueError(
                    f"{label}: max_parallel_tasks × memory_mb_per_task = {memory} MB exceeds "
                    f"total_memory_mb = {self.total_memory_mb}"
                )
        return self


class ExecutionSection(_Section):
    max_attempts: int = Field(default=2, ge=1)
    poll_interval_s: float = Field(default=0.5, gt=0)
    stop_grace_period_s: float = Field(default=30.0, ge=0)
    min_free_disk_mb: int = Field(default=1000, ge=0)


class WorkflowSection(_Section):
    pause_after: list[str] = Field(default_factory=list)


class ShmdConfig(_Section):
    schema_version: Literal[1] = 1
    project: ProjectSection
    molecule: MoleculeSection
    resources: ResourcesSection
    execution: ExecutionSection = Field(default_factory=ExecutionSection)
    workflow: WorkflowSection = Field(default_factory=WorkflowSection)
