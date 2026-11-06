"""Input file schema (core sections; P2 adds electronic, sampling, dynamics, ...)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


class ResourcesSection(_Section):
    max_parallel_tasks: int = Field(default=1, ge=1)
    threads_per_task: int = Field(default=1, ge=1)
    memory_mb_per_task: int = Field(default=2000, ge=100)


class ExecutionSection(_Section):
    max_attempts: int = Field(default=2, ge=1)
    poll_interval_s: float = Field(default=0.5, gt=0)
    stop_grace_period_s: float = Field(default=30.0, ge=0)


class ShmdConfig(_Section):
    schema_version: Literal[1] = 1
    project: ProjectSection
    molecule: MoleculeSection
    resources: ResourcesSection = Field(default_factory=ResourcesSection)
    execution: ExecutionSection = Field(default_factory=ExecutionSection)
