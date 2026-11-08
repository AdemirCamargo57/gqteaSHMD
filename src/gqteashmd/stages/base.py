"""The stage contract: plan (controller) → run_task (worker) → finalize (controller).

Rules for implementers:
- plan() must be deterministic: same inputs → same task ids, same order.
- run_task() runs in a separate process, may be re-run after a crash, and must be
  idempotent. It may only write inside `task_dir`. It returns a small JSON-serializable
  dict; large outputs go to files in `task_dir`.
- finalize() combines task results into the stage's artifacts inside `ctx.stage_dir`
  and raises StageFailedError if physical/sanity checks fail.
- required_modules lists the Python modules the stage's tasks import; checked by preflight.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gqteashmd.config.loader import section_fingerprint
from gqteashmd.config.schema import ShmdConfig
from gqteashmd.errors import ProjectError
from gqteashmd.project.layout import ProjectLayout

SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")


def is_safe_name(name: str) -> bool:
    """True for filesystem-safe task ids and stage names ('.' and '..' are not)."""
    return bool(SAFE_NAME.fullmatch(name)) and name not in {".", ".."}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_safe_name(self.task_id):
            raise ValueError(f"Invalid task id {self.task_id!r}: use letters, digits, _ . -")


@dataclass(frozen=True)
class StageContext:
    config: ShmdConfig
    layout: ProjectLayout
    stage_dir: Path
    upstream: Mapping[str, Path]

    def upstream_dir(self, name: str) -> Path:
        try:
            return self.upstream[name]
        except KeyError as exc:
            raise ProjectError(
                f"Output of stage '{name}' is not available (stage skipped or not run)"
            ) from exc


class Stage(ABC):
    name: str
    version: int = 1
    depends_on: tuple[str, ...] = ()
    config_sections: tuple[str, ...] = ()
    max_failed_fraction: float = 0.0
    required_modules: tuple[str, ...] = ()

    def enabled(self, config: ShmdConfig) -> bool:
        return True

    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        """Inputs that decide whether stored results are still valid.

        Must not read upstream artifacts: the controller also calls this before any stage
        has run, with an empty ``ctx.upstream``. Depend on upstream results through
        ``depends_on`` (their fingerprints are included automatically).
        """
        return section_fingerprint(ctx.config, self.config_sections)

    @abstractmethod
    def plan(self, ctx: StageContext) -> list[TaskSpec]: ...

    @abstractmethod
    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]: ...

    @abstractmethod
    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None: ...
