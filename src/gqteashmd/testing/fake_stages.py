"""Deterministic fake stages for orchestrator tests. Not used by production pipelines.

Behaviour is controlled by the JSON environment variable GQTEASHMD_FAKE_BEHAVIOUR, which
worker subprocesses inherit from the controller:
  sleep: float seconds per task
  fail / unexpected / crash_once / retryable_once: lists of "stage:task_id"
  skip_optimize: bool
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.errors import TaskError
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.stages.prepare import PrepareStage
from gqteashmd.util.fileio import read_json, write_json_atomic

BEHAVIOUR_ENV = "GQTEASHMD_FAKE_BEHAVIOUR"


def _behaviour() -> dict[str, Any]:
    return json.loads(os.environ.get(BEHAVIOUR_ENV, "{}"))


class FakeStage(Stage):
    def __init__(
        self,
        name: str,
        n_tasks: int,
        depends_on: Sequence[str] = (),
        *,
        max_failed_fraction: float = 0.0,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.n_tasks = n_tasks
        self.depends_on = tuple(depends_on)
        self.max_failed_fraction = max_failed_fraction
        self._enabled = enabled

    def enabled(self, config: ShmdConfig) -> bool:
        return self._enabled

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        return [TaskSpec(f"t{i:03d}", {"index": i}) for i in range(self.n_tasks)]

    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        key = f"{self.name}:{task.task_id}"
        behaviour = _behaviour()
        time.sleep(float(behaviour.get("sleep", 0.0)))
        task_dir.mkdir(parents=True, exist_ok=True)
        if key in behaviour.get("crash_once", []):
            marker = task_dir / "crashed_once"
            if not marker.exists():
                marker.write_text("crashed")
                os._exit(137)
        if key in behaviour.get("retryable_once", []):
            marker = task_dir / "failed_once"
            if not marker.exists():
                marker.write_text("failed")
                raise TaskError(f"transient fake failure in {key}", retryable=True)
        if key in behaviour.get("fail", []):
            raise TaskError(f"fake failure in {key}", retryable=False)
        if key in behaviour.get("unexpected", []):
            raise RuntimeError(f"unexpected fake error in {key}")
        upstream = sorted(
            name for name, path in ctx.upstream.items() if (path / "summary.json").exists()
        )
        (task_dir / "out.txt").write_text(key)
        return {"index": task.payload["index"], "upstream": upstream}

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        values = sorted(int(r["index"]) for r in results.values())
        write_json_atomic(ctx.stage_dir / "summary.json", {"values": values})


def make_pipeline(config: ShmdConfig) -> list[Stage]:
    behaviour = _behaviour()
    return [
        PrepareStage(),
        FakeStage("optimize", 1, ("prepare",), enabled=not behaviour.get("skip_optimize", False)),
        FakeStage("frequencies", 1, ("prepare",)),
        FakeStage("sampling", 3, ("frequencies",)),
        FakeStage("dynamics", 4, ("sampling",), max_failed_fraction=0.25),
    ]


def read_summary(stage_dir: Path) -> dict[str, Any]:
    return read_json(stage_dir / "summary.json")
