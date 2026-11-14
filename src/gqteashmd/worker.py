"""Task worker: `python -m gqteashmd.worker <task.json>`. Runs exactly one task of one stage."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from gqteashmd.config.loader import load_snapshot
from gqteashmd.errors import ProjectError, ShmdError, TaskError
from gqteashmd.logging_setup import configure_task_logging
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.stages.base import StageContext, TaskSpec
from gqteashmd.stages.pipeline import import_pipeline
from gqteashmd.util.fileio import read_json, write_json_atomic

TASK_FILE = "task.json"
RESULT_FILE = "result.json"


def write_task_file(
    path: Path,
    *,
    pipeline_ref: str,
    config_snapshot: Path,
    ctx: StageContext,
    stage_name: str,
    task: TaskSpec,
    task_dir: Path,
) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "pipeline_ref": pipeline_ref,
            "config_snapshot": str(config_snapshot),
            "project_root": str(ctx.layout.root),
            "stage": stage_name,
            "stage_dir": str(ctx.stage_dir),
            "upstream": {name: str(p) for name, p in ctx.upstream.items()},
            "task_id": task.task_id,
            "payload": dict(task.payload),
            "task_dir": str(task_dir),
        },
    )


def read_result(task_dir: Path) -> dict[str, Any] | None:
    try:
        result: dict[str, Any] = read_json(Path(task_dir) / RESULT_FILE)
        return result
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _error(exc: BaseException, *, retryable: bool, with_traceback: bool) -> dict[str, Any]:
    data: dict[str, Any] = {
        "status": "error",
        "error_type": type(exc).__name__,
        "message": str(exc),
        "retryable": retryable,
    }
    if with_traceback:
        data["traceback"] = traceback.format_exc()
    return data


def run_task_file(task_file: Path) -> int:
    try:
        spec = read_json(task_file)
        task_dir = Path(spec["task_dir"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        message = f"Cannot read task file {task_file}: {type(exc).__name__}: {exc}"
        print(message, file=sys.stderr)
        # The controller rewrites task.json before every attempt: an I/O error may be transient,
        # a malformed file will not improve.
        write_json_atomic(
            Path(task_file).parent / RESULT_FILE,
            _error(ProjectError(message), retryable=isinstance(exc, OSError), with_traceback=False),
        )
        return 1
    try:
        loaded = load_snapshot(Path(spec["config_snapshot"]))
        stages = {s.name: s for s in import_pipeline(spec["pipeline_ref"])(loaded.config)}
        stage = stages.get(spec["stage"])
        if stage is None:
            raise ProjectError(f"Stage {spec['stage']!r} is not part of the pipeline")
        ctx = StageContext(
            config=loaded.config,
            layout=ProjectLayout(Path(spec["project_root"])),
            stage_dir=Path(spec["stage_dir"]),
            upstream={name: Path(p) for name, p in spec["upstream"].items()},
        )
        result = stage.run_task(ctx, TaskSpec(spec["task_id"], spec["payload"]), task_dir)
    except TaskError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        write_json_atomic(
            task_dir / RESULT_FILE, _error(exc, retryable=exc.retryable, with_traceback=False)
        )
        return 1
    except ShmdError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        write_json_atomic(
            task_dir / RESULT_FILE, _error(exc, retryable=False, with_traceback=False)
        )
        return 1
    except Exception as exc:  # a bug: keep the traceback for the developer
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        write_json_atomic(task_dir / RESULT_FILE, _error(exc, retryable=False, with_traceback=True))
        return 1
    try:
        write_json_atomic(task_dir / RESULT_FILE, {"status": "ok", "result": result})
    except (TypeError, ValueError) as exc:
        kind = type(exc).__name__
        print(f"{kind}: task result is not JSON-serializable: {exc}", file=sys.stderr)
        write_json_atomic(
            task_dir / RESULT_FILE,
            {
                "status": "error",
                "error_type": kind,
                "message": f"task result is not JSON-serializable: {exc}",
                "retryable": False,
            },
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m gqteashmd.worker TASK_FILE", file=sys.stderr)
        return 2
    configure_task_logging()
    return run_task_file(Path(args[0]))


if __name__ == "__main__":
    sys.exit(main())
