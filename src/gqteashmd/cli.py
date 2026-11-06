"""Command-line interface: gqteashmd init | validate | run | status | stop."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from gqteashmd import __version__
from gqteashmd.config.loader import load_config
from gqteashmd.errors import ConfigError, ProjectError, ShmdError
from gqteashmd.logging_setup import FILE_ONLY, close_logging, configure_logging
from gqteashmd.orchestrator import Orchestrator
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.project.lock import locked_by
from gqteashmd.project.state import StateStore, TaskStatus
from gqteashmd.stages.pipeline import DEFAULT_PIPELINE, import_pipeline

EXIT_STOPPED = 5

TEMPLATE = """\
# gqteashmd input file. Paths are relative to this file.
schema_version = 1

[project]
name = "my_molecule"          # letters, digits, _ . -
directory = "run"             # results go here; use a local Linux disk (not OneDrive, not /mnt/c)
seed = 12345                  # master random seed for the whole workflow

[molecule]
geometry = "molecule.xyz"     # XYZ file, Angstrom
charge = 0
multiplicity = 1

[resources]
max_parallel_tasks = 1        # tasks running at the same time
threads_per_task = 1          # OpenMP/BLAS threads per task
memory_mb_per_task = 2000     # passed to PySCF as max_memory (MB)

[execution]
max_attempts = 2              # attempts per task for retryable failures
stop_grace_period_s = 30.0    # time running tasks get to stop after `gqteashmd stop`
"""


def _cmd_init(args: argparse.Namespace) -> int:
    path: Path = args.path
    if path.exists() and not args.force:
        raise ConfigError(f"{path} already exists; use --force to overwrite it.")
    try:
        path.write_text(TEMPLATE, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot write {path}: {exc.strerror or exc}") from exc
    print(f"Wrote template input file {path}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    loaded = load_config(args.input)
    cfg = loaded.config
    if not cfg.molecule.geometry.is_file():
        raise ConfigError(f"Geometry file not found or not a regular file: {cfg.molecule.geometry}")
    print(f"Input file {loaded.source} is valid.")
    print(f"  project directory: {cfg.project.directory}")
    print(f"  geometry:          {cfg.molecule.geometry}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    loaded = load_config(args.input)
    orchestrator = Orchestrator(loaded, pipeline_ref=args.pipeline, rerun_from=args.rerun_from)
    orchestrator.layout.ensure()
    configure_logging(orchestrator.layout.logs_dir / "gqteashmd.log")
    logger = logging.getLogger("gqteashmd.cli")
    stop_requested = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        if not stop_requested.is_set():
            logger.warning("Stop requested (signal %d): stopping running tasks...", signum)
        stop_requested.set()

    previous: dict[int, Any] = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, _on_signal)
        summary = orchestrator.run(should_stop=stop_requested.is_set)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if summary.stopped:
        print("Stopped. Run the same command again to resume.")
        return EXIT_STOPPED
    print(
        f"Workflow complete: {len(summary.completed)} stage(s) run, "
        f"{len(summary.up_to_date)} already up to date, {len(summary.skipped)} skipped."
    )
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(args.input).config
    layout = ProjectLayout(cfg.project.directory)
    if not layout.state_db.exists():
        print(f"Project '{cfg.project.name}' has not been started (no state in {layout.root}).")
        return 0
    pid = locked_by(layout.lock_file)
    print(f"Project: {cfg.project.name}  ({layout.root})")
    print(f"Controller: {f'running (pid {pid})' if pid else 'not running'}")
    print(
        f"{'stage':<16}{'status':<13}{'done':>6}{'failed':>8}{'pending':>9}{'running':>9}  message"
    )
    with StateStore(layout.state_db, readonly=True) as state:
        for stage in import_pipeline(args.pipeline)(cfg):
            record = state.get_stage(stage.name)
            counts = state.task_counts(stage.name)
            status = record.status.value if record else "not started"
            message = (record.message or "").splitlines()[0] if record and record.message else ""
            print(
                f"{stage.name:<16}{status:<13}{counts[TaskStatus.DONE]:>6}"
                f"{counts[TaskStatus.FAILED]:>8}{counts[TaskStatus.PENDING]:>9}"
                f"{counts[TaskStatus.RUNNING]:>9}  {message}"
            )
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    cfg = load_config(args.input).config
    pid = locked_by(ProjectLayout(cfg.project.directory).lock_file)
    if pid is None:
        print("No running controller for this project.")
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        raise ProjectError(
            f"Cannot signal controller pid {pid}: {exc.strerror or exc}. "
            "It may have just exited, or belong to another user."
        ) from exc
    print(f"Stop requested for controller pid {pid}; running tasks are being stopped.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gqteashmd", description="Surface-hopping MD workflow")
    parser.add_argument("--version", action="version", version=f"gqteashmd {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="write a commented template input file")
    p_init.add_argument("path", type=Path)
    p_init.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_init.set_defaults(func=_cmd_init)

    p_validate = sub.add_parser("validate", help="check an input file without running anything")
    p_validate.add_argument("input", type=Path)
    p_validate.set_defaults(func=_cmd_validate)

    p_run = sub.add_parser("run", help="run (or resume) the workflow")
    p_run.add_argument("input", type=Path)
    p_run.add_argument("--rerun-from", metavar="STAGE", help="discard results from STAGE onward")
    p_run.add_argument("--pipeline", default=DEFAULT_PIPELINE, help=argparse.SUPPRESS)
    p_run.set_defaults(func=_cmd_run)

    p_status = sub.add_parser("status", help="show progress of a project")
    p_status.add_argument("input", type=Path)
    p_status.add_argument("--pipeline", default=DEFAULT_PIPELINE, help=argparse.SUPPRESS)
    p_status.set_defaults(func=_cmd_status)

    p_stop = sub.add_parser("stop", help="ask the running controller to stop gracefully")
    p_stop.add_argument("input", type=Path)
    p_stop.set_defaults(func=_cmd_stop)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = logging.getLogger("gqteashmd.cli")
    try:
        return args.func(args)
    except ShmdError as exc:
        log.error("%s", exc, extra=FILE_ONLY)
        print(f"gqteashmd: error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        log.error("Unexpected error", exc_info=True, extra=FILE_ONLY)
        print(
            f"gqteashmd: error: internal error: {type(exc).__name__}: {exc} "
            "(please report this; details in the project log if one was configured)",
            file=sys.stderr,
        )
        return 1
    finally:
        close_logging()
