"""The workflow controller: runs stages in order, tasks in parallel, and survives restarts."""

from __future__ import annotations

import hashlib
import json
import logging
import shlex
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gqteashmd.config.loader import LoadedConfig, write_snapshot
from gqteashmd.errors import ConfigError, ProjectError, ShmdError, StageFailedError, StaleStageError
from gqteashmd.execution.local import Job, JobOutcome, LocalExecutor
from gqteashmd.execution.processes import is_worker_process, terminate_process_group
from gqteashmd.preflight import run_preflight
from gqteashmd.project.layout import ProjectLayout
from gqteashmd.project.lock import ProjectLock
from gqteashmd.project.provenance import append_provenance, collect_provenance
from gqteashmd.project.state import StageStatus, StateStore, TaskRecord, TaskStatus
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.stages.pipeline import (
    DEFAULT_PIPELINE,
    check_stage_names,
    import_pipeline,
    validate_pipeline,
)
from gqteashmd.worker import RESULT_FILE, TASK_FILE, read_result, write_task_file

log = logging.getLogger("gqteashmd.orchestrator")

SKIPPED_FINGERPRINT = "skipped"


@dataclass
class RunSummary:
    completed: list[str] = field(default_factory=list)
    up_to_date: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    stopped: bool = False
    paused_after: str | None = None


def _fingerprint(data: Mapping[str, Any]) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Orchestrator:
    def __init__(
        self,
        loaded: LoadedConfig,
        *,
        pipeline_ref: str = DEFAULT_PIPELINE,
        rerun_from: str | None = None,
    ) -> None:
        self.loaded = loaded
        self.config = loaded.config
        self.pipeline_ref = pipeline_ref
        self.stages: list[Stage] = import_pipeline(pipeline_ref)(self.config)
        validate_pipeline(self.stages)
        names = [s.name for s in self.stages]
        if rerun_from is not None and rerun_from not in names:
            raise ConfigError(
                f"Unknown stage {rerun_from!r} for --rerun-from; stages are: {', '.join(names)}"
            )
        self.rerun_from = rerun_from
        check_stage_names(self.config, names)
        self.layout = ProjectLayout(self.config.project.directory)
        self._preflight_warnings: list[str] | None = None
        self._session_depth = 0
        self._provenance: dict[str, Any] | None = None

    def _executor_for(self, stage: Stage) -> LocalExecutor:
        res = self.config.resources.for_stage(stage.name)
        execution = self.config.execution
        return LocalExecutor(
            max_parallel=res.max_parallel_tasks,
            threads_per_task=res.threads_per_task,
            memory_mb_per_task=res.memory_mb_per_task,
            poll_interval=execution.poll_interval_s,
            stop_grace_period=execution.stop_grace_period_s,
        )

    # ------------------------------------------------------------------ public API
    def preflight(self) -> list[str]:
        """Run the preflight checks once; raises ConfigError, returns warnings."""
        if self._preflight_warnings is None:
            self._preflight_warnings = run_preflight(self.config, self.stages)
        return self._preflight_warnings

    @contextmanager
    def session(self) -> Iterator[None]:
        """Hold the project lock; create the project skeleton only once the lock is ours."""
        if self._session_depth:
            self._session_depth += 1
            try:
                yield
            finally:
                self._session_depth -= 1
            return
        provenance = collect_provenance()  # before the lock: git may be slow
        with ProjectLock(self.layout.lock_file):
            self.layout.ensure()
            for stale in self.layout.meta_dir.glob(".*.tmp"):
                stale.unlink(missing_ok=True)  # left by a crashed controller; we hold the lock
            self._provenance = provenance
            self._session_depth = 1
            try:
                yield
            finally:
                self._session_depth = 0
                self._provenance = None

    def run(self, should_stop: Callable[[], bool] = lambda: False) -> RunSummary:
        already_checked = self._preflight_warnings is not None
        warnings = self.preflight()
        if not already_checked:
            for message in warnings:
                log.warning("Preflight: %s", message)
        with self.session(), StateStore(self.layout.state_db) as state:
            write_snapshot(self.loaded, self.layout.config_snapshot)
            assert self._provenance is not None
            append_provenance(self.layout.provenance_file, self._provenance)
            if self.rerun_from is not None:
                self._reset_from(state, self.rerun_from)
                self.rerun_from = None
            return self._run_pipeline(state, should_stop)

    # ------------------------------------------------------------------ pipeline
    def _run_pipeline(self, state: StateStore, should_stop: Callable[[], bool]) -> RunSummary:
        summary = RunSummary()
        fingerprints: dict[str, str] = {}
        upstream_dirs: dict[str, Path] = {}
        for index, stage in enumerate(self.stages):
            if should_stop():
                summary.stopped = True
                break
            stage_dir = self.layout.stage_dir(index, stage.name)
            ctx = StageContext(
                self.config,
                self.layout,
                stage_dir,
                {d: upstream_dirs[d] for d in stage.depends_on if d in upstream_dirs},
            )
            if not stage.enabled(self.config):
                fingerprints[stage.name] = SKIPPED_FINGERPRINT
                state.set_stage(stage.name, StageStatus.SKIPPED, fingerprint=SKIPPED_FINGERPRINT)
                summary.skipped.append(stage.name)
                log.info("Stage %s: skipped (disabled in the input file)", stage.name)
                continue
            try:
                fp = _fingerprint(
                    {
                        "stage": stage.name,
                        "version": stage.version,
                        "inputs": stage.fingerprint_inputs(ctx),
                        "upstream": {d: fingerprints[d] for d in stage.depends_on},
                    }
                )
            except TypeError as exc:
                raise ProjectError(
                    f"Stage {stage.name!r}: fingerprint inputs are not JSON-serializable: {exc}"
                ) from exc
            fingerprints[stage.name] = fp
            record = state.get_stage(stage.name)
            if record is not None and record.fingerprint != fp:
                has_finished_tasks = state.task_counts(stage.name)[TaskStatus.DONE] > 0
                if has_finished_tasks or record.status in (StageStatus.DONE, StageStatus.RUNNING):
                    source = shlex.quote(str(self.loaded.source))
                    raise StaleStageError(
                        f"Stage '{stage.name}' has results produced from different inputs "
                        "(the input file or an earlier stage changed). Run "
                        f"`gqteashmd run {source} --rerun-from {stage.name}` "
                        "to move the old results aside and recompute from this stage onward."
                    )
                log.info("Stage %s: inputs changed since last attempt; starting over", stage.name)
                self._reset_stage(state, index, stage)
                record = None
            if record is not None and record.status == StageStatus.DONE and not stage_dir.exists():
                log.warning(
                    "Stage %s: results directory %s is missing; recomputing", stage.name, stage_dir
                )
                state.reset_stage(stage.name)
                record = None
            if record is not None and record.status == StageStatus.DONE:
                summary.up_to_date.append(stage.name)
                upstream_dirs[stage.name] = stage_dir
                log.info("Stage %s: up to date", stage.name)
                continue
            try:
                finished = self._run_stage(state, stage, ctx, fp, should_stop)
            except Exception as exc:
                current = state.get_stage(stage.name)
                if current is None or current.status != StageStatus.FAILED:
                    state.set_stage(
                        stage.name, StageStatus.FAILED, message=f"{type(exc).__name__}: {exc}"
                    )
                raise
            if not finished:
                summary.stopped = True
                break
            upstream_dirs[stage.name] = stage_dir
            summary.completed.append(stage.name)
            if stage.name in self.config.workflow.pause_after and index < len(self.stages) - 1:
                summary.paused_after = stage.name
                log.info(
                    "Paused after stage %s (workflow.pause_after); run again to continue",
                    stage.name,
                )
                break
        return summary

    def _reset_from(self, state: StateStore, name: str) -> None:
        start = [s.name for s in self.stages].index(name)
        for index in range(start, len(self.stages)):
            self._reset_stage(state, index, self.stages[index])

    def _reset_stage(self, state: StateStore, index: int, stage: Stage) -> None:
        stage_dir = self.layout.stage_dir(index, stage.name)
        for record in state.get_tasks(stage.name):
            if record.status == TaskStatus.RUNNING:
                self._terminate_orphan(
                    stage, record, self.layout.task_dir(stage_dir, record.task_id)
                )
        moved = self.layout.supersede(stage_dir)
        state.reset_stage(stage.name)
        if moved is not None:
            log.info("Stage %s: previous results moved to %s", stage.name, moved)

    # ------------------------------------------------------------------ one stage
    def _run_stage(
        self,
        state: StateStore,
        stage: Stage,
        ctx: StageContext,
        fp: str,
        should_stop: Callable[[], bool],
    ) -> bool:
        log.info("Stage %s: starting", stage.name)
        state.set_stage(stage.name, StageStatus.RUNNING, fingerprint=fp)
        ctx.stage_dir.mkdir(parents=True, exist_ok=True)
        tasks = stage.plan(ctx)
        ids = [t.task_id for t in tasks]
        if len(set(ids)) != len(ids):
            raise ProjectError(f"Stage {stage.name!r} planned duplicate task ids")
        known = {r.task_id for r in state.get_tasks(stage.name)}
        if known and known != set(ids):
            raise ProjectError(
                f"Stage {stage.name!r} planned different tasks than recorded; "
                f"run with --rerun-from {stage.name}"
            )
        state.upsert_tasks(stage.name, ids)
        state.requeue_failed(stage.name)
        self._reconcile_running(state, stage, ctx)
        specs = {t.task_id: t for t in tasks}
        max_attempts = self.config.execution.max_attempts

        executor = self._executor_for(stage)
        while True:
            runnable = [
                r
                for r in state.get_tasks(stage.name)
                if r.status == TaskStatus.PENDING and r.attempts < max_attempts
            ]
            if not runnable:
                break
            jobs = [self._prepare_job(stage, ctx, specs[r.task_id]) for r in runnable]
            finished = executor.run(
                jobs,
                on_start=lambda job, pid: state.set_task(
                    stage.name, job.key, TaskStatus.RUNNING, pid=pid, attempts_delta=1
                ),
                on_finish=lambda outcome: self._record_outcome(
                    state, stage, ctx, outcome, max_attempts
                ),
                should_stop=should_stop,
            )
            if not finished:
                state.set_stage(stage.name, StageStatus.PENDING, message="stopped by user")
                log.warning("Stage %s: stopped; run the same command again to resume", stage.name)
                return False

        records = state.get_tasks(stage.name)
        failed = [r for r in records if r.status != TaskStatus.DONE]
        if records and len(failed) / len(records) > stage.max_failed_fraction:
            message = self._failure_message(stage, ctx, failed, len(records))
            state.set_stage(stage.name, StageStatus.FAILED, message=message)
            raise StageFailedError(message)
        if failed:
            log.warning(
                "Stage %s: %d of %d tasks failed (tolerated)", stage.name, len(failed), len(records)
            )
        results = self._collect_results(stage, ctx, records)
        try:
            stage.finalize(ctx, results)
        except ShmdError as exc:
            message = f"Stage '{stage.name}' could not finalize its results: {exc}"
            state.set_stage(stage.name, StageStatus.FAILED, message=message)
            raise StageFailedError(message) from exc
        state.set_stage(
            stage.name,
            StageStatus.DONE,
            message=f"{len(results)}/{len(records)} tasks succeeded",
        )
        log.info("Stage %s: done (%d/%d tasks)", stage.name, len(results), len(records))
        return True

    def _prepare_job(self, stage: Stage, ctx: StageContext, task: TaskSpec) -> Job:
        task_dir = self.layout.task_dir(ctx.stage_dir, task.task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / RESULT_FILE).unlink(missing_ok=True)
        task_file = task_dir / TASK_FILE
        write_task_file(
            task_file,
            pipeline_ref=self.pipeline_ref,
            config_snapshot=self.layout.config_snapshot,
            ctx=ctx,
            stage_name=stage.name,
            task=task,
            task_dir=task_dir,
        )
        return Job(key=task.task_id, task_file=task_file, log_file=task_dir / "worker.log")

    def _record_outcome(
        self,
        state: StateStore,
        stage: Stage,
        ctx: StageContext,
        outcome: JobOutcome,
        max_attempts: int,
    ) -> None:
        task_id = outcome.job.key
        result = read_result(self.layout.task_dir(ctx.stage_dir, task_id))
        if result is not None and result.get("status") == "ok":
            state.set_task(stage.name, task_id, TaskStatus.DONE)
            return
        if outcome.returncode is None:
            state.set_task(
                stage.name,
                task_id,
                TaskStatus.PENDING,
                message="interrupted by stop request",
                attempts_delta=-1,
            )
            return
        if result is None:
            retryable = True
            message = (
                f"worker exited with code {outcome.returncode} without writing a result "
                f"(crash, out-of-memory or kill); see {outcome.job.log_file}"
            )
        else:
            retryable = bool(result.get("retryable", False))
            message = f"{result.get('error_type')}: {result.get('message')}"
        attempts = state.get_task(stage.name, task_id).attempts
        if retryable and attempts < max_attempts:
            log.warning("Task %s/%s failed, will retry: %s", stage.name, task_id, message)
            state.set_task(
                stage.name, task_id, TaskStatus.PENDING, message=f"retrying after: {message}"
            )
        else:
            log.error("Task %s/%s failed: %s", stage.name, task_id, message)
            state.set_task(stage.name, task_id, TaskStatus.FAILED, message=message)

    def _terminate_orphan(self, stage: Stage, record: TaskRecord, task_dir: Path) -> None:
        """Kill a worker left behind by a controller that died, if it is still ours."""
        if record.pid is not None and is_worker_process(record.pid, task_dir / TASK_FILE):
            log.warning(
                "Terminating orphaned worker %d of %s/%s", record.pid, stage.name, record.task_id
            )
            terminate_process_group(record.pid, grace=self.config.execution.stop_grace_period_s)

    def _reconcile_running(self, state: StateStore, stage: Stage, ctx: StageContext) -> None:
        """Tasks left RUNNING by a controller that died: finish or requeue them."""
        for record in state.get_tasks(stage.name):
            if record.status != TaskStatus.RUNNING:
                continue
            task_dir = self.layout.task_dir(ctx.stage_dir, record.task_id)
            self._terminate_orphan(stage, record, task_dir)
            result = read_result(task_dir)
            if result is not None and result.get("status") == "ok":
                state.set_task(stage.name, record.task_id, TaskStatus.DONE)
            else:
                state.set_task(
                    stage.name,
                    record.task_id,
                    TaskStatus.PENDING,
                    message="previous controller stopped while this task was running",
                    attempts_delta=-1,
                )

    def _collect_results(
        self, stage: Stage, ctx: StageContext, records: list[TaskRecord]
    ) -> dict[str, Mapping[str, Any]]:
        results: dict[str, Mapping[str, Any]] = {}
        for record in records:
            if record.status != TaskStatus.DONE:
                continue
            result = read_result(self.layout.task_dir(ctx.stage_dir, record.task_id))
            if result is None or result.get("status") != "ok":
                raise ProjectError(
                    f"Result of completed task {stage.name}/{record.task_id} is missing or "
                    f"corrupt; run with --rerun-from {stage.name}"
                )
            results[record.task_id] = result["result"]
        return results

    def _failure_message(
        self, stage: Stage, ctx: StageContext, failed: list[TaskRecord], total: int
    ) -> str:
        lines = [
            f"Stage '{stage.name}' failed: {len(failed)} of {total} tasks did not succeed "
            f"(tolerated fraction: {stage.max_failed_fraction:.0%})."
        ]
        for record in failed[:5]:
            log_path = self.layout.task_dir(ctx.stage_dir, record.task_id) / "worker.log"
            lines.append(f"  - {record.task_id}: {record.message} (log: {log_path})")
        if len(failed) > 5:
            lines.append(f"  ... and {len(failed) - 5} more; see `gqteashmd status`.")
        lines.append("Fix the cause and run the same command again; failed tasks will be retried.")
        return "\n".join(lines)
