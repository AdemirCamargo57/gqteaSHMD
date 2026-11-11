"""Stage 7: populations, a kinetic fit, hop and energy statistics (P8).

`run_task` reads every selected trajectory's output.h5, applies the inclusion rule and the
common grid (P8 spec section 3), computes the observables with `gqteashmd.analysis` and writes
every output in its task directory; it raises only for unreadable or inconsistent inputs
(non-retryable TaskError naming the `--rerun-from` stage). A physics or statistics stop is
recorded in analysis.json (plan ruling PR6); `finalize` adds the excluded-fraction warning,
copies the files to the stage directory, writes analysis.json and report.md, then raises it.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.analysis import artifacts, energy, grid, hops, kinetics, populations, report
from gqteashmd.config.schema import AnalysisSection, ShmdConfig
from gqteashmd.core.states import states_from_counts
from gqteashmd.errors import StageFailedError, TaskError
from gqteashmd.excite.artifacts import SelectedTrajectory, read_trajectories_json
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.trajectory.outcomes import TrajectoryOutcome
from gqteashmd.trajectory.runner import OUTPUT_FILE
from gqteashmd.trajectory.store import TrajectoryData, read_output_h5
from gqteashmd.util.fileio import (
    read_json,
    write_bytes_atomic,
    write_json_atomic,
    write_text_atomic,
)
from gqteashmd.util.seeds import task_rng

log = logging.getLogger("gqteashmd.stages.analysis")
NORM_TOLERANCE = 1e-6
FILES = ("populations.h5", "populations.csv", "hops.csv")
DYNAMICS_ADVICE = "rerun with --rerun-from dynamics"
TASK = "analysis"
TABLE_ROWS = 25
_TIME_TOLERANCE_FS = 1e-9

FIT_FLAG_MEANINGS = {
    "no_decay": "the fitted population never drops below 0.99 of its initial value; "
    "no lifetime is reported",
    "no_population": "the fitted population is zero at t = 0; no lifetime is reported",
    "tau_beyond_horizon": "the fitted lifetime is longer than the analysed time; "
    "it is an extrapolation",
    "little_decay": "the fitted population is above 0.9 of its initial value at the horizon; "
    "the lifetime is poorly determined",
    "t0_at_upper_bound": "the fitted delay is at the end of the analysed time; "
    "the decay is not resolved",
    "fit_failed": "the least-squares fit failed from every start; no lifetime is reported",
}


def _task_error(message: str) -> TaskError:
    return TaskError(message, retryable=False)


class AnalysisStage(Stage):
    name = "analysis"
    version = 1
    depends_on = ("selection", "dynamics")
    config_sections = ("analysis",)
    required_modules: tuple[str, ...] = ("h5py>=3.9", "scipy>=1.11")
    max_failed_fraction = 0.0

    def enabled(self, config: ShmdConfig) -> bool:
        return config.dynamics is not None and config.analysis.enabled

    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        section = ctx.config.analysis.model_dump(mode="json")
        section.pop("enabled")
        section.pop("warn_excluded_fraction")  # finalize-only
        return {"analysis": section, "project_seed": ctx.config.project.seed}

    def reopen_tasks(
        self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]
    ) -> list[str]:
        """Redo the analysis when tmax_fs (not hashed upstream) moved the end of the grid.

        Lowering tmax_fs reopens no trajectory, so the analysis would otherwise keep the old
        horizon (controller ruling P8-3). Reads only the config and the recorded result.
        """
        dyn = ctx.config.dynamics
        if dyn is None or TASK not in results:
            return []
        n_total = round(dyn.tmax_fs / dyn.dt_fs)
        return [] if results[TASK].get("n_total") == n_total else [TASK]

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        return [TaskSpec(TASK)]

    # ------------------------------------------------------------------------------ task
    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        section = ctx.config.analysis
        dyn = ctx.config.dynamics
        assert dyn is not None
        n_total = round(dyn.tmax_fs / dyn.dt_fs)
        dt_out = dyn.output_every * dyn.dt_fs
        task_dir = Path(task_dir)

        selected = _read_selection(ctx.upstream_dir("selection") / "trajectories.json")
        dynamics_dir = ctx.upstream_dir("dynamics")
        rows = _read_summary_rows(dynamics_dir / "dynamics_summary.json")

        trajectories: dict[str, grid.Trajectory | None] = {}
        files: dict[str, Path] = {}
        first: TrajectoryData | None = None
        for entry in selected:
            path = ctx.layout.task_dir(dynamics_dir, entry.task_id) / OUTPUT_FILE
            if entry.task_id not in rows or not path.exists():
                trajectories[entry.task_id] = None
                continue
            try:
                data = read_output_h5(path)
            except ValueError as exc:
                raise _task_error(f"{exc}; {DYNAMICS_ADVICE}") from exc
            _check_consistent(data, first, entry, dyn.output_every, dyn.dt_fs)
            if first is None:
                first = data
            trajectories[entry.task_id] = _trajectory(entry.task_id, data)
            files[entry.task_id] = path

        if first is not None:
            labels = list(first.labels)
        else:
            electronic = ctx.config.electronic
            singlets = electronic.singlets if electronic is not None else 1
            labels = [str(label) for label in states_from_counts(singlets)]
        fit_states = list(section.fit_states) if section.fit_states else labels[1:]
        unknown = [s for s in fit_states if s not in labels]
        if unknown:
            raise _task_error(
                f"analysis.fit_states {unknown} are not among the computed states {labels} of "
                "output.h5; change [analysis] fit_states (and run with --rerun-from analysis)"
            )
        fit_index = [labels.index(s) for s in fit_states]

        jumps = {
            name: energy.max_jump_eV(t.e_total[t.grid_step <= n_total])
            for name, t in trajectories.items()
            if t is not None
        }
        included, excluded = grid.select(
            [e.task_id for e in selected],
            trajectories,
            jumps,
            energy_jump_eV=section.energy_jump_eV,
            exclude_energy_jumps=section.exclude_energy_jumps,
        )
        for t in included:
            _check_included(t, files[t.task])

        counts = Counter(t.outcome for t in trajectories.values() if t is not None)
        outcomes = {o.value: int(counts.get(o.value, 0)) for o in TrajectoryOutcome}
        outcomes["task_failed"] = sum(1 for t in trajectories.values() if t is None)
        summary = _base_summary(ctx, section, selected, included, excluded, outcomes, dt_out)

        if not included:
            seen = ", ".join(f"{k} {v}" for k, v in outcomes.items() if v) or "none"
            error = (
                f"No trajectory can be analysed: outcomes {seen}; only 'finished' and "
                "'stopped_in_s0' trajectories enter the averages (USER_MANUAL 8.13)"
            )
            if any(e.reason == "energy_jump" for e in excluded):
                error += (
                    "; set [analysis] exclude_energy_jumps = false and run with "
                    "--rerun-from analysis"
                )
            return _stop(task_dir, summary, error, 0, n_total)

        deviation, worst = 0.0, included[0].task
        for t in included:
            d = float(np.abs(t.populations.sum(axis=1) - 1.0).max()) if t.populations.size else 0.0
            if d > deviation:
                deviation, worst = d, t.task
        if deviation > NORM_TOLERANCE:
            error = (
                f"populations of {worst} sum to 1 within {deviation:.3e} only "
                "(tolerance 1e-6): this indicates a program error"
            )
            return _stop(task_dir, summary, error, len(included), n_total)

        try:
            g = grid.on_grid(
                included,
                output_every=dyn.output_every,
                dt_fs=dyn.dt_fs,
                n_total=n_total,
                n_states=len(labels),
            )
        except grid.InconsistentData as exc:
            raise _task_error(f"{exc}; {DYNAMICS_ADVICE}") from exc

        n = len(included)
        p_classical, n_counts = populations.classical_populations(g.active, len(labels))
        p_lo, p_hi = populations.wilson_intervals(n_counts, n)
        rho = populations.quantum_populations(g.populations)
        idx = populations.bootstrap_indices(
            task_rng(ctx.config.project.seed, "analysis", 0), n, section.n_bootstrap
        )
        rho_lo, rho_hi = populations.bootstrap_intervals(g.populations, idx)

        if section.fit_population == "classical":
            per = np.isin(g.active, fit_index).astype(float)
        else:
            per = g.populations[:, :, fit_index].sum(axis=-1)
        target = per.mean(axis=0)
        fitu = kinetics.bootstrap_fit(g.time_fs, per, idx, dt_out=dt_out)
        fit = fitu.fit
        if fit.tau_fs is not None and fit.t0_fs is not None:
            curve = kinetics.model_curve(g.time_fs, fit.a, fit.t0_fs, fit.tau_fs)
        else:
            curve = np.full(g.time_fs.shape, np.nan)
        warnings: list[str] = summary["warnings"]
        for flag in fit.flags:
            meaning = FIT_FLAG_MEANINGS.get(flag)
            warnings.append(f"fit: {flag}" + (f" ({meaning})" if meaning else ""))
        if fitu.bootstrap_failures:
            warnings.append(
                f"fit: bootstrap_failures ({fitu.bootstrap_failures} of {section.n_bootstrap} "
                "bootstrap resamples gave no fit and are left out of the intervals)"
            )

        included_tasks = {t.task for t in included}
        events: list[hops.HopEvent] = []
        for entry in selected:
            traj = trajectories[entry.task_id]
            if traj is not None:
                events += hops.hop_events(
                    traj, included=traj.task in included_tasks, horizon_fs=g.horizon_fs
                )
        hop_stats = hops.hop_statistics(events, included, labels, g.horizon_fs)

        drifts = [
            energy.max_drift_eV(t.e_total[t.time_fs <= g.horizon_fs + _TIME_TOLERANCE_FS])
            for t in included
        ]
        flagged = [
            {"task": name, "energy_jump_eV": jumps[name]}
            for name in (e.task_id for e in selected)
            if name in jumps and jumps[name] > section.energy_jump_eV
        ]

        summary.update(
            {
                "cut": list(g.cut),
                "horizon_fs": g.horizon_fs,
                "fit": {
                    "states": fit_states,
                    "population": section.fit_population,
                    "A": fit.a,
                    "t0_fs": fit.t0_fs,
                    "tau_fs": fit.tau_fs,
                    "t0_interval_fs": fitu.t0_interval_fs,
                    "tau_interval_fs": fitu.tau_interval_fs,
                    "t0_std_fs": fitu.t0_std_fs,
                    "tau_std_fs": fitu.tau_std_fs,
                    "rms": fit.rms,
                    "flags": list(fit.flags),
                    "bootstrap_failures": fitu.bootstrap_failures,
                },
                "hops": hop_stats,
                "energy": {
                    "drift_eV": energy.summary_eV(drifts),
                    "jump_threshold_eV": section.energy_jump_eV,
                    "flagged": flagged,
                },
                "consistency": dict(
                    zip(labels, populations.consistency(p_classical, rho).tolist(), strict=True)
                ),
                "table": _table(g.time_fs, labels, p_classical, rho),
            }
        )

        arrays: dict[str, Any] = {
            "time_fs": g.time_fs,
            "labels": labels,
            "classical": p_classical,
            "classical_lower": p_lo,
            "classical_upper": p_hi,
            "quantum": rho,
            "quantum_lower": rho_lo,
            "quantum_upper": rho_hi,
            "fit_target": target,
            "fit_curve": curve,
        }
        artifacts.write_populations_h5(
            task_dir / "populations.h5",
            **arrays,
            attrs={
                "n_included": int(n),
                "n_selected": len(selected),
                "horizon_fs": float(g.horizon_fs),
                "dt_out_fs": float(dt_out),
                "fit_population": section.fit_population,
                "fit_states": [str(s) for s in fit_states],
            },
        )
        comments = [
            "gqteashmd analysis: classical (P) and quantum (rho) populations with 95 % "
            "intervals, the fit target and the fitted curve",
            "units: time_fs in fs; populations are fractions (0-1)",
            f"n_included = {n}",
            f"horizon_fs = {g.horizon_fs:.6f}",
            f"fit_population = {section.fit_population}",
            f"fit_states = {' '.join(fit_states)}",
        ]
        write_text_atomic(
            task_dir / "populations.csv",
            artifacts.format_populations_csv(**arrays, comments=comments),
        )
        write_text_atomic(task_dir / "hops.csv", artifacts.format_hops_csv(events, labels))
        write_json_atomic(task_dir / "analysis.json", report.json_safe(summary))
        return {
            "stage_failed": False,
            "n_total": int(n_total),
            "n_included": int(n),
            "horizon_fs": float(g.horizon_fs),
            "tau_fs": fit.tau_fs,
        }

    # -------------------------------------------------------------------------- finalize
    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        task_dir = ctx.layout.task_dir(ctx.stage_dir, TASK)
        summary = read_json(task_dir / "analysis.json")
        n_selected = int(summary["n_selected"])
        fraction = (n_selected - int(summary["n_included"])) / n_selected if n_selected else 0.0
        limit = ctx.config.analysis.warn_excluded_fraction
        if fraction > limit:
            summary["warnings"].append(
                f"{fraction:.0%} of the selected trajectories are excluded (more than "
                f"warn_excluded_fraction = {limit}): possible selection bias, see 'excluded'"
            )
        for name in FILES:
            source = task_dir / name
            if source.exists():
                write_bytes_atomic(ctx.stage_dir / name, source.read_bytes())
        write_json_atomic(ctx.stage_dir / "analysis.json", summary)
        write_text_atomic(ctx.stage_dir / "report.md", report.analysis_markdown(summary))
        for warning in summary["warnings"]:
            log.warning("Stage analysis: %s", warning)
        if summary["stage_failed"]:
            raise StageFailedError(str(summary["error"]))


# ---------------------------------------------------------------------------- helpers
def _read_selection(path: Path) -> list[SelectedTrajectory]:
    try:
        return read_trajectories_json(path)
    except ValueError as exc:
        raise _task_error(f"{exc}; rerun with --rerun-from selection") from exc


def _read_summary_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    try:
        summary = read_json(path)
        return {str(r["task"]): r for r in summary["trajectories"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise _task_error(
            f"{path.name} is missing or malformed ({exc!r}); {DYNAMICS_ADVICE}"
        ) from exc


def _check_consistent(
    data: TrajectoryData,
    first: TrajectoryData | None,
    entry: SelectedTrajectory,
    output_every: int,
    dt_fs: float,
) -> None:
    """Plan ruling PR5, plus the task name and the initial state (every file; the record at
    t = 0 is checked for included trajectories only, in `_check_included`)."""

    def fail(field: str, detail: str) -> TaskError:
        return _task_error(f"{data.path}: {field} {detail}; {DYNAMICS_ADVICE}")

    try:
        labels, every, dt = data.labels, data.output_every, data.dt_fs
        task, initial = data.task, data.initial_state
    except (TypeError, ValueError) as exc:
        raise fail("attributes", f"are malformed ({exc})") from exc
    if first is not None and labels != first.labels:
        raise fail(
            "state_labels", f"{list(labels)} differ from {list(first.labels)} of {first.path}"
        )
    if every != output_every:
        raise fail("output_every", f"{every} differs from [dynamics] output_every = {output_every}")
    if not abs(dt - dt_fs) <= 1e-9 * dt_fs:
        raise fail("dt_fs", f"{dt} differs from [dynamics] dt_fs = {dt_fs}")
    if task != entry.task_id:
        raise fail("task", f"{task!r} is not {entry.task_id!r}")
    if initial not in labels:
        raise fail("initial_state", f"{initial!r} is not among the state_labels {list(labels)}")
    if data["populations"].ndim != 2 or data["populations"].shape[1] != len(labels):
        raise fail("populations", f"do not have {len(labels)} states per record")


def _check_included(t: grid.Trajectory, path: Path) -> None:
    """Checks of an included trajectory (spec section 8). An excluded one may have no record
    at all (a failure at step 0, P7b ruling R6)."""
    if not bool(np.any((t.grid_step == 0) & (t.ticks == 0))):
        raise _task_error(
            f"{path}: records have no record at t = 0 (outcome {t.outcome!r}); {DYNAMICS_ADVICE}"
        )
    for name, values in (
        ("e_total", t.e_total),
        ("populations", t.populations),
        ("active", t.active),
    ):
        if not bool(np.all(np.isfinite(values))):
            raise _task_error(
                f"{path}: non-finite {name} in an included trajectory; {DYNAMICS_ADVICE}"
            )


def _trajectory(task: str, data: TrajectoryData) -> grid.Trajectory:
    return grid.Trajectory(
        task=task,
        outcome=data.outcome,
        initial_state=data.labels.index(data.initial_state),
        grid_step=np.asarray(data["grid_step"], dtype=np.int64),
        ticks=np.asarray(data["ticks"], dtype=np.int64),
        time_fs=np.asarray(data["time_fs"], dtype=float),
        active=np.asarray(data["active"], dtype=np.int64),
        hop_from=np.asarray(data["hop_from"], dtype=np.int64),
        hop_to=np.asarray(data["hop_to"], dtype=np.int64),
        frustrated=np.asarray(data["frustrated"]) != 0,
        e_total=np.asarray(data["e_total"], dtype=float),
        populations=np.asarray(data["populations"], dtype=float),
    )


def _base_summary(
    ctx: StageContext,
    section: AnalysisSection,
    selected: Sequence[SelectedTrajectory],
    included: Sequence[grid.Trajectory],
    excluded: Sequence[grid.Excluded],
    outcomes: Mapping[str, int],
    dt_out: float,
) -> dict[str, Any]:
    """Every key of analysis.json; `None` for what is not computed (yet)."""
    return {
        "schema_version": 1,
        "n_selected": len(selected),
        "n_included": len(included),
        "outcomes": dict(outcomes),
        "included": [t.task for t in included],
        "excluded": [
            {
                "task": e.task,
                "reason": e.reason,
                "t_final_fs": e.t_final_fs,
                "energy_jump_eV": e.energy_jump_eV,
            }
            for e in excluded
        ],
        "cut": None,
        "horizon_fs": None,
        "dt_out_fs": dt_out,
        "fit": None,
        "hops": None,
        "energy": None,
        "consistency": None,
        "table": None,
        "settings": {
            **section.model_dump(mode="json"),
            "project_seed": ctx.config.project.seed,
        },
        "warnings": [],
        "stage_failed": False,
        "error": None,
    }


def _stop(
    task_dir: Path, summary: dict[str, Any], error: str, n_included: int, n_total: int
) -> dict[str, Any]:
    """Record a stop (raised by `finalize`, PR6); stale outputs of an earlier attempt go.

    The result carries `n_total` like every result (`reopen_tasks`, ruling P8-3)."""
    for name in FILES:
        (task_dir / name).unlink(missing_ok=True)
    summary["stage_failed"] = True
    summary["error"] = error
    write_json_atomic(task_dir / "analysis.json", report.json_safe(summary))
    return {"stage_failed": True, "n_total": int(n_total), "n_included": n_included}


def _table(
    time_fs: np.ndarray, labels: Sequence[str], classical: np.ndarray, quantum: np.ndarray
) -> list[dict[str, Any]]:
    """At most about TABLE_ROWS rows: every ceil(G / TABLE_ROWS)-th grid point and the last."""
    g = int(time_fs.size)
    if g == 0:
        return []
    every = max(1, math.ceil(g / TABLE_ROWS))
    rows = list(range(0, g, every))
    if rows[-1] != g - 1:
        rows.append(g - 1)
    return [
        {
            "time_fs": float(time_fs[m]),
            "P": {lab: float(classical[m, j]) for j, lab in enumerate(labels)},
            "rho": {lab: float(quantum[m, j]) for j, lab in enumerate(labels)},
        }
        for m in rows
    ]
