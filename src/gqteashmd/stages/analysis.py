"""Stage 7: populations, a kinetic fit, hop and energy statistics (P8; censored analysis).

`run_task` reads every selected trajectory's output.h5, applies the two inclusion rules (the
complete rule of P8 spec section 3 and the censored rule of the censored analysis spec, section
2.1) on the common grid (with no complete trajectory the horizon is capped at the last grid
step at which any trajectory is observed, spec 2026-11-26, 3.1), computes both estimators with
`gqteashmd.analysis` and writes every output in its task directory; the estimator in use
(`[analysis] estimator`) fills the top-level keys and the files, the other one is reported
beside it. It raises only for unreadable or inconsistent inputs (non-retryable TaskError naming
the `--rerun-from` stage). A physics or statistics stop is recorded in analysis.json (plan
ruling PR6); `finalize` adds the excluded-fraction warning, copies the files to the stage
directory, writes analysis.json and report.md, then raises it.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.analysis import (
    artifacts,
    censoring,
    energy,
    estimators,
    grid,
    hops,
    kinetics,
    populations,
    report,
)
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
HOLD_RULE = "lost trajectories held in S0 from their censoring step"
FILES = ("populations.h5", "populations.csv", "hops.csv")
DYNAMICS_ADVICE = "rerun with --rerun-from dynamics"
TASK = "analysis"
TABLE_ROWS = 25
NO_DATA_AFTER_START = "no trajectory is complete and none is observed after t = 0"
# P9b (spec 6): the smallest of N = 20, 50, 100 whose 95 % tau interval covered the true tau in
# at least 90 % of 400 synthetic repetitions (tools/validation/tau_coverage.py); fewer included
# trajectories: warn and `small_ensemble` (ruling P8-5)
SMALL_ENSEMBLE = 50
SMALL_ENSEMBLE_COVERAGE = "coverage 0.895 at N = 20, 0.9275 at N = 50, 0.935 at N = 100"
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
    # 4: the horizon cap without complete trajectories and the hold-in-S0 estimate (spec 2026-11-26)
    version = 4
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
        use_censored = section.estimator == "censored"

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
            n_singlets, n_triplets = _spin_counts(first)
        else:
            electronic = ctx.config.electronic
            n_singlets = electronic.singlets if electronic is not None else 1
            n_triplets = electronic.triplets if electronic is not None else 0
            labels = [str(label) for label in states_from_counts(n_singlets, n_triplets)]
        fit_states = (
            list(section.fit_states)
            if section.fit_states
            else _default_fit_states(labels, n_singlets, n_triplets)
        )
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
        tasks = [e.task_id for e in selected]
        complete, excluded_complete = grid.select(
            tasks,
            trajectories,
            jumps,
            energy_jump_eV=section.energy_jump_eV,
            exclude_energy_jumps=section.exclude_energy_jumps,
        )
        used, censor_steps, censored, excluded_censored = censoring.select_censored(
            tasks,
            trajectories,
            output_every=dyn.output_every,
            n_total=n_total,
            dt_fs=dyn.dt_fs,
            energy_jump_eV=section.energy_jump_eV if section.exclude_energy_jumps else None,
        )
        # Both estimators are computed, so every trajectory either uses is checked. `used`
        # holds every complete trajectory except a `finished` file without records, which
        # must still fail its check.
        checked = list({t.task: t for t in (*complete, *used)}.values())
        for t in checked:
            _check_included(t, files[t.task])
        h = grid.horizon_steps(complete, n_total)  # ruling PR7
        # spec 2026-11-26, 3.1: with no complete trajectory, the last observed grid step,
        # never beyond n_total (final review I1: records past a lowered tmax_fs)
        horizon_capped = not complete and bool(censor_steps)
        if horizon_capped:
            h = min(n_total, max(censor_steps.values()))
        jump_censored = {c.task for c in censored if c.reason == "energy_jump"}
        censor_steps = {k: v for k, v in censor_steps.items() if v < h}  # ruling PR5
        censored = [c for c in censored if c.task in censor_steps]
        in_use = used if use_censored else complete
        excluded = excluded_censored if use_censored else excluded_complete

        counts = Counter(t.outcome for t in trajectories.values() if t is not None)
        outcomes = {o.value: int(counts.get(o.value, 0)) for o in TrajectoryOutcome}
        outcomes["task_failed"] = sum(1 for t in trajectories.values() if t is None)
        summary = _base_summary(
            ctx,
            section,
            selected,
            in_use,
            excluded,
            complete,
            censored,
            outcomes,
            dt_out,
            horizon_capped=horizon_capped,
            triplets=n_triplets > 0,
        )

        if not in_use:
            error = _nothing_to_analyse(outcomes, excluded, use_censored)
            return _stop(task_dir, summary, error, 0, n_total)
        if horizon_capped and h == 0:
            return _stop(task_dir, summary, NO_DATA_AFTER_START, len(in_use), n_total)

        deviation, worst = 0.0, checked[0].task
        for t in checked:
            d = float(np.abs(t.populations.sum(axis=1) - 1.0).max()) if t.populations.size else 0.0
            if d > deviation:
                deviation, worst = d, t.task
        if deviation > NORM_TOLERANCE:
            error = (
                f"populations of {worst} sum to 1 within {deviation:.3e} only "
                "(tolerance 1e-6): this indicates a program error"
            )
            return _stop(task_dir, summary, error, len(in_use), n_total)

        def _grid(trajs: Sequence[grid.Trajectory], at: Mapping[str, int] | None) -> grid.GridData:
            try:
                return grid.on_grid(
                    trajs,
                    output_every=dyn.output_every,
                    dt_fs=dyn.dt_fs,
                    n_total=n_total,
                    n_states=len(labels),
                    censored_at=at,
                    horizon=h,
                )
            except grid.InconsistentData as exc:
                raise _task_error(f"{exc}; {DYNAMICS_ADVICE}") from exc

        seed = ctx.config.project.seed
        g_complete = _grid(complete, None) if complete else None
        g_censored = _grid(used, censor_steps)
        est_complete = (
            estimators.complete_estimate(
                g_complete,
                fit_index,
                fit_population=section.fit_population,
                rng=task_rng(seed, "analysis", 0),
                n_bootstrap=section.n_bootstrap,
                dt_out=dt_out,
            )
            if g_complete is not None
            else None
        )
        est_censored = estimators.censored_estimate(
            g_censored,
            fit_index,
            rng=task_rng(seed, "analysis", 1),  # ruling PR3
            n_bootstrap=section.n_bootstrap,
            dt_out=dt_out,
        )
        # spec 2026-11-26, 3.2: reported only; stream 2, so streams 0 and 1 are unchanged
        est_hold = (
            estimators.hold_in_s0_estimate(
                g_censored,
                fit_index,
                fit_population=section.fit_population,
                rng=task_rng(seed, "analysis", 2),
                n_bootstrap=section.n_bootstrap,
                dt_out=dt_out,
            )
            if censor_steps
            else None
        )
        est = est_censored if use_censored else est_complete
        assert est is not None  # `in_use` is not empty
        g = g_censored if use_censored else g_complete
        assert g is not None
        n = est.n
        fit_population = "classical" if use_censored else section.fit_population

        warnings: list[str] = summary["warnings"]
        if n < SMALL_ENSEMBLE:
            warnings.append(_small_ensemble_warning(n, len(censored), use_censored))
        fitu = est.fit
        fit = fitu.fit
        if fit.tau_fs is not None and fit.t0_fs is not None:
            curve = kinetics.model_curve(g.time_fs, fit.a, fit.t0_fs, fit.tau_fs)
        else:
            curve = np.full(g.time_fs.shape, np.nan)
        for flag in fit.flags:
            meaning = FIT_FLAG_MEANINGS.get(flag)
            warnings.append(f"fit: {flag}" + (f" ({meaning})" if meaning else ""))
        if fitu.bootstrap_failures:
            warnings.append(
                f"fit: bootstrap_failures ({fitu.bootstrap_failures} of {section.n_bootstrap} "
                "bootstrap resamples gave no fit and are left out of the intervals)"
            )
        # final review I2, ruling F1: name the identity rows that carry probability; then,
        # with no complete trajectory, the capped horizon
        if est_censored.empty_rows:
            warnings.append(
                f"the censored estimator held {est_censored.empty_rows} state probabilities over "
                "intervals with no trajectory at risk (empty_rows = "
                f"{est_censored.empty_rows}); the censored curve and its fit are not supported "
                "by data there"
            )
        if horizon_capped:
            warnings.append(
                f"no trajectory is complete: the analysis horizon is capped at "
                f"{g_censored.horizon_fs:g} fs, the last time any trajectory is observed"
            )

        # hops, drifts and consistency stay over the complete trajectories (ruling PR4)
        complete_tasks = {t.task for t in complete}
        events: list[hops.HopEvent] = []
        for entry in selected:
            traj = trajectories[entry.task_id]
            if traj is not None:
                events += hops.hop_events(
                    traj, included=traj.task in complete_tasks, horizon_fs=g.horizon_fs
                )
        hop_stats = hops.hop_statistics(events, complete, labels, g.horizon_fs)
        if n_triplets:  # spec 4.3: like the hops, over the complete trajectories
            changes: list[hops.Transition] = []
            for entry in selected:
                traj = trajectories[entry.task_id]
                if traj is not None:
                    changes += hops.transitions(
                        traj, included=traj.task in complete_tasks, horizon_fs=g.horizon_fs
                    )
            summary["transitions"] = hops.transition_statistics(
                changes, complete, labels, g.horizon_fs
            )

        drifts = [
            energy.max_drift_eV(t.e_total[t.time_fs <= g.horizon_fs + _TIME_TOLERANCE_FS])
            for t in complete
        ]
        flagged = [
            {"task": name, "energy_jump_eV": jumps[name]}
            for name in (e.task_id for e in selected)
            if name in jumps and jumps[name] > section.energy_jump_eV
        ]
        # a trajectory censored before its jump contributes no data with the jump
        in_use_tasks = {t.task for t in in_use} - jump_censored
        k = sum(1 for f in flagged if f["task"] in in_use_tasks)
        if k:
            warnings.append(
                f"{k} included trajectories have an energy jump above energy_jump_eV = "
                f"{section.energy_jump_eV} eV (see energy.flagged); consider "
                "exclude_energy_jumps = true"
            )

        consistency = None
        if est_complete is not None:
            values = populations.consistency(est_complete.classical, est_complete.quantum)
            consistency = dict(zip(labels, values.tolist(), strict=True))
        summary.update(
            {
                "cut": list(g.cut),
                "horizon_fs": g.horizon_fs,
                "fit": _fit_summary(est, fit_states, fit_population),
                "estimators": {
                    "complete": None
                    if est_complete is None or g_complete is None
                    else _estimate_summary(
                        est_complete, g_complete, labels, fit_states, section.fit_population
                    ),
                    "censored": _estimate_summary(
                        est_censored, g_censored, labels, fit_states, "classical"
                    ),
                    "hold_in_s0": None
                    if est_hold is None
                    else _estimate_summary(
                        est_hold, g_censored, labels, fit_states, section.fit_population
                    ),
                },
                "tau_range_fs": _tau_range(
                    {"complete": est_complete, "censored": est_censored, "hold_in_s0": est_hold}
                ),
                "hops": hop_stats,
                "energy": {
                    "drift_eV": energy.summary_eV(drifts),
                    "jump_threshold_eV": section.energy_jump_eV,
                    "flagged": flagged,
                },
                "consistency": consistency,
                "table": _table(g.time_fs, labels, est.classical, est.quantum),
            }
        )

        arrays: dict[str, Any] = {
            "time_fs": g.time_fs,
            "labels": labels,
            "classical": est.classical,
            "classical_lower": est.classical_lower,
            "classical_upper": est.classical_upper,
            "quantum": est.quantum,
            "quantum_lower": est.quantum_lower,
            "quantum_upper": est.quantum_upper,
            "fit_target": est.target,
            "fit_curve": curve,
        }
        extra, extra_blocks = _extra_curves(
            est_complete if use_censored else est_censored, est_hold
        )
        artifacts.write_populations_h5(
            task_dir / "populations.h5",
            **arrays,
            attrs={
                "estimator": section.estimator,
                "n_included": int(n),
                "n_complete": len(complete),
                "n_censored": len(censored),
                "n_selected": len(selected),
                "horizon_fs": float(g.horizon_fs),
                "dt_out_fs": float(dt_out),
                "fit_population": fit_population,
                "fit_states": [str(s) for s in fit_states],
            },
            extra=extra,
        )
        comments = [
            "gqteashmd analysis: classical (P) and quantum (rho) populations with 95 % "
            "intervals, the fit target and the fitted curve",
            "units: time_fs in fs; populations are fractions (0-1)",
            f"estimator = {section.estimator}",
            f"n_included = {n}",
            f"horizon_fs = {g.horizon_fs:.6f}",
            f"fit_population = {fit_population}",
            f"fit_states = {' '.join(fit_states)}",
        ]
        write_text_atomic(
            task_dir / "populations.csv",
            artifacts.format_populations_csv(
                **arrays, comments=comments, extra_blocks=extra_blocks
            ),
        )
        write_text_atomic(task_dir / "hops.csv", artifacts.format_hops_csv(events, labels))
        write_json_atomic(task_dir / "analysis.json", report.json_safe(summary))
        return {
            "stage_failed": False,
            "estimator": section.estimator,
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
        limit = ctx.config.analysis.warn_excluded_fraction
        if summary["estimator"] == "censored":
            n_not_complete = n_selected - (int(summary["n_included"]) - int(summary["n_censored"]))
            fraction = n_not_complete / n_selected if n_selected else 0.0
            if fraction > limit:
                summary["warnings"].append(
                    f"{fraction:.0%} of the selected trajectories are not complete (more than "
                    f"warn_excluded_fraction = {limit}): {summary['n_censored']} censored, "
                    f"{len(summary['excluded'])} without data; the censored estimator assumes "
                    "that a censored trajectory would have gone on like those still running in "
                    "its state (USER_MANUAL 8.13)"
                )
        else:
            n_included = int(summary["n_included"])
            fraction = (n_selected - n_included) / n_selected if n_selected else 0.0
            if fraction > limit:
                summary["warnings"].append(
                    f"{fraction:.0%} of the selected trajectories are excluded (more than "
                    f"warn_excluded_fraction = {limit}): possible selection bias, see 'excluded'"
                )
        for name in FILES:  # the stage directory mirrors the task: no stale copy survives
            source = task_dir / name
            if source.exists():
                write_bytes_atomic(ctx.stage_dir / name, source.read_bytes())
            else:
                (ctx.stage_dir / name).unlink(missing_ok=True)
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
    if not _spin_counts_fit(data):  # schema 2 only; a schema-1 file is checked as before
        raise fail(
            "n_singlets and n_triplets",
            f"({data.attrs.get('n_singlets')!r}, {data.attrs.get('n_triplets')!r}) do not fit "
            f"the state_labels {list(labels)}",
        )
    if data["populations"].ndim != 2 or data["populations"].shape[1] != len(labels):
        raise fail("populations", f"do not have {len(labels)} states per record")


def _spin_counts(data: TrajectoryData) -> tuple[int, int]:
    """(singlets including S0, triplets) of an output.h5: its `n_singlets`/`n_triplets`
    attributes (schema 2, spec 4.2); a file without triplets has neither, every state is a
    singlet."""
    n_triplets = int(data.attrs.get("n_triplets", 0))
    if not n_triplets:
        return len(data.labels), 0
    return int(data.attrs.get("n_singlets", 0)), n_triplets


def _spin_counts_fit(data: TrajectoryData) -> bool:
    """True for a file without spin counts (schema 1) or whose counts give its labels."""
    if "n_singlets" not in data.attrs and "n_triplets" not in data.attrs:
        return True
    try:
        n_singlets, n_triplets = _spin_counts(data)
        return [str(s) for s in states_from_counts(n_singlets, n_triplets)] == list(data.labels)
    except (TypeError, ValueError):
        return False


def _default_fit_states(labels: Sequence[str], n_singlets: int, n_triplets: int) -> list[str]:
    """The fit default: every excited state without triplets (P8), the excited singlets
    S1 ... S(n-1) with triplets (spec 4.3, owner decision 6)."""
    return list(labels[1:n_singlets]) if n_triplets else list(labels[1:])


def _check_included(t: grid.Trajectory, path: Path) -> None:
    """Checks of an included (complete or censored) trajectory (P8 spec section 8; censored
    analysis spec 2.1). An excluded one may have no record at all (a failure at step 0, P7b
    ruling R6)."""
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


def _nothing_to_analyse(
    outcomes: Mapping[str, int], excluded: Sequence[grid.Excluded], use_censored: bool
) -> str:
    seen = ", ".join(f"{k} {v}" for k, v in outcomes.items() if v) or "none"
    if use_censored:
        return (
            f"No trajectory can be analysed: outcomes {seen}; no selected trajectory has a "
            "record (USER_MANUAL 8.13)"
        )
    error = (
        f"No trajectory can be analysed: outcomes {seen}; only 'finished' and "
        "'stopped_in_s0' trajectories enter the averages (USER_MANUAL 8.13)"
    )
    if any(e.reason == "energy_jump" for e in excluded):
        error += "; set [analysis] exclude_energy_jumps = false and run with --rerun-from analysis"
    return error


def _small_ensemble_warning(n: int, n_censored: int, use_censored: bool) -> str:
    if use_censored:
        return (
            f"only {n} trajectories are used ({n - n_censored} complete, {n_censored} censored; "
            f"fewer than {SMALL_ENSEMBLE}): populations, intervals and the fit are not "
            f"statistically reliable (measured 95 % tau interval {SMALL_ENSEMBLE_COVERAGE}, "
            "for complete ensembles)"
        )
    return (
        f"only {n} trajectories are included (fewer than {SMALL_ENSEMBLE}): populations, "
        "intervals and the fit are not statistically reliable (measured 95 % "
        f"tau interval {SMALL_ENSEMBLE_COVERAGE})"
    )


def _base_summary(
    ctx: StageContext,
    section: AnalysisSection,
    selected: Sequence[SelectedTrajectory],
    in_use: Sequence[grid.Trajectory],
    excluded: Sequence[grid.Excluded],
    complete: Sequence[grid.Trajectory],
    censored: Sequence[censoring.Censored],
    outcomes: Mapping[str, int],
    dt_out: float,
    *,
    horizon_capped: bool,
    triplets: bool = False,
) -> dict[str, Any]:
    """Every key of analysis.json; `None` for what is not computed (yet). The top-level
    `n_included`, `included`, `excluded`, `small_ensemble`, `fit` and `table` belong to the
    estimator in use; `estimators` holds both (ruling PR4). `transitions` exists only with
    triplets (spec 4.3), so a singlet project's analysis.json is unchanged."""
    return {
        "schema_version": 3,
        "estimator": section.estimator,
        "n_selected": len(selected),
        "n_included": len(in_use),
        "n_complete": len(complete),
        "n_censored": len(censored),
        "small_ensemble": len(in_use) < SMALL_ENSEMBLE,
        "outcomes": dict(outcomes),
        "included": [t.task for t in in_use],
        "excluded": [
            {
                "task": e.task,
                "reason": e.reason,
                "t_final_fs": e.t_final_fs,
                "energy_jump_eV": e.energy_jump_eV,
            }
            for e in excluded
        ],
        "censored": [
            {
                "task": c.task,
                "outcome": c.outcome,
                "reason": c.reason,
                "censored_at_fs": c.time_fs,
            }
            for c in censored
        ],
        "cut": None,
        "horizon_fs": None,
        "horizon_capped": horizon_capped,
        "dt_out_fs": dt_out,
        "fit": None,
        "estimators": None,
        "tau_range_fs": None,
        "hops": None,
        **({"transitions": None} if triplets else {}),
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


def _fit_summary(
    est: estimators.Estimate, fit_states: Sequence[str], population: str
) -> dict[str, Any]:
    fitu, fit = est.fit, est.fit.fit
    return {
        "states": list(fit_states),
        "population": population,
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
    }


def _estimate_summary(
    est: estimators.Estimate,
    g: grid.GridData,
    labels: Sequence[str],
    fit_states: Sequence[str],
    population: str,
) -> dict[str, Any]:
    out = {
        "n": est.n,
        "small_ensemble": est.n < SMALL_ENSEMBLE,
        "fit": _fit_summary(est, fit_states, population),
        "table": _table(g.time_fs, labels, est.classical, est.quantum),
    }
    if est.name == "censored":
        out["empty_rows"] = est.empty_rows
        out["quantum_rule"] = "at_risk_average"
    if est.name == "hold_in_s0":
        out["rule"] = HOLD_RULE
    return out


def _extra_curves(
    *others: estimators.Estimate | None,
) -> tuple[dict[str, np.ndarray], list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]]:
    """The curves of the estimators not in use, under their names as prefix (ruling PR6; spec
    2026-11-26, 3.3); nothing for an estimator without trajectories."""
    extra: dict[str, np.ndarray] = {}
    extra_blocks: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for other in others:
        if other is None:
            continue
        prefix = other.name  # "complete", "censored" or "hold_in_s0"
        extra |= {
            f"{prefix}_classical": other.classical,
            f"{prefix}_classical_lower": other.classical_lower,
            f"{prefix}_classical_upper": other.classical_upper,
            f"{prefix}_quantum": other.quantum,
            f"{prefix}_quantum_lower": other.quantum_lower,
            f"{prefix}_quantum_upper": other.quantum_upper,
        }
        extra_blocks += [
            (f"{prefix}_P", other.classical, other.classical_lower, other.classical_upper),
            (f"{prefix}_rho", other.quantum, other.quantum_lower, other.quantum_upper),
        ]
    return extra, extra_blocks


def _tau_range(blocks: Mapping[str, estimators.Estimate | None]) -> dict[str, Any] | None:
    """Spec 2026-11-26, 3.3: over the estimators that exist and have a fitted tau."""
    taus = [
        (name, float(e.fit.fit.tau_fs))
        for name, e in blocks.items()
        if e is not None and e.fit.fit.tau_fs is not None
    ]
    if len(taus) < 2:
        return None
    low = min(taus, key=lambda x: x[1])
    high = max(taus, key=lambda x: x[1])
    return {"min": low[1], "max": high[1], "min_estimator": low[0], "max_estimator": high[0]}


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
    return {
        "stage_failed": True,
        "estimator": str(summary["estimator"]),
        "n_total": int(n_total),
        "n_included": n_included,
    }


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
