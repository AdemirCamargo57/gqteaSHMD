"""One trajectory of the dynamics stage (spec sections 3.2-6, plan rulings R2-R10).

The core's ``driver.step`` advances one (sub)step; this module owns everything around it: the
integer time grid, step rejection with halving, outcomes, the step log, per-step restart
generations, resume and extension. It reimplements no part of the step order.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.core.units import AU_TIME_IN_FS, HARTREE_IN_EV
from gqteashmd.dynamics.driver import initial_state, kinetic_energy, step
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import RandomSource, source_from_state
from gqteashmd.dynamics.records import StepRecord, TrajectoryState
from gqteashmd.dynamics.restart import load_state, save_state
from gqteashmd.errors import ElectronicStructureError
from gqteashmd.trajectory.model import BackendModel
from gqteashmd.trajectory.outcomes import TrajectoryOutcome
from gqteashmd.trajectory.store import RestartStore, StepLog, log_row, write_output_h5
from gqteashmd.util.fileio import read_json, write_json_atomic

log = logging.getLogger("gqteashmd.trajectory.runner")

STATE_FILE = "state.npz"
ELECTRONIC_FILE = "electronic.npz"
META_FILE = "meta.json"
LOG_FILE = "steps.log"
OUTPUT_FILE = "output.h5"
PROGRESS_FILE = "progress.json"
GRID_TOLERANCE = 1e-9
MAX_HALVINGS = 6


class RestartError(ValueError):
    """The trajectory's restart or step log cannot be used to resume (raised by ``run``)."""


@dataclass(frozen=True)
class RunSettings:
    options: DynamicsOptions
    tmax_au: float
    overlap_threshold: float = 0.5
    max_step_halvings: int = 3
    output_every: int = 1
    stop_after_s0_au: float | None = None
    energy_jump_limit_au: float | None = None

    def __post_init__(self) -> None:
        ratio = float(self.tmax_au) / self.options.dt_au
        if (
            not math.isfinite(ratio)
            or round(ratio) < 1
            or abs(ratio - round(ratio)) > (GRID_TOLERANCE * ratio)
        ):
            raise ValueError(
                f"tmax_au = {self.tmax_au} must be an integer multiple (>= 1) of "
                f"dt_au = {self.options.dt_au}"
            )
        if not 0 < self.overlap_threshold < 1:
            raise ValueError(f"overlap_threshold must be in (0, 1); got {self.overlap_threshold}")
        if not 0 <= self.max_step_halvings <= MAX_HALVINGS:
            raise ValueError(
                f"max_step_halvings must be in 0..{MAX_HALVINGS}; got {self.max_step_halvings}"
            )
        if self.output_every < 1:
            raise ValueError(f"output_every must be >= 1; got {self.output_every}")
        if self.stop_after_s0_au is not None and not self.stop_after_s0_au > 0:
            raise ValueError(f"stop_after_s0_au must be > 0; got {self.stop_after_s0_au}")
        if self.energy_jump_limit_au is not None and not (
            math.isfinite(self.energy_jump_limit_au) and self.energy_jump_limit_au > 0
        ):
            raise ValueError(
                f"energy_jump_limit_au must be finite and > 0, or None; got "
                f"{self.energy_jump_limit_au}"
            )

    @property
    def n_grid_steps(self) -> int:
        return int(round(float(self.tmax_au) / self.options.dt_au))

    @property
    def ticks_per_step(self) -> int:
        return 1 << self.max_step_halvings


@dataclass(frozen=True)
class InitialConditions:
    q: np.ndarray
    v: np.ndarray
    active: int


@dataclass
class Progress:
    grid_step: int = 0
    ticks: int = 0
    depth: int = 0
    n_steps: int = 0
    n_hops: int = 0
    n_frustrated: int = 0
    n_rejections: int = 0
    n_flagged: int = 0
    pending_rejections: int = 0
    max_drift_au: float = 0.0
    s0_time_au: float = 0.0
    log_records: int = 0
    outcome: str | None = None
    message: str | None = None
    rejections_by_signal: dict[str, int] = field(default_factory=dict)
    resumes: list[str] = field(default_factory=list)
    last_e_total_au: float | None = None

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        if not math.isfinite(self.max_drift_au):
            data["max_drift_au"] = None  # strict JSON: None means non-finite
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Progress:
        values = dict(data)
        if values.get("max_drift_au") is None:
            values["max_drift_au"] = math.inf
        return cls(**values)


class TrajectoryRunner:
    def __init__(
        self,
        model: BackendModel,
        settings: RunSettings,
        directory: Path,
        *,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.directory = Path(directory)
        self.attrs = dict(attrs or {})
        self.restarts = RestartStore(self.directory / "restart")
        self.log = StepLog(self.directory / LOG_FILE, model.n_states, model.n_dof)

    # ---------------------------------------------------------------- public
    def run(self, initial: InitialConditions, source: RandomSource) -> dict[str, Any]:
        started = time.perf_counter()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.restarts.clean_temporary()
        resumed = self._resume()
        state: TrajectoryState | None
        if resumed is None:
            state, source, progress = self._start(initial, source)
        else:
            state, source, progress = resumed
            if (
                progress.outcome == TrajectoryOutcome.FINISHED
                and progress.grid_step < self.settings.n_grid_steps
            ):
                progress.outcome = None  # O16: tmax was raised after the trajectory finished
                progress.message = None
        while progress.outcome is None and state is not None:
            if progress.ticks == 0 and progress.grid_step >= self.settings.n_grid_steps:
                progress.outcome = TrajectoryOutcome.FINISHED
                self._save(state, source, progress)
                break
            state, source = self._advance(state, source, progress)
        return self._finish(progress, time.perf_counter() - started)

    # ---------------------------------------------------------------- start and resume
    def _start(
        self, initial: InitialConditions, source: RandomSource
    ) -> tuple[TrajectoryState | None, RandomSource, Progress]:
        self.log.truncate(0)
        progress = Progress()
        self.model.backend.reset()
        self.model.active = int(initial.active)
        try:
            state = initial_state(
                self.model,
                initial.q,
                initial.v,
                int(initial.active),
                options=self.settings.options,
            )
        except ElectronicStructureError as exc:
            if exc.retryable:
                raise
            progress.outcome = TrajectoryOutcome.ELECTRONIC_FAILURE
            progress.message = f"step 0: {exc}"
            return None, source, progress
        self._append(state, None, progress, depth=0, flagged=False)
        progress.last_e_total_au = float(state.e_total_initial)
        self._save(state, source, progress)
        return state, source, progress

    def _resume(self) -> tuple[TrajectoryState, RandomSource, Progress] | None:
        """The last restart generation, or None; any failure to use it is a RestartError."""
        try:
            generation = self.restarts.current()
            if generation is None:
                return None
            state, source = load_state(generation / STATE_FILE, self.model)
            self.model.backend.load_state(generation / ELECTRONIC_FILE)
            progress = Progress.from_json(read_json(generation / META_FILE))
            self.log.truncate(progress.log_records)
            progress.resumes.append(generation.name)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RestartError(
                f"cannot resume from the restart in {self.restarts.directory}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        log.info(
            "resumed from restart generation %s at t = %.4f fs",
            generation.name,
            self._time_au(progress) * AU_TIME_IN_FS,
        )
        return state, source, progress

    # ---------------------------------------------------------------- one (sub)step
    def _advance(
        self, state: TrajectoryState, source: RandomSource, progress: Progress
    ) -> tuple[TrajectoryState, RandomSource]:
        settings = self.settings
        dt = settings.options.dt_au / (1 << progress.depth)
        options = (
            settings.options
            if progress.depth == 0
            else dataclasses.replace(settings.options, dt_au=dt)
        )
        saved = source.state()
        self.model.active = state.active
        signal: str | None = None
        message = ""
        try:
            new_state, record = step(self.model, state, options, source)
            if record.hop_to is not None:
                new_state = self.model.refresh_active_gradient(new_state)
        except ElectronicStructureError as exc:
            if exc.retryable:
                raise
            signal, message = "electronic", str(exc)
        else:
            sv = self.model.min_singular_value
            jump = self._energy_jump(state, record, progress)
            if sv is not None and not sv >= settings.overlap_threshold:
                signal = "overlap"
                message = (
                    f"smallest singular value of the state overlap matrix {sv:.3g} is below "
                    f"overlap_threshold = {settings.overlap_threshold}"
                )
            elif jump is not None:
                assert settings.energy_jump_limit_au is not None
                signal = "energy"
                message = (
                    f"|E_tot - E_tot(previous record)| = {jump * HARTREE_IN_EV:.4g} eV exceeds "
                    f"energy_jump_limit_eV = {settings.energy_jump_limit_au * HARTREE_IN_EV:.4g}"
                )
            elif self.model.rung_changed:
                signal = "rung"
                message = f"CASSCF ladder rung {self.model.last_meta.get('casscf')!r}"
        flagged = False
        if signal == "rung" and progress.depth >= settings.max_step_halvings:
            signal, flagged = None, True
        if signal is not None:
            source = self._rollback(saved)
            progress.n_rejections += 1
            progress.rejections_by_signal[signal] = progress.rejections_by_signal.get(signal, 0) + 1
            progress.pending_rejections += 1
            if progress.depth < settings.max_step_halvings:
                progress.depth += 1
                return state, source
            progress.outcome = {
                "overlap": TrajectoryOutcome.OVERLAP_FAILURE,
                "energy": TrajectoryOutcome.ELECTRONIC_DISCONTINUITY,
            }.get(signal, TrajectoryOutcome.ELECTRONIC_FAILURE)
            when = self._time_au(progress) * AU_TIME_IN_FS
            progress.message = f"t = {when:.4f} fs, step dt/{1 << progress.depth}: {message}"
            self._save(state, source, progress)
            return state, source
        return self._accept(new_state, record, source, progress, dt, flagged), source

    def _energy_jump(
        self, state: TrajectoryState, record: StepRecord, progress: Progress
    ) -> float | None:
        """|E_tot - E_tot(previous record)| if it exceeds energy_jump_limit_au, else None."""
        limit = self.settings.energy_jump_limit_au
        if limit is None:
            return None
        previous = progress.last_e_total_au
        if previous is None:  # a restart written before this field existed
            previous = float(
                state.snapshot.energies[state.active] + kinetic_energy(state.v, self.model.masses)
            )
        # A non-finite E_tot gives a NaN jump, rejected here as an energy jump (practically
        # unreachable: BackendModel already rejects non-finite energies as electronic failures).
        jump = abs(float(record.e_total) - previous)
        return None if jump <= limit else jump

    def _rollback(self, saved: dict[str, Any]) -> RandomSource:
        generation = self.restarts.current()
        if generation is None:  # a step is only attempted after a restart was written
            raise RuntimeError(
                f"no restart generation in {self.restarts.directory} to roll a rejected step "
                "back to (a program error)"
            )
        self.model.backend.load_state(generation / ELECTRONIC_FILE)
        return source_from_state(saved)

    def _accept(
        self,
        state: TrajectoryState,
        record: StepRecord,
        source: RandomSource,
        progress: Progress,
        dt: float,
        flagged: bool,
    ) -> TrajectoryState:
        settings = self.settings
        depth_used = progress.depth
        progress.n_steps += 1
        progress.ticks += settings.ticks_per_step >> progress.depth
        if progress.ticks >= settings.ticks_per_step:
            progress.grid_step += 1
            progress.ticks = 0
            progress.depth = 0
        progress.n_hops += record.hop_to is not None
        progress.n_frustrated += bool(record.frustrated)
        progress.n_flagged += flagged
        drift = abs(record.e_total - state.e_total_initial)
        if not math.isfinite(drift):
            drift = math.inf
        progress.max_drift_au = max(progress.max_drift_au, drift)
        progress.s0_time_au = progress.s0_time_au + dt if state.active == 0 else 0.0
        self._append(state, record, progress, depth=depth_used, flagged=flagged)
        progress.last_e_total_au = float(record.e_total)
        if not drift <= settings.options.energy_drift_limit_au:
            progress.outcome = TrajectoryOutcome.ENERGY_DRIFT
            progress.message = (
                f"|E_tot - E_tot(0)| = {drift * HARTREE_IN_EV:.4g} eV exceeds "
                f"energy_drift_limit_eV = "
                f"{settings.options.energy_drift_limit_au * HARTREE_IN_EV:.4g}"
            )
        elif (
            settings.stop_after_s0_au is not None
            and progress.ticks == 0
            and progress.s0_time_au >= settings.stop_after_s0_au * (1 - 1e-12)
        ):
            progress.outcome = TrajectoryOutcome.STOPPED_IN_S0
        self._save(state, source, progress)
        return state

    # ---------------------------------------------------------------- files
    def _time_au(self, progress: Progress) -> float:
        s = self.settings
        return (progress.grid_step + progress.ticks / s.ticks_per_step) * s.options.dt_au

    def _append(
        self,
        state: TrajectoryState,
        record: StepRecord | None,
        progress: Progress,
        *,
        depth: int,
        flagged: bool,
    ) -> None:
        n = self.model.n_states
        sv = self.model.min_singular_value if record is not None else None
        if record is None:
            values: dict[str, Any] = {
                "step": 0,
                "active": state.active,
                "hop_from": -1,
                "hop_to": -1,
                "random_number": math.nan,
                "e_total": state.e_total_initial,
                "e_kinetic": kinetic_energy(state.v, self.model.masses),
                "energies": state.snapshot.energies,
                "c_real": state.c.real,
                "c_imag": state.c.imag,
                "probabilities": np.zeros(n),
                "q": state.q,
                "v": state.v,
            }
        else:
            values = {
                "step": progress.n_steps,
                "active": record.active,
                "hop_from": -1 if record.hop_from is None else record.hop_from,
                "hop_to": -1 if record.hop_to is None else record.hop_to,
                "frustrated": int(record.frustrated),
                "random_number": record.random_number,
                "e_total": record.e_total,
                "e_kinetic": record.e_kinetic,
                "energies": record.energies,
                "c_real": record.c.real,
                "c_imag": record.c.imag,
                "probabilities": record.probabilities,
                "q": record.q,
                "v": record.v,
            }
        values.update(
            grid_step=progress.grid_step,
            ticks=progress.ticks,
            depth=depth,
            time_au=self._time_au(progress),
            rung_changed=int(flagged),
            rejections=progress.pending_rejections,
            min_singular_value=math.nan if sv is None else sv,
        )
        self.log.append(log_row(self.log.dtype, **values))
        progress.log_records += 1
        progress.pending_rejections = 0

    def _save(self, state: TrajectoryState, source: RandomSource, progress: Progress) -> None:
        def fill(directory: Path) -> None:
            save_state(directory / STATE_FILE, state, source)
            self.model.backend.save_state(directory / ELECTRONIC_FILE)
            write_json_atomic(directory / META_FILE, progress.to_json())

        self.restarts.write(fill)
        self._write_progress(progress, state.active)

    def _write_progress(self, progress: Progress, active: int | None) -> None:
        write_json_atomic(
            self.directory / PROGRESS_FILE,
            {
                "grid_step": progress.grid_step,
                "time_fs": self._time_au(progress) * AU_TIME_IN_FS,
                "active": None if active is None else str(self.model.labels[active]),
                "outcome": None if progress.outcome is None else str(progress.outcome),
            },
        )

    def _finish(self, progress: Progress, elapsed: float) -> dict[str, Any]:
        outcome = str(progress.outcome)
        attrs = {
            **self.attrs,
            "outcome": outcome,
            "dt_fs": self.settings.options.dt_au * AU_TIME_IN_FS,
        }
        if progress.message is not None:
            attrs["message"] = progress.message
        write_output_h5(
            self.directory / OUTPUT_FILE,
            self.log.read(),
            labels=[str(label) for label in self.model.labels],
            n_atoms=self.model.n_dof // 3,
            output_every=self.settings.output_every,
            attrs=attrs,
        )
        rows = self.log.read()
        self._write_progress(progress, int(rows["active"][-1]) if rows.size else None)
        drift = progress.max_drift_au
        return {
            "outcome": outcome,
            "t_final_fs": self._time_au(progress) * AU_TIME_IN_FS,
            "n_grid_steps": progress.grid_step,
            "n_steps": progress.n_steps,
            "n_hops": progress.n_hops,
            "n_frustrated": progress.n_frustrated,
            "n_rejections": progress.n_rejections,
            "n_flagged": progress.n_flagged,
            "rejections_by_signal": dict(sorted(progress.rejections_by_signal.items())),
            "resumes": list(progress.resumes),
            "max_drift_eV": drift * HARTREE_IN_EV if math.isfinite(drift) else None,
            "message": progress.message,
            "wall_time_s": elapsed,
        }
