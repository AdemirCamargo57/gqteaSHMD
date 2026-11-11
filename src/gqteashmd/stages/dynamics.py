"""Stage 6: one surface-hopping trajectory per entry of trajectories.json (P7b)."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.config.schema import DynamicsStageSection, ShmdConfig
from gqteashmd.core.states import StateLabel
from gqteashmd.core.units import AU_TIME_IN_FS, DALTON_IN_ME, HARTREE_IN_EV
from gqteashmd.dynamics.options import DynamicsOptions
from gqteashmd.dynamics.randomness import GeneratorSource
from gqteashmd.electronic.base import ElectronicBackend, MoleculeSpec
from gqteashmd.errors import ProjectError, StageFailedError, TaskError
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.stages.geometry import prepared_molecule, stage_backend
from gqteashmd.trajectory.model import BackendModel
from gqteashmd.trajectory.outcomes import TrajectoryOutcome
from gqteashmd.trajectory.runner import (
    OUTPUT_FILE,
    InitialConditions,
    RestartError,
    RunSettings,
    TrajectoryRunner,
)
from gqteashmd.util.fileio import read_json, write_json_atomic, write_text_atomic
from gqteashmd.util.seeds import task_rng
from gqteashmd.vib.artifacts import read_samples_h5

log = logging.getLogger("gqteashmd.stages.dynamics")
BackendFactory = Callable[[StageContext, MoleculeSpec, Path], ElectronicBackend]


def _pyscf_backend(ctx: StageContext, molecule: MoleculeSpec, task_dir: Path) -> ElectronicBackend:
    return stage_backend(ctx, "dynamics", molecule, task_dir, settings=ctx.config.electronic)


def run_settings(section: DynamicsStageSection) -> RunSettings:
    options = DynamicsOptions(
        dt_au=section.dt_fs / AU_TIME_IN_FS,
        nsubsteps=section.nsubsteps,
        hopping=section.hopping,
        ekin_correction=section.ekin_correction,
        frustrated=section.frustrated,
        decoherence=section.decoherence,
        decoherence_param_Eh=section.decoherence_param_Eh,
        energy_drift_limit_au=section.energy_drift_limit_eV / HARTREE_IN_EV,
        edc_convention=section.edc_convention,
        substep_interpolation=section.substep_interpolation,
        coupling=section.coupling,
    )
    return RunSettings(
        options,
        tmax_au=section.tmax_fs / AU_TIME_IN_FS,
        overlap_threshold=section.overlap_threshold,
        max_step_halvings=section.max_step_halvings,
        output_every=section.output_every,
        stop_after_s0_au=(
            None if section.stop_after_s0_fs is None else section.stop_after_s0_fs / AU_TIME_IN_FS
        ),
    )


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    x = float(value)
    return x if math.isfinite(x) else None


class DynamicsStage(Stage):
    name = "dynamics"
    version = 1
    depends_on = ("prepare", "sampling", "selection")
    config_sections = ("electronic", "dynamics")
    required_modules: tuple[str, ...] = ("pyscf>=2.6", "h5py>=3.9")

    def __init__(
        self,
        *,
        max_failed_fraction: float = 0.1,
        backend_factory: BackendFactory | None = None,
        required_modules: tuple[str, ...] | None = None,
    ) -> None:
        self.max_failed_fraction = max_failed_fraction
        self._backend_factory = backend_factory or _pyscf_backend
        if required_modules is not None:
            self.required_modules = required_modules

    def enabled(self, config: ShmdConfig) -> bool:
        return config.dynamics is not None  # plan ruling R1

    def check_config(self, config: ShmdConfig) -> list[str]:
        if config.electronic is not None and config.electronic.singlets < 2:
            return [
                "Stage 'dynamics' needs [electronic] singlets >= 2 (S0 and at least one "
                "excited state)"
            ]
        return []

    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        assert ctx.config.dynamics is not None and ctx.config.electronic is not None
        dynamics = ctx.config.dynamics.model_dump(mode="json")
        dynamics.pop("tmax_fs")  # O16: raising it extends finished trajectories
        dynamics.pop("max_failed_fraction")  # finalize-only
        return {
            "electronic": ctx.config.electronic.model_dump(mode="json"),
            "dynamics": dynamics,
            "project_seed": ctx.config.project.seed,
        }

    def reopen_tasks(
        self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]
    ) -> list[str]:
        assert ctx.config.dynamics is not None
        n_total = run_settings(ctx.config.dynamics).n_grid_steps
        return sorted(
            task_id
            for task_id, result in results.items()
            if result.get("outcome") == TrajectoryOutcome.FINISHED
            and int(result.get("n_grid_steps", n_total)) < n_total
        )

    # ------------------------------------------------------------------------------- plan
    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        electronic = ctx.config.electronic
        if (
            electronic is not None
            and electronic.method in ("tda", "tddft")
            and electronic.functional is not None
            and electronic.functional.lower() != "hf"
            and electronic.grid_level < 5
        ):
            log.warning(
                "DFT gradients in PySCF omit the grid response; use [electronic] "
                "grid_level >= 5 for dynamics"
            )
        path = ctx.upstream_dir("selection") / "trajectories.json"
        advice = f"; rerun the selection stage with --rerun-from selection ({path})"
        try:
            data = read_json(path)
            return [
                TaskSpec(
                    f"traj_{int(e['index']):05d}",
                    {
                        "index": int(e["index"]),
                        "sample": int(e["sample"]),
                        "state": str(e["state"]),
                        "rng": e["rng"],
                    },
                )
                for e in data["trajectories"]
            ]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ProjectError(
                f"trajectories.json is missing or malformed ({exc!r}){advice}"
            ) from exc

    # ------------------------------------------------------------------------------ tasks
    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        section = ctx.config.dynamics
        assert section is not None
        payload = task.payload
        index = int(payload["index"])
        sample = int(payload["sample"])
        state = str(payload["state"])
        molecule, _ = prepared_molecule(ctx.upstream_dir("prepare"))
        try:
            samples = read_samples_h5(ctx.upstream_dir("sampling") / "samples.h5")
        except ValueError as exc:
            raise TaskError(str(exc), retryable=False) from exc
        if not 1 <= sample <= samples["n_samples"]:
            raise TaskError(
                f"sample {sample} is outside 1..{samples['n_samples']} of samples.h5; "
                "rerun with --rerun-from sampling",
                retryable=False,
            )
        if tuple(samples["symbols"]) != tuple(molecule.symbols):
            raise TaskError(
                f"the atoms of samples.h5 {tuple(samples['symbols'])} differ from the prepared "
                f"molecule {tuple(molecule.symbols)}; rerun with --rerun-from sampling",
                retryable=False,
            )
        if payload.get("rng") != {"stage": "dynamics", "task_index": index}:
            raise TaskError(
                f"trajectory {task.task_id} has the random stream {payload.get('rng')!r}, "
                f"expected stage 'dynamics' task_index {index}; rerun with --rerun-from "
                "selection",
                retryable=False,
            )
        backend = self._backend_factory(ctx, molecule, task_dir)
        masses = np.repeat(np.asarray(samples["masses_amu"], dtype=float) * DALTON_IN_ME, 3)
        try:
            model = BackendModel(backend, masses, coupling=section.coupling)
            active = model.labels.index(StateLabel.parse(state))
        except ValueError as exc:
            raise TaskError(f"trajectory {task.task_id}: {exc}", retryable=False) from exc
        runner = TrajectoryRunner(
            model,
            run_settings(section),
            task_dir,
            attrs={
                "symbols": list(samples["symbols"]),
                "sample": sample,
                "initial_state": state,
                "task": task.task_id,
            },
        )
        initial = InitialConditions(
            np.asarray(samples["positions_bohr"][sample - 1], dtype=float).reshape(-1),
            np.asarray(samples["velocities_au"][sample - 1], dtype=float).reshape(-1),
            active,
        )
        try:
            result = runner.run(
                initial, GeneratorSource(task_rng(ctx.config.project.seed, "dynamics", index))
            )
        except RestartError as exc:
            raise TaskError(
                f"trajectory {task.task_id}: {exc}; the restart or step log is unusable, "
                "rerun with --rerun-from dynamics",
                retryable=False,
            ) from exc
        except ValueError as exc:
            raise TaskError(f"trajectory {task.task_id}: {exc}", retryable=False) from exc
        return {**result, "sample": sample, "initial_state": state}

    # -------------------------------------------------------------------------- finalize
    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        section = ctx.config.dynamics
        assert section is not None
        if not results:
            raise StageFailedError("The dynamics stage produced no trajectory results")
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for task_id in sorted(results):
            r = results[task_id]
            if not (ctx.layout.task_dir(ctx.stage_dir, task_id) / OUTPUT_FILE).exists():
                missing.append(task_id)
            rows.append(
                {
                    "task": task_id,
                    "sample": r.get("sample"),
                    "initial_state": r.get("initial_state"),
                    "outcome": str(r.get("outcome")),
                    "t_final_fs": _finite_or_none(r.get("t_final_fs")),
                    "n_hops": int(r.get("n_hops", 0)),
                    "n_rejections": int(r.get("n_rejections", 0)),
                    "n_flagged": int(r.get("n_flagged", 0)),
                    "max_drift_eV": _finite_or_none(r.get("max_drift_eV")),
                    "message": r.get("message"),
                }
            )
        if missing:
            log.warning("Stage dynamics: no %s for: %s", OUTPUT_FILE, ", ".join(missing))
        counts = Counter(row["outcome"] for row in rows)
        outcomes = {o.value: int(counts.get(o.value, 0)) for o in TrajectoryOutcome}
        drifts = [row["max_drift_eV"] for row in rows if row["max_drift_eV"] is not None]
        totals = {
            "n_hops": sum(row["n_hops"] for row in rows),
            "n_frustrated": sum(int(results[row["task"]].get("n_frustrated", 0)) for row in rows),
            "n_rejections": sum(row["n_rejections"] for row in rows),
            "n_flagged": sum(row["n_flagged"] for row in rows),
        }
        summary = {
            "schema_version": 1,
            "n_results": len(rows),
            "outcomes": outcomes,
            "tmax_fs": section.tmax_fs,
            "dt_fs": section.dt_fs,
            "totals": totals,
            "max_drift_eV": max(drifts) if drifts else None,
            "missing_output": missing,
            "trajectories": rows,
        }
        write_json_atomic(ctx.stage_dir / "dynamics_summary.json", summary)
        write_text_atomic(ctx.stage_dir / "dynamics_summary.md", _markdown(summary))


def _fmt(value: Any, spec: str = "") -> str:
    return "-" if value is None else format(value, spec)


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Dynamics summary",
        "",
        f"- trajectories: {summary['n_results']}",
        f"- tmax: {summary['tmax_fs']} fs, dt: {summary['dt_fs']} fs",
        "- outcomes: " + ", ".join(f"{k} {v}" for k, v in summary["outcomes"].items()),
        "- totals: " + ", ".join(f"{k} {v}" for k, v in summary["totals"].items()),
        f"- largest energy drift (eV): {_fmt(summary['max_drift_eV'], '.4g')}",
    ]
    if summary["missing_output"]:
        lines.append("- missing output.h5: " + ", ".join(summary["missing_output"]))
    lines += [
        "",
        "| task | sample | state | outcome | t_final (fs) | hops | rejections | flagged "
        "| max drift (eV) | message |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["trajectories"]:
        lines.append(
            f"| {r['task']} | {_fmt(r['sample'])} | {_fmt(r['initial_state'])} | {r['outcome']} "
            f"| {_fmt(r['t_final_fs'], '.3f')} | {r['n_hops']} | {r['n_rejections']} "
            f"| {r['n_flagged']} | {_fmt(r['max_drift_eV'], '.4g')} "
            f"| {(r['message'] or '').replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"
