"""Stage 5: choose each trajectory's initial excited state from the vertical excitations."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.errors import StageFailedError, TaskError
from gqteashmd.excite.artifacts import read_excitations_h5
from gqteashmd.excite.selection import candidates_from, select_bright, select_fixed
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.util.fileio import write_json_atomic, write_text_atomic
from gqteashmd.util.seeds import task_rng

log = logging.getLogger("gqteashmd.stages.selection")
REPORT_ROWS = 20


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None or not math.isfinite(value) else format(value, spec)


class SelectionStage(Stage):
    name = "selection"
    version = 1
    depends_on = ("excitations",)
    config_sections = ("selection",)
    required_modules = ("h5py>=3.9",)

    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        return {
            "selection": ctx.config.selection.model_dump(mode="json"),
            "project_seed": ctx.config.project.seed,
        }

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        return [TaskSpec("select")]

    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        options = ctx.config.selection
        try:
            data = read_excitations_h5(ctx.upstream_dir("excitations") / "excitations.h5")
        except ValueError as exc:
            raise TaskError(str(exc), retryable=False) from exc
        states = list(options.states) if options.states else list(data["state_labels"][1:])
        window = (
            (float(options.energy_window_eV[0]), float(options.energy_window_eV[1]))
            if options.energy_window_eV
            else None
        )
        candidates = candidates_from(
            data["excitation_energies_eV"],
            data["oscillator_strengths"],
            data["ok"],
            states,
            window,
        )
        rng = task_rng(ctx.config.project.seed, self.name, 0)
        n_all = max(len(candidates), 1)  # all accepted pairs, truncated below
        if options.mode == "bright":
            full = select_bright(candidates, n_all, rng)
        else:
            full = select_fixed(candidates, n_all)
        chosen = full.selected[: options.n_trajectories]
        cand_count = Counter(c.state for c in candidates)
        acc_count = Counter(c.state for c in full.selected)
        per_state: dict[str, dict[str, Any]] = {}
        for s in sorted(set(states) | set(cand_count), key=lambda x: int(x[1:])):
            sel = [c for c in chosen if c.state == s]
            per_state[s] = {
                "candidates": cand_count[s],
                "accepted": acc_count[s],
                "selected": len(sel),
                "mean_excitation_energy_eV": _mean([c.excitation_energy_eV for c in sel]),
                "mean_oscillator_strength": _mean([c.oscillator_strength for c in sel]),
            }
        return {
            "mode": options.mode,
            "states": states,
            "window_eV": list(window) if window else None,
            "candidates": full.candidates,
            "accepted": full.accepted,
            "f_max": full.f_max,
            "per_state": per_state,
            "selected": [
                [c.sample, c.state, c.excitation_energy_eV, c.oscillator_strength] for c in chosen
            ],
        }

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        options = ctx.config.selection
        summary = dict(results["select"])
        rows = summary["selected"]
        requested = options.n_trajectories
        window = summary["window_eV"]
        if not rows or (len(rows) < requested and not options.allow_fewer):
            raise StageFailedError(
                f"Only {len(rows)} of {requested} requested trajectories can be selected "
                f"(candidates: {summary['candidates']}, accepted: {summary['accepted']}, "
                f"f_max: {summary['f_max']:.4g}, window: {window}, "
                f"states: {', '.join(summary['states'])}). Increase [sampling] n_samples "
                "(then run with --rerun-from sampling), widen energy_window_eV, add states, "
                "lower n_trajectories" + (", or set allow_fewer = true." if rows else ".")
            )
        if len(rows) < requested:
            log.warning(
                "Stage selection: only %d of %d requested trajectories were selected "
                "(allow_fewer = true)",
                len(rows),
                requested,
            )
        trajectories = [
            {
                "id": f"t{k + 1:05d}",
                "index": k,
                "sample": int(sample),
                "state": state,
                "excitation_energy_eV": float(de),
                "oscillator_strength": float(f),
                "rng": {"stage": "dynamics", "task_index": k},
            }
            for k, (sample, state, de, f) in enumerate(rows)
        ]
        seed = ctx.config.project.seed
        write_json_atomic(
            ctx.stage_dir / "trajectories.json",
            {
                "schema_version": 1,
                "n_trajectories": len(trajectories),
                "mode": summary["mode"],
                "project_seed": seed,
                "trajectories": trajectories,
            },
        )
        write_json_atomic(
            ctx.stage_dir / "selection_report.json",
            {
                "schema_version": 1,
                "mode": summary["mode"],
                "states": summary["states"],
                "window_eV": window,
                "n_requested": requested,
                "n_selected": len(trajectories),
                "candidates": summary["candidates"],
                "accepted": summary["accepted"],
                "f_max": summary["f_max"],
                "per_state": summary["per_state"],
            },
        )
        exc_name = ctx.upstream_dir("excitations").name
        lines = [
            "# Initial-state selection",
            "",
            f"- mode: {summary['mode']}",
            f"- states: {', '.join(summary['states'])}",
            f"- energy window (eV): {window if window else 'none'}",
            f"- trajectories: {len(trajectories)} selected of {requested} requested",
            f"- candidates: {summary['candidates']}, accepted: {summary['accepted']}, "
            f"f_max: {summary['f_max']:.4g}",
            f"- project seed: {seed}",
            "",
            "## Per state",
            "",
            "| state | candidates | accepted | selected | mean dE (eV) | mean f |",
            "|---|---|---|---|---|---|",
        ]
        for s, d in summary["per_state"].items():
            lines.append(
                f"| {s} | {d['candidates']} | {d['accepted']} | {d['selected']} | "
                f"{_fmt(d['mean_excitation_energy_eV'], '.3f')} | "
                f"{_fmt(d['mean_oscillator_strength'], '.4f')} |"
            )
        lines += [
            "",
            f"## First {min(REPORT_ROWS, len(trajectories))} trajectories",
            "",
            "| id | sample | state | dE (eV) | f |",
            "|---|---|---|---|---|",
        ]
        for t in trajectories[:REPORT_ROWS]:
            lines.append(
                f"| {t['id']} | {t['sample']} | {t['state']} | "
                f"{t['excitation_energy_eV']:.3f} | {t['oscillator_strength']:.4f} |"
            )
        lines += [
            "",
            "## Spectrum and excitations",
            "",
            f"- spectrum: `../{exc_name}/spectrum.csv`",
            f"- summary: `../{exc_name}/excitations.json`",
            "",
            "Review, then run `gqteashmd run` again to continue.",
            "",
        ]
        write_text_atomic(ctx.stage_dir / "selection_report.md", "\n".join(lines))
