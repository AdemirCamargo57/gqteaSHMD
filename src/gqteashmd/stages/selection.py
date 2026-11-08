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
from gqteashmd.excite.selection import (
    candidate_weight,
    candidates_from,
    select_bright,
    select_fixed,
)
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.util.fileio import write_json_atomic, write_text_atomic
from gqteashmd.util.seeds import task_rng

log = logging.getLogger("gqteashmd.stages.selection")
REPORT_ROWS = 20
# The w_max pair is an outlier when its weight exceeds this multiple of the 99th percentile of
# the candidate weights: it then sets the normalisation for every other pair.
OUTLIER_FACTOR = 10.0


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None or not math.isfinite(value) else format(value, spec)


def _outlier(summary: Mapping[str, Any]) -> str | None:
    """A warning text when the w_max pair dominates the normalisation, else None."""
    pair, p99 = summary.get("weight_max_pair"), summary.get("weight_p99")
    if pair is None or p99 is None or pair["weight"] <= OUTLIER_FACTOR * p99:
        return None
    ratio = "inf" if p99 <= 0.0 else f"{pair['weight'] / p99:.3g}"
    return (
        f"the w_max pair (sample {pair['sample']}, {pair['state']}, "
        f"dE = {pair['excitation_energy_eV']:.4g} eV, f = {pair['oscillator_strength']:.4g}, "
        f"w = {pair['weight']:.4g}) is {ratio} times the 99th percentile of the candidate "
        f"weights ({p99:.4g}), so it suppresses the acceptance of every other pair; set "
        "[selection] energy_window_eV to exclude it (then run with --rerun-from selection)"
    )


class SelectionStage(Stage):
    name = "selection"
    version = 1
    depends_on = ("excitations",)
    config_sections = ("selection",)
    required_modules = ("h5py>=3.9",)

    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        """The [selection] keys that change the selected pairs, and the project seed.

        `allow_fewer` acts in finalize only (so the O20 remedy re-runs finalize, not the task),
        and `weighting` is ignored by `mode = "fixed"`; neither is hashed then."""
        options = ctx.config.selection.model_dump(mode="json")
        options.pop("allow_fewer")
        if options["mode"] == "fixed":
            options.pop("weighting")
        return {"selection": options, "project_seed": ctx.config.project.seed}

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
        try:
            candidates = candidates_from(
                data["excitation_energies_eV"],
                data["oscillator_strengths"],
                data["ok"],
                states,
                window,
            )
        except ValueError as exc:
            raise TaskError(
                f"excitations.h5 does not match [selection]: {exc}", retryable=False
            ) from exc
        rng = task_rng(ctx.config.project.seed, self.name, 0)
        n_all = max(len(candidates), 1)  # all accepted pairs, truncated below
        weight_max_pair: dict[str, Any] | None = None
        weight_p99: float | None = None
        if options.mode == "bright":
            full = select_bright(candidates, n_all, rng, weighting=options.weighting)
            if candidates:
                weights = np.array([candidate_weight(c, options.weighting) for c in candidates])
                top = candidates[int(np.argmax(weights))]
                weight_max_pair = {
                    "sample": top.sample,
                    "state": top.state,
                    "excitation_energy_eV": top.excitation_energy_eV,
                    "oscillator_strength": top.oscillator_strength,
                    "weight": float(weights.max()),
                }
                weight_p99 = float(np.percentile(weights, 99))
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
            "weighting": options.weighting,
            "weight_max": full.weight_max,
            "weight_max_pair": weight_max_pair,
            "weight_p99": weight_p99,
            "per_state": per_state,
            "selected": [
                [c.sample, c.state, c.excitation_energy_eV, c.oscillator_strength] for c in chosen
            ],
        }

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        """Write the reports, then fail (O20) or write `trajectories.json`.

        The reports are written first, also when the stage fails, so the user can see why."""
        options = ctx.config.selection
        summary = dict(results["select"])
        rows = summary["selected"]
        requested = options.n_trajectories
        window = summary["window_eV"]
        too_few = not rows or (len(rows) < requested and not options.allow_fewer)
        outlier = _outlier(summary)
        if outlier is not None:
            log.warning("Stage selection: %s.", outlier)
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
        self._write_reports(ctx, summary, trajectories, failed=too_few, outlier=outlier)
        if too_few:
            raise StageFailedError(
                f"Only {len(rows)} of {requested} requested trajectories can be selected "
                f"(candidates: {summary['candidates']}, accepted: {summary['accepted']}, "
                f"f_max: {summary['f_max']:.4g}, weighting: {summary['weighting']}, "
                f"window: {window}, states: {', '.join(summary['states'])}). "
                "Widen energy_window_eV, add states or lower n_trajectories (then run with "
                "--rerun-from selection); or increase [sampling] n_samples (then run with "
                "--rerun-from sampling, which recomputes the excitations of every sample)"
                + (
                    "; or set allow_fewer = true and run again (no --rerun-from needed)."
                    if rows
                    else "."
                )
                + (f" Note: {outlier}." if outlier is not None else "")
                + f" See {ctx.stage_dir / 'selection_report.md'}."
            )
        if len(rows) < requested:
            log.warning(
                "Stage selection: only %d of %d requested trajectories were selected "
                "(allow_fewer = true)",
                len(rows),
                requested,
            )
        write_json_atomic(
            ctx.stage_dir / "trajectories.json",
            {
                "schema_version": 1,
                "n_trajectories": len(trajectories),
                "mode": summary["mode"],
                "project_seed": ctx.config.project.seed,
                "trajectories": trajectories,
            },
        )

    def _write_reports(
        self,
        ctx: StageContext,
        summary: Mapping[str, Any],
        trajectories: list[dict[str, Any]],
        *,
        failed: bool,
        outlier: str | None,
    ) -> None:
        requested = ctx.config.selection.n_trajectories
        window = summary["window_eV"]
        seed = ctx.config.project.seed
        pair = summary.get("weight_max_pair")
        write_json_atomic(
            ctx.stage_dir / "selection_report.json",
            {
                "schema_version": 1,
                "mode": summary["mode"],
                "states": summary["states"],
                "window_eV": window,
                "n_requested": requested,
                "n_selected": len(trajectories),
                "stage_failed": failed,
                "candidates": summary["candidates"],
                "accepted": summary["accepted"],
                "f_max": summary["f_max"],
                "weighting": summary["weighting"],
                "weight_max": summary["weight_max"],
                "weight_max_pair": pair,
                "weight_p99": summary.get("weight_p99"),
                "weight_outlier": outlier is not None,
                "per_state": summary["per_state"],
            },
        )
        exc_name = ctx.upstream_dir("excitations").name
        lines = [
            "# Initial-state selection",
            "",
            f"- mode: {summary['mode']}",
            f"- weighting: {summary['weighting']} (w_max = {summary['weight_max']:.4g})",
        ]
        if pair is not None:
            lines.append(
                f"- w_max pair: sample {pair['sample']}, {pair['state']}, "
                f"dE = {pair['excitation_energy_eV']:.4f} eV, "
                f"f = {pair['oscillator_strength']:.4f}, w = {pair['weight']:.4g} "
                f"(99th percentile of the weights: {summary['weight_p99']:.4g})"
            )
        lines += [
            f"- states: {', '.join(summary['states'])}",
            f"- energy window (eV): {window if window else 'none'}",
            f"- trajectories: {len(trajectories)} selected of {requested} requested",
            f"- candidates: {summary['candidates']}, accepted: {summary['accepted']}, "
            f"f_max: {summary['f_max']:.4g}",
            f"- project seed: {seed}",
        ]
        if outlier is not None:
            lines += ["", f"**Warning:** {outlier}."]
        if failed:
            lines += [
                "",
                "**The stage failed: too few trajectories (see the error message); "
                "`trajectories.json` was not written.**",
            ]
        lines += [
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
            'If the run paused here (`pause_after = ["selection"]`, effective once a later '
            "stage follows `selection`), review this report, then run `gqteashmd run` again "
            "to continue.",
            "",
        ]
        write_text_atomic(ctx.stage_dir / "selection_report.md", "\n".join(lines))
