"""Stage 4: vertical excitation energies, transition dipoles and the absorption spectrum."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.core.states import states_from_counts
from gqteashmd.core.units import HARTREE_IN_EV
from gqteashmd.electronic.base import Request
from gqteashmd.electronic.properties import oscillator_strength
from gqteashmd.errors import StageFailedError, TaskError
from gqteashmd.excite.artifacts import format_spectrum_csv, write_excitations_h5
from gqteashmd.excite.spectrum import broadened_spectrum
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.stages.geometry import prepared_molecule, stage_backend
from gqteashmd.util.fileio import write_json_atomic, write_text_atomic
from gqteashmd.vib.artifacts import read_samples_h5

log = logging.getLogger("gqteashmd.stages.excitations")


def _chunks(config: ShmdConfig) -> list[list[int]]:
    n = config.sampling.n_samples
    size = config.excitations.samples_per_task
    return [list(range(first, min(first + size, n + 1))) for first in range(1, n + 1, size)]


def _task_id(first: int) -> str:
    return f"s{first:05d}"


class ExcitationsStage(Stage):
    name = "excitations"
    version = 1
    depends_on = ("prepare", "sampling")
    config_sections = ("electronic", "excitations")
    required_modules = ("pyscf>=2.6", "h5py>=3.9")

    def __init__(self, max_failed_fraction: float = 0.05) -> None:
        self.max_failed_fraction = max_failed_fraction

    def check_config(self, config: ShmdConfig) -> list[str]:
        if config.electronic is not None and config.electronic.singlets < 2:
            return [
                "the excitations stage needs at least one excited state: "
                "[electronic] singlets must be >= 2"
            ]
        return []

    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        """Electronic settings (without `overlap_norm_kept`), the task-defining [excitations]
        keys and the number of samples. The broadening and the failure limit act in finalize."""
        electronic = ctx.config.electronic
        electronic_dump = None if electronic is None else electronic.model_dump(mode="json")
        if electronic_dump is not None:
            electronic_dump.pop("overlap_norm_kept", None)
        options = ctx.config.excitations.model_dump(mode="json")
        for key in ("max_failed_fraction", "broadening_fwhm_eV", "spectrum_step_eV"):
            options.pop(key)
        return {
            "electronic": electronic_dump,
            "excitations": options,
            "n_samples": ctx.config.sampling.n_samples,
        }

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        tasks = [TaskSpec("ref")] if ctx.config.excitations.include_reference else []
        tasks += [TaskSpec(_task_id(chunk[0]), {"samples": chunk}) for chunk in _chunks(ctx.config)]
        return tasks

    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        molecule, _ = prepared_molecule(ctx.upstream_dir("prepare"))
        try:
            data = read_samples_h5(ctx.upstream_dir("sampling") / "samples.h5")
        except ValueError as exc:
            raise TaskError(str(exc), retryable=False) from exc
        backend = stage_backend(ctx, self.name, molecule, task_dir, settings=ctx.config.electronic)
        if task.task_id == "ref":
            jobs = [(0, np.asarray(data["equilibrium_bohr"], dtype=float))]
        else:
            jobs = [
                (int(k), np.asarray(data["positions_bohr"][int(k) - 1], dtype=float))
                for k in task.payload["samples"]
            ]
        rows: list[dict[str, Any]] = []
        for sample, coords in jobs:
            backend.reset()
            try:
                result = backend.compute(Request(coords, transition_dipoles=True))
                states = result.states
                energies = [float(result.energies[s]) for s in states]
                dipoles = [
                    np.asarray(result.transition_dipoles[s], dtype=float).tolist()
                    for s in states[1:]
                ]
                strengths = [
                    oscillator_strength(result.excitation_energy(s), result.transition_dipoles[s])
                    for s in states[1:]
                ]
            except TaskError as exc:
                if exc.retryable:
                    raise
                log.warning("Stage excitations: sample %d failed: %s", sample, exc)
                rows.append({"sample": sample, "ok": False, "error": str(exc)})
                continue
            rows.append(
                {
                    "sample": sample,
                    "ok": True,
                    "energies_hartree": energies,
                    "transition_dipoles_au": dipoles,
                    "oscillator_strengths": strengths,
                }
            )
        return {"samples": rows}

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        config = ctx.config
        options = config.excitations
        electronic = config.electronic
        assert electronic is not None
        labels = [str(s) for s in states_from_counts(electronic.singlets)]
        n_states = len(labels)
        n_samples = config.sampling.n_samples

        rows: dict[int, dict[str, Any]] = {}
        for task in self.plan(ctx):
            if task.task_id in results:
                for row in results[task.task_id]["samples"]:
                    rows[int(row["sample"])] = dict(row)
            elif task.task_id == "ref":
                rows[0] = {"sample": 0, "ok": False, "error": "task failed"}
            else:
                for k in task.payload["samples"]:
                    rows[int(k)] = {"sample": int(k), "ok": False, "error": "task failed"}
        for k in range(1, n_samples + 1):
            rows.setdefault(k, {"sample": k, "ok": False, "error": "no result"})

        failed = [rows[k] for k in range(1, n_samples + 1) if not rows[k]["ok"]]
        if len(failed) / n_samples > self.max_failed_fraction:
            shown = "; ".join(f"sample {r['sample']}: {r['error']}" for r in failed[:5])
            more = f" (and {len(failed) - 5} more)" if len(failed) > 5 else ""
            raise StageFailedError(
                f"{len(failed)} of {n_samples} samples failed, more than "
                f"max_failed_fraction = {self.max_failed_fraction:g}: {shown}{more}. "
                "Fix the cause (see the task logs in the excitations stage directory), or "
                "raise [excitations] max_failed_fraction."
            )
        reference: dict[str, Any] | None = None
        if options.include_reference:
            ref_row = rows[0]
            if not ref_row["ok"]:
                raise StageFailedError(
                    "the equilibrium geometry itself failed: "
                    f"{ref_row['error']}. Fix the electronic-structure settings first; the "
                    "reference is needed to judge the sampled spectrum "
                    "(or set [excitations] include_reference = false)."
                )
            samples = read_samples_h5(ctx.upstream_dir("sampling") / "samples.h5")
            reference = {
                "energies_hartree": np.asarray([ref_row["energies_hartree"]]),
                "transition_dipoles_au": np.asarray([ref_row["transition_dipoles_au"]]),
                "oscillator_strengths": np.asarray([ref_row["oscillator_strengths"]]),
                "coordinates_bohr": samples["equilibrium_bohr"],
            }

        n_exc = n_states - 1
        energies = np.full((n_samples, n_states), np.nan)
        dipoles = np.full((n_samples, n_exc, 3), np.nan)
        strengths = np.full((n_samples, n_exc), np.nan)
        ok = np.zeros(n_samples, dtype=bool)
        for k in range(1, n_samples + 1):
            row = rows[k]
            if row["ok"]:
                ok[k - 1] = True
                energies[k - 1] = row["energies_hartree"]
                dipoles[k - 1] = row["transition_dipoles_au"]
                strengths[k - 1] = row["oscillator_strengths"]
        sampling_dir = ctx.upstream_dir("sampling")
        symbols = read_samples_h5(sampling_dir / "samples.h5")["symbols"]
        write_excitations_h5(
            ctx.stage_dir / "excitations.h5",
            symbols=symbols,
            state_labels=labels,
            sample_indices=np.arange(1, n_samples + 1),
            ok=ok,
            energies_hartree=energies,
            transition_dipoles_au=dipoles,
            oscillator_strengths=strengths,
            reference=reference,
            project_seed=config.project.seed,
        )
        de_ev = (energies[ok][:, 1:] - energies[ok][:, :1]) * HARTREE_IN_EV
        f_ok = strengths[ok]
        spectrum = broadened_spectrum(
            de_ev,
            f_ok,
            fwhm_eV=options.broadening_fwhm_eV,
            step_eV=options.spectrum_step_eV,
        )
        write_text_atomic(
            ctx.stage_dir / "spectrum.csv",
            format_spectrum_csv(
                spectrum,
                labels,
                n_ok=int(ok.sum()),
                fwhm_eV=options.broadening_fwhm_eV,
                step_eV=options.spectrum_step_eV,
            ),
        )
        per_state: dict[str, Any] = {}
        for j, label in enumerate(labels[1:]):
            entry: dict[str, Any] = {
                "mean_excitation_eV": float(de_ev[:, j].mean()),
                "std_excitation_eV": float(de_ev[:, j].std()),
                "mean_oscillator_strength": float(f_ok[:, j].mean()),
                "std_oscillator_strength": float(f_ok[:, j].std()),
                "reference_excitation_eV": None,
                "reference_oscillator_strength": None,
            }
            if reference is not None:
                ref_e = reference["energies_hartree"][0]
                entry["reference_excitation_eV"] = float((ref_e[j + 1] - ref_e[0]) * HARTREE_IN_EV)
                entry["reference_oscillator_strength"] = float(
                    reference["oscillator_strengths"][0][j]
                )
            per_state[label] = entry
            log.info(
                "Stage excitations: %s  mean dE = %.3f +- %.3f eV, mean f = %.4f +- %.4f (%d ok)",
                label,
                entry["mean_excitation_eV"],
                entry["std_excitation_eV"],
                entry["mean_oscillator_strength"],
                entry["std_oscillator_strength"],
                int(ok.sum()),
            )
        write_json_atomic(
            ctx.stage_dir / "excitations.json",
            {
                "schema_version": 1,
                "n_samples": n_samples,
                "n_ok": int(ok.sum()),
                "failed": [{"sample": r["sample"], "error": r["error"]} for r in failed],
                "state_labels": labels,
                "states": per_state,
                "include_reference": options.include_reference,
                "broadening_fwhm_eV": options.broadening_fwhm_eV,
                "spectrum_step_eV": options.spectrum_step_eV,
            },
        )
