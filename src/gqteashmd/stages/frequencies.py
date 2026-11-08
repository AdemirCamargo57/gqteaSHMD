"""Stage 2: S0 Hessian (analytic or finite differences of gradients) and harmonic analysis."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.chem.masses import isotope_mass
from gqteashmd.config.schema import ShmdConfig
from gqteashmd.electronic.base import Request
from gqteashmd.electronic.factory import ground_state_fingerprint
from gqteashmd.errors import StageFailedError
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.stages.geometry import prepared_molecule, stage_backend, working_geometry
from gqteashmd.util.fileio import read_json, write_json_atomic, write_text_atomic
from gqteashmd.vib.artifacts import write_frequencies_h5
from gqteashmd.vib.harmonic import harmonic_analysis
from gqteashmd.vib.molden import format_molden

log = logging.getLogger("gqteashmd.stages.frequencies")
EXTERNAL_RESIDUAL_WARNING_CM1 = 100.0
STATIONARY_GRADIENT = 1e-3  # Eh/bohr
FD_CONSISTENCY_TOLERANCE = 1e-4  # Eh/bohr: |(E+ - E-)/2h - g_ref| over all displacements
OPTIMIZED_ENERGY_TOLERANCE = 1e-6  # Eh: reference energy vs the optimize stage's energy


def resolved_method(config: ShmdConfig) -> str:
    requested = config.frequencies.method
    electronic = config.electronic
    if requested == "auto":
        casscf = electronic is not None and electronic.method == "casscf"
        return "finite_difference" if casscf else "analytic"
    return requested


def _task_id(k: int, sign: int) -> str:
    return f"fd_{k:03d}_{'plus' if sign > 0 else 'minus'}"


class FrequenciesStage(Stage):
    name = "frequencies"
    version = 1
    depends_on = ("prepare", "optimize")
    config_sections = ("electronic", "frequencies")
    required_modules = ("pyscf>=2.6", "h5py>=3.9")

    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        """S0-determining electronic settings and [frequencies] without `imaginary_cutoff_cm1`.

        The cutoff is a finalize-only check: after a stop on it, the next run re-runs finalize
        with the new value from the stored task results."""
        electronic = ctx.config.electronic
        options = ctx.config.frequencies.model_dump(mode="json")
        options.pop("imaginary_cutoff_cm1")
        return {
            "electronic": None if electronic is None else ground_state_fingerprint(electronic),
            "frequencies": options,
        }

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        if resolved_method(ctx.config) == "analytic":
            return [TaskSpec("hessian")]
        spec, _ = prepared_molecule(ctx.upstream_dir("prepare"))
        tasks = [TaskSpec("reference")]
        for k in range(3 * len(spec.symbols)):
            tasks += [TaskSpec(_task_id(k, s), {"coordinate": k, "sign": s}) for s in (1, -1)]
        return tasks

    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        molecule, coords, _ = working_geometry(ctx)
        backend = stage_backend(ctx, self.name, molecule, task_dir)
        s0 = backend.states[0]
        if task.task_id in ("hessian", "reference"):
            result = backend.compute(Request(coords, gradients=frozenset({s0})))
            out: dict[str, Any] = {
                "energy_hartree": result.energies[s0],
                "gradient": result.gradients[s0].ravel().tolist(),
            }
            if task.task_id == "hessian":
                np.save(task_dir / "hessian.npy", backend.ground_state_hessian(coords))
            return out
        displaced = coords.copy().ravel()
        step = ctx.config.frequencies.fd_step_bohr
        displaced[int(task.payload["coordinate"])] += int(task.payload["sign"]) * step
        result = backend.compute(Request(displaced.reshape(-1, 3), gradients=frozenset({s0})))
        return {
            "energy_hartree": result.energies[s0],
            "gradient": result.gradients[s0].ravel().tolist(),
        }

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        options = ctx.config.frequencies
        molecule, coords, source = working_geometry(ctx)
        method = resolved_method(ctx.config)
        n3 = 3 * len(molecule.symbols)
        consistency_problems: list[str] = []
        fd_residual: float | None = None
        energy_difference: float | None = None
        if method == "analytic":
            hessian = np.load(ctx.layout.task_dir(ctx.stage_dir, "hessian") / "hessian.npy")
            centre = results["hessian"]
        else:
            hessian = np.zeros((n3, n3))
            step = options.fd_step_bohr
            for k in range(n3):
                plus = np.asarray(results[_task_id(k, 1)]["gradient"])
                minus = np.asarray(results[_task_id(k, -1)]["gradient"])
                hessian[:, k] = (plus - minus) / (2.0 * step)
            hessian = 0.5 * (hessian + hessian.T)
            centre = results["reference"]
            reference_gradient = np.asarray(centre["gradient"])
            slopes = np.array(
                [
                    (
                        results[_task_id(k, 1)]["energy_hartree"]
                        - results[_task_id(k, -1)]["energy_hartree"]
                    )
                    / (2.0 * step)
                    for k in range(n3)
                ]
            )
            fd_residuals = np.abs(slopes - reference_gradient)
            worst = int(np.argmax(fd_residuals))
            fd_residual = float(fd_residuals[worst])
            if fd_residual > FD_CONSISTENCY_TOLERANCE:
                consistency_problems.append(
                    f"the finite-difference energies disagree with the reference gradient by "
                    f"{fd_residual:.2e} Eh/bohr (tolerance {FD_CONSISTENCY_TOLERANCE:.0e}), worst "
                    f"for Cartesian coordinate {worst} (atom {molecule.symbols[worst // 3]}"
                    f"{worst // 3 + 1}, {'xyz'[worst % 3]}). The displaced calculations probably "
                    "landed on different electronic solutions than the reference: check the "
                    "active space and orbitals, or reduce [frequencies] fd_step_bohr."
                )
        if source == "optimize":
            optimized = read_json(ctx.upstream_dir("optimize") / "optimize.json")
            energy_difference = abs(
                float(centre["energy_hartree"]) - float(optimized["energy_hartree"])
            )
            if energy_difference > OPTIMIZED_ENERGY_TOLERANCE:
                consistency_problems.append(
                    f"the energy at the optimized geometry ({float(centre['energy_hartree']):.10f} "
                    f"Eh) differs from the optimize stage's energy "
                    f"({float(optimized['energy_hartree']):.10f} Eh) by {energy_difference:.2e} Eh "
                    f"(tolerance {OPTIMIZED_ENERGY_TOLERANCE:.0e}): the frequencies surface is not "
                    "the optimized one. Re-run with --rerun-from optimize."
                )
        masses = np.array([isotope_mass(s) for s in molecule.symbols])
        analysis = harmonic_analysis(masses, coords, hessian)
        freqs = analysis.frequencies_cm1
        cutoff = options.imaginary_cutoff_cm1
        big = [(i + 1, float(f)) for i, f in enumerate(freqs) if f < -cutoff]
        small = [(i + 1, float(f)) for i, f in enumerate(freqs) if -cutoff <= f < 0]
        max_gradient = float(np.abs(np.asarray(centre["gradient"])).max())
        warnings: list[str] = []
        if small:
            warnings.append(
                "small imaginary modes (within the cutoff): "
                + ", ".join(f"mode {i}: {f:.1f} cm^-1" for i, f in small)
            )
        if analysis.external_residual_cm1 > EXTERNAL_RESIDUAL_WARNING_CM1:
            warnings.append(
                f"translation/rotation contamination up to "
                f"{analysis.external_residual_cm1:.0f} cm^-1 (Hessian or geometry inaccurate)"
            )
        if max_gradient > STATIONARY_GRADIENT:
            warnings.append(
                f"max |gradient| = {max_gradient:.2e} Eh/bohr: not a stationary point; "
                "frequencies are approximate"
            )
        for message in warnings:
            log.warning("Stage frequencies: %s", message)
        if not consistency_problems:
            write_frequencies_h5(
                ctx.stage_dir / "frequencies.h5",
                symbols=molecule.symbols,
                masses_amu=masses,
                coords_bohr=coords,
                hessian=hessian,
                analysis=analysis,
            )
            write_text_atomic(
                ctx.stage_dir / "frequencies.molden",
                format_molden(molecule.symbols, coords, freqs, analysis.modes_cartesian),
            )
        write_json_atomic(
            ctx.stage_dir / "frequencies.json",
            {
                "schema_version": 1,
                "method": method,
                "geometry_source": source,
                "energy_hartree": centre.get("energy_hartree"),
                "max_gradient_hartree_per_bohr": max_gradient,
                "linear": analysis.linear,
                "n_external": analysis.n_external,
                "external_residual_cm1": analysis.external_residual_cm1,
                "fd_consistency_residual_hartree_per_bohr": fd_residual,
                "optimized_energy_difference_hartree": energy_difference,
                "consistency_problems": consistency_problems,
                "frequencies_cm1": freqs.tolist(),
                "reduced_masses_amu": analysis.reduced_masses_amu.tolist(),
                "imaginary_modes": [{"mode": i, "frequency_cm1": f} for i, f in big],
                "small_imaginary_modes": [{"mode": i, "frequency_cm1": f} for i, f in small],
                "warnings": warnings,
            },
        )
        if consistency_problems:
            raise StageFailedError(
                "The finite-difference/optimized-surface consistency check failed: "
                + " ".join(consistency_problems)
            )
        if big:
            listing = ", ".join(f"mode {i}: {f:.1f} cm^-1" for i, f in big)
            raise StageFailedError(
                f"The structure is not a minimum: {len(big)} imaginary mode(s) beyond "
                f"{cutoff:g} cm^-1 ({listing}). Re-optimize (e.g. [optimize] convergence = "
                '"GAU_TIGHT"), check the starting structure, or raise [frequencies] '
                "imaginary_cutoff_cm1 if the mode is a known numerical artefact. The modes are in "
                f"{ctx.stage_dir / 'frequencies.molden'}."
            )
