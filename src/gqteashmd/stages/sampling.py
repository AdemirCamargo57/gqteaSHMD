"""Stage 3: harmonic Wigner sampling of initial geometries and velocities."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.core.units import DALTON_IN_ME
from gqteashmd.errors import StageFailedError, TaskError
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.util.fileio import write_json_atomic
from gqteashmd.util.seeds import task_rng
from gqteashmd.vib.artifacts import format_samples_xyz, read_frequencies_h5, write_samples_h5
from gqteashmd.vib.wigner import sample_wigner

log = logging.getLogger("gqteashmd.stages.sampling")
LINEAR_MOMENTUM_TOLERANCE = 1e-8  # a.u.
ANGULAR_MOMENTUM_TOLERANCE = 1e-6  # a.u.
ENERGY_FALSE_ALARM_PROBABILITY = 1e-9  # total, for a correct program, whatever N


def mode_energy_p_values(ratio_sums: Any, n_samples: int) -> np.ndarray:
    """Two-sided tail probability of S_k ~ Gamma(n_samples, 1), per sampled mode."""
    from scipy.stats import gamma

    sums = np.asarray(ratio_sums, dtype=float)
    p = 2.0 * np.minimum(gamma.cdf(sums, n_samples), gamma.sf(sums, n_samples))
    return np.where(np.isfinite(sums), np.minimum(p, 1.0), 0.0)


def _momenta(
    masses_me: np.ndarray, positions: np.ndarray, velocities: np.ndarray
) -> tuple[float, float]:
    """Largest |P| and |L| (about the centre of mass) over the samples, in atomic units."""
    m = masses_me[None, :, None]
    linear = np.linalg.norm((m * velocities).sum(axis=1), axis=1)
    centre = (m * positions).sum(axis=1) / masses_me.sum()
    angular = np.linalg.norm(
        (m * np.cross(positions - centre[:, None, :], velocities)).sum(axis=1), axis=1
    )
    return float(linear.max()), float(angular.max())


class SamplingStage(Stage):
    name = "sampling"
    version = 1
    depends_on = ("frequencies",)
    config_sections = ("sampling",)
    required_modules = ("h5py>=3.9",)

    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        return {
            "sampling": ctx.config.sampling.model_dump(mode="json"),
            "project_seed": ctx.config.project.seed,
        }

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        return [TaskSpec("samples")]

    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        options = ctx.config.sampling
        seed = ctx.config.project.seed
        try:
            data = read_frequencies_h5(ctx.upstream_dir("frequencies") / "frequencies.h5")
        except ValueError as exc:
            raise TaskError(str(exc), retryable=False) from exc
        rng = task_rng(seed, self.name, 0)
        samples = sample_wigner(
            data["masses_amu"],
            data["coords_bohr"],
            data["frequencies_cm1"],
            data["modes_mass_weighted"],
            n_samples=options.n_samples,
            temperature_K=options.temperature_K,
            low_frequency_cutoff_cm1=options.low_frequency_cutoff_cm1,
            rng=rng,
        )
        if not (
            np.all(np.isfinite(samples.positions_bohr))
            and np.all(np.isfinite(samples.velocities_au))
            and np.all(np.isfinite(samples.harmonic_energy_hartree))
        ):
            raise TaskError(
                "The sampler produced non-finite positions, velocities or energies "
                f"(temperature_K = {options.temperature_K:g}, "
                f"low_frequency_cutoff_cm1 = {options.low_frequency_cutoff_cm1:g}); "
                "check the [sampling] values and frequencies.h5",
                retryable=False,
            )
        write_samples_h5(
            task_dir / "samples.h5",
            symbols=data["symbols"],
            masses_amu=data["masses_amu"],
            coords_bohr=data["coords_bohr"],
            frequencies_cm1=data["frequencies_cm1"],
            samples=samples,
            temperature_K=options.temperature_K,
            low_frequency_cutoff_cm1=options.low_frequency_cutoff_cm1,
            project_seed=seed,
        )
        (task_dir / "samples.xyz").write_text(
            format_samples_xyz(
                data["symbols"], samples.positions_bohr, samples.harmonic_energy_hartree
            )
        )
        max_p, max_l = _momenta(
            np.asarray(data["masses_amu"]) * DALTON_IN_ME,
            samples.positions_bohr,
            samples.velocities_au,
        )
        freqs = np.asarray(data["frequencies_cm1"], dtype=float)
        return {
            "n_samples": options.n_samples,
            "n_vibrations": int(freqs.size),
            "n_sampled_modes": int(samples.sampled_modes.sum()),
            "mean_energy_hartree": float(samples.harmonic_energy_hartree.mean()),
            "expected_mean_energy_hartree": samples.expected_mean_energy_hartree,
            "expected_energy_std_hartree": samples.expected_energy_std_hartree,
            "max_linear_momentum_au": max_p,
            "max_angular_momentum_au": max_l,
            "mean_kinetic_energy_hartree": float(samples.kinetic_energy_hartree.mean()),
            "mean_removed_kinetic_energy_hartree": float(
                samples.removed_kinetic_energy_hartree.mean()
            ),
            "mode_energy_ratio_sums": samples.mode_energy_ratio_sums().tolist(),
            "unsampled_frequencies_cm1": freqs[~samples.sampled_modes].tolist(),
        }

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        options = ctx.config.sampling
        summary = dict(results["samples"])
        unsampled = summary["unsampled_frequencies_cm1"]
        if summary["n_vibrations"] > 0 and summary["n_sampled_modes"] == 0:
            raise StageFailedError(
                f"No vibrational mode is sampled: every mode is below the cutoff "
                f"{options.low_frequency_cutoff_cm1:g} cm^-1 (frequencies: "
                + ", ".join(f"{f:.1f}" for f in unsampled)
                + "). Please lower [sampling] low_frequency_cutoff_cm1."
            )
        if unsampled:
            log.warning(
                "Stage sampling: %d mode(s) below the cutoff are not sampled "
                "(frequencies in cm^-1: %s)",
                len(unsampled),
                ", ".join(f"{f:.1f}" for f in unsampled),
            )
        n = int(summary["n_samples"])
        measured = float(summary["mean_energy_hartree"])
        expected = float(summary["expected_mean_energy_hartree"])
        ratio_sums = [float(x) for x in summary["mode_energy_ratio_sums"]]
        p_values = mode_energy_p_values(ratio_sums, n)
        threshold = ENERGY_FALSE_ALARM_PROBABILITY / max(len(ratio_sums), 1)
        min_p = float(p_values.min()) if ratio_sums else None
        task_dir = ctx.layout.task_dir(ctx.stage_dir, "samples")
        for name in ("samples.h5", "samples.xyz"):
            tmp = ctx.stage_dir / f".{name}.{os.getpid()}.tmp"
            try:
                shutil.copyfile(task_dir / name, tmp)
                os.replace(tmp, ctx.stage_dir / name)
            finally:
                tmp.unlink(missing_ok=True)
        write_json_atomic(
            ctx.stage_dir / "samples.json",
            {
                "schema_version": 1,
                **summary,
                "temperature_K": options.temperature_K,
                "low_frequency_cutoff_cm1": options.low_frequency_cutoff_cm1,
                "project_seed": ctx.config.project.seed,
                "mode_energy_min_p_value": min_p,
                "mode_energy_p_value_threshold": threshold if ratio_sums else None,
            },
        )
        if min_p is not None and not min_p >= threshold:
            k = int(np.argmin(p_values))
            raise StageFailedError(
                "The sampled harmonic energies are inconsistent with the Wigner distribution "
                f"(mode {k}: sum of E/<E> over {n} samples = {ratio_sums[k]:.6g}, expected "
                f"Gamma({n}, 1); two-sided p = {p_values[k]:.2e} < {threshold:.1e}; mean "
                f"{measured:.8e} vs expected {expected:.8e} Eh); "
                "this indicates a program error — please report it"
            )
        if not (
            summary["max_linear_momentum_au"] <= LINEAR_MOMENTUM_TOLERANCE
            and summary["max_angular_momentum_au"] <= ANGULAR_MOMENTUM_TOLERANCE
        ):
            raise StageFailedError(
                "Total momentum was not removed from the samples (max |P| = "
                f"{summary['max_linear_momentum_au']:.2e}, max |L| = "
                f"{summary['max_angular_momentum_au']:.2e} a.u.); this indicates a program "
                "error — please report it"
            )
