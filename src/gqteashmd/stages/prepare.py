"""Stage 0: validate the molecule and store a normalized copy in the project."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gqteashmd.chem.elements import atomic_number
from gqteashmd.chem.xyz import Geometry, format_xyz, parse_xyz
from gqteashmd.errors import ConfigError, TaskError
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.util.fileio import write_json_atomic, write_text_atomic


class PrepareStage(Stage):
    name = "prepare"
    version = 1
    config_sections = ("molecule",)

    def fingerprint_inputs(self, ctx: StageContext) -> dict[str, Any]:
        molecule = ctx.config.molecule
        try:
            digest = hashlib.sha256(molecule.geometry.read_bytes()).hexdigest()
        except FileNotFoundError as exc:
            raise ConfigError(f"Geometry file not found: {molecule.geometry}") from exc
        except OSError as exc:
            raise ConfigError(
                f"Cannot read geometry file {molecule.geometry}: {exc.strerror or exc}"
            ) from exc
        return {
            "charge": molecule.charge,
            "multiplicity": molecule.multiplicity,
            "geometry_sha256": digest,
        }

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        return [TaskSpec("molecule")]

    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        molecule = ctx.config.molecule
        try:
            geometry = parse_xyz(molecule.geometry.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise TaskError(
                f"Geometry file not found: {molecule.geometry}", retryable=False
            ) from exc
        except OSError as exc:
            raise TaskError(
                f"Cannot read geometry file {molecule.geometry}: {exc.strerror or exc}",
                retryable=False,
            ) from exc
        except ValueError as exc:
            raise TaskError(
                f"Invalid geometry file {molecule.geometry}: {exc}", retryable=False
            ) from exc
        numbers = [atomic_number(symbol) for symbol in geometry.symbols]
        n_electrons = sum(numbers) - molecule.charge
        unpaired = molecule.multiplicity - 1
        if n_electrons <= 0 or unpaired > n_electrons or n_electrons % 2 != unpaired % 2:
            raise TaskError(
                f"Charge {molecule.charge} and multiplicity {molecule.multiplicity} are "
                f"inconsistent with {n_electrons} electrons. Check [molecule] in the input file.",
                retryable=False,
            )
        return {
            "symbols": list(geometry.symbols),
            "atomic_numbers": numbers,
            "coords_angstrom": [list(c) for c in geometry.coords_angstrom],
            "charge": molecule.charge,
            "multiplicity": molecule.multiplicity,
            "n_electrons": n_electrons,
        }

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        data = dict(results["molecule"])
        geometry = Geometry(
            tuple(data["symbols"]),
            tuple((c[0], c[1], c[2]) for c in data["coords_angstrom"]),
            comment=ctx.config.project.name,
        )
        write_text_atomic(ctx.stage_dir / "geometry.xyz", format_xyz(geometry))
        write_json_atomic(ctx.stage_dir / "molecule.json", {"schema_version": 1, **data})
