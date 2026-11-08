"""Pipelines for tests that exercise orchestration without quantum chemistry."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.stages.base import Stage, StageContext, TaskSpec
from gqteashmd.stages.prepare import PrepareStage
from gqteashmd.stages.selection import SelectionStage


def prepare_only(config: ShmdConfig) -> list[Stage]:
    return [PrepareStage()]


class CannedExcitationsStage(Stage):
    """Stands in for `excitations`: writes a fixed `excitations.h5` (20 samples, S1 and S2)."""

    name = "excitations"
    version = 1
    required_modules = ("h5py>=3.9",)
    N_SAMPLES = 20

    def plan(self, ctx: StageContext) -> list[TaskSpec]:
        return [TaskSpec("write")]

    def run_task(self, ctx: StageContext, task: TaskSpec, task_dir: Path) -> dict[str, Any]:
        return {}

    def finalize(self, ctx: StageContext, results: Mapping[str, Mapping[str, Any]]) -> None:
        from gqteashmd.excite.artifacts import write_excitations_h5

        n = self.N_SAMPLES
        energies = np.full((n, 3), -75.0)
        energies[:, 1] += 0.20 + 0.001 * np.arange(n)
        energies[:, 2] += 0.30 + 0.001 * np.arange(n)
        write_excitations_h5(
            ctx.stage_dir / "excitations.h5",
            symbols=("O", "H", "H"),
            state_labels=("S0", "S1", "S2"),
            sample_indices=np.arange(1, n + 1),
            ok=np.ones(n, dtype=bool),
            energies_hartree=energies,
            transition_dipoles_au=np.zeros((n, 2, 3)),
            oscillator_strengths=np.tile([0.01, 0.2], (n, 1)),
        )


def canned_selection(config: ShmdConfig) -> list[Stage]:
    """`excitations` (canned data) followed by the real `selection` stage."""
    return [CannedExcitationsStage(), SelectionStage()]
