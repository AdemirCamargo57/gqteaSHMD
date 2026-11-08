"""Reproducible random streams: one master seed (project.seed) → per-stage → per-task (D12)."""

from __future__ import annotations

import zlib

import numpy as np


def stage_seed_sequence(project_seed: int, stage_name: str) -> np.random.SeedSequence:
    return np.random.SeedSequence([project_seed, zlib.crc32(stage_name.encode("utf-8"))])


def task_rng(project_seed: int, stage_name: str, task_index: int) -> np.random.Generator:
    """The generator of task `task_index`; independent of the total number of tasks."""
    child = stage_seed_sequence(project_seed, stage_name).spawn(task_index + 1)[task_index]
    return np.random.default_rng(child)
