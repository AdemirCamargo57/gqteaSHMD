"""Pipelines for tests that exercise orchestration without quantum chemistry."""

from __future__ import annotations

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.stages.base import Stage
from gqteashmd.stages.prepare import PrepareStage


def prepare_only(config: ShmdConfig) -> list[Stage]:
    return [PrepareStage()]
