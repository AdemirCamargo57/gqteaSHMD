"""The production pipeline. Later sub-plans append optimize, frequencies, sampling, ..."""

from __future__ import annotations

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.stages.base import Stage
from gqteashmd.stages.prepare import PrepareStage


def default_pipeline(config: ShmdConfig) -> list[Stage]:
    return [PrepareStage()]
