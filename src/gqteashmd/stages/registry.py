"""The production pipeline: prepare, optimize, frequencies. Later sub-plans append sampling, ..."""

from __future__ import annotations

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.stages.base import Stage
from gqteashmd.stages.frequencies import FrequenciesStage
from gqteashmd.stages.optimize import OptimizeStage
from gqteashmd.stages.prepare import PrepareStage


def default_pipeline(config: ShmdConfig) -> list[Stage]:
    return [PrepareStage(), OptimizeStage(), FrequenciesStage()]
