"""The production pipeline: prepare, optimize, frequencies, sampling, excitations, selection,
dynamics (optional: a [dynamics] table), analysis (after dynamics; [analysis] enabled)."""

from __future__ import annotations

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.stages.analysis import AnalysisStage
from gqteashmd.stages.base import Stage
from gqteashmd.stages.dynamics import DynamicsStage
from gqteashmd.stages.excitations import ExcitationsStage
from gqteashmd.stages.frequencies import FrequenciesStage
from gqteashmd.stages.optimize import OptimizeStage
from gqteashmd.stages.prepare import PrepareStage
from gqteashmd.stages.sampling import SamplingStage
from gqteashmd.stages.selection import SelectionStage


def default_pipeline(config: ShmdConfig) -> list[Stage]:
    return [
        PrepareStage(),
        OptimizeStage(),
        FrequenciesStage(),
        SamplingStage(),
        ExcitationsStage(max_failed_fraction=config.excitations.max_failed_fraction),
        SelectionStage(),
        DynamicsStage(max_failed_fraction=dynamics_max_failed_fraction(config)),
        AnalysisStage(),
    ]


def dynamics_max_failed_fraction(config: ShmdConfig) -> float:
    return config.dynamics.max_failed_fraction if config.dynamics is not None else 0.1
