"""Locate and validate the ordered list of stages."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.errors import ConfigError, ProjectError
from gqteashmd.stages.base import Stage, is_safe_name

PipelineFactory = Callable[[ShmdConfig], list[Stage]]
DEFAULT_PIPELINE = "gqteashmd.stages.registry:default_pipeline"


def import_pipeline(ref: str) -> PipelineFactory:
    module_name, sep, attribute = ref.partition(":")
    if not sep or not module_name or not attribute:
        raise ProjectError(f"Invalid pipeline reference {ref!r}; expected 'module:function'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ProjectError(f"Cannot import pipeline module {module_name!r}: {exc}") from exc
    try:
        factory: PipelineFactory = getattr(module, attribute)
        return factory
    except AttributeError as exc:
        raise ProjectError(f"Module {module_name!r} has no attribute {attribute!r}") from exc


def check_stage_names(config: ShmdConfig, stage_names: Sequence[str]) -> None:
    """Raise ConfigError if the input file names a stage the pipeline does not have."""
    names = list(stage_names)
    unknown = sorted(set(config.resources.stages) - set(names))
    if unknown:
        raise ConfigError(
            f"[resources.stages] names unknown stage(s): {', '.join(unknown)}; "
            f"stages are: {', '.join(names)}"
        )
    unknown_pauses = sorted(set(config.workflow.pause_after) - set(names))
    if unknown_pauses:
        raise ConfigError(
            f"workflow.pause_after names unknown stage(s): {', '.join(unknown_pauses)}; "
            f"stages are: {', '.join(names)}"
        )


def check_stage_inputs(config: ShmdConfig, stages: Sequence[Stage]) -> None:
    """Every enabled stage's input sections must be present (optional sections default)."""
    missing = [
        f"Stage '{stage.name}' needs an [{section}] section in the input file"
        for stage in stages
        if stage.enabled(config)
        for section in stage.config_sections
        if getattr(config, section, None) is None
    ]
    missing += [
        problem
        for stage in stages
        if stage.enabled(config)
        for problem in stage.check_config(config)
    ]
    if missing:
        raise ConfigError("; ".join(missing) + ". See USER_MANUAL.md §4.")


def validate_pipeline(stages: Sequence[Stage]) -> None:
    seen: set[str] = set()
    for stage in stages:
        if not is_safe_name(stage.name):
            raise ProjectError(f"Invalid stage name {stage.name!r}")
        if stage.name in seen:
            raise ProjectError(f"Duplicate stage name {stage.name!r} in pipeline")
        for dependency in stage.depends_on:
            if dependency not in seen:
                raise ProjectError(
                    f"Stage {stage.name!r} depends on {dependency!r}, "
                    "which does not come earlier in the pipeline"
                )
        seen.add(stage.name)
