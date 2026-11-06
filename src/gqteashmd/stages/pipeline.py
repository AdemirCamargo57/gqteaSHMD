"""Locate and validate the ordered list of stages."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.errors import ProjectError
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
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ProjectError(f"Module {module_name!r} has no attribute {attribute!r}") from exc


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
