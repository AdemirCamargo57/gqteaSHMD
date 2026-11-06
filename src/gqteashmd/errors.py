"""Error taxonomy. Every deliberate error raised by gqteashmd derives from ShmdError.

Exit codes: 0 ok, 1 generic, 2 configuration, 3 project/state, 4 stage failed,
5 stopped by user (defined in gqteashmd.cli).
"""

from __future__ import annotations


class ShmdError(Exception):
    """Base class for all errors raised deliberately by gqteashmd."""

    exit_code: int = 1


class ConfigError(ShmdError):
    """The input file or command-line arguments are invalid. Nothing has been run."""

    exit_code = 2


class ProjectError(ShmdError):
    """The project directory or its state cannot be used (lock held, corrupt state, ...)."""

    exit_code = 3


class StaleStageError(ProjectError):
    """A completed stage has results that were produced from different inputs."""


class StageFailedError(ShmdError):
    """A stage could not complete (too many failed tasks or failed physical checks)."""

    exit_code = 4


class TaskError(ShmdError):
    """A single task failed. `retryable` tells the controller whether to try again."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable
