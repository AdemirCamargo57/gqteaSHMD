"""Logging: INFO to the console, DEBUG to the project log file."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# Pass as `extra=FILE_ONLY` for records that must reach the log file but not the console
# (e.g. errors the CLI already prints in its own format).
FILE_ONLY: dict[str, bool] = {"console": False}


class _ConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "console", True))


def _remove_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def configure_logging(log_file: Path | None, *, verbose: bool = False) -> None:
    logger = logging.getLogger("gqteashmd")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    _remove_handlers(logger)
    formatter = logging.Formatter(_FORMAT, "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    console.addFilter(_ConsoleFilter())
    logger.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def close_logging() -> None:
    """Close all gqteashmd handlers and return to the silent import-time state."""
    logger = logging.getLogger("gqteashmd")
    _remove_handlers(logger)
    logger.addHandler(logging.NullHandler())
    logger.propagate = True
