"""Load and validate the TOML input file; snapshot the resolved configuration."""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from gqteashmd.config.schema import ShmdConfig
from gqteashmd.errors import ConfigError
from gqteashmd.util.fileio import write_text_atomic


@dataclass(frozen=True)
class LoadedConfig:
    config: ShmdConfig
    source: Path  # absolute path of the input file (or of the snapshot it was read from)


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"Invalid input file {path}:"]
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "<top level>"
        lines.append(f"  - {location}: {err['msg']}")
    return "\n".join(lines)


def _resolve_paths(cfg: ShmdConfig, base: Path) -> ShmdConfig:
    def absolute(p: Path) -> Path:
        p = p.expanduser()
        return p if p.is_absolute() else (base / p).resolve()

    return cfg.model_copy(
        update={
            "project": cfg.project.model_copy(
                update={"directory": absolute(cfg.project.directory)}
            ),
            "molecule": cfg.molecule.model_copy(
                update={"geometry": absolute(cfg.molecule.geometry)}
            ),
        }
    )


def load_config(path: Path) -> LoadedConfig:
    path = Path(path).resolve()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Input file not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"Input file {path} is not valid UTF-8 text ({exc.reason} at byte {exc.start}). "
            "Re-save it as UTF-8."
        ) from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read input file {path}: {exc.strerror or exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Input file {path} is not valid TOML: {exc}") from exc
    try:
        cfg = ShmdConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc
    return LoadedConfig(config=_resolve_paths(cfg, path.parent), source=path)


def write_snapshot(loaded: LoadedConfig, path: Path) -> None:
    write_text_atomic(path, loaded.config.model_dump_json(indent=2) + "\n")


def load_snapshot(path: Path) -> LoadedConfig:
    path = Path(path)
    try:
        cfg = ShmdConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ConfigError(f"Cannot read config snapshot {path}: {exc}") from exc
    return LoadedConfig(config=cfg, source=path.resolve())


def section_fingerprint(config: ShmdConfig, sections: Iterable[str]) -> dict[str, Any]:
    """JSON-compatible dump of the named top-level sections (used for stage fingerprints)."""
    dumped = config.model_dump(mode="json")
    return {name: dumped[name] for name in sections}
