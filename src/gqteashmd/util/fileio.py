"""Atomic file writes and JSON helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_text_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` so that readers never see a partially written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: Path, data: Any) -> None:
    write_text_atomic(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
