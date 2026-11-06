"""Directory layout of a gqteashmd project."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ProjectLayout:
    root: Path

    @property
    def meta_dir(self) -> Path:
        return self.root / ".gqteashmd"

    @property
    def state_db(self) -> Path:
        return self.meta_dir / "state.db"

    @property
    def lock_file(self) -> Path:
        return self.meta_dir / "controller.lock"

    @property
    def config_snapshot(self) -> Path:
        return self.meta_dir / "config.json"

    @property
    def provenance_file(self) -> Path:
        return self.meta_dir / "provenance.json"

    @property
    def superseded_dir(self) -> Path:
        return self.meta_dir / "superseded"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def stage_dir(self, index: int, name: str) -> Path:
        return self.root / f"{index:02d}_{name}"

    def task_dir(self, stage_dir: Path, task_id: str) -> Path:
        return stage_dir / "tasks" / task_id

    def ensure(self) -> None:
        for directory in (self.meta_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def supersede(self, stage_dir: Path) -> Path | None:
        """Move an old stage directory into .gqteashmd/superseded/ (never delete results)."""
        if not stage_dir.exists():
            return None
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        target = self.superseded_dir / f"{stamp}_{stage_dir.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stage_dir), str(target))
        return target
