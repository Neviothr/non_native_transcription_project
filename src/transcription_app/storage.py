"""Project persistence helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import ProjectData


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def save_project(project: ProjectData, path: str | Path) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".ntproject":
        target = target.with_suffix(".ntproject")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not project.metadata.created_at:
        project.metadata.created_at = _now_iso()
    project.metadata.updated_at = _now_iso()
    project.project_file = str(target.resolve())
    target.write_text(
        json.dumps(project.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def load_project(path: str | Path) -> ProjectData:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    project = ProjectData.from_dict(data)
    project.project_file = str(source.resolve())
    return project