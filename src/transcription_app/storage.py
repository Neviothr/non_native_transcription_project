"""Project persistence helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import ProjectData


class ProjectLoadError(RuntimeError):
    """Raised when a saved project cannot be read or decoded safely."""


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
    """Load one project and convert low-level failures into useful messages."""
    source = Path(path).expanduser()
    if not source.exists():
        raise ProjectLoadError(f"Project file not found: {source}")
    if not source.is_file():
        raise ProjectLoadError(f"Project path is not a file: {source}")

    try:
        raw_text = source.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise ProjectLoadError(
            f"The project is not valid UTF-8 text: {source.name}"
        ) from exc
    except OSError as exc:
        raise ProjectLoadError(f"Could not read project file: {exc}") from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ProjectLoadError(
            f"The project file contains invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise ProjectLoadError("The project file must contain one JSON object.")

    try:
        project = ProjectData.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectLoadError(
            f"The project data is corrupt or incompatible: {exc}"
        ) from exc

    project.project_file = str(source.resolve())
    return project
