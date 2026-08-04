from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app import __version__
from transcription_app.models import ProjectData
from transcription_app.storage import ProjectLoadError, load_project, save_project


class ProjectLoadTests(unittest.TestCase):
    def test_invalid_json_has_a_clear_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.ntproject"
            path.write_text('{"turns": [}', encoding="utf-8")

            with self.assertRaises(ProjectLoadError) as caught:
                load_project(path)

        message = str(caught.exception)
        self.assertIn("invalid JSON", message)
        self.assertIn("line", message)
        self.assertIn("column", message)

    def test_unknown_fields_are_rejected(self) -> None:
        raw = ProjectData().to_dict()
        raw["application_version"] = __version__
        raw["future_project_field"] = True

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unexpected-field.ntproject"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ProjectLoadError) as caught:
                load_project(path)

        self.assertIn("unexpected fields: future_project_field", str(caught.exception))

    def test_saved_project_records_current_application_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = save_project(ProjectData(), Path(directory) / "current.ntproject")
            raw = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(raw["application_version"], __version__)

    def test_project_without_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unversioned.ntproject"
            path.write_text(json.dumps({"metadata": {}, "turns": []}), encoding="utf-8")

            with self.assertRaises(ProjectLoadError) as caught:
                load_project(path)

        self.assertIn("Saved version: missing", str(caught.exception))

    def test_project_from_older_version_is_rejected(self) -> None:
        raw = {
            "application_version": "1.6.3",
            "metadata": {},
            "turns": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "older.ntproject"
            path.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaises(ProjectLoadError) as caught:
                load_project(path)

        message = str(caught.exception)
        self.assertIn(__version__, message)
        self.assertIn("1.6.3", message)


if __name__ == "__main__":
    unittest.main()
