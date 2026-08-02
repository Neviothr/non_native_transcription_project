from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.gui import (
    MAX_REVIEW_TREE_LINES,
    TranscriptionApp,
    _review_tree_rowheight,
)
from transcription_app import __version__
from transcription_app.models import ProjectData
from transcription_app.storage import ProjectLoadError, load_project, save_project


class ProjectOpenGuiTests(unittest.TestCase):
    def test_open_runs_file_work_in_a_worker_and_dispatches_to_tk(self) -> None:
        source = inspect.getsource(TranscriptionApp.open_project)

        self.assertIn("threading.Thread(", source)
        self.assertIn("project-open-worker", source)
        self.assertIn("self._post_to_ui(", source)
        self.assertIn('mode="determinate"', source)
        self.assertIn("report_progress(", source)

    def test_open_does_not_reload_or_realign_saved_transcripts(self) -> None:
        source = inspect.getsource(TranscriptionApp.open_project)

        self.assertNotIn("reload_selected_transcripts(", source)
        self.assertNotIn("align_all_sources(", source)
        self.assertNotIn("recover_speaker_mapping(", source)
        self.assertNotIn("analyze_turns(", source)

    def test_open_has_success_and_failure_logging(self) -> None:
        success_source = inspect.getsource(TranscriptionApp._project_open_finished)
        failure_source = inspect.getsource(TranscriptionApp._project_open_failed)

        self.assertIn("PROJECT OPEN COMPLETED", success_source)
        self.assertIn("were not reloaded or re-aligned", success_source)
        self.assertIn("PROJECT OPEN FAILED", failure_source)
        self.assertIn("current project was left unchanged", failure_source)

    def test_review_row_height_is_capped_for_extreme_turns(self) -> None:
        capped = _review_tree_rowheight(MAX_REVIEW_TREE_LINES)

        self.assertEqual(_review_tree_rowheight(1_000_000), capped)
        self.assertGreater(capped, _review_tree_rowheight(2))


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
