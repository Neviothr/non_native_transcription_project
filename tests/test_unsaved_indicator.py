from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.gui import TranscriptionApp


class _TitleHarness:
    def __init__(self, project_file: str = "") -> None:
        self.project = type("Project", (), {"project_file": project_file})()
        self._project_dirty = False
        self.rendered_title = ""

    def title(self, value: str) -> None:
        self.rendered_title = value

    def _update_window_title(self) -> None:
        TranscriptionApp._update_window_title(self)


class UnsavedIndicatorTests(unittest.TestCase):
    def test_dirty_saved_project_shows_filename_and_asterisk(self) -> None:
        app = _TitleHarness("C:/projects/interview.ntproject")

        TranscriptionApp._set_project_dirty(app, True)

        self.assertIn("interview.ntproject", app.rendered_title)
        self.assertTrue(app.rendered_title.endswith(" *"))

    def test_clean_project_removes_asterisk(self) -> None:
        app = _TitleHarness("C:/projects/interview.ntproject")
        TranscriptionApp._set_project_dirty(app, True)

        TranscriptionApp._set_project_dirty(app, False)

        self.assertFalse(app.rendered_title.endswith(" *"))

    def test_successful_save_paths_clear_dirty_state(self) -> None:
        manual_source = inspect.getsource(TranscriptionApp.save_project)
        autosave_source = inspect.getsource(TranscriptionApp._autosave_review_changes)

        self.assertIn("self._set_project_dirty(False)", manual_source)
        self.assertIn("self._set_project_dirty(False)", autosave_source)


if __name__ == "__main__":
    unittest.main()
