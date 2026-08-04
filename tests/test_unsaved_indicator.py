from __future__ import annotations

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
    def test_title_tracks_dirty_state_for_a_saved_project(self) -> None:
        app = _TitleHarness("C:/projects/interview.ntproject")

        TranscriptionApp._set_project_dirty(app, True)

        self.assertIn("interview.ntproject", app.rendered_title)
        self.assertTrue(app.rendered_title.endswith(" *"))

        TranscriptionApp._set_project_dirty(app, False)

        self.assertFalse(app.rendered_title.endswith(" *"))


if __name__ == "__main__":
    unittest.main()
