from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from transcription_app.gui import REVIEW_AUTOSAVE_DELAY_MS, TranscriptionApp


class ReviewAutosaveTests(unittest.TestCase):
    def test_review_widgets_schedule_debounced_autosave(self) -> None:
        source = inspect.getsource(TranscriptionApp._build_review_tab)

        self.assertIn('trace_add("write", self._schedule_review_autosave)', source)
        self.assertIn('bind("<<Modified>>", self._on_final_text_modified)', source)
        self.assertGreaterEqual(REVIEW_AUTOSAVE_DELAY_MS, 1_000)

    def test_autosave_requires_existing_project_path_and_uses_no_dialog(self) -> None:
        source = inspect.getsource(TranscriptionApp._autosave_review_changes)

        self.assertIn("not self.project.project_file", source)
        self.assertIn("save_project(self.project, self.project.project_file)", source)
        self.assertNotIn("asksaveasfilename", source)
        self.assertIn("Autosaved", source)

    def test_manual_save_and_close_cancel_pending_autosave(self) -> None:
        save_source = inspect.getsource(TranscriptionApp.save_project)
        close_source = inspect.getsource(TranscriptionApp._on_close)

        self.assertIn("self._cancel_review_autosave()", save_source)
        self.assertIn("self._cancel_review_autosave()", close_source)


if __name__ == "__main__":
    unittest.main()
