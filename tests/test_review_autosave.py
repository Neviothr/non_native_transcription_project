from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.gui import REVIEW_AUTOSAVE_DELAY_MS, TranscriptionApp
from transcription_app.models import ProjectData


class _ScheduleHarness:
    def __init__(self) -> None:
        self._loading_editor = False
        self._saving_editor = False
        self._closing = False
        self.current_turn_index = 0
        self._review_autosave_after_id = "old"
        self.dirty = False
        self.cancelled: list[str] = []
        self.scheduled: list[tuple[int, object]] = []

    def _mark_project_dirty(self) -> None:
        self.dirty = True

    def _cancel_review_autosave(self) -> None:
        TranscriptionApp._cancel_review_autosave(self)

    def _autosave_review_changes(self) -> None:
        TranscriptionApp._autosave_review_changes(self)

    def after_cancel(self, identifier: str) -> None:
        self.cancelled.append(identifier)

    def after(self, delay: int, callback) -> str:
        self.scheduled.append((delay, callback))
        return "new"


class _SaveHarness:
    def __init__(self, project_file: str = "") -> None:
        self.project = ProjectData(project_file=project_file)
        self._closing = False
        self._review_autosave_after_id = "pending"
        self.dirty = True
        self.cancel_count = 0
        self.editor_saves: list[tuple[bool, bool]] = []
        self.status = ""

    def _cancel_review_autosave(self) -> None:
        self.cancel_count += 1

    def save_editor_to_turn(self, silent=False, *, refresh_table=True) -> None:
        self.editor_saves.append((silent, refresh_table))

    def _sync_metadata_from_ui(self) -> None:
        pass

    def _append_log(self, _message: str) -> None:
        pass

    def _set_project_dirty(self, dirty: bool) -> None:
        self.dirty = dirty

    def _set_status(self, message: str) -> None:
        self.status = message


class ReviewAutosaveTests(unittest.TestCase):
    def test_changes_replace_the_pending_debounced_autosave(self) -> None:
        app = _ScheduleHarness()

        TranscriptionApp._schedule_review_autosave(app)

        self.assertTrue(app.dirty)
        self.assertEqual(app.cancelled, ["old"])
        self.assertEqual(app._review_autosave_after_id, "new")
        self.assertEqual(app.scheduled[0][0], REVIEW_AUTOSAVE_DELAY_MS)
        self.assertGreaterEqual(REVIEW_AUTOSAVE_DELAY_MS, 1_000)

    def test_autosave_requires_an_existing_project_path(self) -> None:
        app = _SaveHarness()
        with patch("transcription_app.gui.save_project") as save:
            TranscriptionApp._autosave_review_changes(app)
            save.assert_not_called()

            app.project.project_file = "C:/projects/interview.ntproject"
            save.return_value = Path(app.project.project_file)
            TranscriptionApp._autosave_review_changes(app)

        save.assert_called_once_with(app.project, app.project.project_file)
        self.assertEqual(app.editor_saves, [(True, False)])
        self.assertFalse(app.dirty)
        self.assertTrue(app.status.startswith("Autosaved interview.ntproject"))

    def test_manual_save_cancels_autosave_and_clears_dirty_state(self) -> None:
        app = _SaveHarness("C:/projects/interview.ntproject")
        with patch(
            "transcription_app.gui.save_project",
            return_value=Path(app.project.project_file),
        ):
            saved = TranscriptionApp.save_project(app)

        self.assertTrue(saved)
        self.assertEqual(app.cancel_count, 1)
        self.assertFalse(app.dirty)


if __name__ == "__main__":
    unittest.main()
