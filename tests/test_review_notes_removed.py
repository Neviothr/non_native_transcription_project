from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUI_SOURCE = ROOT / "src" / "transcription_app" / "gui.py"


class ReviewNotesRemovalTests(unittest.TestCase):
    def test_review_editor_has_no_notes_widget(self) -> None:
        source = GUI_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn('text="Notes"', source)
        self.assertNotIn("self.notes_text", source)

    def test_save_turn_tooltip_does_not_mention_notes(self) -> None:
        source = GUI_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("final transcript, notes, and correction time", source)


if __name__ == "__main__":
    unittest.main()