from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUI_SOURCE = ROOT / "src" / "transcription_app" / "gui.py"


class ReviewCorrectionTimerRemovalTests(unittest.TestCase):
    def test_review_editor_has_no_correction_timer_controls(self) -> None:
        source = GUI_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("Correction timer", source)
        self.assertNotIn("Start Timer", source)
        self.assertNotIn("Stop Timer", source)
        self.assertNotIn("toggle_timer", source)
        self.assertNotIn("_stop_timer", source)
        self.assertNotIn("timer_started_at", source)
        self.assertNotIn("timer_turn_index", source)

    def test_manual_correction_seconds_entry_is_retained(self) -> None:
        source = GUI_SOURCE.read_text(encoding="utf-8")
        self.assertIn('text="Manual correction seconds"', source)
        self.assertIn("self.correction_seconds_var", source)


if __name__ == "__main__":
    unittest.main()
