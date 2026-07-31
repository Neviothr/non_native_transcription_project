from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUI_SOURCE = ROOT / "src" / "transcription_app" / "gui.py"


class ManualReviewCheckboxRemovalTests(unittest.TestCase):
    def test_review_editor_has_no_manual_review_checkbox(self) -> None:
        source = GUI_SOURCE.read_text(encoding="utf-8")

        self.assertNotIn('text="Manual review required"', source)
        self.assertNotIn("editor_review_var", source)


if __name__ == "__main__":
    unittest.main()
