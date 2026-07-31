from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.gui import REVIEW_TURN_COLUMNS


class ReviewAudioControlTests(unittest.TestCase):
    def test_review_table_omits_redundant_review_column(self) -> None:
        self.assertNotIn("review", REVIEW_TURN_COLUMNS)

    def test_review_table_contains_audio_column_before_transcript(self) -> None:
        self.assertIn("listen", REVIEW_TURN_COLUMNS)
        self.assertLess(
            REVIEW_TURN_COLUMNS.index("listen"),
            REVIEW_TURN_COLUMNS.index("text"),
        )


if __name__ == "__main__":
    unittest.main()
