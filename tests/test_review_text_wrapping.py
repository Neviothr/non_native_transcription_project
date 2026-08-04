from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.gui import (
    MAX_REVIEW_TREE_LINES,
    _review_tree_rowheight,
    _wrap_turn_table_text,
)


class ReviewTextWrappingTests(unittest.TestCase):
    def test_wrapping_preserves_the_complete_transcript(self) -> None:
        original = (
            "This is a deliberately long final transcription containing enough "
            "words to require several wrapped lines without losing any content."
        )

        wrapped = _wrap_turn_table_text(original, width=24)

        self.assertIn("\n", wrapped)
        self.assertEqual(wrapped.replace("\n", " "), original)
        self.assertNotIn("...", wrapped)

    def test_existing_newlines_are_normalized_before_wrapping(self) -> None:
        wrapped = _wrap_turn_table_text("first line\nsecond line", width=40)

        self.assertEqual(wrapped, "first line second line")

    def test_row_height_expands_with_wrapped_line_count(self) -> None:
        self.assertEqual(_review_tree_rowheight(1), 28)
        self.assertGreater(_review_tree_rowheight(4), _review_tree_rowheight(2))
        self.assertEqual(
            _review_tree_rowheight(1_000_000),
            _review_tree_rowheight(MAX_REVIEW_TREE_LINES),
        )


if __name__ == "__main__":
    unittest.main()
