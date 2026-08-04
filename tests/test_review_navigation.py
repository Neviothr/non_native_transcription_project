from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.gui import _relative_review_index, _review_position
from transcription_app.models import Turn


class ReviewNavigationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.turns = [
            Turn(turn_id=1, manual_review=False),
            Turn(turn_id=2, manual_review=True),
            Turn(turn_id=3, manual_review=False),
            Turn(turn_id=4, manual_review=True),
        ]

    def test_navigation_skips_unflagged_turns_and_wraps(self) -> None:
        cases = (
            (1, 1, 3),
            (3, -1, 1),
            (3, 1, 1),
            (1, -1, 3),
            (None, 1, 1),
            (None, -1, 3),
        )
        for current, direction, expected in cases:
            with self.subTest(current=current, direction=direction):
                self.assertEqual(
                    _relative_review_index(self.turns, current, direction),
                    expected,
                )

    def test_review_position_counts_only_flagged_turns(self) -> None:
        self.assertEqual(_review_position(self.turns, 3), (2, 2))
        self.assertEqual(_review_position(self.turns, 2), (None, 2))

    def test_empty_review_queue_has_no_destination(self) -> None:
        turns = [Turn(turn_id=1, manual_review=False)]

        self.assertIsNone(_relative_review_index(turns, 0, 1))
        self.assertEqual(_review_position(turns, 0), (None, 0))


if __name__ == "__main__":
    unittest.main()
