from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.models import Turn
from transcription_app.workflow import _mark_overlaps


class OverlapDetectionTests(unittest.TestCase):
    def test_marks_both_turns_when_different_speakers_overlap(self) -> None:
        later = Turn(turn_id=2, speaker_raw="Teacher", start=1.0, end=2.0)
        earlier = Turn(turn_id=1, speaker_raw="Student", start=0.0, end=1.5)

        _mark_overlaps([later, earlier])

        self.assertTrue(earlier.overlapping_speech)
        self.assertTrue(later.overlapping_speech)

    def test_ignores_same_speaker_and_threshold_overlap(self) -> None:
        turns = [
            Turn(turn_id=1, speaker_raw="Student", start=0.0, end=1.0),
            Turn(turn_id=2, speaker_raw="Student", start=0.5, end=1.5),
            Turn(turn_id=3, speaker_raw="Teacher", start=1.35, end=2.0),
        ]

        _mark_overlaps(turns)

        self.assertFalse(any(turn.overlapping_speech for turn in turns))

    def test_clears_stale_flags_and_skips_untimed_turns(self) -> None:
        turn = Turn(turn_id=1, overlapping_speech=True)

        _mark_overlaps([turn])

        self.assertFalse(turn.overlapping_speech)


if __name__ == "__main__":
    unittest.main()
