from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.gui import (
    PROGRESS_UPDATE_INTERVAL_SECONDS,
    _should_emit_item_progress,
    _transcription_progress,
)


class LongRunningProgressTests(unittest.TestCase):
    def test_whisper_stages_map_into_seven_stage_pipeline(self) -> None:
        progress = _transcription_progress(
            "Stage 5/5 - Temporary working files were removed."
        )

        self.assertIsNotNone(progress)
        percentage, summary = progress
        self.assertAlmostEqual(percentage, 5 / 7 * 100)
        self.assertIn("Stage 5 of 7", summary)

    def test_turn_count_advances_within_final_stage(self) -> None:
        percentage, summary = _transcription_progress(
            "Stage 7/7 - Analyzing turn 50 of 100..."
        )

        self.assertAlmostEqual(percentage, 6.5 / 7 * 100)
        self.assertIn("turn 50 of 100", summary)

    def test_completed_final_stage_reaches_one_hundred_percent(self) -> None:
        percentage, _summary = _transcription_progress(
            "Stage 7/7 - Initial analysis complete in 2.0 seconds."
        )

        self.assertEqual(percentage, 100.0)

    def test_intermediate_items_are_throttled_but_boundaries_are_kept(self) -> None:
        short = PROGRESS_UPDATE_INTERVAL_SECONDS / 2

        self.assertTrue(_should_emit_item_progress("Analyzing turn 1 of 100", short))
        self.assertFalse(_should_emit_item_progress("Analyzing turn 2 of 100", short))
        self.assertTrue(_should_emit_item_progress("Analyzing turn 50 of 100", PROGRESS_UPDATE_INTERVAL_SECONDS))
        self.assertTrue(_should_emit_item_progress("Analyzing turn 100 of 100", short))

    def test_unstructured_messages_are_not_suppressed(self) -> None:
        self.assertTrue(_should_emit_item_progress("Loading model", 0.0))
        self.assertIsNone(_transcription_progress("Loading model"))


if __name__ == "__main__":
    unittest.main()
