from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.models import Turn
from transcription_app.quality import update_turn_quality
from transcription_app.workflow import choose_initial_text


class _FixedPredictor:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities

    def predict_proba(self, _row: list[float]) -> list[float]:
        return list(self.probabilities)


class ReviewOptimizationTests(unittest.TestCase):
    def test_duplicate_source_wording_counts_as_two_votes(self) -> None:
        turn = Turn(
            turn_id=1,
            model_text="I went to the school yesterday",
            chatgpt_text="I go to school yesterday",
            zoom_text="I go to school yesterday",
        )

        selected = choose_initial_text(turn)

        self.assertEqual(selected, "I go to school yesterday")

    def test_high_consensus_minor_turn_is_auto_cleared(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker="Student",
            zoom_text="I go to school yesterday",
            chatgpt_text="I go to school yesterday",
            model_text="I went to the school yesterday",
            final_text="I go to school yesterday",
            model_confidence=0.70,
        )

        update_turn_quality(turn)

        self.assertEqual(turn.quality_label, "Needs minor correction")
        self.assertFalse(turn.manual_review)

    def test_low_consensus_minor_turn_still_requires_review(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker="Student",
            zoom_text="I go to school yesterday",
            chatgpt_text="I went school yesterday",
            model_text="I go to the school last day",
            final_text="I go to school yesterday",
            model_confidence=0.70,
        )

        update_turn_quality(turn)

        self.assertEqual(turn.quality_label, "Needs minor correction")
        self.assertTrue(turn.manual_review)

    def test_hard_risk_overrides_acceptable_ml_prediction(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker="Student",
            zoom_text="I said [unclear] yesterday",
            chatgpt_text="I said [unclear] yesterday",
            model_text="I said [unclear] yesterday",
            final_text="I said [unclear] yesterday",
            model_confidence=0.90,
        )

        update_turn_quality(
            turn,
            _FixedPredictor([0.95, 0.04, 0.01]),
        )

        self.assertEqual(turn.quality_label, "Transcript acceptable")
        self.assertTrue(turn.manual_review)

    def test_ml_boundary_minor_is_auto_cleared_only_with_consensus(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker="Student",
            zoom_text="I go to school yesterday",
            chatgpt_text="I go to school yesterday",
            model_text="I went to the school yesterday",
            final_text="I go to school yesterday",
            model_confidence=0.70,
        )

        update_turn_quality(
            turn,
            _FixedPredictor([0.49, 0.50, 0.01]),
        )

        self.assertEqual(turn.quality_label, "Needs minor correction")
        self.assertFalse(turn.manual_review)

    def test_unknown_speaker_is_never_auto_cleared(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker="Unknown",
            zoom_text="I go to school yesterday",
            chatgpt_text="I go to school yesterday",
            model_text="I go to school yesterday",
            final_text="I go to school yesterday",
            model_confidence=0.90,
        )

        update_turn_quality(turn)

        self.assertTrue(turn.manual_review)


if __name__ == "__main__":
    unittest.main()
