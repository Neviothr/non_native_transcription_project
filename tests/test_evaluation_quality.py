from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.evaluation import character_error_rate, evaluate_turns, word_error_rate
from transcription_app.models import Turn
from transcription_app.quality import update_turn_quality


class EvaluationQualityTests(unittest.TestCase):
    def test_error_rates(self) -> None:
        self.assertAlmostEqual(word_error_rate("I go to school", "I went to school"), 0.25)
        self.assertGreater(character_error_rate("abc", "axc"), 0.0)

    def test_evaluation_and_feature_flags(self) -> None:
        turn = Turn(
            turn_id=1,
            start=0.0,
            end=4.0,
            speaker="Learner",
            gold_speaker="Learner",
            zoom_text="I I go yesterday",
            chatgpt_text="I I go yesterday",
            model_text="I I go yesterday",
            final_text="I I go yesterday, um, סליחה",
            gold_text="I I go yesterday, um, סליחה",
        )
        update_turn_quality(turn)
        self.assertTrue(turn.hesitation_or_repetition)
        self.assertTrue(turn.hebrew_switch)
        metrics = evaluate_turns([turn])
        self.assertEqual(metrics["word_error_rate"], 0.0)
        self.assertEqual(metrics["speaker_accuracy"], 1.0)

    def test_unavailable_metrics_are_not_reported_as_zero(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker="Learner",
            gold_speaker="Unknown",
            final_text="ordinary sentence",
            gold_text="ordinary sentence",
        )

        metrics = evaluate_turns([turn])

        self.assertEqual(metrics["speaker_labels_evaluated"], 0)
        self.assertIsNone(metrics["speaker_accuracy"])
        self.assertEqual(metrics["speech_error_events_evaluated"], 0)
        self.assertIsNone(metrics["speech_error_preservation_rate"])

    def test_known_gold_speaker_counts_unknown_prediction_as_incorrect(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker="Unknown",
            gold_speaker="Learner",
            final_text="hello",
            gold_text="hello",
        )

        metrics = evaluate_turns([turn])

        self.assertEqual(metrics["speaker_labels_evaluated"], 1)
        self.assertEqual(metrics["speaker_labels_correct"], 0)
        self.assertEqual(metrics["speaker_accuracy"], 0.0)

    def test_speaker_role_aliases_are_compared_canonically(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker="Student",
            gold_speaker="Learner",
            final_text="hello",
            gold_text="hello",
        )

        metrics = evaluate_turns([turn])

        self.assertEqual(metrics["speaker_accuracy"], 1.0)

    def test_speech_error_preservation_counts_matching_events(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker="Learner",
            gold_speaker="Learner",
            gold_text="um um I I [unclear] שלום",
            final_text="um I [unclear]",
        )

        metrics = evaluate_turns([turn])

        self.assertEqual(metrics["speech_error_events_evaluated"], 5)
        self.assertEqual(metrics["speech_error_events_preserved"], 2)
        self.assertAlmostEqual(metrics["speech_error_preservation_rate"], 0.4)



if __name__ == "__main__":
    unittest.main()
