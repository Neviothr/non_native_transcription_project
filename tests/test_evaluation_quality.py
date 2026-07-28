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
        self.assertNotIn("manual_correction_minutes_per_audio_minute", metrics)


if __name__ == "__main__":
    unittest.main()
