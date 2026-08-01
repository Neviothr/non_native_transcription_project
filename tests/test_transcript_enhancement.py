from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.models import ProjectData, Turn
from transcription_app.transcript_enhancement import (
    ENHANCEMENT_FEATURE_NAMES,
    select_enhanced_transcript,
)
from transcription_app.workflow import (
    apply_transcript_enhancement,
    enhancement_training_examples_from_project,
    train_transcript_enhancer,
)


class FixedPredictor:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities

    def predict_proba(self, row: list[float]) -> list[float]:
        if len(row) != len(ENHANCEMENT_FEATURE_NAMES):
            raise AssertionError("Unexpected enhancement feature length")
        return list(self.probabilities)


class TranscriptEnhancementTests(unittest.TestCase):
    def test_ml_selects_available_source_without_using_gold(self) -> None:
        turn = Turn(
            turn_id=1,
            model_text="I go yesterday",
            chatgpt_text="I went yesterday",
            zoom_text="I go yes today",
            gold_text="This must never be copied during inference",
        )
        selection = select_enhanced_transcript(
            turn, FixedPredictor([0.1, 0.8, 0.1])
        )
        self.assertEqual(selection.text, "I went yesterday")
        self.assertEqual(selection.source_name, "ChatGPT")
        self.assertEqual(selection.method, "machine_learning")
        self.assertNotEqual(selection.text, turn.gold_text)

    def test_missing_predicted_source_is_masked(self) -> None:
        turn = Turn(
            turn_id=1,
            model_text="hello",
            zoom_text="hello there",
        )
        selection = select_enhanced_transcript(
            turn, FixedPredictor([0.2, 0.75, 0.05])
        )
        self.assertEqual(selection.source_name, "Whisper")
        self.assertEqual(selection.text, "hello")

    def test_manual_final_text_is_not_overwritten(self) -> None:
        turn = Turn(
            turn_id=1,
            model_text="raw whisper",
            chatgpt_text="aligned alternative",
            final_text="manual reviewer wording",
        )
        apply_transcript_enhancement(
            turn,
            FixedPredictor([0.1, 0.9, 0.0]),
            update_safe_final_text=True,
        )
        self.assertEqual(turn.enhanced_text, "aligned alternative")
        self.assertEqual(turn.final_text, "manual reviewer wording")

    def test_automatic_final_text_is_replaced(self) -> None:
        turn = Turn(
            turn_id=1,
            model_text="raw whisper",
            chatgpt_text="aligned alternative",
            final_text="raw whisper",
        )
        apply_transcript_enhancement(
            turn,
            FixedPredictor([0.1, 0.9, 0.0]),
            update_safe_final_text=True,
        )
        self.assertEqual(turn.final_text, "aligned alternative")

    def test_gold_labels_best_source_for_training(self) -> None:
        project = ProjectData(
            turns=[
                Turn(
                    turn_id=1,
                    model_text="I go yesterday",
                    chatgpt_text="I went yesterday",
                    zoom_text="I go yes today",
                    gold_text="I went yesterday",
                )
            ]
        )
        rows, labels = enhancement_training_examples_from_project(project)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), len(ENHANCEMENT_FEATURE_NAMES))
        self.assertEqual(labels, [1])  # ChatGPT

    def test_enhancer_can_be_trained_and_saved(self) -> None:
        records = []
        for index in range(12):
            model_wins = index % 2 == 0
            turn = Turn(
                turn_id=index + 1,
                model_text="the accurate phrase" if model_wins else "wrong phrase",
                chatgpt_text="wrong phrase" if model_wins else "the accurate phrase",
                zoom_text="different words here",
                gold_text="the accurate phrase",
                model_confidence=0.9 if model_wins else 0.3,
            )
            project = ProjectData(turns=[turn])
            rows, labels = enhancement_training_examples_from_project(project)
            records.append({"features": rows[0], "label": labels[0]})

        with tempfile.TemporaryDirectory() as directory:
            training_path = Path(directory) / "training.json"
            model_path = Path(directory) / "enhancer.json"
            import json

            training_path.write_text(json.dumps(records), encoding="utf-8")
            model, comparison = train_transcript_enhancer(
                training_path, model_path
            )
            self.assertTrue(model_path.exists())
            self.assertEqual(len(comparison), 3)
            self.assertEqual(len(model.predict_proba(records[0]["features"])), 3)


if __name__ == "__main__":
    unittest.main()
