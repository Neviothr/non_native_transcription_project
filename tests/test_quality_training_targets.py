from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.models import ProjectData, Turn
from transcription_app.quality import extract_features
from transcription_app.workflow import (
    QUALITY_LABEL_TARGET,
    QUALITY_TRAINING_SCHEMA_VERSION,
    append_training_examples,
    ensure_quality_target_text,
    training_examples_from_project,
)


class QualityTrainingTargetTests(unittest.TestCase):
    def test_label_tracks_selected_initial_transcript_not_local_model(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker="Student",
            zoom_text="I go to school yesterday",
            chatgpt_text="I go to school yesterday",
            model_text="Completely unrelated local model output",
            final_text="I go to school yesterday",
            gold_text="I go to school yesterday",
        )
        project = ProjectData(turns=[turn])

        _rows, labels = training_examples_from_project(project)

        self.assertEqual(labels, [0])
        self.assertEqual(
            turn.quality_target_text,
            "I go to school yesterday",
        )

    def test_manual_edit_does_not_change_saved_quality_target(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker="Student",
            model_text="I school",
            final_text="I go to school",
            quality_target_text="I school",
            gold_text="I go to school",
        )
        project = ProjectData(turns=[turn])

        _rows, labels = training_examples_from_project(project)

        self.assertEqual(labels, [2])

    def test_new_turn_uses_source_candidate_for_quality_target(self) -> None:
        turn = Turn(
            turn_id=1,
            zoom_text="I school",
            chatgpt_text="I school",
            model_text="I go school",
            final_text="I go to school",
            gold_text="I go to school",
        )

        target = ensure_quality_target_text(turn)

        self.assertEqual(target, "I school")

    def test_feature_extraction_uses_unedited_quality_target(self) -> None:
        turn = Turn(
            turn_id=1,
            model_text="I said [unclear]",
            final_text="I said hello",
            quality_target_text="I said [unclear]",
        )

        features = extract_features(turn)

        self.assertEqual(features["unclear_penalty"], 1.0)

    def test_distinct_turns_with_same_features_are_retained(self) -> None:
        project = ProjectData(
            turns=[
                Turn(
                    turn_id=1,
                    start=0.0,
                    end=1.0,
                    model_text="hello",
                    final_text="hello",
                    quality_target_text="hello",
                    gold_text="hello",
                ),
                Turn(
                    turn_id=2,
                    start=2.0,
                    end=3.0,
                    model_text="hello",
                    final_text="hello",
                    quality_target_text="hello",
                    gold_text="hello",
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality_training.json"
            added = append_training_examples(project, path)
            records = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(added, 2)
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0]["example_id"], records[1]["example_id"])

    def test_append_rejects_incompatible_training_schema(self) -> None:
        turn = Turn(
            turn_id=1,
            model_text="hello",
            final_text="hello",
            quality_target_text="hello",
            gold_text="hello",
        )
        project = ProjectData(turns=[turn])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality_training.json"
            path.write_text(
                json.dumps([{"features": [0.0], "label": 2}]),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                append_training_examples(project, path)


if __name__ == "__main__":
    unittest.main()
