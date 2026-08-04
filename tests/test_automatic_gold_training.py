from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.gui import TranscriptionApp, _training_record_summary
from transcription_app.models import ProjectData
from transcription_app.quality import FEATURE_NAMES
from transcription_app.workflow import QUALITY_LABEL_TARGET, QUALITY_TRAINING_SCHEMA_VERSION


class _TrainingHarness:
    def __init__(self, support: Path) -> None:
        self.project = ProjectData()
        self.support = support
        self.worker_result = None
        self.messages: list[str] = []

    def _begin_evaluation_operation(self, _name: str) -> float:
        return 1.0

    def _log_evaluation_context(self) -> None:
        pass

    def _project_support_dir(self) -> Path:
        return self.support

    def _append_evaluation_log(self, message: str) -> None:
        self.messages.append(message)

    def save_editor_to_turn(self, silent: bool = False) -> None:
        pass

    def _set_status(self, _message: str) -> None:
        pass

    def _run_evaluation_background(self, worker, _on_success, _on_error) -> None:
        self.worker_result = worker()


def _training_record(label: int, identifier: str) -> dict[str, object]:
    return {
        "schema_version": QUALITY_TRAINING_SCHEMA_VERSION,
        "label_target": QUALITY_LABEL_TARGET,
        "example_id": identifier * 64,
        "features": [0.1] * len(FEATURE_NAMES),
        "label": label,
    }


class AutomaticGoldTrainingTests(unittest.TestCase):
    def test_gold_examples_are_collected_before_preflight_and_training(self) -> None:
        events: list[str] = []

        def append_examples(snapshot, path):
            events.append("collect")
            self.assertIsNot(snapshot, app.project)
            return 3

        def summarize(path):
            events.append("preflight")
            return 9, {0: 5, 1: 4}

        def train(path, model_path):
            events.append("train")
            return "trained"

        with tempfile.TemporaryDirectory() as directory:
            app = _TrainingHarness(Path(directory))
            with patch(
                "transcription_app.gui.append_training_examples",
                side_effect=append_examples,
            ), patch(
                "transcription_app.gui._training_record_summary",
                side_effect=summarize,
            ), patch(
                "transcription_app.gui.train_quality_model",
                side_effect=train,
            ):
                TranscriptionApp.train_models(app)

        self.assertEqual(events, ["collect", "preflight", "train"])
        self.assertEqual(app.worker_result["added"], 3)
        self.assertEqual(app.worker_result["training_result"], "trained")

    def test_preflight_requires_nine_records_from_two_classes(self) -> None:
        for summary in ((8, {0: 4, 1: 4}), (9, {0: 9})):
            with self.subTest(summary=summary), tempfile.TemporaryDirectory() as directory:
                app = _TrainingHarness(Path(directory))
                with patch(
                    "transcription_app.gui.append_training_examples",
                    return_value=0,
                ), patch(
                    "transcription_app.gui._training_record_summary",
                    return_value=summary,
                ), patch("transcription_app.gui.train_quality_model") as train:
                    TranscriptionApp.train_models(app)

                train.assert_not_called()
                self.assertIsNone(app.worker_result["training_result"])

    def test_training_summary_ignores_incompatible_records(self) -> None:
        records = [
            _training_record(0, "a"),
            _training_record(1, "b"),
            {**_training_record(2, "c"), "schema_version": -1},
            {"features": [0.2] * len(FEATURE_NAMES), "label": 2},
            {"features": "invalid", "label": 2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality_training.json"
            path.write_text(json.dumps(records), encoding="utf-8")

            self.assertEqual(_training_record_summary(path), (2, {0: 1, 1: 1}))


if __name__ == "__main__":
    unittest.main()
