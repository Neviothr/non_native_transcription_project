from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.gui import TranscriptionApp


class AutomaticGoldTrainingTests(unittest.TestCase):
    def test_training_collects_gold_examples_before_model_training(self) -> None:
        source = inspect.getsource(TranscriptionApp.train_models)

        collect_position = source.index(
            "append_training_examples(snapshot, training_path)"
        )
        train_position = source.index(
            "train_quality_model(training_path, model_path)"
        )
        self.assertLess(collect_position, train_position)

    def test_training_uses_combined_dataset_for_preflight(self) -> None:
        source = inspect.getsource(TranscriptionApp.train_models)

        self.assertIn("record_count >= 9", source)
        self.assertIn("len(distribution) >= 2", source)
        self.assertIn("Automatically added", source)

    def test_missing_examples_no_longer_require_separate_button(self) -> None:
        evaluation_source = inspect.getsource(TranscriptionApp._build_evaluation_tab)
        menu_source = inspect.getsource(TranscriptionApp._build_menu)

        self.assertNotIn("Add Gold Examples", evaluation_source)
        self.assertNotIn("Add Gold Examples", menu_source)
        self.assertFalse(hasattr(TranscriptionApp, "add_training_examples"))


if __name__ == "__main__":
    unittest.main()
