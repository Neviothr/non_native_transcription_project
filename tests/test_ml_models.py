from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.ml_models import load_model, save_model, train_and_compare
from transcription_app.quality import FEATURE_NAMES


class MLModelTests(unittest.TestCase):
    def test_train_compare_and_serialize(self) -> None:
        rng = random.Random(33)
        rows = []
        labels = []
        for label, center in enumerate((0.85, 0.52, 0.18)):
            for _ in range(12):
                rows.append([max(0.0, min(1.0, center + rng.uniform(-0.08, 0.08))) for _ in range(len(FEATURE_NAMES))])
                labels.append(label)
        model, comparison = train_and_compare(rows, labels)
        self.assertEqual(len(comparison), 3)
        probabilities = model.predict_proba(rows[0])
        self.assertEqual(len(probabilities), 3)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_model(model, path)
            loaded = load_model(path)
            loaded_probabilities = loaded.predict_proba(rows[0])
        self.assertEqual(len(loaded_probabilities), 3)


if __name__ == "__main__":
    unittest.main()