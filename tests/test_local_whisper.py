from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.local_whisper import _finite_probability, _segments_from_whisper


class FakeSegment:
    def __init__(self, t0, t1, text, probability):
        self.t0 = t0
        self.t1 = t1
        self.text = text
        self.probability = probability


class LocalWhisperConversionTests(unittest.TestCase):
    def test_converts_whisper_timestamp_units_to_seconds(self):
        result = _segments_from_whisper([FakeSegment(376, 1344, "  hello  ", 0.81)])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].start, 3.76)
        self.assertAlmostEqual(result[0].end, 13.44)
        self.assertEqual(result[0].text, "hello")
        self.assertAlmostEqual(result[0].confidence, 0.81)
        self.assertEqual(result[0].speaker, "Unknown")

    def test_skips_empty_text_and_handles_nan_probability(self):
        result = _segments_from_whisper(
            [FakeSegment(0, 10, "   ", 0.5), FakeSegment(10, 20, "speech", math.nan)]
        )
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].confidence)

    def test_probability_is_clamped(self):
        self.assertEqual(_finite_probability(1.4), 1.0)
        self.assertEqual(_finite_probability(-0.3), 0.0)
        self.assertIsNone(_finite_probability("not a number"))


if __name__ == "__main__":
    unittest.main()
