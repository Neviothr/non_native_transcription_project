from __future__ import annotations

import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.local_whisper import (
    _finite_probability,
    _format_duration,
    _segment_log_message,
    _segments_from_whisper,
)


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

    def test_verbose_segment_log_contains_number_time_and_text(self):
        segment = FakeSegment(376, 1344, "  learner speech here  ", 0.81)
        message = _segment_log_message(segment, 7)
        self.assertIn("Segment 0007", message)
        self.assertIn("00:00:03.76", message)
        self.assertIn("00:00:13.44", message)
        self.assertIn("learner speech here", message)

    def test_duration_formatter(self):
        self.assertEqual(_format_duration(0.0), "00:00:00.00")
        self.assertEqual(_format_duration(3661.25), "01:01:01.25")

    def test_transcription_emits_verbose_stage_messages_and_timings(self):
        from transcription_app.local_whisper import create_local_transcription

        produced = [FakeSegment(0, 100, "hello learner", 0.9)]

        class FakeModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, _path, new_segment_callback, extract_probability):
                self.assert_extract = extract_probability
                for segment in produced:
                    new_segment_callback(segment)
                return produced

        fake_package = types.ModuleType("pywhispercpp")
        fake_model_module = types.ModuleType("pywhispercpp.model")
        fake_model_module.Model = FakeModel
        messages: list[str] = []

        def fake_convert(_source: Path, target: Path) -> None:
            target.write_bytes(b"temporary wav")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.wav"
            source.write_bytes(b"audio")
            with patch.dict(
                sys.modules,
                {"pywhispercpp": fake_package, "pywhispercpp.model": fake_model_module},
            ), patch(
                "transcription_app.local_whisper._convert_audio", fake_convert
            ), patch(
                "transcription_app.local_whisper._wav_duration_seconds", return_value=10.0
            ):
                segments, details = create_local_transcription(
                    source,
                    model_name="tiny-q5_1",
                    language="auto",
                    threads=2,
                    status_callback=messages.append,
                )

        combined = "\n".join(messages)
        self.assertEqual(len(segments), 1)
        self.assertIn("Stage 1/5", combined)
        self.assertIn("Stage 2/5", combined)
        self.assertIn("Segment 0001", combined)
        self.assertIn("Stage 5/5", combined)
        self.assertEqual(details["audio_duration_seconds"], 10.0)
        self.assertIn("inference_seconds", details)


if __name__ == "__main__":
    unittest.main()