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

import transcription_app.local_whisper as local_whisper
from transcription_app.speech_delays import (
    DetectedDelay,
    SpeechDelayDetectionError,
)


class FakeSegment:
    def __init__(self, t0=0, t1=100, text="hello", probability=0.9):
        self.t0 = t0
        self.t1 = t1
        self.text = text
        self.probability = probability


class NativeSegment:
    def __init__(self) -> None:
        self.valid = True

    def _value(self, value):
        if not self.valid:
            raise RuntimeError("native segment accessed after transcribe")
        return value

    t0 = property(lambda self: self._value(0))
    t1 = property(lambda self: self._value(100))
    text = property(lambda self: self._value("callback copy survives"))
    probability = property(lambda self: self._value(0.88))


class LocalWhisperTests(unittest.TestCase):
    def setUp(self) -> None:
        local_whisper._MODEL_CACHE.clear()

    def tearDown(self) -> None:
        local_whisper._MODEL_CACHE.clear()

    def _transcribe(self, model_type, **kwargs):
        fake_package = types.ModuleType("pywhispercpp")
        fake_model_module = types.ModuleType("pywhispercpp.model")
        fake_model_module.Model = model_type
        messages: list[str] = []

        def fake_convert(_source: Path, target: Path) -> None:
            target.write_bytes(b"temporary wav")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.wav"
            source.write_bytes(b"audio")
            with patch.dict(
                sys.modules,
                {"pywhispercpp": fake_package, "pywhispercpp.model": fake_model_module},
            ), patch.object(
                local_whisper, "_convert_audio", fake_convert
            ), patch.object(
                local_whisper, "_wav_duration_seconds", return_value=10.0
            ):
                result = local_whisper.create_local_transcription(
                    source,
                    model_name="tiny-q5_1",
                    language="auto",
                    threads=2,
                    status_callback=messages.append,
                    **kwargs,
                )
        return result, messages

    def test_converts_native_segments_to_project_seconds(self) -> None:
        result = local_whisper._segments_from_whisper(
            [FakeSegment(376, 1344, "  hello  ", 0.81)]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(
            (result[0].start, result[0].end, result[0].text, result[0].speaker),
            (3.76, 13.44, "hello", "Unknown"),
        )
        self.assertAlmostEqual(result[0].confidence, 0.81)

    def test_empty_segments_and_invalid_probabilities_are_normalized(self) -> None:
        result = local_whisper._segments_from_whisper(
            [FakeSegment(text="   ", probability=0.5), FakeSegment(text="speech", probability=math.nan)]
        )

        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].confidence)
        for value, expected in ((1.4, 1.0), (-0.3, 0.0), ("invalid", None)):
            with self.subTest(value=value):
                self.assertEqual(local_whisper._finite_probability(value), expected)

    def test_duration_and_segment_log_formatting(self) -> None:
        message = local_whisper._segment_log_message(
            FakeSegment(376, 1344, "  learner speech here  ", 0.81),
            7,
        )

        self.assertIn("Segment 0007", message)
        self.assertIn("00:00:03.76 -> 00:00:13.44", message)
        self.assertIn("learner speech here", message)
        self.assertEqual(local_whisper._format_duration(0.0), "00:00:00.00")
        self.assertEqual(local_whisper._format_duration(3661.25), "01:01:01.25")

    def test_thread_count_defaults_to_the_system_maximum(self) -> None:
        with patch.object(local_whisper.os, "cpu_count", return_value=32):
            self.assertEqual(local_whisper.available_cpu_threads(), 32)
            self.assertEqual(local_whisper._normalise_thread_count(None), 32)
            self.assertEqual(local_whisper._normalise_thread_count("invalid"), 32)
            self.assertEqual(local_whisper._normalise_thread_count(64), 32)

    def test_successful_transcription_uses_auto_language_and_reuses_cached_model(self) -> None:
        constructor_options: list[dict[str, object]] = []

        class FakeModel:
            def __init__(self, *_args, **kwargs):
                constructor_options.append(kwargs)

            def transcribe(self, _path, new_segment_callback, extract_probability):
                self.extract_probability = extract_probability
                segment = FakeSegment(text="hello learner")
                new_segment_callback(segment)
                return [segment]

        (segments, details), messages = self._transcribe(FakeModel)
        self._transcribe(FakeModel)

        self.assertEqual(len(constructor_options), 1)
        self.assertEqual(constructor_options[0]["language"], "auto")
        self.assertIs(constructor_options[0]["detect_language"], False)
        self.assertEqual(len(local_whisper._MODEL_CACHE), 1)
        self.assertEqual([segment.text for segment in segments], ["hello learner"])
        self.assertEqual(details["audio_duration_seconds"], 10.0)
        self.assertIn("inference_seconds", details)
        combined = "\n".join(messages)
        for marker in ("Stage 1/5", "Stage 2/5", "Segment 0001", "Stage 5/5"):
            self.assertIn(marker, combined)

    def test_native_segments_are_copied_before_transcribe_returns(self) -> None:
        native = NativeSegment()

        class FakeModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, _path, new_segment_callback, extract_probability):
                new_segment_callback(native)
                native.valid = False
                return [native]

        (segments, _details), messages = self._transcribe(FakeModel)

        self.assertEqual([segment.text for segment in segments], ["callback copy survives"])
        self.assertIn(
            "No native Whisper segment objects will be re-read",
            "\n".join(messages),
        )

    def test_pause_detections_are_returned_before_temporary_wav_is_removed(self) -> None:
        class FakeModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, _path, new_segment_callback, extract_probability):
                new_segment_callback(FakeSegment(text="hello learner"))

        delay = DetectedDelay(
            interval_index=0,
            interval_start_seconds=0.0,
            interval_end_seconds=10.0,
            start_seconds=2.0,
            end_seconds=2.6,
            duration_seconds=0.6,
            loudest_frame_dbfs=None,
        )
        with patch.object(
            local_whisper,
            "_detect_speech_delays",
            return_value=[delay],
        ) as detector:
            (_segments, details), _messages = self._transcribe(
                FakeModel,
                minimum_pause_seconds=0.5,
            )

        detector.assert_called_once()
        self.assertEqual(details["minimum_pause_seconds"], 0.5)
        self.assertEqual(len(details["speech_delay_events"]), 1)
        self.assertEqual(
            details["speech_delay_events"][0]["event_type"],
            "silent_pause",
        )

    def test_pause_detector_failure_does_not_discard_transcription(self) -> None:
        class FakeModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, _path, new_segment_callback, extract_probability):
                new_segment_callback(FakeSegment(text="speech remains"))

        with patch.object(
            local_whisper,
            "_detect_speech_delays",
            side_effect=SpeechDelayDetectionError("unreadable WAV"),
        ):
            (segments, details), messages = self._transcribe(FakeModel)

        self.assertEqual([segment.text for segment in segments], ["speech remains"])
        self.assertEqual(details["speech_delay_events"], [])
        self.assertEqual(details["speech_delay_detection_error"], "unreadable WAV")
        self.assertIn("WARNING", "\n".join(messages))


if __name__ == "__main__":
    unittest.main()
