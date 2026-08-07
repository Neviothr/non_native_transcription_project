from __future__ import annotations

import tempfile
import unittest
import wave
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.speech_delays import (
    DetectedDelay,
    SpeechDelayConfig,
    SpeechDelayDetectionError,
    detect_speech_delays,
    detect_speech_delays_in_interval,
)


SAMPLE_RATE = 1_000
SAMPLE_WIDTH = 2


def _constant_samples(duration: float, amplitude: int) -> list[int]:
    count = round(duration * SAMPLE_RATE)
    return [amplitude if index % 2 == 0 else -amplitude for index in range(count)]


def _write_wav(path: Path, samples: list[int], *, channels: int = 1) -> None:
    raw = b"".join(
        int(sample).to_bytes(SAMPLE_WIDTH, byteorder="little", signed=True)
        for sample in samples
        for _channel in range(channels)
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(raw)


class SpeechDelayDetectionTests(unittest.TestCase):
    def test_returns_absolute_timestamps_for_subinterval(self) -> None:
        samples = (
            _constant_samples(0.8, 10_000)
            + _constant_samples(0.4, 10_000)
            + _constant_samples(0.4, 0)
            + _constant_samples(0.4, 10_000)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.wav"
            _write_wav(path, samples)
            original = path.read_bytes()

            detections = detect_speech_delays_in_interval(
                path,
                0.8,
                2.0,
                config=SpeechDelayConfig(
                    silence_threshold_dbfs=-40.0,
                    minimum_pause_seconds=0.30,
                    frame_seconds=0.01,
                    edge_padding_seconds=0.0,
                ),
            )

            self.assertEqual(path.read_bytes(), original)

        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.event_type, "silent_pause")
        self.assertAlmostEqual(detection.interval_start_seconds, 0.8)
        self.assertAlmostEqual(detection.interval_end_seconds, 2.0)
        self.assertAlmostEqual(detection.start_seconds, 1.2, places=3)
        self.assertAlmostEqual(detection.end_seconds, 1.6, places=3)
        self.assertAlmostEqual(detection.duration_seconds, 0.4, places=3)
        self.assertIsNone(detection.loudest_frame_dbfs)

    def test_threshold_and_minimum_duration_are_configurable(self) -> None:
        samples = (
            _constant_samples(0.4, 10_000)
            + _constant_samples(0.4, 100)
            + _constant_samples(0.2, 10_000)
            + _constant_samples(0.2, 0)
            + _constant_samples(0.4, 10_000)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.wav"
            _write_wav(path, samples)

            quiet_detection = detect_speech_delays_in_interval(
                path,
                0.0,
                1.6,
                config=SpeechDelayConfig(
                    silence_threshold_dbfs=-40.0,
                    minimum_pause_seconds=0.30,
                    frame_seconds=0.01,
                    edge_padding_seconds=0.0,
                ),
            )
            stricter_threshold = detect_speech_delays_in_interval(
                path,
                0.0,
                1.6,
                config=SpeechDelayConfig(
                    silence_threshold_dbfs=-60.0,
                    minimum_pause_seconds=0.30,
                    frame_seconds=0.01,
                    edge_padding_seconds=0.0,
                ),
            )

        self.assertEqual(len(quiet_detection), 1)
        self.assertAlmostEqual(quiet_detection[0].start_seconds, 0.4, places=3)
        self.assertAlmostEqual(quiet_detection[0].end_seconds, 0.8, places=3)
        self.assertEqual(stricter_threshold, [])

    def test_edge_padding_excludes_near_boundary_silence(self) -> None:
        samples = (
            _constant_samples(0.25, 10_000)
            + _constant_samples(0.35, 0)
            + _constant_samples(0.60, 10_000)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.wav"
            _write_wav(path, samples)

            guarded = detect_speech_delays_in_interval(
                path,
                0.2,
                1.2,
                config=SpeechDelayConfig(
                    minimum_pause_seconds=0.30,
                    frame_seconds=0.01,
                    edge_padding_seconds=0.10,
                ),
            )
            unguarded = detect_speech_delays_in_interval(
                path,
                0.2,
                1.2,
                config=SpeechDelayConfig(
                    minimum_pause_seconds=0.30,
                    frame_seconds=0.01,
                    edge_padding_seconds=0.0,
                ),
            )

        self.assertEqual(guarded, [])
        self.assertEqual(len(unguarded), 1)
        self.assertAlmostEqual(unguarded[0].start_seconds, 0.25, places=3)
        self.assertAlmostEqual(unguarded[0].end_seconds, 0.60, places=3)

    def test_interval_edge_silence_is_opt_in(self) -> None:
        samples = _constant_samples(0.4, 0) + _constant_samples(0.6, 10_000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.wav"
            _write_wav(path, samples)

            default_results = detect_speech_delays_in_interval(path, 0.0, 1.0)
            included_results = detect_speech_delays_in_interval(
                path,
                0.0,
                1.0,
                config=SpeechDelayConfig(include_interval_edges=True),
            )

        self.assertEqual(default_results, [])
        self.assertEqual(len(included_results), 1)
        self.assertAlmostEqual(included_results[0].start_seconds, 0.0)
        self.assertAlmostEqual(included_results[0].end_seconds, 0.4, places=3)

    def test_multiple_intervals_retain_caller_index(self) -> None:
        samples = (
            _constant_samples(0.3, 10_000)
            + _constant_samples(0.4, 0)
            + _constant_samples(0.6, 10_000)
            + _constant_samples(0.4, 0)
            + _constant_samples(0.3, 10_000)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.wav"
            _write_wav(path, samples)
            detections = detect_speech_delays(
                path,
                SpeechDelayConfig(
                    frame_seconds=0.01,
                    edge_padding_seconds=0.0,
                ),
                intervals=[(0.0, 1.0), (1.0, 2.0)],
            )

        self.assertEqual([item.interval_index for item in detections], [0, 1])
        self.assertAlmostEqual(detections[0].start_seconds, 0.3, places=3)
        self.assertAlmostEqual(detections[1].start_seconds, 1.3, places=3)
        self.assertTrue(all(isinstance(item, DetectedDelay) for item in detections))

    def test_full_file_api_returns_asdict_serializable_detection(self) -> None:
        samples = (
            _constant_samples(0.3, 10_000)
            + _constant_samples(0.4, 0)
            + _constant_samples(0.3, 10_000)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.wav"
            _write_wav(path, samples)
            detections = detect_speech_delays(
                path,
                SpeechDelayConfig(
                    frame_seconds=0.01,
                    edge_padding_seconds=0.0,
                ),
            )

        self.assertEqual(len(detections), 1)
        serialized = asdict(detections[0])
        self.assertEqual(serialized["event_type"], "silent_pause")
        self.assertAlmostEqual(serialized["start_seconds"], 0.3, places=3)
        self.assertAlmostEqual(serialized["end_seconds"], 0.7, places=3)

    def test_rejects_non_mono_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            _write_wav(path, _constant_samples(0.5, 10_000), channels=2)

            with self.assertRaisesRegex(SpeechDelayDetectionError, "Expected mono"):
                detect_speech_delays_in_interval(path, 0.0, 0.5)

    def test_rejects_invalid_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum_pause_seconds"):
            SpeechDelayConfig(minimum_pause_seconds=0.0)


if __name__ == "__main__":
    unittest.main()
