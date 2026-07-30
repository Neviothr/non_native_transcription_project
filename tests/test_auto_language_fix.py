from __future__ import annotations

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


class Segment:
    t0 = 0
    t1 = 100
    text = "hello"
    probability = 0.9


class AutoLanguageTests(unittest.TestCase):
    def setUp(self):
        local_whisper._MODEL_CACHE.clear()

    def test_auto_language_does_not_enable_detection_only_flag(self):
        captured = {}

        class FakeModel:
            def __init__(self, *_args, **kwargs):
                captured.update(kwargs)

            def transcribe(self, _path, new_segment_callback, extract_probability):
                segment = Segment()
                new_segment_callback(segment)
                return [segment]

        fake_package = types.ModuleType("pywhispercpp")
        fake_model_module = types.ModuleType("pywhispercpp.model")
        fake_model_module.Model = FakeModel

        def fake_convert(_source: Path, target: Path):
            target.write_bytes(b"wav")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.wav"
            source.write_bytes(b"audio")
            with patch.dict(sys.modules, {
                "pywhispercpp": fake_package,
                "pywhispercpp.model": fake_model_module,
            }), patch.object(local_whisper, "_convert_audio", fake_convert), patch.object(
                local_whisper, "_wav_duration_seconds", return_value=1.0
            ):
                segments, _ = local_whisper.create_local_transcription(
                    source,
                    model_name="tiny-q5_1",
                    language="auto",
                    threads=1,
                )

        self.assertEqual(len(segments), 1)
        self.assertEqual(captured["language"], "auto")
        self.assertIs(captured["detect_language"], False)


if __name__ == "__main__":
    unittest.main()