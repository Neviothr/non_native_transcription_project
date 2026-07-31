from __future__ import annotations

import gc
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


class FakeSegment:
    t0 = 0
    t1 = 100
    text = "hello"
    probability = 0.9


class RetentionTests(unittest.TestCase):
    def setUp(self):
        local_whisper._MODEL_CACHE.clear()

    def test_model_is_retained_after_successful_run(self):
        destroyed = []

        class FakeModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, _path, new_segment_callback, extract_probability):
                segment = FakeSegment()
                new_segment_callback(segment)
                return [segment]

            def __del__(self):
                destroyed.append(True)

        fake_package = types.ModuleType("pywhispercpp")
        fake_model_module = types.ModuleType("pywhispercpp.model")
        fake_model_module.Model = FakeModel
        messages = []

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
                segments, _details = local_whisper.create_local_transcription(
                    source,
                    model_name="tiny-q5_1",
                    language="auto",
                    threads=1,
                    status_callback=messages.append,
                )

        gc.collect()
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(local_whisper._MODEL_CACHE), 1)
        self.assertEqual(destroyed, [])
        combined = "\n".join(messages)
        self.assertIn("stage4-auto-language-fix-v3", combined)
        self.assertIn("model retained in memory", combined)
        self.assertIn("Stage 5/5", combined)


if __name__ == "__main__":
    unittest.main()
