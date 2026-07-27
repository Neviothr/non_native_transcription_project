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

from transcription_app.local_whisper import create_local_transcription


class NativeSegment:
    def __init__(self) -> None:
        self.valid = True

    @property
    def t0(self):
        if not self.valid:
            raise RuntimeError("native segment accessed after transcribe")
        return 0

    @property
    def t1(self):
        if not self.valid:
            raise RuntimeError("native segment accessed after transcribe")
        return 100

    @property
    def text(self):
        if not self.valid:
            raise RuntimeError("native segment accessed after transcribe")
        return "callback copy survives"

    @property
    def probability(self):
        if not self.valid:
            raise RuntimeError("native segment accessed after transcribe")
        return 0.88


class Stage4NativeObjectRegressionTests(unittest.TestCase):
    def test_stage4_uses_callback_copy_without_rereading_native_object(self):
        native = NativeSegment()

        class FakeModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def transcribe(self, _path, new_segment_callback, extract_probability):
                self.assertTrue if False else None
                new_segment_callback(native)
                native.valid = False
                return [native]

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
                "transcription_app.local_whisper._wav_duration_seconds",
                return_value=1.0,
            ):
                segments, _details = create_local_transcription(
                    source,
                    model_name="tiny-q5_1",
                    status_callback=messages.append,
                )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "callback copy survives")
        self.assertIn("No native Whisper segment objects will be re-read", "\n".join(messages))


if __name__ == "__main__":
    unittest.main()
