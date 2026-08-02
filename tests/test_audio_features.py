from __future__ import annotations

import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.audio_features import analyze_wav_intervals


class AudioFeatureBatchTests(unittest.TestCase):
    def test_multiple_intervals_open_wav_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            rate = 8_000
            samples = [
                int(5_000 * math.sin(2 * math.pi * 220 * index / rate))
                for index in range(rate)
            ]
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))

            with patch("transcription_app.audio_features.wave.open", wraps=wave.open) as opened:
                results = analyze_wav_intervals(
                    path,
                    [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0)],
                )

        self.assertEqual(opened.call_count, 1)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(item["volume_dbfs"] is not None for item in results))


if __name__ == "__main__":
    unittest.main()
