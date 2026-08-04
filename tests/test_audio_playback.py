from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.audio_playback import (
    TurnPlaybackError,
    build_clip_command,
    normalize_playback_interval,
)


class AudioPlaybackTests(unittest.TestCase):
    def test_interval_returns_start_and_duration(self) -> None:
        start, duration = normalize_playback_interval(12.5, 16.75)

        self.assertEqual(start, 12.5)
        self.assertEqual(duration, 4.25)

    def test_interval_rejects_missing_or_empty_timestamps(self) -> None:
        for interval in (
            (None, 3.0),
            (5.0, 5.0),
            (6.0, 5.0),
            (float("nan"), 7.0),
        ):
            with self.subTest(interval=interval), self.assertRaises(TurnPlaybackError):
                normalize_playback_interval(*interval)

    def test_ffmpeg_command_extracts_only_the_turn(self) -> None:
        command = build_clip_command(
            "ffmpeg.exe",
            Path("conversation.mp3"),
            Path("turn.wav"),
            8.125,
            2.5,
        )

        self.assertIn("-ss", command)
        self.assertEqual(command[command.index("-ss") + 1], "8.125")
        self.assertIn("-t", command)
        self.assertEqual(command[command.index("-t") + 1], "2.500")
        self.assertEqual(command[-1], "turn.wav")
        self.assertIn("pcm_s16le", command)


if __name__ == "__main__":
    unittest.main()
