from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.gui import (
    _configure_transcribe_row_resizing,
    _normalize_minimum_pause_seconds,
)


class _FakeFrame:
    def __init__(self) -> None:
        self.rows: dict[int, dict[str, int]] = {}

    def rowconfigure(self, row: int, **options: int) -> None:
        self.rows[row] = options


class TranscribeLayoutTests(unittest.TestCase):
    def test_log_row_resizes_while_action_row_keeps_its_height(self) -> None:
        frame = _FakeFrame()

        _configure_transcribe_row_resizing(frame)  # type: ignore[arg-type]

        self.assertEqual(frame.rows[6], {"weight": 0, "minsize": 44})
        self.assertEqual(frame.rows[9], {"weight": 1, "minsize": 80})

    def test_minimum_pause_threshold_is_validated_and_bounded(self) -> None:
        self.assertEqual(_normalize_minimum_pause_seconds("invalid"), 0.3)
        self.assertEqual(_normalize_minimum_pause_seconds(0.01), 0.2)
        self.assertEqual(_normalize_minimum_pause_seconds(0.75), 0.75)
        self.assertEqual(_normalize_minimum_pause_seconds(10.0), 5.0)


if __name__ == "__main__":
    unittest.main()
