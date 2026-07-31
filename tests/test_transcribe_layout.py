from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.gui import _configure_transcribe_row_resizing


class _FakeFrame:
    def __init__(self) -> None:
        self.rows: dict[int, dict[str, int]] = {}

    def rowconfigure(self, row: int, **options: int) -> None:
        self.rows[row] = options


class TranscribeLayoutTests(unittest.TestCase):
    def test_action_row_cannot_shrink_below_button_height(self) -> None:
        frame = _FakeFrame()

        _configure_transcribe_row_resizing(frame)  # type: ignore[arg-type]

        self.assertEqual(frame.rows[5], {"weight": 0, "minsize": 44})

    def test_log_row_absorbs_window_height_changes(self) -> None:
        frame = _FakeFrame()

        _configure_transcribe_row_resizing(frame)  # type: ignore[arg-type]

        self.assertEqual(frame.rows[8], {"weight": 1, "minsize": 80})


if __name__ == "__main__":
    unittest.main()
