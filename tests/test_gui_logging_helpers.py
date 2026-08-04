from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.gui import (
    _format_byte_size,
    _format_elapsed,
)


class GuiLoggingHelperTests(unittest.TestCase):
    def test_elapsed_timer_format(self) -> None:
        self.assertEqual(_format_elapsed(0.0), "00:00:00.0")
        self.assertEqual(_format_elapsed(65.49), "00:01:05.4")
        self.assertEqual(_format_elapsed(3661.99), "01:01:01.9")
        self.assertEqual(_format_elapsed(-5.0), "00:00:00.0")

    def test_file_size_format(self) -> None:
        self.assertEqual(_format_byte_size(900), "900 B")
        self.assertEqual(_format_byte_size(1536), "1.5 KiB")
        self.assertEqual(_format_byte_size(2 * 1024 * 1024), "2.0 MiB")


if __name__ == "__main__":
    unittest.main()
