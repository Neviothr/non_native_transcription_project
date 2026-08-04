from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.gui import (
    ALL_WHISPER_LANGUAGE_CODES,
    LANGUAGE_CHOICES,
    WHISPER_LANGUAGE_NAMES,
    _language_code_from_choice,
)


class LanguageDropdownTests(unittest.TestCase):
    def test_choices_start_with_auto_and_name_every_supported_code_once(self):
        expected = [
            f"{code} ({WHISPER_LANGUAGE_NAMES[code]})"
            for code in ALL_WHISPER_LANGUAGE_CODES
        ]
        self.assertEqual(LANGUAGE_CHOICES[0], "auto")
        self.assertEqual(list(LANGUAGE_CHOICES[1:]), expected)
        self.assertEqual(len(set(LANGUAGE_CHOICES)), len(LANGUAGE_CHOICES))

    def test_display_value_is_converted_back_to_language_code(self):
        self.assertEqual(_language_code_from_choice("he (Hebrew)"), "he")
        self.assertEqual(_language_code_from_choice("yue (Cantonese)"), "yue")
        self.assertEqual(_language_code_from_choice("auto"), "auto")
        self.assertEqual(_language_code_from_choice(""), "auto")

if __name__ == "__main__":
    unittest.main()
