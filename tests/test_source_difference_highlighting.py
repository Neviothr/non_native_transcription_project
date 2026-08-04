from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.gui import _word_difference_spans


def _highlighted_words(text: str, spans: list[tuple[int, int]]) -> list[str]:
    return [text[start:end] for start, end in spans]


class SourceDifferenceHighlightingTests(unittest.TestCase):
    def test_only_changed_words_are_highlighted(self) -> None:
        cases = (
            ("I go school yesterday", "I went school yesterday", ["go"], ["went"]),
            ("I went school", "I went to school", [], ["to"]),
            ("Hello World", "hello world", [], []),
            ("", "final transcript", [], ["final", "transcript"]),
        )
        for source, final, expected_source, expected_final in cases:
            with self.subTest(source=source, final=final):
                source_spans, final_spans = _word_difference_spans(source, final)
                self.assertEqual(_highlighted_words(source, source_spans), expected_source)
                self.assertEqual(_highlighted_words(final, final_spans), expected_final)


if __name__ == "__main__":
    unittest.main()
