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
    def test_substitution_is_highlighted_on_both_sides(self) -> None:
        source = "I go school yesterday"
        final = "I went school yesterday"

        source_spans, final_spans = _word_difference_spans(source, final)

        self.assertEqual(_highlighted_words(source, source_spans), ["go"])
        self.assertEqual(_highlighted_words(final, final_spans), ["went"])

    def test_inserted_final_words_are_highlighted(self) -> None:
        source = "I went school"
        final = "I went to school"

        source_spans, final_spans = _word_difference_spans(source, final)

        self.assertEqual(source_spans, [])
        self.assertEqual(_highlighted_words(final, final_spans), ["to"])

    def test_case_only_changes_are_not_highlighted(self) -> None:
        source_spans, final_spans = _word_difference_spans("Hello World", "hello world")

        self.assertEqual(source_spans, [])
        self.assertEqual(final_spans, [])

    def test_empty_source_highlights_every_final_word(self) -> None:
        final = "final transcript"

        source_spans, final_spans = _word_difference_spans("", final)

        self.assertEqual(source_spans, [])
        self.assertEqual(
            _highlighted_words(final, final_spans),
            ["final", "transcript"],
        )


if __name__ == "__main__":
    unittest.main()
