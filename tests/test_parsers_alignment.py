from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.alignment import align_source_to_turns, segments_to_turns
from transcription_app.models import TranscriptSegment
from transcription_app.parsers import parse_transcript


class ParserAlignmentTests(unittest.TestCase):
    def test_vtt_parsing_and_alignment(self) -> None:
        content = """WEBVTT

00:00:00.000 --> 00:00:02.000
Learner: I go yesterday to school.

00:00:02.100 --> 00:00:04.000
Teacher: What happened there?
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.vtt"
            path.write_text(content, encoding="utf-8")
            parsed = parse_transcript(path)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].speaker, "Learner")
        turns = segments_to_turns(
            [
                TranscriptSegment(0.0, 2.0, "A", "I go yesterday to school."),
                TranscriptSegment(2.1, 4.0, "B", "What happened there?"),
            ]
        )
        aligned = align_source_to_turns(turns, parsed)
        self.assertEqual(aligned[0], "I go yesterday to school.")
        self.assertEqual(aligned[1], "What happened there?")

    def test_zoom_webvtt_voice_tag_preserves_speaker(self) -> None:
        content = """WEBVTT

00:00:00.000 --> 00:00:02.000
<v Dana Cohen>How are you today?</v>

00:00:02.100 --> 00:00:04.000
<v Zahar Kokin>I am fine.</v>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zoom.vtt"
            path.write_text(content, encoding="utf-8")
            parsed = parse_transcript(path, source_name="zoom")

        self.assertEqual([segment.speaker for segment in parsed], ["Dana Cohen", "Zahar Kokin"])
        self.assertEqual(parsed[0].text, "How are you today?")
        self.assertEqual(parsed[1].text, "I am fine.")

    def test_untimed_monotonic_alignment(self) -> None:
        turns = segments_to_turns(
            [
                TranscriptSegment(text="hello my name is dan"),
                TranscriptSegment(text="I study electrical engineering"),
            ]
        )
        source = [
            TranscriptSegment(text="Hello, my name is Dan."),
            TranscriptSegment(text="I study electrical engineering."),
        ]
        aligned = align_source_to_turns(turns, source)
        self.assertEqual(len(aligned), 2)
        self.assertIn("name", aligned[0])
        self.assertIn("engineering", aligned[1])

    def test_timed_long_segment_is_split_across_turns(self) -> None:
        turns = segments_to_turns(
            [
                TranscriptSegment(0.0, 2.0, "Student", "Hello there"),
                TranscriptSegment(2.0, 4.0, "Student", "how are you"),
            ]
        )
        source = [
            TranscriptSegment(0.0, 4.0, "Student", "Hello there how are you"),
        ]

        aligned = align_source_to_turns(turns, source)

        self.assertEqual(aligned, ["Hello there", "how are you"])
        self.assertEqual(
            " ".join(aligned),
            "Hello there how are you",
        )

    def test_short_source_segments_are_combined_into_one_turn(self) -> None:
        turns = segments_to_turns(
            [TranscriptSegment(0.0, 4.0, "Student", "I went to school yesterday")],
        )
        source = [
            TranscriptSegment(0.0, 1.5, "Student", "I went to"),
            TranscriptSegment(1.5, 4.0, "Student", "school yesterday"),
        ]

        aligned = align_source_to_turns(turns, source)

        self.assertEqual(aligned, ["I went to school yesterday"])

    def test_untimed_alignment_ignores_different_sentence_boundaries(self) -> None:
        turns = segments_to_turns(
            [
                TranscriptSegment(text="I like electrical engineering"),
                TranscriptSegment(text="because it is interesting"),
            ]
        )
        source = [
            TranscriptSegment(text="I like electrical"),
            TranscriptSegment(text="engineering because"),
            TranscriptSegment(text="it is interesting"),
        ]

        aligned = align_source_to_turns(turns, source)

        self.assertEqual(
            aligned,
            ["I like electrical engineering", "because it is interesting"],
        )


if __name__ == "__main__":
    unittest.main()
