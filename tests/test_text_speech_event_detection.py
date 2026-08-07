from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.evaluation import evaluate_speech_events, evaluate_turns
from transcription_app.models import Turn
from transcription_app.text_utils import (
    contains_hesitation_or_repetition,
    contains_self_correction,
    detected_speech_events,
    speech_error_events,
)


class TransparentSpeechEventDetectionTests(unittest.TestCase):
    def test_filler_partial_word_and_exact_phrase_repetition_have_locations(self) -> None:
        source = "Um I wan- I want tea, you know you know, er."

        events = detected_speech_events(source)

        self.assertEqual(
            [(event.kind, event.value) for event in events],
            [
                ("filler", "um"),
                ("partial_word", "wan"),
                ("repetition", "you know"),
                ("filler", "er"),
            ],
        )
        self.assertEqual(
            [source[event.char_start:event.char_end] for event in events],
            ["Um", "wan-", "you know", "er"],
        )
        self.assertEqual(
            [(event.token_start, event.token_end) for event in events],
            [(0, 1), (2, 3), (8, 10), (10, 11)],
        )
        # Detection is observational: spelling and grammar stay verbatim.
        self.assertEqual(source, "Um I wan- I want tea, you know you know, er.")

    def test_internal_hyphens_are_not_partial_words(self) -> None:
        events = detected_speech_events("It is state-of-the-art equipment.")

        self.assertNotIn("partial_word", {event.kind for event in events})
        self.assertFalse(contains_self_correction("state-of-the-art"))

    def test_repeated_fillers_are_counted_as_fillers_not_repetition(self) -> None:
        events = detected_speech_events("um um ah eh")

        self.assertEqual(
            [event.kind for event in events],
            ["filler", "filler", "filler", "filler"],
        )
        self.assertEqual(speech_error_events("um um")["hesitation:um"], 2)

    def test_partial_word_sets_existing_transparent_flags(self) -> None:
        self.assertTrue(contains_hesitation_or_repetition("I wan- I want tea"))
        self.assertTrue(contains_self_correction("I wan- I want tea"))
        self.assertFalse(contains_self_correction("I want tea"))

    def test_invalid_maximum_ngram_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            detected_speech_events("word word", max_repetition_ngram=0)

    def test_model_neutral_persistence_fields_use_half_open_spans(self) -> None:
        event = detected_speech_events("well, um, okay")[0]

        self.assertEqual(
            event.persistence_fields(),
            {
                "event_type": "filled_pause",
                "text": "um",
                "token_start": 1,
                "token_end": 2,
                "details": {
                    "char_start": 6,
                    "char_end": 8,
                    "detector": "transparent_text_v1",
                },
            },
        )


class LocatedSpeechEventEvaluationTests(unittest.TestCase):
    def test_same_occurrence_is_preserved(self) -> None:
        result = evaluate_speech_events(
            "Well, um, I went home",
            "Well um I went home",
        )

        self.assertEqual(result.reference_count, 1)
        self.assertEqual(result.hypothesis_count, 1)
        self.assertEqual(result.matched_count, 1)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.f1, 1.0)

    def test_same_filler_moved_elsewhere_is_not_preserved(self) -> None:
        result = evaluate_speech_events(
            "um I went home",
            "I went home um",
        )

        self.assertEqual(result.reference_count, 1)
        self.assertEqual(result.hypothesis_count, 1)
        self.assertEqual(result.matched_count, 0)

    def test_equivalent_unclear_marker_matches_only_at_same_location(self) -> None:
        same_location = evaluate_speech_events(
            "I heard [unclear] today",
            "I heard [inaudible] today",
        )
        moved = evaluate_speech_events(
            "I heard [unclear] today",
            "I heard today [inaudible]",
        )

        self.assertEqual(same_location.matched_count, 1)
        self.assertEqual(moved.matched_count, 0)

    def test_repeated_phrase_is_one_located_event(self) -> None:
        result = evaluate_speech_events(
            "we can go we can go now",
            "we can go we can go now",
        )

        repetitions = [
            event
            for event in result.reference_events
            if event.kind == "repetition"
        ]
        self.assertEqual(len(repetitions), 1)
        self.assertEqual(repetitions[0].value, "we can go")
        self.assertEqual(result.matched_count, 1)

    def test_turn_metrics_report_event_precision_recall_and_f1(self) -> None:
        metrics = evaluate_turns(
            [
                Turn(
                    turn_id=1,
                    gold_text="um I I [unclear]",
                    final_text="um I [unclear] er",
                )
            ]
        )

        self.assertEqual(metrics["speech_error_events_evaluated"], 3)
        self.assertEqual(metrics["speech_error_events_hypothesized"], 3)
        self.assertEqual(metrics["speech_error_events_preserved"], 2)
        self.assertAlmostEqual(metrics["speech_error_event_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["speech_error_preservation_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["speech_error_event_f1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
