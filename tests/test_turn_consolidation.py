from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.models import ProjectData, SpeechEvent, TranscriptSegment, Turn
from transcription_app.workflow import (
    consolidate_consecutive_speaker_turns,
    initialize_turns_from_model,
)


class TurnConsolidationTests(unittest.TestCase):
    def test_merges_all_consecutive_turns_for_the_same_known_speaker(self) -> None:
        project = ProjectData(
            turns=[
                Turn(
                    4,
                    start=0.0,
                    end=1.0,
                    speaker_raw="Student",
                    speaker="Student",
                    zoom_text="Zoom one",
                    chatgpt_text="Chat one",
                    model_text="one two",
                    gold_text="Gold one",
                    final_text="Final one",
                    quality_target_text="Target one",
                    model_confidence=0.9,
                    notes="First note",
                ),
                Turn(
                    8,
                    start=1.2,
                    end=2.5,
                    speaker_raw="student",
                    speaker=" student ",
                    zoom_text="Zoom two",
                    chatgpt_text="Chat two",
                    model_text="three",
                    gold_text="Gold two",
                    final_text="Final two",
                    quality_target_text="Target two",
                    model_confidence=0.6,
                    notes="Second note",
                ),
                Turn(
                    11,
                    start=2.5,
                    end=3.0,
                    speaker_raw="Teacher",
                    speaker="Teacher",
                    model_text="question",
                    final_text="Question",
                ),
            ],
            speech_events=[
                SpeechEvent(
                    event_id=1,
                    turn_id=8,
                    event_type="repetition",
                    start=1.4,
                    end=1.5,
                    source="manual",
                    details={"following_turn_id": 11},
                )
            ],
        )

        merged = consolidate_consecutive_speaker_turns(project)

        self.assertEqual(merged, 1)
        self.assertEqual(len(project.turns), 2)
        first = project.turns[0]
        self.assertEqual([turn.turn_id for turn in project.turns], [1, 2])
        self.assertEqual((first.start, first.end), (0.0, 2.5))
        self.assertEqual(first.zoom_text, "Zoom one Zoom two")
        self.assertEqual(first.chatgpt_text, "Chat one Chat two")
        self.assertEqual(first.model_text, "one two three")
        self.assertEqual(first.gold_text, "Gold one Gold two")
        self.assertEqual(first.final_text, "Final one Final two")
        self.assertEqual(first.quality_target_text, "Target one Target two")
        self.assertAlmostEqual(first.model_confidence or 0.0, 0.8)
        self.assertEqual(first.notes, "First note Second note")
        self.assertEqual(project.speech_events[0].turn_id, 1)
        self.assertEqual(project.speech_events[0].details["following_turn_id"], 2)

    def test_does_not_merge_consecutive_unknown_speakers(self) -> None:
        project = ProjectData(
            turns=[
                Turn(1, speaker_raw="Unknown", speaker="Unknown", final_text="One"),
                Turn(2, speaker_raw="Unknown", speaker="Unknown", final_text="Two"),
            ]
        )

        self.assertEqual(consolidate_consecutive_speaker_turns(project), 0)
        self.assertEqual([turn.final_text for turn in project.turns], ["One", "Two"])

    @patch("transcription_app.workflow.analyze_turns")
    def test_initialization_consolidates_after_speaker_mapping(self, _analyze) -> None:
        project = ProjectData()
        project.source_transcripts["zoom"] = [
            TranscriptSegment(0.0, 1.0, "Teacher", "First question"),
            TranscriptSegment(1.0, 2.0, "Teacher", "continued"),
            TranscriptSegment(2.0, 3.0, "Learner", "Answer"),
        ]
        local = [
            TranscriptSegment(0.0, 1.0, "Unknown", "First question", 0.9),
            TranscriptSegment(1.0, 2.0, "Unknown", "continued", 0.8),
            TranscriptSegment(2.0, 3.0, "Unknown", "Answer", 0.7),
        ]
        messages: list[str] = []

        initialize_turns_from_model(project, local, status_callback=messages.append)

        self.assertEqual(len(project.turns), 2)
        self.assertEqual(project.turns[0].speaker, "Teacher")
        self.assertEqual(project.turns[0].model_text, "First question continued")
        self.assertEqual(project.turns[1].speaker, "Learner")
        self.assertIn("merged 1 consecutive same-speaker turn(s)", "\n".join(messages))


if __name__ == "__main__":
    unittest.main()
