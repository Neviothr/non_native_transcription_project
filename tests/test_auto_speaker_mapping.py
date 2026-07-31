from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.gui import TranscriptionApp
from transcription_app.models import ProjectData, ProjectMetadata, TranscriptSegment, Turn
from transcription_app.workflow import automatically_map_speakers, recover_speaker_mapping


class AutomaticSpeakerMappingTests(unittest.TestCase):
    def test_explicit_roles_learner_id_and_ai_teacher_are_mapped(self) -> None:
        project = ProjectData(
            metadata=ProjectMetadata(
                learner_id="L-17",
                conversation_type="AI",
            ),
            turns=[
                Turn(1, speaker_raw="L-17", speaker="L-17"),
                Turn(2, speaker_raw="ChatGPT", speaker="ChatGPT"),
                Turn(3, speaker_raw="Unknown", speaker="Unknown"),
            ],
        )

        mapping = automatically_map_speakers(project)

        self.assertEqual(mapping["L-17"], "Learner")
        self.assertEqual(mapping["ChatGPT"], "Teacher")
        self.assertEqual(project.turns[0].speaker, "Learner")
        self.assertEqual(project.turns[1].speaker, "Teacher")
        self.assertEqual(project.turns[2].speaker, "Unknown")

    def test_aligned_gold_roles_map_named_zoom_speakers(self) -> None:
        project = ProjectData(
            turns=[
                Turn(
                    1,
                    start=0.0,
                    end=2.0,
                    speaker_raw="Dana",
                    speaker="Dana",
                    model_text="Hello there",
                ),
                Turn(
                    2,
                    start=2.0,
                    end=4.0,
                    speaker_raw="Alex",
                    speaker="Alex",
                    model_text="I am fine",
                ),
            ],
            source_transcripts={
                "gold": [
                    TranscriptSegment(0.0, 2.0, "Teacher", "Hello there"),
                    TranscriptSegment(2.0, 4.0, "Learner", "I am fine"),
                ]
            },
        )

        mapping = automatically_map_speakers(project)

        self.assertEqual(mapping["Dana"], "Teacher")
        self.assertEqual(mapping["Alex"], "Learner")
        self.assertEqual(project.turns[0].speaker, "Teacher")
        self.assertEqual(project.turns[1].speaker, "Learner")

    def test_generic_two_speaker_labels_are_completed_and_logged(self) -> None:
        project = ProjectData(
            turns=[
                Turn(
                    1,
                    speaker_raw="Speaker 1",
                    speaker="Speaker 1",
                    model_text="How are you today?",
                ),
                Turn(
                    2,
                    speaker_raw="Speaker 2",
                    speaker="Speaker 2",
                    model_text="I am fine today.",
                ),
            ],
        )
        messages: list[str] = []

        mapping = automatically_map_speakers(
            project,
            status_callback=messages.append,
        )

        self.assertEqual(mapping["Speaker 1"], "Teacher")
        self.assertEqual(mapping["Speaker 2"], "Learner")
        self.assertEqual(project.turns[0].speaker, "Teacher")
        self.assertEqual(project.turns[1].speaker, "Learner")
        combined = "\n".join(messages)
        self.assertIn("Automatic speaker mapping started", combined)
        self.assertIn("'Speaker 1' -> Teacher", combined)
        self.assertIn("2 resolved, 0 unresolved", combined)

    def test_existing_unknown_turns_recover_labels_from_imported_transcript(self) -> None:
        project = ProjectData(
            turns=[
                Turn(
                    1,
                    start=0.0,
                    end=2.0,
                    speaker_raw="Unknown",
                    speaker="Unknown",
                    model_text="How are you today?",
                ),
                Turn(
                    2,
                    start=2.0,
                    end=4.0,
                    speaker_raw="Unknown",
                    speaker="Unknown",
                    model_text="I am fine today.",
                ),
            ],
            source_transcripts={
                "zoom": [
                    TranscriptSegment(0.0, 2.0, "Speaker 1", "How are you today?"),
                    TranscriptSegment(2.0, 4.0, "Speaker 2", "I am fine today."),
                ]
            },
        )

        mapping = recover_speaker_mapping(project)

        self.assertEqual(mapping["Speaker 1"], "Teacher")
        self.assertEqual(mapping["Speaker 2"], "Learner")
        self.assertEqual([turn.speaker for turn in project.turns], ["Teacher", "Learner"])


class SpeakerMappingGuiRemovalTests(unittest.TestCase):
    def test_manual_mapping_button_menu_and_handler_are_removed(self) -> None:
        source = (SRC / "transcription_app" / "gui.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("Map Speakers", source)
        self.assertFalse(hasattr(TranscriptionApp, "open_speaker_mapping"))


if __name__ == "__main__":
    unittest.main()
