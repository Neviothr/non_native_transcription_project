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
from transcription_app.workflow import (
    automatically_map_speakers,
    normalize_role_for_conversation_type,
    recover_speaker_mapping,
    speaker_roles_for_conversation_type,
)


class ConversationRoleModelTests(unittest.TestCase):
    def test_ai_conversation_role_choices(self) -> None:
        self.assertEqual(
            speaker_roles_for_conversation_type("AI"),
            ("Student", "Supervisor", "AI"),
        )

    def test_human_teacher_conversation_role_choices(self) -> None:
        self.assertEqual(
            speaker_roles_for_conversation_type("Human teacher"),
            ("Student", "Teacher"),
        )

    def test_legacy_roles_are_normalized_by_conversation_type(self) -> None:
        self.assertEqual(
            normalize_role_for_conversation_type("Learner", "AI"),
            "Student",
        )
        self.assertEqual(
            normalize_role_for_conversation_type("Teacher", "AI"),
            "Supervisor",
        )
        self.assertEqual(
            normalize_role_for_conversation_type("Supervisor", "Human teacher"),
            "Teacher",
        )
        self.assertIsNone(
            normalize_role_for_conversation_type("AI", "Human teacher")
        )


class AutomaticSpeakerMappingTests(unittest.TestCase):
    def test_ai_roles_use_student_supervisor_and_ai(self) -> None:
        project = ProjectData(
            metadata=ProjectMetadata(
                learner_id="L-17",
                conversation_type="AI",
            ),
            turns=[
                Turn(1, speaker_raw="L-17", speaker="L-17"),
                Turn(2, speaker_raw="ChatGPT", speaker="ChatGPT"),
                Turn(3, speaker_raw="Teacher", speaker="Teacher"),
                Turn(4, speaker_raw="Unknown", speaker="Unknown"),
            ],
        )

        mapping = automatically_map_speakers(project)

        self.assertEqual(mapping["L-17"], "Student")
        self.assertEqual(mapping["ChatGPT"], "AI")
        self.assertEqual(mapping["Teacher"], "Supervisor")
        self.assertEqual(project.turns[0].speaker, "Student")
        self.assertEqual(project.turns[1].speaker, "AI")
        self.assertEqual(project.turns[2].speaker, "Supervisor")
        self.assertEqual(project.turns[3].speaker, "Unknown")

    def test_aligned_gold_roles_map_human_teacher_speakers(self) -> None:
        project = ProjectData(
            metadata=ProjectMetadata(conversation_type="Human teacher"),
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
        self.assertEqual(mapping["Alex"], "Student")
        self.assertEqual(project.turns[0].speaker, "Teacher")
        self.assertEqual(project.turns[1].speaker, "Student")

    def test_generic_two_speaker_ai_labels_become_ai_and_student(self) -> None:
        project = ProjectData(
            metadata=ProjectMetadata(conversation_type="AI"),
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

        self.assertEqual(mapping["Speaker 1"], "AI")
        self.assertEqual(mapping["Speaker 2"], "Student")
        self.assertEqual(project.turns[0].speaker, "AI")
        self.assertEqual(project.turns[1].speaker, "Student")
        combined = "\n".join(messages)
        self.assertIn("Automatic speaker mapping started", combined)
        self.assertIn("'Speaker 1' -> AI", combined)
        self.assertIn("2 resolved, 0 unresolved", combined)

    def test_generic_three_speaker_ai_labels_include_supervisor(self) -> None:
        project = ProjectData(
            metadata=ProjectMetadata(conversation_type="AI"),
            turns=[
                Turn(1, speaker_raw="Speaker 1", model_text="Tell me about your school?"),
                Turn(2, speaker_raw="Speaker 2", model_text="I study science and English every day."),
                Turn(3, speaker_raw="Speaker 3", model_text="Continue."),
                Turn(4, speaker_raw="Speaker 1", model_text="Why do you like science?"),
                Turn(5, speaker_raw="Speaker 2", model_text="Because the experiments are interesting."),
            ],
        )

        mapping = automatically_map_speakers(project)

        self.assertEqual(mapping["Speaker 1"], "AI")
        self.assertEqual(mapping["Speaker 2"], "Student")
        self.assertEqual(mapping["Speaker 3"], "Supervisor")

    def test_generic_two_speaker_human_conversation_becomes_teacher_student(self) -> None:
        project = ProjectData(
            metadata=ProjectMetadata(conversation_type="Human teacher"),
            turns=[
                Turn(1, speaker_raw="Speaker 1", model_text="What did you do yesterday?"),
                Turn(2, speaker_raw="Speaker 2", model_text="I went to school."),
            ],
        )

        mapping = automatically_map_speakers(project)

        self.assertEqual(mapping["Speaker 1"], "Teacher")
        self.assertEqual(mapping["Speaker 2"], "Student")

    def test_existing_unknown_turns_recover_labels_from_imported_transcript(self) -> None:
        project = ProjectData(
            metadata=ProjectMetadata(conversation_type="Human teacher"),
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
        self.assertEqual(mapping["Speaker 2"], "Student")
        self.assertEqual([turn.speaker for turn in project.turns], ["Teacher", "Student"])


class SpeakerMappingGuiRemovalTests(unittest.TestCase):
    def test_manual_mapping_button_menu_and_handler_are_removed(self) -> None:
        source = (SRC / "transcription_app" / "gui.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("Map Speakers", source)
        self.assertFalse(hasattr(TranscriptionApp, "open_speaker_mapping"))

    def test_review_role_selector_is_conversation_type_dependent(self) -> None:
        source = (SRC / "transcription_app" / "gui.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("speaker_roles_for_conversation_type", source)
        self.assertIn("<<ComboboxSelected>>", source)
        self.assertNotIn("ROLE_CHOICES", source)


if __name__ == "__main__":
    unittest.main()
