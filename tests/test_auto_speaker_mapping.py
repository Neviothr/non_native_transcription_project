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
from transcription_app.workflow import automatically_map_speakers


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

    def test_ambiguous_labels_remain_unknown_and_are_logged(self) -> None:
        project = ProjectData(
            turns=[
                Turn(1, speaker_raw="Speaker 1", speaker="Speaker 1"),
                Turn(2, speaker_raw="Speaker 2", speaker="Speaker 2"),
            ],
        )
        messages: list[str] = []

        mapping = automatically_map_speakers(
            project,
            status_callback=messages.append,
        )

        self.assertEqual(mapping["Speaker 1"], "Unknown")
        self.assertEqual(mapping["Speaker 2"], "Unknown")
        self.assertEqual(project.turns[0].speaker, "Unknown")
        self.assertEqual(project.turns[1].speaker, "Unknown")
        combined = "\n".join(messages)
        self.assertIn("Automatic speaker mapping started", combined)
        self.assertIn("'Speaker 1' -> Unknown", combined)
        self.assertIn("0 resolved, 2 unresolved", combined)


class SpeakerMappingGuiRemovalTests(unittest.TestCase):
    def test_manual_mapping_button_menu_and_handler_are_removed(self) -> None:
        source = (SRC / "transcription_app" / "gui.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("Map Speakers", source)
        self.assertFalse(hasattr(TranscriptionApp, "open_speaker_mapping"))


if __name__ == "__main__":
    unittest.main()
