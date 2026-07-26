from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.models import ProjectData, TranscriptSegment
from transcription_app.workflow import initialize_turns_from_model


class LocalScaffoldTests(unittest.TestCase):
    @patch("transcription_app.workflow.analyze_turns")
    def test_timed_zoom_speakers_define_turns(self, _analyze):
        project = ProjectData()
        project.source_transcripts["zoom"] = [
            TranscriptSegment(0.0, 2.0, "Teacher", "How are you?"),
            TranscriptSegment(2.0, 5.0, "Learner", "I am good yesterday."),
        ]
        local = [
            TranscriptSegment(0.1, 1.9, "Unknown", "How are you", 0.9),
            TranscriptSegment(2.1, 4.9, "Unknown", "I am good yesterday", 0.8),
        ]

        initialize_turns_from_model(project, local)

        self.assertEqual(len(project.turns), 2)
        self.assertEqual(project.turns[0].speaker_raw, "Teacher")
        self.assertEqual(project.turns[1].speaker_raw, "Learner")
        self.assertIn("How are you", project.turns[0].model_text)
        self.assertIn("I am good yesterday", project.turns[1].model_text)

    @patch("transcription_app.workflow.analyze_turns")
    def test_local_segments_are_used_without_speaker_scaffold(self, _analyze):
        project = ProjectData()
        local = [TranscriptSegment(0.0, 2.0, "Unknown", "Hello", 0.7)]

        initialize_turns_from_model(project, local)

        self.assertEqual(len(project.turns), 1)
        self.assertEqual(project.turns[0].speaker, "Unknown")
        self.assertEqual(project.turns[0].model_text, "Hello")


if __name__ == "__main__":
    unittest.main()
