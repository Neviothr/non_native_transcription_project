from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.gui import TranscriptionApp
from transcription_app.models import ProjectData, ProjectMetadata, TranscriptSegment
from transcription_app.workflow import reload_selected_transcripts


class TranscriptReloadWorkflowTests(unittest.TestCase):
    def test_reload_replaces_selected_sources_and_removes_cleared_sources(self) -> None:
        project = ProjectData(
            metadata=ProjectMetadata(
                zoom_file="old_zoom.txt",
                chatgpt_file="old_chatgpt.txt",
                gold_file="old_gold.txt",
            ),
            source_transcripts={
                "zoom": [TranscriptSegment(text="stale zoom")],
                "chatgpt": [TranscriptSegment(text="stale chatgpt")],
                "gold": [TranscriptSegment(text="stale gold")],
                "model": [TranscriptSegment(text="keep model")],
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zoom = root / "zoom.txt"
            gold = root / "gold.txt"
            zoom.write_text("Teacher: Fresh Zoom text", encoding="utf-8")
            gold.write_text("Learner: Fresh Gold text", encoding="utf-8")

            counts = reload_selected_transcripts(
                project,
                {
                    "zoom": str(zoom),
                    "chatgpt": "",
                    "gold": str(gold),
                },
            )

        self.assertEqual(counts, {"zoom": 1, "chatgpt": 0, "gold": 1})
        self.assertEqual(project.source_transcripts["zoom"][0].text, "Fresh Zoom text")
        self.assertEqual(project.source_transcripts["gold"][0].text, "Fresh Gold text")
        self.assertNotIn("chatgpt", project.source_transcripts)
        self.assertEqual(project.source_transcripts["model"][0].text, "keep model")
        self.assertEqual(project.metadata.zoom_file, str(zoom))
        self.assertEqual(project.metadata.chatgpt_file, "")
        self.assertEqual(project.metadata.gold_file, str(gold))

    def test_failed_reload_does_not_partially_replace_project_sources(self) -> None:
        project = ProjectData(
            metadata=ProjectMetadata(
                zoom_file="old_zoom.txt",
                chatgpt_file="old_chatgpt.txt",
            ),
            source_transcripts={
                "zoom": [TranscriptSegment(text="old zoom")],
                "chatgpt": [TranscriptSegment(text="old chatgpt")],
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zoom = root / "zoom.txt"
            zoom.write_text("Teacher: New Zoom text", encoding="utf-8")
            missing = root / "missing_chatgpt.txt"

            with self.assertRaises(FileNotFoundError):
                reload_selected_transcripts(
                    project,
                    {
                        "zoom": str(zoom),
                        "chatgpt": str(missing),
                        "gold": "",
                    },
                )

        self.assertEqual(project.source_transcripts["zoom"][0].text, "old zoom")
        self.assertEqual(project.source_transcripts["chatgpt"][0].text, "old chatgpt")
        self.assertEqual(project.metadata.zoom_file, "old_zoom.txt")
        self.assertEqual(project.metadata.chatgpt_file, "old_chatgpt.txt")


class TranscriptReloadGuiTests(unittest.TestCase):
    def test_separate_import_button_and_handler_are_removed(self) -> None:
        source = (SRC / "transcription_app" / "gui.py").read_text(encoding="utf-8")

        self.assertNotIn('text="Import Selected Transcripts"', source)
        self.assertFalse(hasattr(TranscriptionApp, "import_selected_transcripts"))

    def test_run_reloads_transcripts_before_copying_project_for_worker(self) -> None:
        source = inspect.getsource(TranscriptionApp.run_transcription)

        reload_position = source.index("reload_selected_transcripts(")
        copy_position = source.index("working_project = ProjectData.from_dict")
        worker_position = source.index("create_local_transcription(")

        self.assertLess(reload_position, copy_position)
        self.assertLess(reload_position, worker_position)


if __name__ == "__main__":
    unittest.main()
