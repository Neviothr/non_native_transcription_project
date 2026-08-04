from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    def test_run_uses_freshly_reloaded_transcripts_in_the_worker_copy(self) -> None:
        class Variable:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class Harness:
            def __init__(self) -> None:
                self.project = ProjectData(
                    metadata=ProjectMetadata(audio_file="audio.wav"),
                    source_transcripts={
                        "zoom": [TranscriptSegment(text="stale")],
                    },
                )
                self.zoom_var = Variable("zoom.txt")
                self.chatgpt_var = Variable("")
                self.gold_var = Variable("")
                self.model_var = Variable("tiny-q5_1")
                self.language_var = Variable("auto")
                self.threads_var = Variable(1)
                self.worker_result = None

            def _sync_metadata_from_ui(self) -> None:
                pass

            def stop_turn_playback(self, silent=False) -> None:
                pass

            def _start_transcription_timer(self) -> None:
                pass

            def _append_log(self, _message: str) -> None:
                pass

            def _set_status(self, _message: str) -> None:
                pass

            def _log_transcription_configuration(self, *_args) -> None:
                pass

            def refresh_all(self) -> None:
                pass

            def _run_background(self, worker, _on_success) -> None:
                self.worker_result = worker()

            def _transcription_finished(self, _result) -> None:
                pass

        app = Harness()
        worker_projects: list[ProjectData] = []

        def reload(project, _selected):
            project.source_transcripts["zoom"] = [
                TranscriptSegment(text="fresh")
            ]
            return {"zoom": 1, "chatgpt": 0, "gold": 0}

        def initialize(project, _segments, status_callback=None):
            worker_projects.append(project)

        with patch(
            "transcription_app.gui.reload_selected_transcripts",
            side_effect=reload,
        ), patch(
            "transcription_app.gui.create_local_transcription",
            return_value=([], {}),
        ), patch(
            "transcription_app.gui.initialize_turns_from_model",
            side_effect=initialize,
        ):
            TranscriptionApp.run_transcription(app)

        self.assertEqual(worker_projects[0].source_transcripts["zoom"][0].text, "fresh")
        self.assertIsNot(worker_projects[0], app.project)


if __name__ == "__main__":
    unittest.main()
