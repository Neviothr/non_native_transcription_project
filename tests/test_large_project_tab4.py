from __future__ import annotations

import sys
import tempfile
import tracemalloc
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.evaluation import evaluate_turns, per_source_metrics
from transcription_app.models import ProjectData, ProjectMetadata, Turn
from transcription_app.xlsx_writer import export_xlsx


class LargeProjectTab4Tests(unittest.TestCase):
    def test_turn_level_evaluation_keeps_peak_memory_bounded(self) -> None:
        turns = []
        for index in range(600):
            reference = (
                f"turn {index} I think the learner said this sentence with um hesitation"
            )
            hypothesis = (
                f"turn {index} I think learner said this sentence with hesitation"
            )
            turns.append(
                Turn(
                    turn_id=index + 1,
                    speaker="Student",
                    gold_speaker="Learner",
                    zoom_text=hypothesis,
                    chatgpt_text=hypothesis,
                    model_text=hypothesis,
                    final_text=hypothesis,
                    gold_text=reference,
                )
            )

        tracemalloc.start()
        metrics = evaluate_turns(turns)
        comparison = per_source_metrics(turns)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertEqual(metrics["turns_evaluated"], 600)
        self.assertEqual(len(comparison), 4)
        self.assertLess(peak, 50 * 1024 * 1024)

    def test_oversized_single_turn_uses_bounded_alignment_fallback(self) -> None:
        reference = "a" * 1_600
        hypothesis = "a" * 1_599 + "b"
        metrics = evaluate_turns(
            [
                Turn(
                    turn_id=1,
                    gold_text=reference,
                    final_text=hypothesis,
                )
            ]
        )

        self.assertEqual(metrics["character_alignment_approximations"], 1)
        self.assertGreater(metrics["character_error_rate"], 0.0)

    def test_excel_export_streams_thousands_of_turns(self) -> None:
        project = ProjectData(
            metadata=ProjectMetadata(
                learner_id="L-large",
                session_number="15",
                conversation_type="AI",
            ),
            turns=[
                Turn(
                    turn_id=index + 1,
                    start=float(index),
                    end=float(index + 1),
                    speaker="Student",
                    model_text=f"model transcript row {index}",
                    final_text=f"final transcript row {index}",
                )
                for index in range(2_000)
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            path = export_xlsx(project, Path(directory) / "large.xlsx")
            self.assertTrue(path.exists())
            with zipfile.ZipFile(path) as archive:
                transcript_xml = archive.read("xl/worksheets/sheet1.xml")

        self.assertEqual(transcript_xml.count(b"<row "), 2_001)
        self.assertIn(b'A1:X2001', transcript_xml)


if __name__ == "__main__":
    unittest.main()
