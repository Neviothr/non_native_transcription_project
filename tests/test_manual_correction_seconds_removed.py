from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.evaluation import evaluate_turns
from transcription_app.models import ProjectData, Turn
from transcription_app.xlsx_writer import export_xlsx


class ManualCorrectionSecondsRemovalTests(unittest.TestCase):
    def test_evaluation_omits_manual_correction_metric(self) -> None:
        metrics = evaluate_turns(
            [Turn(turn_id=1, gold_text="hello", final_text="hello")]
        )

        self.assertNotIn("manual_correction_minutes_per_audio_minute", metrics)

    def test_excel_omits_manual_correction_column(self) -> None:
        project = ProjectData(
            turns=[Turn(turn_id=1, final_text="hello", model_text="hello")]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = export_xlsx(project, Path(directory) / "result.xlsx")
            with zipfile.ZipFile(path) as archive:
                transcript_sheet = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")

        self.assertNotIn("Manual Correction Seconds", transcript_sheet)

    def test_review_ui_has_no_timer_or_seconds_entry(self) -> None:
        source = (SRC / "transcription_app" / "gui.py").read_text(encoding="utf-8")

        self.assertNotIn("Correction timer", source)
        self.assertNotIn("Recorded seconds", source)
        self.assertNotIn("toggle_timer", source)
        self.assertNotIn("correction_seconds_var", source)


if __name__ == "__main__":
    unittest.main()
