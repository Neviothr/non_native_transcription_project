from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.models import ProjectData, ProjectMetadata, Turn
from transcription_app.reporting import export_html_report
from transcription_app.xlsx_writer import export_xlsx


class ExportTests(unittest.TestCase):
    def test_xlsx_and_html_export(self) -> None:
        project = ProjectData(
            metadata=ProjectMetadata(learner_id="L-01", session_number="1", conversation_type="AI"),
            turns=[Turn(turn_id=1, speaker="Learner", model_text="hello", final_text="hello")],
            metrics={"word_error_rate": 0.1, "source_comparison": [{"source": "Final", "wer": 0.1, "cer": 0.05}]},
        )
        with tempfile.TemporaryDirectory() as directory:
            xlsx_path = export_xlsx(project, Path(directory) / "result.xlsx")
            html_path = export_html_report(project, Path(directory) / "report.html")
            self.assertTrue(xlsx_path.exists())
            self.assertTrue(html_path.exists())
            with zipfile.ZipFile(xlsx_path) as archive:
                self.assertIn("xl/workbook.xml", archive.namelist())
                for name in archive.namelist():
                    if name.endswith(".xml"):
                        ElementTree.fromstring(archive.read(name))
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Transcription Evaluation Report", html)
            self.assertIn("<svg", html)


if __name__ == "__main__":
    unittest.main()
