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

from transcription_app.gui import TRANSCRIPT_SUFFIXES_BY_SOURCE
from transcription_app.models import ProjectData, Turn
from transcription_app.parsers import parse_transcript
from transcription_app.xlsx_writer import export_xlsx


class XlsxTranscriptImportTests(unittest.TestCase):
    def test_project_export_can_be_reimported_for_chatgpt_and_gold(self) -> None:
        project = ProjectData(
            turns=[
                Turn(
                    turn_id=1,
                    start=0.0,
                    end=1.0,
                    speaker="Learner",
                    chatgpt_text="ChatGPT learner text",
                    gold_text="Gold learner text",
                ),
                Turn(
                    turn_id=2,
                    start=2.0,
                    end=3.0,
                    speaker="Teacher",
                    chatgpt_text="ChatGPT teacher text",
                    gold_text="Gold teacher text",
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = export_xlsx(project, Path(directory) / "transcripts.xlsx")
            chatgpt = parse_transcript(path, source_name="chatgpt")
            gold = parse_transcript(path, source_name="gold")

        self.assertEqual(
            [segment.text for segment in chatgpt],
            ["ChatGPT learner text", "ChatGPT teacher text"],
        )
        self.assertEqual(
            [segment.text for segment in gold],
            ["Gold learner text", "Gold teacher text"],
        )
        self.assertEqual(chatgpt[0].speaker, "Learner")
        self.assertEqual(gold[1].speaker, "Teacher")
        self.assertEqual(chatgpt[1].start, 2.0)
        self.assertEqual(gold[1].end, 3.0)

    def test_shared_strings_and_excel_time_cells_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.xlsx"
            self._write_shared_string_workbook(path)
            segments = parse_transcript(path, source_name="chatgpt")

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].speaker, "Learner")
        self.assertEqual(segments[0].text, "I go yesterday.")
        self.assertAlmostEqual(segments[0].start or 0.0, 1.0, places=6)
        self.assertAlmostEqual(segments[0].end or 0.0, 2.5, places=6)
        self.assertEqual(segments[1].speaker, "Teacher")

    def test_chatgpt_and_gold_file_pickers_include_xlsx(self) -> None:
        self.assertNotIn(".xlsx", TRANSCRIPT_SUFFIXES_BY_SOURCE["zoom"])
        self.assertIn(".xlsx", TRANSCRIPT_SUFFIXES_BY_SOURCE["chatgpt"])
        self.assertIn(".xlsx", TRANSCRIPT_SUFFIXES_BY_SOURCE["gold"])

    @staticmethod
    def _write_shared_string_workbook(path: Path) -> None:
        content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""
        root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""
        workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="ChatGPT" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
        workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""
        shared_strings = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="10" uniqueCount="10">
<si><t>Start</t></si>
<si><t>End</t></si>
<si><t>Speaker</t></si>
<si><t>ChatGPT Transcript</t></si>
<si><t>Learner</t></si>
<si><t>I go yesterday.</t></si>
<si><t>Teacher</t></si>
<si><t>What happened?</t></si>
<si><t>Unused</t></si>
<si><t>Unused 2</t></si>
</sst>"""
        styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<numFmts count="1"><numFmt numFmtId="164" formatCode="[h]:mm:ss.000"/></numFmts>
<fonts count="1"><font/></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills>
<borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0"/></cellStyleXfs>
<cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="164" applyNumberFormat="1"/></cellXfs>
</styleSheet>"""
        worksheet = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>
<row r="1">
<c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>
<c r="C1" t="s"><v>2</v></c><c r="D1" t="s"><v>3</v></c>
</row>
<row r="2">
<c r="A2" s="1"><v>0.000011574074074074</v></c>
<c r="B2" s="1"><v>0.000028935185185185</v></c>
<c r="C2" t="s"><v>4</v></c><c r="D2" t="s"><v>5</v></c>
</row>
<row r="3">
<c r="A3" s="1"><v>0.000034722222222222</v></c>
<c r="B3" s="1"><v>0.000057870370370370</v></c>
<c r="C3" t="s"><v>6</v></c><c r="D3" t="s"><v>7</v></c>
</row>
</sheetData>
</worksheet>"""
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", content_types)
            archive.writestr("_rels/.rels", root_rels)
            archive.writestr("xl/workbook.xml", workbook)
            archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
            archive.writestr("xl/sharedStrings.xml", shared_strings)
            archive.writestr("xl/styles.xml", styles)
            archive.writestr("xl/worksheets/sheet1.xml", worksheet)


if __name__ == "__main__":
    unittest.main()
