from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app import __version__
from transcription_app.models import (
    SPEECH_EVENT_TYPES,
    ProjectData,
    ProjectMetadata,
    SpeechEvent,
)
from transcription_app.storage import load_project, save_project
from transcription_app.xlsx_writer import export_xlsx


_WORKBOOK_NS = {
    "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
_RELATIONSHIP_NS = {
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _worksheet_xml(
    archive: zipfile.ZipFile,
    sheet_name: str,
) -> ElementTree.Element:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    sheet = next(
        item
        for item in workbook.findall(".//x:sheet", _WORKBOOK_NS)
        if item.attrib.get("name") == sheet_name
    )
    relationship_id = sheet.attrib[
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    ]
    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    relationship = next(
        item
        for item in relationships.findall("p:Relationship", _RELATIONSHIP_NS)
        if item.attrib.get("Id") == relationship_id
    )
    return ElementTree.fromstring(
        archive.read(f"xl/{relationship.attrib['Target']}")
    )


class SpeechEventModelTests(unittest.TestCase):
    def test_supported_event_vocabulary_covers_detection_outputs(self) -> None:
        self.assertEqual(
            SPEECH_EVENT_TYPES,
            {
                "silent_pause",
                "filled_pause",
                "partial_word",
                "repetition",
                "revision",
                "restart",
                "response_gap",
                "grammar_sensitive_difference",
                "unclear",
                "code_switch",
            },
        )

    def test_duration_is_bounded_and_missing_timestamps_are_safe(self) -> None:
        event = SpeechEvent(1, 4, "silent_pause", 2.25, 2.75)
        self.assertAlmostEqual(event.duration(), 0.5)
        self.assertEqual(
            SpeechEvent(2, 4, "silent_pause", 3.0, 2.0).duration(),
            0.0,
        )
        self.assertEqual(
            SpeechEvent(3, None, "response_gap", None, 4.0).duration(),
            0.0,
        )

    def test_save_load_round_trip_preserves_events_and_unknown_details(self) -> None:
        event = SpeechEvent(
            event_id=9,
            turn_id=3,
            event_type="filled_pause",
            start=7.1,
            end=7.52,
            text="um",
            confidence=0.94,
            source="audio_detector",
            token_start=5,
            token_end=6,
            reviewed=True,
            details={
                "detector": {"name": "pause-v2", "revision": 7},
                "alternatives": ["uh", "erm"],
                "language_note": "עברית",
            },
        )
        project = ProjectData(
            metadata=ProjectMetadata(
                detect_speech_delays=False,
                minimum_pause_seconds=0.65,
            ),
            speech_events=[event],
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_project(project, Path(directory) / "events.ntproject")
            loaded = load_project(path)

        self.assertEqual(loaded.speech_events, [event])
        self.assertFalse(loaded.metadata.detect_speech_delays)
        self.assertEqual(loaded.metadata.minimum_pause_seconds, 0.65)

    def test_legacy_current_version_project_defaults_new_fields(self) -> None:
        raw = ProjectData().to_dict()
        raw.pop("speech_events")
        raw["metadata"].pop("detect_speech_delays")
        raw["metadata"].pop("minimum_pause_seconds")
        raw["application_version"] = __version__

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-current.ntproject"
            path.write_text(json.dumps(raw), encoding="utf-8")
            loaded = load_project(path)

        self.assertEqual(loaded.speech_events, [])
        self.assertTrue(loaded.metadata.detect_speech_delays)
        self.assertEqual(loaded.metadata.minimum_pause_seconds, 0.3)

    def test_event_details_must_be_an_object(self) -> None:
        raw = ProjectData(
            speech_events=[SpeechEvent(1, 1, "restart", 1.0, 1.2)]
        ).to_dict()
        raw["speech_events"][0]["details"] = ["not", "an", "object"]

        with self.assertRaisesRegex(TypeError, "details must be a JSON object"):
            ProjectData.from_dict(raw)


class SpeechEventExportTests(unittest.TestCase):
    def test_excel_has_structured_events_and_delay_settings(self) -> None:
        project = ProjectData(
            metadata=ProjectMetadata(
                detect_speech_delays=False,
                minimum_pause_seconds=0.45,
            ),
            speech_events=[
                SpeechEvent(
                    event_id=12,
                    turn_id=None,
                    event_type="response_gap",
                    start=10.0,
                    end=11.25,
                    text="=not_an_excel_formula",
                    confidence=0.88,
                    source="turn_boundaries",
                    details={"between_turns": [4, 5], "note": "ממתין"},
                )
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            path = export_xlsx(project, Path(directory) / "events.xlsx")
            with zipfile.ZipFile(path) as archive:
                events = _worksheet_xml(archive, "Events")
                metadata = _worksheet_xml(archive, "Metadata")

        event_text = [
            node.text or ""
            for node in events.findall(".//x:t", _WORKBOOK_NS)
        ]
        metadata_text = [
            node.text or ""
            for node in metadata.findall(".//x:t", _WORKBOOK_NS)
        ]
        self.assertIn("Event Type", event_text)
        self.assertIn("response_gap", event_text)
        self.assertIn("=not_an_excel_formula", event_text)
        self.assertIn('{"between_turns":[4,5],"note":"ממתין"}', event_text)
        self.assertEqual(events.findall(".//x:f", _WORKBOOK_NS), [])
        self.assertIn("Detect Speech Delays", metadata_text)
        self.assertIn("Minimum Pause (s)", metadata_text)


if __name__ == "__main__":
    unittest.main()
