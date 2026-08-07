"""Minimal dependency-free XLSX exporter using streamed Office Open XML."""

from __future__ import annotations

import json
import math
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from xml.sax.saxutils import escape

from .grammar_events import grammar_review_summary
from .models import ProjectData
from .speech_events import render_turn_with_speech_delays
from .workflow import speaker_label_for_turn


EXCEL_CELL_TEXT_LIMIT = 32_767
_EXCEL_TRUNCATION_MARKER = " … [truncated for Excel]"


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _clean_cell_text(value: object) -> str:
    raw_text = str(value)
    raw_text = "".join(
        character
        for character in raw_text
        if character in "\t\n\r" or ord(character) >= 32
    )
    if len(raw_text) <= EXCEL_CELL_TEXT_LIMIT:
        return raw_text
    keep = EXCEL_CELL_TEXT_LIMIT - len(_EXCEL_TRUNCATION_MARKER)
    return raw_text[:keep] + _EXCEL_TRUNCATION_MARKER


def _cell_xml(row: int, column: int, value: Any, style: int = 0) -> str:
    reference = f"{_column_name(column)}{row}"
    style_attr = f' s="{style}"' if style else ""
    if value is None:
        return f'<c r="{reference}"{style_attr}/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"{style_attr}><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            value = 0
        return f'<c r="{reference}"{style_attr}><v>{value}</v></c>'
    text = escape(_clean_cell_text(value))
    return (
        f'<c r="{reference}" t="inlineStr"{style_attr}>'
        f'<is><t xml:space="preserve">{text}</t></is></c>'
    )


def _write_text(stream: BinaryIO, text: str) -> None:
    stream.write(text.encode("utf-8"))


def _write_sheet_xml(
    archive: zipfile.ZipFile,
    archive_name: str,
    rows: Iterable[list[Any]],
    *,
    row_count: int,
    max_columns: int,
    widths: list[float] | None = None,
    percent_columns: set[int] | None = None,
) -> None:
    """Write a worksheet incrementally instead of assembling one giant string."""

    percent_columns = percent_columns or set()
    bounded_rows = max(1, int(row_count))
    bounded_columns = max(1, int(max_columns))
    dimension = f"A1:{_column_name(bounded_columns)}{bounded_rows}"
    column_xml = ""
    if widths:
        column_xml = "<cols>" + "".join(
            f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
            for index, width in enumerate(widths, start=1)
        ) + "</cols>"

    with archive.open(archive_name, "w", force_zip64=True) as stream:
        _write_text(
            stream,
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="{dimension}"/>'
            '<sheetViews><sheetView workbookViewId="0">'
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            '</sheetView></sheetViews>'
            f'<sheetFormatPr defaultRowHeight="18"/>{column_xml}<sheetData>',
        )

        written_rows = 0
        for row_index, row in enumerate(rows, start=1):
            written_rows = row_index
            cells = []
            for column_index, value in enumerate(row, start=1):
                style = (
                    1
                    if row_index == 1
                    else (2 if column_index in percent_columns else 0)
                )
                cells.append(_cell_xml(row_index, column_index, value, style))
            height_attr = ' ht="30" customHeight="1"' if row_index == 1 else ""
            _write_text(
                stream,
                f'<row r="{row_index}"{height_attr}>{"".join(cells)}</row>',
            )

        if written_rows == 0:
            _write_text(stream, '<row r="1"/>')
            written_rows = 1

        _write_text(stream, "</sheetData>")
        if written_rows > 1:
            _write_text(
                stream,
                f'<autoFilter ref="A1:{_column_name(bounded_columns)}{written_rows}"/>',
            )
        _write_text(stream, "</worksheet>")


_TRANSCRIPT_HEADERS = [
    "Learner ID",
    "Session",
    "Conversation Type",
    "Turn",
    "Start (s)",
    "End (s)",
    "Speaker",
    "Zoom Transcript",
    "ChatGPT Transcript",
    "Additional Model Transcript",
    "Final Transcript",
    "Final Transcript with Delays",
    "Gold Standard",
    "Confidence",
    "Agreement",
    "Quality Classification",
    "Hebrew Switch",
    "Hesitation/Repetition",
    "Self-Correction",
    "Unclear Speech",
    "Overlapping Speech",
    "Manual Review Required",
    "Speech Rate (WPM)",
    "Volume (dBFS)",
    "Estimated SNR (dB)",
    "Grammar Preservation Review",
]


def _iter_transcript_rows(project: ProjectData) -> Iterator[list[Any]]:
    yield _TRANSCRIPT_HEADERS
    metadata = project.metadata
    for turn in project.turns:
        yield [
            metadata.learner_id,
            metadata.session_number,
            metadata.conversation_type,
            turn.turn_id,
            turn.start,
            turn.end,
            speaker_label_for_turn(turn),
            turn.zoom_text,
            turn.chatgpt_text,
            turn.model_text,
            turn.final_text,
            render_turn_with_speech_delays(project, turn),
            turn.gold_text,
            turn.quality_score,
            turn.agreement_score,
            turn.quality_label,
            turn.hebrew_switch,
            turn.hesitation_or_repetition,
            turn.self_correction,
            turn.unclear_speech,
            turn.overlapping_speech,
            turn.manual_review,
            turn.speech_rate_wpm,
            turn.volume_dbfs,
            turn.noise_snr_db,
            grammar_review_summary(project, turn.turn_id),
        ]


def _metric_rows(project: ProjectData) -> list[list[Any]]:
    rows: list[list[Any]] = [["Metric", "Value"]]
    for key, value in project.metrics.items():
        if key == "source_comparison":
            continue
        rows.append([key.replace("_", " ").title(), "N/A" if value is None else value])
    return rows


def _comparison_rows(project: ProjectData) -> list[list[Any]]:
    rows: list[list[Any]] = [
        ["Source", "Word Error Rate", "Character Error Rate"]
    ]
    for item in project.metrics.get("source_comparison", []):
        rows.append(
            [
                item.get("source", ""),
                item.get("wer", 0.0),
                item.get("cer", 0.0),
            ]
        )
    return rows


def _model_rows(project: ProjectData) -> list[list[Any]]:
    rows: list[list[Any]] = [
        [
            "Model",
            "Accuracy",
            "Balanced Accuracy",
            "Macro F1",
            "Selection Score",
            "Selected",
            "Validation Predictions",
        ]
    ]
    for item in project.model_comparison:
        rows.append(
            [
                item.get("model", ""),
                item.get("accuracy", 0.0),
                item.get("balanced_accuracy", 0.0),
                item.get("macro_f1", 0.0),
                item.get("selection_score", 0.0),
                bool(item.get("selected", False)),
                item.get("validation_predictions", 0),
            ]
        )
    return rows


def _metadata_rows(project: ProjectData) -> list[list[Any]]:
    metadata = project.metadata
    return [
        ["Field", "Value"],
        ["Project Title", metadata.title],
        ["Learner ID", metadata.learner_id],
        ["Session Number", metadata.session_number],
        ["Conversation Type", metadata.conversation_type],
        ["Audio File", metadata.audio_file],
        ["Zoom Transcript", metadata.zoom_file],
        ["ChatGPT Transcript", metadata.chatgpt_file],
        ["Gold Standard", metadata.gold_file],
        ["Transcription Model", metadata.transcription_model],
        ["Detect Speech Delays", metadata.detect_speech_delays],
        ["Minimum Pause (s)", metadata.minimum_pause_seconds],
        ["Created", metadata.created_at],
        ["Updated", metadata.updated_at],
    ]


_EVENT_HEADERS = [
    "Event ID",
    "Turn ID",
    "Event Type",
    "Start (s)",
    "End (s)",
    "Duration (s)",
    "Text",
    "Confidence",
    "Source",
    "Token Start",
    "Token End",
    "Reviewed",
    "Details (JSON)",
]


def _iter_event_rows(project: ProjectData) -> Iterator[list[Any]]:
    yield _EVENT_HEADERS
    for event in project.speech_events:
        yield [
            event.event_id,
            event.turn_id,
            event.event_type,
            event.start,
            event.end,
            event.duration(),
            event.text,
            event.confidence,
            event.source,
            event.token_start,
            event.token_end,
            event.reviewed,
            json.dumps(
                event.details,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]


@dataclass(frozen=True, slots=True)
class _SheetSpec:
    name: str
    rows_factory: Callable[[], Iterable[list[Any]]]
    row_count: int
    max_columns: int
    widths: list[float]
    percent_columns: set[int]


def export_xlsx(project: ProjectData, path: str | Path) -> Path:
    target = Path(path)
    if target.suffix.casefold() != ".xlsx":
        target = target.with_suffix(".xlsx")
    target.parent.mkdir(parents=True, exist_ok=True)

    metric_rows = _metric_rows(project)
    comparison_rows = _comparison_rows(project)
    model_rows = _model_rows(project)
    metadata_rows = _metadata_rows(project)
    sheets = [
        _SheetSpec(
            "Transcript",
            lambda: _iter_transcript_rows(project),
            len(project.turns) + 1,
            len(_TRANSCRIPT_HEADERS),
            [
                14, 10, 18, 8, 11, 11, 16, 35, 35, 38, 42, 42, 38,
                12, 12, 24, 12, 18, 15, 14, 16, 18, 17, 14, 16, 44,
            ],
            {14, 15},
        ),
        _SheetSpec(
            "Evaluation",
            lambda: iter(metric_rows),
            len(metric_rows),
            2,
            [42, 22],
            set(),
        ),
        _SheetSpec(
            "Source Comparison",
            lambda: iter(comparison_rows),
            len(comparison_rows),
            3,
            [24, 20, 22],
            {2, 3},
        ),
        _SheetSpec(
            "ML Model Comparison",
            lambda: iter(model_rows),
            len(model_rows),
            7,
            [28, 18, 20, 18, 18, 14, 22],
            {2, 3, 4, 5},
        ),
        _SheetSpec(
            "Metadata",
            lambda: iter(metadata_rows),
            len(metadata_rows),
            2,
            [28, 75],
            set(),
        ),
        _SheetSpec(
            "Events",
            lambda: _iter_event_rows(project),
            len(project.speech_events) + 1,
            len(_EVENT_HEADERS),
            [
                12, 10, 20, 12, 12, 14, 30, 14, 18, 13, 13, 12, 55,
            ],
            {8},
        ),
    ]

    content_overrides = "".join(
        (
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.'
            'spreadsheetml.worksheet+xml"/>'
        )
        for index in range(1, len(sheets) + 1)
    )
    content_types = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
{content_overrides}
</Types>'''

    workbook_sheets = "".join(
        f'<sheet name="{escape(sheet.name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, sheet in enumerate(sheets, start=1)
    )
    workbook = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets>{workbook_sheets}</sheets></workbook>'''
    workbook_relationships = "".join(
        (
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
        for index in range(1, len(sheets) + 1)
    ) + (
        f'<Relationship Id="rId{len(sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )

    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<numFmts count="1"><numFmt numFmtId="164" formatCode="0.00%"/></numFmts>
<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts>
<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF355C7D"/><bgColor indexed="64"/></patternFill></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment vertical="top"/></xf></cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''

    root_relationships = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''

    with tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}_",
        suffix=".xlsx.tmp",
        dir=target.parent,
        delete=False,
    ) as temporary_handle:
        temporary = Path(temporary_handle.name)

    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            archive.writestr("[Content_Types].xml", content_types)
            archive.writestr("_rels/.rels", root_relationships)
            archive.writestr("xl/workbook.xml", workbook)
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                (
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    f"{workbook_relationships}</Relationships>"
                ),
            )
            archive.writestr("xl/styles.xml", styles)
            for index, sheet in enumerate(sheets, start=1):
                _write_sheet_xml(
                    archive,
                    f"xl/worksheets/sheet{index}.xml",
                    sheet.rows_factory(),
                    row_count=sheet.row_count,
                    max_columns=sheet.max_columns,
                    widths=sheet.widths,
                    percent_columns=sheet.percent_columns,
                )
        temporary.replace(target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return target
