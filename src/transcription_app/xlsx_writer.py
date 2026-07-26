"""Minimal dependency-free XLSX exporter using Office Open XML."""

from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape

from .models import ProjectData, Turn


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


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
    raw_text = str(value)
    raw_text = "".join(char for char in raw_text if char in "\t\n\r" or ord(char) >= 32)
    text = escape(raw_text)
    return f'<c r="{reference}" t="inlineStr"{style_attr}><is><t xml:space="preserve">{text}</t></is></c>'


def _sheet_xml(rows: list[list[Any]], widths: list[float] | None = None, percent_columns: set[int] | None = None) -> str:
    percent_columns = percent_columns or set()
    max_columns = max((len(row) for row in rows), default=1)
    dimension = f"A1:{_column_name(max_columns)}{max(1, len(rows))}"
    column_xml = ""
    if widths:
        column_xml = "<cols>" + "".join(
            f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
            for index, width in enumerate(widths, start=1)
        ) + "</cols>"
    row_xml: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells: list[str] = []
        for column_index, value in enumerate(row, start=1):
            style = 1 if row_index == 1 else (2 if column_index in percent_columns else 0)
            cells.append(_cell_xml(row_index, column_index, value, style))
        height_attr = ' ht="30" customHeight="1"' if row_index == 1 else ""
        row_xml.append(f'<row r="{row_index}"{height_attr}>{"".join(cells)}</row>')
    auto_filter = f'<autoFilter ref="A1:{_column_name(max_columns)}{len(rows)}"/>' if len(rows) > 1 else ""
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<dimension ref="{dimension}"/>
<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
<sheetFormatPr defaultRowHeight="18"/>{column_xml}<sheetData>{''.join(row_xml)}</sheetData>{auto_filter}
</worksheet>'''


def _transcript_rows(project: ProjectData) -> list[list[Any]]:
    headers = [
        "Learner ID", "Session", "Conversation Type", "Turn", "Start (s)", "End (s)",
        "Speaker", "Raw Speaker", "Zoom Transcript", "ChatGPT Transcript", "Additional Model Transcript",
        "Final Transcript", "Gold Standard", "Confidence", "Agreement", "Quality Classification",
        "Hebrew Switch", "Hesitation/Repetition", "Self-Correction", "Unclear Speech", "Overlapping Speech",
        "Manual Review Required", "Manual Correction Seconds", "Speech Rate (WPM)", "Volume (dBFS)",
        "Estimated SNR (dB)", "Notes",
    ]
    rows: list[list[Any]] = [headers]
    metadata = project.metadata
    for turn in project.turns:
        rows.append([
            metadata.learner_id,
            metadata.session_number,
            metadata.conversation_type,
            turn.turn_id,
            turn.start,
            turn.end,
            turn.speaker,
            turn.speaker_raw,
            turn.zoom_text,
            turn.chatgpt_text,
            turn.model_text,
            turn.final_text,
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
            turn.manual_correction_seconds,
            turn.speech_rate_wpm,
            turn.volume_dbfs,
            turn.noise_snr_db,
            turn.notes,
        ])
    return rows


def _metric_rows(project: ProjectData) -> list[list[Any]]:
    rows: list[list[Any]] = [["Metric", "Value"]]
    for key, value in project.metrics.items():
        if key == "source_comparison":
            continue
        rows.append([key.replace("_", " ").title(), value])
    return rows


def _comparison_rows(project: ProjectData) -> list[list[Any]]:
    rows: list[list[Any]] = [["Source", "Word Error Rate", "Character Error Rate"]]
    for item in project.metrics.get("source_comparison", []):
        rows.append([item.get("source", ""), item.get("wer", 0.0), item.get("cer", 0.0)])
    return rows


def _model_rows(project: ProjectData) -> list[list[Any]]:
    rows: list[list[Any]] = [["Model", "Accuracy", "Macro F1"]]
    for item in project.model_comparison:
        rows.append([item.get("model", ""), item.get("accuracy", 0.0), item.get("macro_f1", 0.0)])
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
        ["Created", metadata.created_at],
        ["Updated", metadata.updated_at],
    ]


def export_xlsx(project: ProjectData, path: str | Path) -> Path:
    target = Path(path)
    if target.suffix.casefold() != ".xlsx":
        target = target.with_suffix(".xlsx")
    target.parent.mkdir(parents=True, exist_ok=True)

    sheets = [
        ("Transcript", _transcript_rows(project), [14, 10, 18, 8, 11, 11, 16, 16, 35, 35, 38, 42, 38, 12, 12, 24, 12, 18, 15, 14, 16, 18, 18, 17, 14, 16, 30], {14, 15}),
        ("Evaluation", _metric_rows(project), [42, 22], set()),
        ("Source Comparison", _comparison_rows(project), [24, 20, 22], {2, 3}),
        ("ML Model Comparison", _model_rows(project), [28, 18, 18], {2, 3}),
        ("Metadata", _metadata_rows(project), [28, 75], set()),
    ]

    content_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
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
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, (name, _, _, _) in enumerate(sheets, start=1)
    )
    workbook = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets>{workbook_sheets}</sheets></workbook>'''
    workbook_relationships = "".join(
        f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(sheets) + 1)
    ) + f'<Relationship Id="rId{len(sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'

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
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_relationships)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{workbook_relationships}</Relationships>''')
        archive.writestr("xl/styles.xml", styles)
        for index, (_, rows, widths, percent_columns) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(rows, widths, percent_columns))
    return target
