"""Parsers for Zoom, ChatGPT, and manually prepared transcript files."""

from __future__ import annotations

import csv
import math
import posixpath
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from .models import TranscriptSegment

_TIMESTAMP_RE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2}(?:[.,]\d+)?)|(?P<m2>\d{1,2}):(?P<s2>\d{2}(?:[.,]\d+)?)"
)
_RANGE_RE = re.compile(
    r"(?P<start>(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d+)?)\s*--?>\s*"
    r"(?P<end>(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d+)?)"
)
_SPEAKER_LINE_RE = re.compile(
    r"^(?:\[(?P<bracket_time>[^\]]+)\]\s*)?(?P<speaker>[^:\n]{1,80}):\s*(?P<text>.+)$"
)
_CELL_REFERENCE_RE = re.compile(r"([A-Za-z]+)")
_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_NS = {"m": _MAIN_NS, "r": _REL_NS, "pr": _PACKAGE_REL_NS}

_GENERIC_TEXT_HEADERS = (
    "text",
    "transcript",
    "utterance",
    "content",
    "final transcript",
    "manual transcript",
    "corrected transcript",
    "תמלול",
    "תמלול סופי",
)
_SOURCE_TEXT_HEADERS = {
    "zoom": (
        "zoom transcript",
        "zoom transcription",
        "zoom text",
    ),
    "chatgpt": (
        "chatgpt transcript",
        "chat gpt transcript",
        "chatgpt transcription",
        "chatgpt text",
        "chat gpt text",
    ),
    "gold": (
        "gold standard",
        "gold standard transcript",
        "gold transcript",
        "gold text",
        "reference transcript",
        "human transcript",
        "manually corrected transcript",
    ),
}
_START_HEADERS = (
    "start",
    "start time",
    "start_time",
    "start seconds",
    "start (s)",
    "begin",
    "זמן התחלה",
)
_END_HEADERS = (
    "end",
    "end time",
    "end_time",
    "end seconds",
    "end (s)",
    "finish",
    "זמן סיום",
)
_SPEAKER_HEADERS = (
    "speaker",
    "speaker role",
    "raw speaker",
    "gold speaker",
    "name",
    "participant",
    "דובר",
)


@dataclass(slots=True)
class _XlsxCell:
    text: str = ""
    is_time: bool = False


def parse_timestamp(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = "0"
        minutes, seconds = parts
    else:
        raise ValueError(f"Unsupported timestamp: {value}")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_time_value(value: str, *, excel_time: bool = False) -> float | None:
    raw = value.strip().replace(",", ".")
    if not raw:
        return None
    if ":" in raw:
        return parse_timestamp(raw)
    numeric = float(raw)
    if not math.isfinite(numeric):
        raise ValueError(f"Unsupported timestamp: {value}")
    return numeric * 86400.0 if excel_time else numeric


def _clean_vtt_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    return " ".join(text.split())


def _split_speaker(text: str, default: str = "Unknown") -> tuple[str, str]:
    match = _SPEAKER_LINE_RE.match(text.strip())
    if match:
        return match.group("speaker").strip(), match.group("text").strip()
    return default, text.strip()


def parse_vtt_or_srt(path: Path) -> list[TranscriptSegment]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    segments: list[TranscriptSegment] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        range_match = _RANGE_RE.search(line)
        if not range_match:
            index += 1
            continue
        start = parse_timestamp(range_match.group("start"))
        end = parse_timestamp(range_match.group("end"))
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            current = lines[index].strip()
            if not current.isdigit():
                text_lines.append(current)
            index += 1
        text = _clean_vtt_text(" ".join(text_lines))
        speaker, text = _split_speaker(text)
        if text:
            segments.append(TranscriptSegment(start, end, speaker, text))
        index += 1
    return merge_adjacent_segments(segments)


def _normalize_header(value: str) -> str:
    normalized = value.strip().casefold().replace("\n", " ")
    normalized = re.sub(r"[_\-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _text_headers(source_name: str | None) -> tuple[str, ...]:
    source_specific = _SOURCE_TEXT_HEADERS.get((source_name or "").casefold(), ())
    return source_specific + _GENERIC_TEXT_HEADERS


def _first_present(row: dict[str, str], candidates: tuple[str, ...]) -> str:
    lowered = {_normalize_header(key): (value or "") for key, value in row.items()}
    for candidate in candidates:
        value = lowered.get(_normalize_header(candidate), "")
        if value.strip():
            return value.strip()
    return ""


def parse_csv_file(path: Path, source_name: str | None = None) -> list[TranscriptSegment]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        segments: list[TranscriptSegment] = []
        for row in reader:
            text = _first_present(row, _text_headers(source_name))
            if not text:
                continue
            start_raw = _first_present(row, _START_HEADERS)
            end_raw = _first_present(row, _END_HEADERS)
            speaker = _first_present(row, _SPEAKER_HEADERS) or "Unknown"
            try:
                start = _parse_time_value(start_raw) if start_raw else None
            except ValueError:
                start = None
            try:
                end = _parse_time_value(end_raw) if end_raw else None
            except ValueError:
                end = None
            if speaker == "Unknown":
                speaker, text = _split_speaker(text)
            segments.append(TranscriptSegment(start, end, speaker, text))
    return merge_adjacent_segments(segments)


def _column_index(reference: str) -> int:
    match = _CELL_REFERENCE_RE.match(reference)
    if not match:
        return 0
    index = 0
    for character in match.group(1).upper():
        index = index * 26 + ord(character) - 64
    return max(0, index - 1)


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(node.text or "" for node in item.findall(".//m:t", _NS))
        for item in root.findall("m:si", _NS)
    ]


def _is_time_number_format(format_code: str) -> bool:
    cleaned = re.sub(r'"[^"]*"', "", format_code.casefold())
    cleaned = re.sub(r"\\.", "", cleaned)
    return (
        "[h]" in cleaned
        or "[m]" in cleaned
        or "[s]" in cleaned
        or ("h" in cleaned and ("m" in cleaned or "s" in cleaned))
    )


def _xlsx_time_style_indexes(archive: zipfile.ZipFile) -> set[int]:
    try:
        root = ElementTree.fromstring(archive.read("xl/styles.xml"))
    except KeyError:
        return set()
    custom_formats: dict[int, str] = {}
    for node in root.findall("m:numFmts/m:numFmt", _NS):
        try:
            format_id = int(node.attrib.get("numFmtId", ""))
        except ValueError:
            continue
        custom_formats[format_id] = node.attrib.get("formatCode", "")
    built_in_time_ids = {18, 19, 20, 21, 22, 45, 46, 47}
    time_styles: set[int] = set()
    for index, node in enumerate(root.findall("m:cellXfs/m:xf", _NS)):
        try:
            format_id = int(node.attrib.get("numFmtId", "0"))
        except ValueError:
            continue
        if format_id in built_in_time_ids or _is_time_number_format(custom_formats.get(format_id, "")):
            time_styles.add(index)
    return time_styles


def _xlsx_sheet_paths(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    try:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    except KeyError as exc:
        raise ValueError("The XLSX workbook is missing required workbook metadata.") from exc

    targets = {
        relationship.attrib.get("Id", ""): relationship.attrib.get("Target", "")
        for relationship in relationships.findall("pr:Relationship", _NS)
    }
    sheets: list[tuple[str, str]] = []
    for sheet in workbook.findall("m:sheets/m:sheet", _NS):
        relationship_id = sheet.attrib.get(f"{{{_REL_NS}}}id", "")
        target = targets.get(relationship_id, "")
        if not target:
            continue
        if target.startswith("/"):
            path = target.lstrip("/")
        else:
            path = posixpath.normpath(posixpath.join("xl", target))
        sheets.append((sheet.attrib.get("name", "Sheet"), path))
    return sheets


def _xlsx_cell_value(
    cell: ElementTree.Element,
    shared_strings: list[str],
    time_styles: set[int],
) -> _XlsxCell:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        text = "".join(node.text or "" for node in cell.findall(".//m:t", _NS))
    else:
        value_node = cell.find("m:v", _NS)
        text = value_node.text if value_node is not None and value_node.text is not None else ""
        if cell_type == "s" and text:
            try:
                text = shared_strings[int(text)]
            except (ValueError, IndexError):
                text = ""
        elif cell_type == "b":
            text = "TRUE" if text == "1" else "FALSE"
    try:
        style_index = int(cell.attrib.get("s", "0"))
    except ValueError:
        style_index = 0
    return _XlsxCell(text=text, is_time=style_index in time_styles)


def _xlsx_sheet_rows(
    archive: zipfile.ZipFile,
    sheet_path: str,
    shared_strings: list[str],
    time_styles: set[int],
) -> list[list[_XlsxCell]]:
    try:
        root = ElementTree.fromstring(archive.read(sheet_path))
    except KeyError as exc:
        raise ValueError(f"The XLSX workbook references a missing worksheet: {sheet_path}") from exc

    rows: list[list[_XlsxCell]] = []
    for row_node in root.findall("m:sheetData/m:row", _NS):
        cells: dict[int, _XlsxCell] = {}
        next_index = 0
        for cell_node in row_node.findall("m:c", _NS):
            reference = cell_node.attrib.get("r", "")
            index = _column_index(reference) if reference else next_index
            cells[index] = _xlsx_cell_value(cell_node, shared_strings, time_styles)
            next_index = index + 1
        if cells:
            maximum = max(cells)
            rows.append([cells.get(index, _XlsxCell()) for index in range(maximum + 1)])
        else:
            rows.append([])
    return rows


def _header_indexes(
    header_row: list[_XlsxCell],
    source_name: str | None,
) -> tuple[int | None, int | None, int | None, int | None]:
    header_map: dict[str, int] = {}
    for index, cell in enumerate(header_row):
        normalized = _normalize_header(cell.text)
        if normalized and normalized not in header_map:
            header_map[normalized] = index

    def find(candidates: tuple[str, ...]) -> int | None:
        for candidate in candidates:
            index = header_map.get(_normalize_header(candidate))
            if index is not None:
                return index
        return None

    return (
        find(_text_headers(source_name)),
        find(_START_HEADERS),
        find(_END_HEADERS),
        find(_SPEAKER_HEADERS),
    )


def _cell_at(row: list[_XlsxCell], index: int | None) -> _XlsxCell:
    if index is None or index < 0 or index >= len(row):
        return _XlsxCell()
    return row[index]


def _parse_xlsx_table(
    rows: list[list[_XlsxCell]],
    header_index: int,
    source_name: str | None,
) -> list[TranscriptSegment]:
    text_index, start_index, end_index, speaker_index = _header_indexes(
        rows[header_index], source_name
    )
    if text_index is None:
        return []

    segments: list[TranscriptSegment] = []
    for row in rows[header_index + 1 :]:
        text = _cell_at(row, text_index).text.strip()
        if not text:
            continue
        start_cell = _cell_at(row, start_index)
        end_cell = _cell_at(row, end_index)
        speaker = _cell_at(row, speaker_index).text.strip() or "Unknown"
        try:
            start = _parse_time_value(
                start_cell.text,
                excel_time=start_cell.is_time,
            )
        except ValueError:
            start = None
        try:
            end = _parse_time_value(
                end_cell.text,
                excel_time=end_cell.is_time,
            )
        except ValueError:
            end = None
        if speaker == "Unknown":
            speaker, text = _split_speaker(text)
        segments.append(TranscriptSegment(start, end, speaker, text))
    return merge_adjacent_segments(segments)


def _single_column_xlsx_segments(
    sheets: list[tuple[str, list[list[_XlsxCell]]]],
    source_name: str | None,
) -> list[TranscriptSegment]:
    source_key = (source_name or "").casefold()
    ordered = sorted(
        sheets,
        key=lambda item: (
            0 if source_key and source_key in item[0].casefold().replace(" ", "") else 1
        ),
    )
    recognized_headers = {
        _normalize_header(value)
        for value in _text_headers(source_name)
    }
    for _sheet_name, rows in ordered:
        populated_columns = {
            index
            for row in rows
            for index, cell in enumerate(row)
            if cell.text.strip()
        }
        if len(populated_columns) != 1:
            continue
        column = next(iter(populated_columns))
        values = [
            _cell_at(row, column).text.strip()
            for row in rows
            if _cell_at(row, column).text.strip()
        ]
        if values and _normalize_header(values[0]) in recognized_headers:
            values = values[1:]
        segments = []
        for value in values:
            speaker, utterance = _split_speaker(value)
            if utterance:
                segments.append(TranscriptSegment(speaker=speaker, text=utterance))
        if segments:
            return segments
    return []


def parse_xlsx_file(path: Path, source_name: str | None = None) -> list[TranscriptSegment]:
    try:
        with zipfile.ZipFile(path) as archive:
            shared_strings = _xlsx_shared_strings(archive)
            time_styles = _xlsx_time_style_indexes(archive)
            sheets = [
                (
                    sheet_name,
                    _xlsx_sheet_rows(
                        archive,
                        sheet_path,
                        shared_strings,
                        time_styles,
                    ),
                )
                for sheet_name, sheet_path in _xlsx_sheet_paths(archive)
            ]
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid XLSX workbook: {path.name}") from exc
    except ElementTree.ParseError as exc:
        raise ValueError(f"Could not read XLSX workbook XML: {path.name}") from exc

    for _sheet_name, rows in sheets:
        for header_index, row in enumerate(rows[:25]):
            text_index, _start, _end, _speaker = _header_indexes(row, source_name)
            if text_index is None:
                continue
            segments = _parse_xlsx_table(rows, header_index, source_name)
            if segments:
                return segments

    fallback = _single_column_xlsx_segments(sheets, source_name)
    if fallback:
        return fallback

    accepted = ", ".join(f'"{header}"' for header in _text_headers(source_name)[:6])
    raise ValueError(
        "No transcript column was found in the XLSX workbook. "
        f"Use a header such as {accepted}, or place transcript lines in a single column."
    )


def parse_text_file(path: Path) -> list[TranscriptSegment]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    segments: list[TranscriptSegment] = []
    for line in lines:
        range_match = _RANGE_RE.search(line)
        start: float | None = None
        end: float | None = None
        remaining = line
        if range_match:
            start = parse_timestamp(range_match.group("start"))
            end = parse_timestamp(range_match.group("end"))
            remaining = (line[: range_match.start()] + line[range_match.end() :]).strip(" -[]")
        speaker, utterance = _split_speaker(remaining)
        if speaker != "Unknown":
            bracket = _SPEAKER_LINE_RE.match(remaining)
            if bracket and bracket.group("bracket_time") and start is None:
                timestamp_match = _TIMESTAMP_RE.search(bracket.group("bracket_time"))
                if timestamp_match:
                    start = parse_timestamp(timestamp_match.group(0))
        if utterance:
            segments.append(TranscriptSegment(start, end, speaker, utterance))

    if len(segments) == 1 and segments[0].speaker == "Unknown" and len(text) > 300:
        # Untimed prose transcripts are divided at sentence boundaries for alignment.
        sentence_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\u0590-\u05FF])", text.strip())
        return [TranscriptSegment(text=part.strip()) for part in sentence_parts if part.strip()]
    return merge_adjacent_segments(segments)


def merge_adjacent_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    if not segments:
        return []
    merged: list[TranscriptSegment] = []
    for segment in segments:
        if (
            merged
            and segment.speaker == merged[-1].speaker
            and segment.start is not None
            and merged[-1].end is not None
            and segment.start - merged[-1].end <= 0.6
        ):
            merged[-1].text = f"{merged[-1].text} {segment.text}".strip()
            merged[-1].end = segment.end
        else:
            merged.append(segment)
    return merged


def parse_transcript(
    path: str | Path,
    source_name: str | None = None,
) -> list[TranscriptSegment]:
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix in {".vtt", ".srt"}:
        return parse_vtt_or_srt(source)
    if suffix in {".csv", ".tsv"}:
        return parse_csv_file(source, source_name)
    if suffix == ".xlsx":
        return parse_xlsx_file(source, source_name)
    if suffix in {".txt", ".text", ".md"}:
        return parse_text_file(source)
    raise ValueError(
        f"Unsupported transcript format '{source.suffix}'. "
        "Use VTT, SRT, TXT, CSV, TSV, Markdown, or XLSX."
    )