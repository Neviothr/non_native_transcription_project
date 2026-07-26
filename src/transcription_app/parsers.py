"""Parsers for Zoom, ChatGPT, and manually prepared transcript files."""

from __future__ import annotations

import csv
import re
from pathlib import Path

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


def _first_present(row: dict[str, str], candidates: tuple[str, ...]) -> str:
    lowered = {key.strip().casefold(): (value or "") for key, value in row.items()}
    for candidate in candidates:
        if candidate in lowered and lowered[candidate].strip():
            return lowered[candidate].strip()
    return ""


def parse_csv_file(path: Path) -> list[TranscriptSegment]:
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
            text = _first_present(row, ("text", "transcript", "utterance", "content", "תמלול"))
            if not text:
                continue
            start_raw = _first_present(row, ("start", "start time", "start_time", "begin", "זמן התחלה"))
            end_raw = _first_present(row, ("end", "end time", "end_time", "finish", "זמן סיום"))
            speaker = _first_present(row, ("speaker", "name", "participant", "דובר")) or "Unknown"
            try:
                start = parse_timestamp(start_raw) if start_raw else None
            except ValueError:
                start = None
            try:
                end = parse_timestamp(end_raw) if end_raw else None
            except ValueError:
                end = None
            segments.append(TranscriptSegment(start, end, speaker, text))
    return merge_adjacent_segments(segments)


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


def parse_transcript(path: str | Path) -> list[TranscriptSegment]:
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix in {".vtt", ".srt"}:
        return parse_vtt_or_srt(source)
    if suffix in {".csv", ".tsv"}:
        return parse_csv_file(source)
    if suffix in {".txt", ".text", ".md"}:
        return parse_text_file(source)
    raise ValueError(
        f"Unsupported transcript format '{source.suffix}'. Use VTT, SRT, TXT, CSV, TSV, or Markdown."
    )
