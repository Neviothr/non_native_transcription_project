"""Core data structures for the transcription project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TranscriptSegment:
    start: float | None = None
    end: float | None = None
    speaker: str = "Unknown"
    text: str = ""
    confidence: float | None = None

    def duration(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class Turn:
    turn_id: int
    start: float | None = None
    end: float | None = None
    speaker_raw: str = "Unknown"
    speaker: str = "Unknown"
    zoom_text: str = ""
    chatgpt_text: str = ""
    model_text: str = ""
    gold_text: str = ""
    gold_speaker: str = ""
    final_text: str = ""
    model_confidence: float | None = None
    agreement_score: float = 0.0
    quality_score: float = 0.0
    quality_label: str = "Needs major correction"
    hebrew_switch: bool = False
    hesitation_or_repetition: bool = False
    self_correction: bool = False
    unclear_speech: bool = False
    overlapping_speech: bool = False
    manual_review: bool = True
    speech_rate_wpm: float | None = None
    volume_dbfs: float | None = None
    noise_snr_db: float | None = None
    notes: str = ""

    def duration(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class ProjectMetadata:
    learner_id: str = ""
    session_number: str = ""
    conversation_type: str = "AI"
    title: str = ""
    audio_file: str = ""
    zoom_file: str = ""
    chatgpt_file: str = ""
    gold_file: str = ""
    transcription_model: str = "small-q5_1"
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class ProjectData:
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    turns: list[Turn] = field(default_factory=list)
    speaker_mapping: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    model_comparison: list[dict[str, Any]] = field(default_factory=list)
    source_transcripts: dict[str, list[TranscriptSegment]] = field(default_factory=dict)
    project_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectData":
        metadata = ProjectMetadata(**data.get("metadata", {}))
        turns: list[Turn] = []
        for item in data.get("turns", []):
            turn_data = dict(item)
            # Older project files may contain the removed correction-timer field.
            turn_data.pop("manual_correction_seconds", None)
            turns.append(Turn(**turn_data))
        source_transcripts: dict[str, list[TranscriptSegment]] = {}
        for key, items in data.get("source_transcripts", {}).items():
            source_transcripts[key] = [TranscriptSegment(**item) for item in items]
        return cls(
            metadata=metadata,
            turns=turns,
            speaker_mapping=dict(data.get("speaker_mapping", {})),
            metrics=dict(data.get("metrics", {})),
            model_comparison=list(data.get("model_comparison", [])),
            source_transcripts=source_transcripts,
            project_file=data.get("project_file", ""),
        )

    def base_directory(self) -> Path:
        if self.project_file:
            return Path(self.project_file).resolve().parent
        return Path.cwd()
