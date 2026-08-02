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
    quality_target_text: str = ""
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
        """Build a project from the exact current persisted structure."""
        if not isinstance(data, dict):
            raise TypeError("Project data must be a JSON object.")

        def require_fields(
            model_type: type,
            raw: object,
            expected: set[str],
        ) -> dict[str, Any]:
            if not isinstance(raw, dict):
                raise TypeError(f"{model_type.__name__} data must be a JSON object.")
            actual = set(raw)
            if actual != expected:
                missing = sorted(expected - actual)
                unexpected = sorted(actual - expected)
                details = []
                if missing:
                    details.append(f"missing fields: {', '.join(missing)}")
                if unexpected:
                    details.append(f"unexpected fields: {', '.join(unexpected)}")
                raise ValueError(f"{model_type.__name__} has " + "; ".join(details))
            return dict(raw)

        project_fields = {
            "metadata", "turns", "speaker_mapping", "metrics",
            "model_comparison", "source_transcripts", "project_file",
        }
        project_data = require_fields(cls, data, project_fields)
        metadata_fields = set(ProjectMetadata.__dataclass_fields__)
        metadata = ProjectMetadata(**require_fields(
            ProjectMetadata,
            project_data["metadata"],
            metadata_fields,
        ))

        raw_turns = project_data["turns"]
        if not isinstance(raw_turns, list):
            raise TypeError("Project turns must be a JSON array.")
        turn_fields = set(Turn.__dataclass_fields__)
        turns = [Turn(**require_fields(Turn, item, turn_fields)) for item in raw_turns]

        raw_sources = project_data["source_transcripts"]
        if not isinstance(raw_sources, dict):
            raise TypeError("Project source transcripts must be a JSON object.")
        segment_fields = set(TranscriptSegment.__dataclass_fields__)
        source_transcripts: dict[str, list[TranscriptSegment]] = {}
        for key, items in raw_sources.items():
            if not isinstance(items, list):
                raise TypeError(f"Source transcript {key!r} must be a JSON array.")
            source_transcripts[str(key)] = [
                TranscriptSegment(**require_fields(TranscriptSegment, item, segment_fields))
                for item in items
            ]

        return cls(
            metadata=metadata,
            turns=turns,
            speaker_mapping=dict(project_data["speaker_mapping"]),
            metrics=dict(project_data["metrics"]),
            model_comparison=list(project_data["model_comparison"]),
            source_transcripts=source_transcripts,
            project_file=str(project_data["project_file"]),
        )

    def base_directory(self) -> Path:
        if self.project_file:
            return Path(self.project_file).resolve().parent
        return Path.cwd()
