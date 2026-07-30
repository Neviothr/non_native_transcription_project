"""High-level project workflow operations used by the GUI."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .alignment import align_source_segments_to_turns, align_source_to_turns, segments_to_turns
from .audio_features import AudioFeatureError, analyze_wav_interval
from .evaluation import evaluate_turns, per_source_metrics, word_error_rate
from .ml_models import load_model, save_model, train_and_compare
from .models import ProjectData, TranscriptSegment, Turn
from .parsers import parse_transcript
from .quality import FEATURE_NAMES, extract_features, update_turn_quality


DEFAULT_ROLE_MAP = {
    "learner": "Learner",
    "student": "Learner",
    "pupil": "Learner",
    "teacher": "Teacher",
    "tutor": "Teacher",
    "instructor": "Teacher",
    "supervisor": "Supervisor",
    "observer": "Supervisor",
    "monitor": "Supervisor",
}


def import_source(project: ProjectData, source_name: str, path: str) -> list[TranscriptSegment]:
    segments = parse_transcript(path, source_name=source_name)
    project.source_transcripts[source_name] = segments
    if source_name == "zoom":
        project.metadata.zoom_file = path
    elif source_name == "chatgpt":
        project.metadata.chatgpt_file = path
    elif source_name == "gold":
        project.metadata.gold_file = path
    return segments


def initialize_turns_from_model(
    project: ProjectData,
    segments: list[TranscriptSegment],
    status_callback: Callable[[str], None] | None = None,
) -> None:
    """Create review turns from the local model and available speaker timing.

    Local Whisper provides timestamped text but not reliable speaker identities.
    When a timed Zoom transcript with speaker labels exists, it is used as the
    turn and speaker scaffold, and the local-model text is aligned onto those
    turns. Otherwise the local model segments become turns with an Unknown
    speaker until the reviewer assigns or splits them manually.
    """

    project.source_transcripts["model"] = segments
    zoom_segments = project.source_transcripts.get("zoom", [])
    uses_zoom_scaffold = _usable_speaker_scaffold(zoom_segments)
    if status_callback:
        if uses_zoom_scaffold:
            status_callback(
                "Stage 6/7 - Building review turns from "
                f"{len(zoom_segments)} timed Zoom segments with speaker labels."
            )
        else:
            status_callback(
                "Stage 6/7 - No usable timed speaker scaffold was found; "
                f"building provisional turns from {len(segments)} Whisper segments."
            )
    if uses_zoom_scaffold:
        project.turns = segments_to_turns(zoom_segments)
        for turn in project.turns:
            turn.zoom_text = turn.model_text
            turn.model_text = ""
            turn.final_text = ""
            turn.model_confidence = None
        aligned_model_text = align_source_to_turns(project.turns, segments)
        aligned_model_segments = align_source_segments_to_turns(project.turns, segments)
        for turn, text, model_segment in zip(project.turns, aligned_model_text, aligned_model_segments):
            turn.model_text = text
            turn.model_confidence = model_segment.confidence if model_segment else None
    else:
        project.turns = segments_to_turns(segments)

    _mark_overlaps(project.turns)
    for turn in project.turns:
        mapped = project.speaker_mapping.get(turn.speaker_raw)
        if mapped:
            turn.speaker = mapped
        else:
            turn.speaker = infer_role_from_name(turn.speaker_raw)
    if status_callback:
        status_callback(
            "Stage 6/7 - Aligning Zoom, ChatGPT, Gold Standard, and local-model "
            f"text across {len(project.turns)} review turns. Source counts: "
            f"Zoom={len(project.source_transcripts.get('zoom', []))}, "
            f"ChatGPT={len(project.source_transcripts.get('chatgpt', []))}, "
            f"Gold={len(project.source_transcripts.get('gold', []))}."
        )
    alignment_started = time.monotonic()
    align_all_sources(project)
    alignment_seconds = time.monotonic() - alignment_started
    if status_callback:
        status_callback(
            f"Stage 6/7 - Source alignment completed in {alignment_seconds:.2f} seconds."
        )

    if status_callback:
        status_callback(
            "Stage 7/7 - Calculating speech features, transcript agreement, "
            "quality labels, and manual-review flags."
        )

    def quality_status(message: str) -> None:
        if status_callback:
            status_callback(f"Stage 7/7 - {message}")

    analysis_started = time.monotonic()
    analyze_turns(project, status_callback=quality_status if status_callback else None)
    analysis_seconds = time.monotonic() - analysis_started
    if status_callback:
        status_callback(
            "Stage 7/7 - Initial analysis complete in "
            f"{analysis_seconds:.2f} seconds: "
            f"{sum(turn.manual_review for turn in project.turns)} of "
            f"{len(project.turns)} turns require manual review."
        )


def _usable_speaker_scaffold(segments: list[TranscriptSegment]) -> bool:
    timed = [segment for segment in segments if segment.start is not None and segment.end is not None]
    named_speakers = {
        segment.speaker.strip()
        for segment in timed
        if segment.speaker.strip() and segment.speaker.strip().casefold() != "unknown"
    }
    return bool(timed) and bool(named_speakers)


def infer_role_from_name(name: str) -> str:
    folded = name.casefold()
    for key, role in DEFAULT_ROLE_MAP.items():
        if key in folded:
            return role
    return name or "Unknown"


def align_all_sources(project: ProjectData) -> None:
    if not project.turns:
        return
    mapping = {
        "zoom": "zoom_text",
        "chatgpt": "chatgpt_text",
        "gold": "gold_text",
    }
    for source_name, attribute in mapping.items():
        segments = project.source_transcripts.get(source_name, [])
        aligned = align_source_to_turns(project.turns, segments)
        for turn, text in zip(project.turns, aligned):
            setattr(turn, attribute, text)
        if source_name == "gold":
            matched_segments = align_source_segments_to_turns(project.turns, segments)
            for turn, segment in zip(project.turns, matched_segments):
                if segment:
                    turn.gold_speaker = project.speaker_mapping.get(
                        segment.speaker, infer_role_from_name(segment.speaker)
                    )
                else:
                    turn.gold_speaker = ""

    for turn in project.turns:
        if not turn.final_text.strip():
            turn.final_text = choose_initial_text(turn)


def choose_initial_text(turn: Turn) -> str:
    candidates = [turn.model_text, turn.chatgpt_text, turn.zoom_text]
    present = [candidate for candidate in candidates if candidate.strip()]
    if not present:
        return ""
    if len(present) == 1:
        return present[0]
    # Select the candidate with greatest average agreement while preserving raw wording.
    from .alignment import text_similarity

    return max(
        present,
        key=lambda candidate: sum(text_similarity(candidate, other) for other in present if other != candidate)
        / max(1, len(present) - 1),
    )


def _mark_overlaps(turns: list[Turn]) -> None:
    for index, turn in enumerate(turns):
        turn.overlapping_speech = False
        if turn.start is None or turn.end is None:
            continue
        for other_index, other in enumerate(turns):
            if index == other_index or other.start is None or other.end is None:
                continue
            overlap = min(turn.end, other.end) - max(turn.start, other.start)
            if overlap > 0.15 and turn.speaker_raw != other.speaker_raw:
                turn.overlapping_speech = True
                break


def analyze_turns(
    project: ProjectData,
    predictor: object | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> None:
    audio_path = Path(project.metadata.audio_file) if project.metadata.audio_file else None
    for index, turn in enumerate(project.turns, start=1):
        if status_callback:
            status_callback(f"Analyzing turn {index} of {len(project.turns)}...")
        if audio_path and audio_path.exists() and audio_path.suffix.casefold() == ".wav":
            try:
                signal = analyze_wav_interval(audio_path, turn.start, turn.end)
                turn.volume_dbfs = signal["volume_dbfs"]
                turn.noise_snr_db = signal["noise_snr_db"]
            except AudioFeatureError:
                turn.volume_dbfs = None
                turn.noise_snr_db = None
        update_turn_quality(turn, predictor)
    project.metrics = evaluate_turns(project.turns)
    project.metrics["source_comparison"] = per_source_metrics(project.turns)


def apply_speaker_mapping(project: ProjectData, mapping: dict[str, str]) -> None:
    project.speaker_mapping.update(mapping)
    for turn in project.turns:
        turn.speaker = project.speaker_mapping.get(turn.speaker_raw, turn.speaker)
        if turn.gold_speaker:
            turn.gold_speaker = project.speaker_mapping.get(turn.gold_speaker, turn.gold_speaker)
    # Reapply Gold Standard mapping from the original source labels when available.
    gold_segments = project.source_transcripts.get("gold", [])
    if gold_segments and project.turns:
        matched_segments = align_source_segments_to_turns(project.turns, gold_segments)
        for turn, segment in zip(project.turns, matched_segments):
            if segment:
                turn.gold_speaker = project.speaker_mapping.get(
                    segment.speaker, infer_role_from_name(segment.speaker)
                )


def training_examples_from_project(project: ProjectData) -> tuple[list[list[float]], list[int]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for turn in project.turns:
        if not turn.gold_text.strip() or not turn.model_text.strip():
            continue
        error = word_error_rate(turn.gold_text, turn.model_text)
        if error <= 0.10:
            label = 0
        elif error <= 0.30:
            label = 1
        else:
            label = 2
        features = extract_features(turn)
        rows.append([features[name] for name in FEATURE_NAMES])
        labels.append(label)
    return rows, labels


def append_training_examples(project: ProjectData, path: str | Path) -> int:
    rows, labels = training_examples_from_project(project)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, object]] = []
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = []
    existing_keys = {
        (tuple(round(float(value), 10) for value in item.get("features", [])), int(item.get("label", -1)))
        for item in existing
        if isinstance(item, dict)
    }
    added = 0
    for row, label in zip(rows, labels):
        key = (tuple(round(float(value), 10) for value in row), int(label))
        if key in existing_keys:
            continue
        existing.append({"features": row, "label": label})
        existing_keys.add(key)
        added += 1
    target.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return added


def train_quality_model(training_path: str | Path, model_path: str | Path) -> tuple[object, list[dict[str, float | str]]]:
    data = json.loads(Path(training_path).read_text(encoding="utf-8"))
    rows = [list(map(float, item["features"])) for item in data]
    labels = [int(item["label"]) for item in data]
    model, comparison = train_and_compare(rows, labels)
    save_model(model, model_path)
    return model, comparison


def load_quality_model_if_available(path: str | Path) -> object | None:
    target = Path(path)
    if not target.exists():
        return None
    return load_model(target)