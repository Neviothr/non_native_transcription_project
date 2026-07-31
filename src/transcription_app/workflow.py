"""High-level project workflow operations used by the GUI."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
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
from .text_utils import normalize_for_comparison, words


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

PROJECT_SPEAKER_ROLES = ("Learner", "Teacher", "Supervisor")
_UNKNOWN_SPEAKER_LABELS = {
    "",
    "unknown",
    "unmapped",
    "none",
    "n a",
    "na",
    "not available",
    "speaker",
}
_AI_TEACHER_LABELS = {
    "ai",
    "assistant",
    "bot",
    "chatgpt",
    "chat gpt",
    "virtual teacher",
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


def reload_selected_transcripts(
    project: ProjectData,
    selected_paths: dict[str, str],
) -> dict[str, int]:
    """Reload the selected transcript files as one atomic operation.

    Zoom, ChatGPT, and Gold Standard sources are parsed before the project is
    changed. If any selected file fails, the existing imported sources and
    metadata remain untouched. Empty path fields remove the corresponding stale
    source so a previous selection cannot silently leak into a new run.
    """

    source_names = ("zoom", "chatgpt", "gold")
    normalized_paths = {
        source_name: str(selected_paths.get(source_name, "")).strip()
        for source_name in source_names
    }
    parsed_sources: dict[str, list[TranscriptSegment]] = {}
    for source_name, path in normalized_paths.items():
        if path:
            parsed_sources[source_name] = parse_transcript(
                path,
                source_name=source_name,
            )

    metadata_fields = {
        "zoom": "zoom_file",
        "chatgpt": "chatgpt_file",
        "gold": "gold_file",
    }
    for source_name in source_names:
        path = normalized_paths[source_name]
        setattr(project.metadata, metadata_fields[source_name], path)
        if path:
            project.source_transcripts[source_name] = parsed_sources[source_name]
        else:
            project.source_transcripts.pop(source_name, None)

    return {
        source_name: len(parsed_sources.get(source_name, []))
        for source_name in source_names
    }


def initialize_turns_from_model(
    project: ProjectData,
    segments: list[TranscriptSegment],
    status_callback: Callable[[str], None] | None = None,
) -> None:
    """Create review turns from the local model and available speaker timing.

    Local Whisper provides timestamped text but not reliable speaker identities.
    A timed labeled transcript is used as the preferred turn scaffold. When only
    an untimed labeled transcript is available, its speaker labels are transferred
    to the timestamped local-model turns by monotonic text alignment. Roles are
    then assigned automatically using direct evidence and a logged dialogue-role
    fallback for ordinary participant names or generic labels.
    """

    project.source_transcripts["model"] = segments
    scaffold_name, scaffold_segments = _best_speaker_source(project)
    uses_timed_scaffold = _usable_speaker_scaffold(scaffold_segments)
    if status_callback:
        if uses_timed_scaffold:
            status_callback(
                "Stage 6/7 - Building review turns from "
                f"{len(scaffold_segments)} timed {scaffold_name} segments with speaker labels."
            )
        else:
            status_callback(
                "Stage 6/7 - No usable timed speaker scaffold was found; "
                f"building provisional turns from {len(segments)} Whisper segments."
            )
    if uses_timed_scaffold:
        project.turns = segments_to_turns(scaffold_segments)
        scaffold_attribute = {
            "zoom": "zoom_text",
            "chatgpt": "chatgpt_text",
            "gold": "gold_text",
        }.get(scaffold_name, "zoom_text")
        for turn in project.turns:
            setattr(turn, scaffold_attribute, turn.model_text)
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
        if scaffold_segments:
            transferred = _transfer_speaker_labels_to_turns(
                project.turns,
                scaffold_segments,
            )
            if status_callback:
                status_callback(
                    "Stage 6/7 - Transferred speaker labels from the untimed "
                    f"{scaffold_name} transcript to {transferred} of "
                    f"{len(project.turns)} Whisper turn(s)."
                )
        elif status_callback:
            status_callback(
                "Stage 6/7 - No imported transcript contains usable speaker labels; "
                "local Whisper cannot distinguish speakers by itself."
            )

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

    automatically_map_speakers(project, status_callback=status_callback)

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


def _usable_speaker_label(value: str) -> bool:
    return normalize_for_comparison(value) not in _UNKNOWN_SPEAKER_LABELS


def _best_speaker_source(
    project: ProjectData,
) -> tuple[str, list[TranscriptSegment]]:
    """Choose the strongest imported source carrying speaker labels."""
    candidates: list[tuple[int, int, str, list[TranscriptSegment]]] = []
    source_priority = {"zoom": 3, "gold": 2, "chatgpt": 1}
    for source_name in ("zoom", "gold", "chatgpt"):
        source_segments = project.source_transcripts.get(source_name, [])
        labeled = [
            segment for segment in source_segments
            if _usable_speaker_label(segment.speaker)
        ]
        if not labeled:
            continue
        timed_count = sum(
            segment.start is not None and segment.end is not None
            for segment in labeled
        )
        candidates.append(
            (
                1 if timed_count else 0,
                source_priority[source_name],
                source_name,
                source_segments,
            )
        )
    if not candidates:
        return "none", []
    _timed, _priority, source_name, source_segments = max(candidates)
    return source_name, source_segments


def _transfer_speaker_labels_to_turns(
    turns: list[Turn],
    source_segments: list[TranscriptSegment],
) -> int:
    """Transfer aligned imported labels onto timestamped Whisper turns."""
    matched_segments = align_source_segments_to_turns(turns, source_segments)
    transferred = 0
    for turn, segment in zip(turns, matched_segments):
        if segment is None or not _usable_speaker_label(segment.speaker):
            continue
        turn.speaker_raw = segment.speaker.strip()
        turn.speaker = turn.speaker_raw
        transferred += 1
    return transferred


def infer_role_from_name(name: str) -> str:
    folded = name.casefold()
    for key, role in DEFAULT_ROLE_MAP.items():
        if key in folded:
            return role
    return name or "Unknown"


def _normalized_project_role(value: str) -> str | None:
    normalized = normalize_for_comparison(value)
    for role in PROJECT_SPEAKER_ROLES:
        if normalized == normalize_for_comparison(role):
            return role
    return None


def _role_from_label(
    label: str,
    project: ProjectData,
) -> tuple[str | None, str]:
    normalized = normalize_for_comparison(label)
    if normalized in _UNKNOWN_SPEAKER_LABELS:
        return None, "placeholder or missing label"

    learner_id = normalize_for_comparison(project.metadata.learner_id)
    if learner_id:
        label_tokens = set(words(normalized))
        learner_tokens = set(words(learner_id))
        if normalized == learner_id or (
            learner_tokens
            and learner_tokens.issubset(label_tokens)
        ):
            return "Learner", "matched the project learner ID"

    label_tokens = set(words(normalized))
    for alias, role in DEFAULT_ROLE_MAP.items():
        if alias in label_tokens:
            return role, f"recognized role word '{alias}'"

    if (
        project.metadata.conversation_type.casefold() == "ai"
        and normalized in _AI_TEACHER_LABELS
    ):
        return "Teacher", "recognized AI teacher label"

    return None, "no reliable role evidence"


_TEACHER_PROMPT_RE = re.compile(
    r"(?:\?|\b(?:what|why|when|where|who|how|can you|could you|would you|"
    r"tell me|describe|explain|please|let us|let's|next question)\b)",
    re.IGNORECASE,
)


def _raw_speaker_profiles(project: ProjectData) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    for index, turn in enumerate(project.turns):
        label = turn.speaker_raw.strip()
        if not _usable_speaker_label(label):
            continue
        profile = profiles.setdefault(
            label,
            {
                "first_index": float(index),
                "turns": 0.0,
                "words": 0.0,
                "teacher_score": 0.0,
            },
        )
        text = turn.final_text or turn.zoom_text or turn.chatgpt_text or turn.model_text
        profile["turns"] += 1.0
        profile["words"] += float(len(words(text)))
        profile["teacher_score"] += float(len(_TEACHER_PROMPT_RE.findall(text)))
    return profiles


def _apply_dialogue_role_fallbacks(
    project: ProjectData,
    raw_labels: list[str],
    mapping: dict[str, str],
    reasons: dict[str, str],
) -> None:
    """Complete ordinary two/three-person conversations without a dialog."""
    profiles = _raw_speaker_profiles(project)
    unresolved = [label for label in raw_labels if label not in mapping]
    if not unresolved:
        return

    resolved_roles = {mapping[label] for label in raw_labels if label in mapping}
    expected_roles = list(
        ("Learner", "Teacher")
        if len(raw_labels) == 2
        else PROJECT_SPEAKER_ROLES
    )
    missing_roles = [role for role in expected_roles if role not in resolved_roles]

    if "Teacher" in missing_roles and unresolved:
        teacher_label = max(
            unresolved,
            key=lambda label: (
                profiles.get(label, {}).get("teacher_score", 0.0),
                -profiles.get(label, {}).get("first_index", float("inf")),
            ),
        )
        mapping[teacher_label] = "Teacher"
        teacher_score = profiles.get(teacher_label, {}).get("teacher_score", 0.0)
        reasons[teacher_label] = (
            "dialogue-role fallback: most teacher-like prompts/questions"
            if teacher_score > 0
            else "dialogue-role fallback: first participant in a two/three-speaker exchange"
        )
        unresolved.remove(teacher_label)
        missing_roles.remove("Teacher")

    if "Supervisor" in missing_roles and len(raw_labels) >= 3 and unresolved:
        supervisor_label = min(
            unresolved,
            key=lambda label: (
                profiles.get(label, {}).get("words", 0.0),
                profiles.get(label, {}).get("turns", 0.0),
                profiles.get(label, {}).get("first_index", float("inf")),
            ),
        )
        mapping[supervisor_label] = "Supervisor"
        reasons[supervisor_label] = (
            "dialogue-role fallback: least speaking activity among three participants"
        )
        unresolved.remove(supervisor_label)
        missing_roles.remove("Supervisor")

    for label, role in zip(
        sorted(
            unresolved,
            key=lambda item: profiles.get(item, {}).get("first_index", float("inf")),
        ),
        missing_roles,
    ):
        mapping[label] = role
        reasons[label] = "dialogue-role fallback: remaining participant role"


def automatically_map_speakers(
    project: ProjectData,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Infer and apply speaker-role mappings without opening a GUI dialog.

    The mapper first uses saved mappings, explicit role words, the configured
    learner ID, and aligned Gold/ChatGPT evidence. For ordinary two- or
    three-participant conversations it then completes unresolved roles using
    logged dialogue structure and speaking-activity fallbacks.
    """

    labels = {
        turn.speaker_raw.strip()
        for turn in project.turns
        if turn.speaker_raw.strip()
    }
    for source_segments in project.source_transcripts.values():
        labels.update(
            segment.speaker.strip()
            for segment in source_segments
            if segment.speaker.strip()
        )

    usable_labels = sorted(
        label
        for label in labels
        if normalize_for_comparison(label) not in _UNKNOWN_SPEAKER_LABELS
    )
    if status_callback:
        status_callback(
            "Stage 6/7 - Automatic speaker mapping started for "
            f"{len(usable_labels)} usable label(s)."
        )

    mapping: dict[str, str] = {}
    reasons: dict[str, str] = {}
    evidence_votes: dict[str, Counter[str]] = {
        label: Counter() for label in usable_labels
    }

    for label in usable_labels:
        saved_role = _normalized_project_role(project.speaker_mapping.get(label, ""))
        if saved_role is not None:
            mapping[label] = saved_role
            reasons[label] = "reused saved project mapping"
            continue

        role, reason = _role_from_label(label, project)
        if role is not None:
            mapping[label] = role
            reasons[label] = reason

    # A role-labeled Gold or ChatGPT transcript can identify a raw Zoom speaker
    # by alignment even when the Zoom label itself is only a participant name.
    for source_name in ("gold", "chatgpt"):
        source_segments = project.source_transcripts.get(source_name, [])
        if not source_segments or not project.turns:
            continue
        matched_segments = align_source_segments_to_turns(
            project.turns,
            source_segments,
        )
        for turn, segment in zip(project.turns, matched_segments):
            raw_label = turn.speaker_raw.strip()
            if (
                not raw_label
                or raw_label not in evidence_votes
                or segment is None
            ):
                continue
            role, _reason = _role_from_label(segment.speaker, project)
            if role is not None:
                evidence_votes[raw_label][role] += 1

    for label in usable_labels:
        if label in mapping or not evidence_votes[label]:
            continue
        ranked = evidence_votes[label].most_common()
        top_role, top_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else 0
        if top_count > second_count:
            mapping[label] = top_role
            reasons[label] = (
                f"aligned transcript role evidence ({top_count} supporting turn(s))"
            )

    # Limited elimination is safe only when exactly one participant and exactly
    # one expected role remain. No first-speaker or speaking-time guess is used.
    raw_labels = sorted(
        {
            turn.speaker_raw.strip()
            for turn in project.turns
            if normalize_for_comparison(turn.speaker_raw)
            not in _UNKNOWN_SPEAKER_LABELS
        }
    )
    unresolved_raw = [label for label in raw_labels if label not in mapping]
    resolved_roles = {
        mapping[label]
        for label in raw_labels
        if label in mapping and mapping[label] in PROJECT_SPEAKER_ROLES
    }
    expected_roles = (
        ("Learner", "Teacher")
        if len(raw_labels) == 2
        else PROJECT_SPEAKER_ROLES
    )
    missing_roles = [
        role for role in expected_roles if role not in resolved_roles
    ]
    if len(unresolved_raw) == 1 and len(missing_roles) == 1:
        label = unresolved_raw[0]
        mapping[label] = missing_roles[0]
        reasons[label] = "only remaining participant role"

    _apply_dialogue_role_fallbacks(project, raw_labels, mapping, reasons)

    complete_mapping = {
        label: mapping.get(label, "Unknown")
        for label in usable_labels
    }
    apply_speaker_mapping(project, complete_mapping)

    if status_callback:
        for label in usable_labels:
            role = complete_mapping[label]
            reason = reasons.get(label, "no reliable role evidence")
            status_callback(
                f"Stage 6/7 - Speaker mapping: {label!r} -> {role} ({reason})."
            )
        resolved_count = sum(
            role != "Unknown" for role in complete_mapping.values()
        )
        unresolved_count = len(complete_mapping) - resolved_count
        status_callback(
            "Stage 6/7 - Automatic speaker mapping complete: "
            f"{resolved_count} resolved, {unresolved_count} unresolved."
        )

    return complete_mapping


def recover_speaker_mapping(
    project: ProjectData,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Recover labels and roles for existing turns without rerunning Whisper."""
    if not project.turns:
        return {}
    source_name, source_segments = _best_speaker_source(project)
    if source_segments:
        transferred = _transfer_speaker_labels_to_turns(
            project.turns,
            source_segments,
        )
        if status_callback:
            status_callback(
                "Speaker recovery: transferred labels from the "
                f"{source_name} transcript to {transferred} of "
                f"{len(project.turns)} existing turn(s)."
            )
    elif status_callback:
        status_callback(
            "Speaker recovery: no imported transcript contains usable speaker labels."
        )
    return automatically_map_speakers(
        project,
        status_callback=status_callback,
    )



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
