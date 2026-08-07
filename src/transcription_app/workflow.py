"""High-level project workflow operations used by the GUI."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .alignment import align_source_segments_to_turns, align_source_to_turns, segments_to_turns
from .audio_features import AudioFeatureError, analyze_wav_intervals
from .evaluation import evaluate_turns, per_source_metrics, word_error_rate
from .grammar_events import (
    GRAMMAR_EVENT_TYPE,
    GRAMMAR_GUARD_SOURCE,
    is_likely_learner_turn,
    refresh_grammar_preservation_events,
)
from .ml_models import load_model, save_model, train_and_compare
from .models import ProjectData, TranscriptSegment, Turn
from .parsers import parse_transcript
from .quality import FEATURE_NAMES, extract_features, update_turn_quality
from .speech_events import (
    reassociate_automatic_delay_events,
    remap_nonautomatic_event_turn_ids,
    replace_detected_delay_events,
)
from .text_utils import (
    DetectedSpeechEvent,
    detected_speech_events,
    normalize_for_comparison,
    words,
)


STUDENT_ROLE = "Student"
TEACHER_ROLE = "Teacher"
SUPERVISOR_ROLE = "Supervisor"
AI_ROLE = "AI"
UNKNOWN_ROLE = "Unknown"

QUALITY_TRAINING_SCHEMA_VERSION = 4
QUALITY_LABEL_TARGET = "initial_transcript_wer"

AI_CONVERSATION_ROLES = (STUDENT_ROLE, SUPERVISOR_ROLE, AI_ROLE)
HUMAN_TEACHER_CONVERSATION_ROLES = (STUDENT_ROLE, TEACHER_ROLE)
ALL_SPEAKER_ROLES = (
    STUDENT_ROLE,
    TEACHER_ROLE,
    SUPERVISOR_ROLE,
    AI_ROLE,
)

_STUDENT_ALIASES = {"student", "pupil"}
_TEACHER_ALIASES = {"teacher", "tutor", "instructor"}
_SUPERVISOR_ALIASES = {"supervisor", "observer", "monitor"}
_AI_LABELS = {
    "ai",
    "assistant",
    "bot",
    "chatgpt",
    "chat gpt",
    "virtual teacher",
    "artificial intelligence",
}
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
_GENERIC_SPEAKER_LABEL_RE = re.compile(
    r"^(?:speaker|participant|person|voice|talker|spk)(?:\s*[-_#]?\s*[a-z0-9]+)?$",
    re.IGNORECASE,
)
_NAME_TRAILING_TAG_RE = re.compile(
    r"\s*(?:\([^)]*(?:guest|host|participant|student|learner)[^)]*\)"
    r"|\[[^]]*(?:guest|host|participant|student|learner)[^]]*\])\s*$",
    re.IGNORECASE,
)
_NAME_TOKEN_RE = re.compile(r"^[^\W\d_]+(?:[-'’][^\W\d_]+)*$", re.UNICODE)
_NAME_STOP_WORDS = {
    "a", "an", "and", "are", "because", "but", "can", "class", "could", "did",
    "do", "everybody", "everyone", "friend", "friends", "from", "guys", "here",
    "how", "in", "is", "much", "of", "on", "please", "student", "students", "the",
    "there", "this", "today", "very", "what", "when", "where", "who", "why",
    "with", "would", "your", "you",
}
_SELF_INTRO_RE = re.compile(
    r"\b(?:my name is|you can call me|call me|i am called|i'm called)\s+"
    r"(?P<name>[^,.;:!?\n]{1,80})",
    re.IGNORECASE,
)
_CAPITALIZED_SELF_INTRO_RE = re.compile(
    r"\b(?:I am|I'm)\s+(?P<name>[A-Z][^\W\d_]*(?:[-'’][A-Z][^\W\d_]*)?"
    r"(?:\s+[A-Z][^\W\d_]*(?:[-'’][A-Z][^\W\d_]*)?){0,2})"
    r"(?=\s*[,.;:!?]|\s*$)",
)
_GREETING_NAME_RE = re.compile(
    r"\b(?:hello|hi|hey|welcome|thanks|thank you|good morning|good afternoon|good evening)"
    r"[,:]?\s+(?P<name>[^,.;:!?\n]{1,60})",
    re.IGNORECASE,
)
_INITIAL_ADDRESS_RE = re.compile(
    r"^\s*(?P<name>[A-Z][^\W\d_]*(?:[-'’][A-Z][^\W\d_]*)?"
    r"(?:\s+[A-Z][^\W\d_]*(?:[-'’][A-Z][^\W\d_]*)?){0,2})\s*,",
)


def speaker_roles_for_conversation_type(
    conversation_type: str,
) -> tuple[str, ...]:
    """Return the only valid speaker roles for a conversation type."""
    if conversation_type.strip().casefold() == "ai":
        return AI_CONVERSATION_ROLES
    return HUMAN_TEACHER_CONVERSATION_ROLES


def _expected_roles_for_project(
    project: ProjectData,
    participant_count: int,
) -> tuple[str, ...]:
    roles = speaker_roles_for_conversation_type(
        project.metadata.conversation_type
    )
    if (
        project.metadata.conversation_type.strip().casefold() == "ai"
        and participant_count <= 2
    ):
        # A supervisor is optional in AI conversations.
        return (STUDENT_ROLE, AI_ROLE)
    return roles


def normalize_role_for_conversation_type(
    value: str,
    conversation_type: str,
) -> str | None:
    """Normalize supported role names to the selected role model."""
    normalized = normalize_for_comparison(value)
    if not normalized or normalized in _UNKNOWN_SPEAKER_LABELS:
        return None

    if normalized in _STUDENT_ALIASES:
        return STUDENT_ROLE

    is_ai_conversation = conversation_type.strip().casefold() == "ai"
    if is_ai_conversation:
        if normalized in _AI_LABELS:
            return AI_ROLE
        if normalized in _TEACHER_ALIASES or normalized in _SUPERVISOR_ALIASES:
            # In AI conversations, a human teacher-like label represents the supervisor.
            return SUPERVISOR_ROLE
    else:
        if normalized in _TEACHER_ALIASES or normalized in _SUPERVISOR_ALIASES:
            return TEACHER_ROLE

    return None


def normalize_speaker_identity(
    value: str,
    conversation_type: str,
) -> str | None:
    """Normalize a role while preserving a detected or manually selected name."""
    role = normalize_role_for_conversation_type(value, conversation_type)
    if role is not None:
        return role
    cleaned = " ".join(str(value).split()).strip()
    if normalize_for_comparison(cleaned) in _UNKNOWN_SPEAKER_LABELS:
        return None
    return _clean_human_name_candidate(cleaned)


def _clean_human_name_candidate(value: str) -> str | None:
    """Return a conservative human-name candidate or ``None``."""
    cleaned = _NAME_TRAILING_TAG_RE.sub("", " ".join(value.split())).strip(" \t,.;:!?-–—")
    if not cleaned or len(cleaned) > 80 or any(character.isdigit() for character in cleaned):
        return None
    normalized = normalize_for_comparison(cleaned)
    if (
        normalized in _UNKNOWN_SPEAKER_LABELS
        or normalized in _AI_LABELS
        or normalized in _STUDENT_ALIASES
        or normalized in _TEACHER_ALIASES
        or normalized in _SUPERVISOR_ALIASES
        or _GENERIC_SPEAKER_LABEL_RE.fullmatch(normalized)
    ):
        return None
    tokens = cleaned.split()
    if not 1 <= len(tokens) <= 4:
        return None
    accepted: list[str] = []
    for token in tokens:
        bare = token.strip(" \t,.;:!?()[]{}")
        normalized_token = normalize_for_comparison(bare)
        if normalized_token in _NAME_STOP_WORDS:
            break
        if not bare or _NAME_TOKEN_RE.fullmatch(bare) is None:
            return None
        accepted.append(bare)
    if not accepted:
        return None
    candidate = " ".join(accepted)
    if candidate.islower():
        candidate = " ".join(part.capitalize() for part in candidate.split())
    return candidate


def _human_name_from_label(label: str) -> str | None:
    """Treat a non-generic transcript speaker label as possible human name."""
    return _clean_human_name_candidate(label)


def _names_declared_in_text(text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in (_SELF_INTRO_RE, _CAPITALIZED_SELF_INTRO_RE):
        for match in pattern.finditer(text):
            candidate = _clean_human_name_candidate(match.group("name"))
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _names_used_as_address(text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in (_GREETING_NAME_RE, _INITIAL_ADDRESS_RE):
        for match in pattern.finditer(text):
            candidate = _clean_human_name_candidate(match.group("name"))
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _turn_text_versions(turn: Turn) -> tuple[str, ...]:
    """Return every distinct transcript version available for one turn."""
    versions: list[str] = []
    seen: set[str] = set()
    for value in (
        turn.final_text,
        turn.zoom_text,
        turn.chatgpt_text,
        turn.model_text,
        turn.gold_text,
    ):
        cleaned = " ".join(value.split()).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        versions.append(cleaned)
    return tuple(versions)


def _declared_name_for_turn(turn: Turn) -> str | None:
    """Return one unambiguous self-declared name found in any transcript version."""
    votes: Counter[str] = Counter()
    for text in _turn_text_versions(turn):
        for name in _names_declared_in_text(text):
            votes[name] += 1
    ranked = votes.most_common()
    if not ranked:
        return None
    top_name, top_count = ranked[0]
    second_count = ranked[1][1] if len(ranked) > 1 else 0
    return top_name if top_count > second_count else None


def _speaker_mapping_lookup(project: ProjectData, label: str) -> str | None:
    """Look up a raw label despite harmless whitespace or capitalization changes."""
    direct = project.speaker_mapping.get(label)
    if direct:
        return direct
    cleaned = " ".join(label.split()).strip()
    direct = project.speaker_mapping.get(cleaned)
    if direct:
        return direct
    normalized = normalize_for_comparison(cleaned)
    if not normalized:
        return None
    for stored_label, identity in project.speaker_mapping.items():
        if normalize_for_comparison(stored_label) == normalized:
            return identity
    return None


def resolve_turn_speaker_identity(project: ProjectData, turn: Turn) -> str:
    """Resolve the identity displayed for a review turn.

    A self-declared learner name is allowed to replace ``Student`` or ``Unknown``.
    Fixed facilitator roles are never replaced by a name extracted from dialogue.
    """
    conversation_type = project.metadata.conversation_type
    mapped = _speaker_mapping_lookup(project, turn.speaker_raw)
    raw_label = " ".join(turn.speaker_raw.split()).strip()
    if (
        mapped
        and _usable_speaker_label(raw_label)
        and normalize_for_comparison(mapped) == normalize_for_comparison(raw_label)
    ):
        # A self-mapping records that this exact label came from an uploaded
        # transcript. Do not normalize it into a conversation-role alias.
        return raw_label
    current = normalize_speaker_identity(mapped or turn.speaker, conversation_type)

    if current in (None, STUDENT_ROLE):
        declared_name = _declared_name_for_turn(turn)
        if declared_name:
            return declared_name

    if current is not None:
        return current

    raw_identity = normalize_speaker_identity(turn.speaker_raw, conversation_type)
    return raw_identity or UNKNOWN_ROLE



def _detected_project_learner_name(project: ProjectData) -> str | None:
    """Return one project-wide learner name supported by transcript evidence."""
    conversation_type = project.metadata.conversation_type
    votes: Counter[str] = Counter()

    def add(value: str | None, weight: int) -> None:
        if not value:
            return
        name = _human_name_from_label(value)
        if not name:
            return
        if normalize_role_for_conversation_type(name, conversation_type) is not None:
            return
        votes[name] += weight

    # A self-introduction is the strongest evidence because it explicitly belongs
    # to the speaker of that turn.
    for turn in project.turns:
        add(_declared_name_for_turn(turn), 12)

    # Retain names already established by automatic mapping or a previous review.
    for identity in project.speaker_mapping.values():
        add(identity, 7)
    for turn in project.turns:
        add(turn.speaker, 5)

    # A human-looking learner ID is useful supporting evidence, but weaker than
    # transcript evidence because many projects use numeric or coded learner IDs.
    add(project.metadata.learner_id, 3)

    ranked = votes.most_common()
    if not ranked:
        return None
    top_name, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    return top_name if top_score > second_score else None


def propagate_detected_learner_identity(project: ProjectData) -> int:
    """Apply one detected learner name to every turn from the same learner.

    Version 1.5.7 resolved a self-introduction only on the turn containing the
    name. This function treats that result as project-level identity evidence and
    propagates it to turns already marked Student and to turns sharing the same
    raw speaker label, including a repeated ``Unknown`` placeholder. Fixed AI,
    Teacher, and Supervisor turns are never overwritten.
    """
    learner_name = _detected_project_learner_name(project)
    if not learner_name:
        return 0

    conversation_type = project.metadata.conversation_type
    fixed_roles = {AI_ROLE, TEACHER_ROLE, SUPERVISOR_ROLE}
    learner_raw_keys: set[str] = set()

    for turn in project.turns:
        mapped = _speaker_mapping_lookup(project, turn.speaker_raw)
        raw_label = " ".join(turn.speaker_raw.split()).strip()
        if (
            mapped
            and _usable_speaker_label(raw_label)
            and normalize_for_comparison(mapped) == normalize_for_comparison(raw_label)
        ):
            continue
        current = normalize_speaker_identity(
            mapped or turn.speaker,
            conversation_type,
        )
        declared_name = _declared_name_for_turn(turn)
        if (
            declared_name == learner_name
            or current == STUDENT_ROLE
            or current == learner_name
        ):
            learner_raw_keys.add(normalize_for_comparison(turn.speaker_raw))

    changed = 0
    for turn in project.turns:
        mapped = _speaker_mapping_lookup(project, turn.speaker_raw)
        raw_label = " ".join(turn.speaker_raw.split()).strip()
        if (
            mapped
            and _usable_speaker_label(raw_label)
            and normalize_for_comparison(mapped) == normalize_for_comparison(raw_label)
        ):
            continue
        current = normalize_speaker_identity(
            mapped or turn.speaker,
            conversation_type,
        )
        raw_role = normalize_role_for_conversation_type(
            turn.speaker_raw,
            conversation_type,
        )
        if current in fixed_roles or raw_role in fixed_roles:
            continue

        raw_key = normalize_for_comparison(turn.speaker_raw)
        gold_identity = normalize_speaker_identity(
            turn.gold_speaker,
            conversation_type,
        )
        should_use_name = (
            current == STUDENT_ROLE
            or current == learner_name
            or raw_key in learner_raw_keys
            or gold_identity == STUDENT_ROLE
            or gold_identity == learner_name
        )
        if not should_use_name:
            continue

        if turn.speaker != learner_name:
            turn.speaker = learner_name
            changed += 1

    # Preserve the project-wide decision for usable raw labels. Placeholder
    # labels such as Unknown are intentionally not stored as a global mapping,
    # because the same placeholder could be reused for a different participant.
    for label, identity in list(project.speaker_mapping.items()):
        normalized_label = normalize_for_comparison(label)
        if normalized_label not in learner_raw_keys:
            continue
        if normalized_label in _UNKNOWN_SPEAKER_LABELS:
            continue
        current = normalize_speaker_identity(identity, conversation_type)
        if current in (None, STUDENT_ROLE, learner_name):
            project.speaker_mapping[label] = learner_name

    if changed:
        # Speaker identity is part of the pause-versus-response-gap decision.
        # Keep structured acoustic events synchronized even when the learner
        # name is discovered lazily while the review table is refreshed.
        reassociate_automatic_delay_events(project)
    return changed


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
    detected_delays: Iterable[Mapping[str, object]] | None = None,
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
            turn.quality_target_text = ""
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
            turn.speaker = infer_role_from_name(
                turn.speaker_raw,
                project.metadata.conversation_type,
            )
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

    turns_before_consolidation = len(project.turns)
    merged_turn_count = consolidate_consecutive_speaker_turns(project)
    if status_callback:
        status_callback(
            "Stage 6/7 - Post-transcription speaker consolidation complete: "
            f"merged {merged_turn_count} consecutive same-speaker turn(s); "
            f"{turns_before_consolidation} provisional turn(s) became "
            f"{len(project.turns)} review turn(s)."
        )

    if detected_delays is not None:
        added_delay_events = replace_detected_delay_events(
            project,
            detected_delays,
        )
        if status_callback:
            internal_pauses = sum(
                event.event_type == "silent_pause"
                for event in added_delay_events
            )
            response_gaps = sum(
                event.event_type == "response_gap"
                for event in added_delay_events
            )
            status_callback(
                "Stage 6/7 - Associated audio delay evidence with review turns: "
                f"{internal_pauses} silent pause(s), "
                f"{response_gaps} inter-speaker response gap(s)."
            )
    else:
        # ``None`` means the supplementary detector did not produce a valid
        # update (for example, a transient failure), not a successful empty
        # result. Preserve and reassociate compatible prior evidence.
        reassociate_automatic_delay_events(project)

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


def _same_known_speaker(first: Turn, second: Turn) -> bool:
    """Return whether two adjacent turns have the same resolved speaker."""
    first_speaker = normalize_for_comparison(first.speaker)
    second_speaker = normalize_for_comparison(second.speaker)
    return (
        first_speaker not in _UNKNOWN_SPEAKER_LABELS
        and second_speaker not in _UNKNOWN_SPEAKER_LABELS
        and first_speaker == second_speaker
    )


def _join_turn_text(first: str, second: str) -> str:
    return " ".join(part.strip() for part in (first, second) if part.strip())


def _merged_model_confidence(first: Turn, second: Turn) -> float | None:
    """Combine model confidence in proportion to the represented model words."""
    if first.model_confidence is None:
        return second.model_confidence
    if second.model_confidence is None:
        return first.model_confidence
    first_weight = max(1, len(words(first.model_text)))
    second_weight = max(1, len(words(second.model_text)))
    return (
        first.model_confidence * first_weight
        + second.model_confidence * second_weight
    ) / (first_weight + second_weight)


def _merge_turn_into(first: Turn, second: Turn) -> None:
    """Merge ``second`` into ``first`` without discarding transcript evidence."""
    model_confidence = _merged_model_confidence(first, second)
    starts = [value for value in (first.start, second.start) if value is not None]
    ends = [value for value in (first.end, second.end) if value is not None]
    first.start = min(starts) if starts else None
    first.end = max(ends) if ends else None

    for attribute in (
        "zoom_text",
        "chatgpt_text",
        "model_text",
        "gold_text",
        "final_text",
        "quality_target_text",
        "notes",
    ):
        setattr(
            first,
            attribute,
            _join_turn_text(getattr(first, attribute), getattr(second, attribute)),
        )

    first.model_confidence = model_confidence
    if not first.gold_speaker.strip() and second.gold_speaker.strip():
        first.gold_speaker = second.gold_speaker
    first.hebrew_switch = first.hebrew_switch or second.hebrew_switch
    first.hesitation_or_repetition = (
        first.hesitation_or_repetition or second.hesitation_or_repetition
    )
    first.self_correction = first.self_correction or second.self_correction
    first.unclear_speech = first.unclear_speech or second.unclear_speech
    first.overlapping_speech = first.overlapping_speech or second.overlapping_speech
    first.manual_review = first.manual_review or second.manual_review


def consolidate_consecutive_speaker_turns(project: ProjectData) -> int:
    """Merge adjacent post-transcription turns belonging to one known speaker.

    Speaker comparison uses the resolved identity because imported labels and
    automatic mapping are finalized only after the initial source alignment.
    Consecutive ``Unknown`` turns are deliberately retained: the shared
    placeholder is not evidence that they came from one participant.

    The surviving turns are renumbered, retained structured events are remapped,
    and automatic delay events are reassociated with the consolidated timeline.
    The return value is the number of turns removed.
    """
    original_turn_count = len(project.turns)
    if original_turn_count < 2:
        return 0

    consolidated: list[Turn] = []
    original_ids_by_turn: list[list[int]] = []
    for turn in project.turns:
        if consolidated and _same_known_speaker(consolidated[-1], turn):
            _merge_turn_into(consolidated[-1], turn)
            original_ids_by_turn[-1].append(turn.turn_id)
        else:
            consolidated.append(turn)
            original_ids_by_turn.append([turn.turn_id])

    merged_turn_count = original_turn_count - len(consolidated)
    if not merged_turn_count:
        return 0

    turn_id_mapping: dict[int, int] = {}
    for new_turn_id, (turn, original_ids) in enumerate(
        zip(consolidated, original_ids_by_turn),
        start=1,
    ):
        turn.turn_id = new_turn_id
        turn_id_mapping.update(
            (original_turn_id, new_turn_id)
            for original_turn_id in original_ids
        )

    project.turns = consolidated
    remap_nonautomatic_event_turn_ids(project, turn_id_mapping)
    _mark_overlaps(project.turns)
    reassociate_automatic_delay_events(project)
    return merged_turn_count


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


def speaker_label_for_turn(turn: Turn) -> str:
    """Prefer a usable uploaded label, otherwise return the inferred label."""
    uploaded_label = " ".join(turn.speaker_raw.split()).strip()
    if _usable_speaker_label(uploaded_label):
        return uploaded_label
    inferred_label = " ".join(turn.speaker.split()).strip()
    return inferred_label if _usable_speaker_label(inferred_label) else UNKNOWN_ROLE


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


def _uploaded_speaker_labels_by_turn(project: ProjectData) -> dict[int, str]:
    """Return usable uploaded labels aligned to their review-turn indexes.

    A label already carried by a turn is retained when that same label exists in
    an uploaded source. Missing turn labels are filled from the strongest aligned
    source, preferring the same source ordering used for the turn scaffold.
    """
    imported_sources = {
        source_name: project.source_transcripts.get(source_name, [])
        for source_name in ("zoom", "gold", "chatgpt")
        if project.source_transcripts.get(source_name)
    }
    if not project.turns or not imported_sources:
        return {}

    uploaded_labels = {
        normalize_for_comparison(segment.speaker)
        for segments in imported_sources.values()
        for segment in segments
        if _usable_speaker_label(segment.speaker)
    }
    selected: dict[int, str] = {}
    for index, turn in enumerate(project.turns):
        raw_label = " ".join(turn.speaker_raw.split()).strip()
        if (
            _usable_speaker_label(raw_label)
            and normalize_for_comparison(raw_label) in uploaded_labels
        ):
            selected[index] = raw_label

    best_source, _segments = _best_speaker_source(project)
    source_order = [
        source_name
        for source_name in (best_source, "zoom", "gold", "chatgpt")
        if source_name in imported_sources
    ]
    source_order = list(dict.fromkeys(source_order))
    for source_name in source_order:
        matched_segments = align_source_segments_to_turns(
            project.turns,
            imported_sources[source_name],
        )
        for index, segment in enumerate(matched_segments):
            if (
                index in selected
                or segment is None
                or not _usable_speaker_label(segment.speaker)
            ):
                continue
            selected[index] = " ".join(segment.speaker.split()).strip()

    return selected


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


def infer_role_from_name(
    name: str,
    conversation_type: str = "Human teacher",
) -> str:
    """Infer an explicit role label without guessing ordinary participant names."""
    normalized = normalize_role_for_conversation_type(
        name,
        conversation_type,
    )
    return normalized or name or UNKNOWN_ROLE


def _normalized_project_identity(
    value: str,
    project: ProjectData,
) -> str | None:
    return normalize_speaker_identity(
        value,
        project.metadata.conversation_type,
    )


def _effective_role_for_identity(
    value: str,
    project: ProjectData,
) -> str | None:
    """Treat a retained human name as the learner role for role accounting."""
    role = normalize_role_for_conversation_type(
        value,
        project.metadata.conversation_type,
    )
    if role is not None:
        return role
    return STUDENT_ROLE if _clean_human_name_candidate(value) else None


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
            return STUDENT_ROLE, "matched the project learner ID"

    role = normalize_role_for_conversation_type(
        label,
        project.metadata.conversation_type,
    )
    if role is not None:
        if normalized in _AI_LABELS:
            return role, "recognized AI participant label"
        label_tokens = set(words(normalized))
        aliases = _STUDENT_ALIASES | _TEACHER_ALIASES | _SUPERVISOR_ALIASES
        matched_alias = next(
            (alias for alias in aliases if alias in label_tokens),
            None,
        )
        if matched_alias:
            return role, f"recognized role word '{matched_alias}'"
        return role, "recognized role label"

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
    """Complete role mappings using the selected conversation-type constraints."""
    profiles = _raw_speaker_profiles(project)
    unresolved = [label for label in raw_labels if label not in mapping]
    if not unresolved:
        return

    expected_roles = list(
        _expected_roles_for_project(project, len(raw_labels))
    )
    resolved_roles = {
        effective_role
        for label in raw_labels
        if label in mapping
        for effective_role in [_effective_role_for_identity(mapping[label], project)]
        if effective_role in expected_roles
    }
    missing_roles = [
        role for role in expected_roles if role not in resolved_roles
    ]

    is_ai_conversation = (
        project.metadata.conversation_type.strip().casefold() == "ai"
    )
    facilitator_role = AI_ROLE if is_ai_conversation else TEACHER_ROLE

    if facilitator_role in missing_roles and unresolved:
        facilitator_label = max(
            unresolved,
            key=lambda label: (
                profiles.get(label, {}).get("teacher_score", 0.0),
                -profiles.get(label, {}).get("first_index", float("inf")),
            ),
        )
        mapping[facilitator_label] = facilitator_role
        facilitator_score = profiles.get(
            facilitator_label, {}
        ).get("teacher_score", 0.0)
        reasons[facilitator_label] = (
            f"dialogue-role fallback: most {facilitator_role.lower()}-like prompts/questions"
            if facilitator_score > 0
            else f"dialogue-role fallback: first participant assigned as {facilitator_role}"
        )
        unresolved.remove(facilitator_label)
        missing_roles.remove(facilitator_role)

    if (
        is_ai_conversation
        and SUPERVISOR_ROLE in missing_roles
        and len(raw_labels) >= 3
        and unresolved
    ):
        supervisor_label = min(
            unresolved,
            key=lambda label: (
                profiles.get(label, {}).get("words", 0.0),
                profiles.get(label, {}).get("turns", 0.0),
                profiles.get(label, {}).get("first_index", float("inf")),
            ),
        )
        mapping[supervisor_label] = SUPERVISOR_ROLE
        reasons[supervisor_label] = (
            "dialogue-role fallback: least speaking activity among three participants"
        )
        unresolved.remove(supervisor_label)
        missing_roles.remove(SUPERVISOR_ROLE)

    for label, role in zip(
        sorted(
            unresolved,
            key=lambda item: profiles.get(item, {}).get(
                "first_index", float("inf")
            ),
        ),
        missing_roles,
    ):
        mapping[label] = role
        reasons[label] = (
            "dialogue-role fallback: remaining participant role allowed by "
            f"{project.metadata.conversation_type or 'Human teacher'} conversation type"
        )


def _speaker_name_evidence(
    project: ProjectData,
    raw_labels: list[str],
) -> tuple[dict[str, Counter[str]], dict[tuple[str, str], set[str]]]:
    """Collect transcript-backed human-name evidence for each raw speaker."""
    votes: dict[str, Counter[str]] = {label: Counter() for label in raw_labels}
    evidence: dict[tuple[str, str], set[str]] = {}

    def add(label: str, name: str | None, weight: int, reason: str) -> None:
        if label not in votes or not name:
            return
        votes[label][name] += weight
        evidence.setdefault((label, name), set()).add(reason)

    for label in raw_labels:
        add(label, _human_name_from_label(label), 8, "speaker label contains a human name")

    learner_name = _human_name_from_label(project.metadata.learner_id)
    if learner_name and len(raw_labels) == 1:
        add(raw_labels[0], learner_name, 4, "project learner ID contains a human name")

    for turn_index, turn in enumerate(project.turns):
        label = turn.speaker_raw.strip()
        if label not in votes:
            continue
        addressed_names: list[str] = []
        for text in _turn_text_versions(turn):
            for name in _names_declared_in_text(text):
                add(label, name, 7, "speaker states their name in the transcript")
            for name in _names_used_as_address(text):
                if name not in addressed_names:
                    addressed_names.append(name)

        if addressed_names:
            for next_turn in project.turns[turn_index + 1 : turn_index + 3]:
                next_label = next_turn.speaker_raw.strip()
                if next_label and next_label != label and next_label in votes:
                    for name in addressed_names:
                        add(next_label, name, 3, "previous speaker addresses them by name")
                    break

    for source_name in ("zoom", "gold", "chatgpt"):
        source_segments = project.source_transcripts.get(source_name, [])
        if not source_segments or not project.turns:
            continue
        matched_segments = align_source_segments_to_turns(project.turns, source_segments)
        for turn, segment in zip(project.turns, matched_segments):
            if segment is None:
                continue
            label = turn.speaker_raw.strip()
            add(
                label,
                _human_name_from_label(segment.speaker),
                5,
                f"aligned {source_name} speaker label contains a human name",
            )

    return votes, evidence


def _replace_student_roles_with_detected_names(
    project: ProjectData,
    raw_labels: list[str],
    mapping: dict[str, str],
    reasons: dict[str, str],
) -> None:
    """Replace a learner role with a transcript-backed name when confidence wins."""
    votes, evidence = _speaker_name_evidence(project, raw_labels)
    student_labels = [label for label in raw_labels if mapping.get(label) == STUDENT_ROLE]
    learner_name = _human_name_from_label(project.metadata.learner_id)
    if learner_name and len(student_labels) == 1:
        label = student_labels[0]
        votes[label][learner_name] += 4
        evidence.setdefault((label, learner_name), set()).add(
            "project learner ID contains a human name"
        )

    for label in student_labels:
        ranked = votes[label].most_common()
        if not ranked:
            continue
        name, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0
        if top_score <= second_score:
            continue
        mapping[label] = name
        details = sorted(evidence.get((label, name), set()))
        reasons[label] = "identified learner name: " + "; ".join(details)


def _propagate_aligned_identities_to_source_labels(
    project: ProjectData,
    mapping: dict[str, str],
    reasons: dict[str, str],
) -> None:
    """Give aligned source labels the same final identity as their review turn."""
    if not project.turns:
        return
    for source_name in ("zoom", "gold", "chatgpt"):
        source_segments = project.source_transcripts.get(source_name, [])
        if not source_segments:
            continue
        matched_segments = align_source_segments_to_turns(project.turns, source_segments)
        for turn, segment in zip(project.turns, matched_segments):
            if segment is None or not _usable_speaker_label(segment.speaker):
                continue
            source_label = segment.speaker.strip()
            raw_label = turn.speaker_raw.strip()
            identity = mapping.get(raw_label)
            if not identity or identity == UNKNOWN_ROLE or source_label in mapping:
                continue
            mapping[source_label] = identity
            reasons[source_label] = (
                f"aligned {source_name} label inherited the review-turn identity"
            )


def automatically_map_speakers(
    project: ProjectData,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Apply uploaded speaker labels, then infer labels only where they are absent.

    Any usable speaker label aligned from Zoom, Gold, or ChatGPT is preserved
    verbatim. Saved mappings, role evidence, dialogue structure, and detected names
    are fallback evidence only for turns without an uploaded label.
    """

    uploaded_labels_by_turn = _uploaded_speaker_labels_by_turn(project)
    for turn_index, uploaded_label in uploaded_labels_by_turn.items():
        turn = project.turns[turn_index]
        turn.speaker_raw = uploaded_label
        turn.speaker = uploaded_label

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
        saved_identity = _normalized_project_identity(
            project.speaker_mapping.get(label, ""),
            project,
        )
        if saved_identity is not None:
            mapping[label] = saved_identity
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
    valid_roles = speaker_roles_for_conversation_type(
        project.metadata.conversation_type
    )
    resolved_roles = {
        effective_role
        for label in raw_labels
        if label in mapping
        for effective_role in [_effective_role_for_identity(mapping[label], project)]
        if effective_role in valid_roles
    }
    expected_roles = _expected_roles_for_project(
        project,
        len(raw_labels),
    )
    missing_roles = [
        role for role in expected_roles if role not in resolved_roles
    ]
    if len(unresolved_raw) == 1 and len(missing_roles) == 1:
        label = unresolved_raw[0]
        mapping[label] = missing_roles[0]
        reasons[label] = "only remaining participant role"

    _apply_dialogue_role_fallbacks(project, raw_labels, mapping, reasons)
    _replace_student_roles_with_detected_names(
        project,
        raw_labels,
        mapping,
        reasons,
    )
    _propagate_aligned_identities_to_source_labels(
        project,
        mapping,
        reasons,
    )

    # Source labels have final authority. A self-mapping lets later project loads
    # and Review Turns edits distinguish preserved labels from inferred roles.
    preserved_uploaded_labels = set(uploaded_labels_by_turn.values())
    for label in preserved_uploaded_labels:
        mapping[label] = label
        reasons[label] = "preserved verbatim from an uploaded transcript"

    complete_mapping = {
        label: mapping.get(label, "Unknown")
        for label in usable_labels
    }
    apply_speaker_mapping(project, complete_mapping)
    learner_name = _detected_project_learner_name(project)
    learner_turn_count = (
        sum(turn.speaker == learner_name for turn in project.turns)
        if learner_name
        else 0
    )

    if status_callback:
        if learner_name and learner_turn_count:
            status_callback(
                "Stage 6/7 - Learner identity propagation: "
                f"{learner_name!r} applied to {learner_turn_count} learner turn(s)."
            )
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
                        segment.speaker,
                        infer_role_from_name(
                            segment.speaker,
                            project.metadata.conversation_type,
                        ),
                    )
                else:
                    turn.gold_speaker = ""

    for turn in project.turns:
        if not turn.final_text.strip():
            turn.final_text = choose_initial_text(turn)
        ensure_quality_target_text(turn)


_VERBATIM_DISFLUENCY_KINDS = frozenset(
    {"filler", "partial_word", "repetition", "self_correction"}
)
_MIN_FLUENT_SKELETON_SIMILARITY = 0.65


def _verbatim_disfluency_events(text: str) -> list[DetectedSpeechEvent]:
    """Return explicit disfluency evidence that must survive source voting."""

    return [
        event
        for event in detected_speech_events(text)
        if event.kind in _VERBATIM_DISFLUENCY_KINDS
    ]


def _fluent_skeleton_for_comparison(text: str) -> str:
    """Hide explicit disfluency spans for comparison without rewriting output.

    This is used only to decide whether a more verbatim source says the same
    underlying thing as a smoother majority source. The selected source is
    always returned byte-for-byte; no cleaned or synthetic text is produced.
    """

    spans = sorted(
        (event.char_start, event.char_end)
        for event in _verbatim_disfluency_events(text)
    )
    if not spans:
        return text

    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))

    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return " ".join("".join(pieces).split())


def choose_initial_text(turn: Turn) -> str:
    """Select the source wording with the strongest independent support.

    Candidate slots are kept separate even when two sources contain identical
    text. The previous implementation compared string values and accidentally
    removed duplicate wording from the vote, allowing one disagreeing source to
    beat two matching sources.
    """

    candidates = [
        turn.model_text,
        turn.chatgpt_text,
        turn.zoom_text,
    ]
    present = [candidate for candidate in candidates if candidate.strip()]
    if not present:
        return ""
    if len(present) == 1:
        return present[0]

    from .alignment import text_similarity

    scored: list[tuple[float, int, str]] = []
    for index, candidate in enumerate(present):
        similarities = [
            text_similarity(candidate, other)
            for other_index, other in enumerate(present)
            if other_index != index
        ]
        average_similarity = sum(similarities) / len(similarities)
        exact_votes = sum(
            normalize_for_comparison(candidate)
            == normalize_for_comparison(other)
            for other in present
        )
        scored.append((average_similarity, exact_votes, candidate))

    # Average semantic agreement is primary. Exact source votes break ties and
    # make the majority behavior explicit without synthesizing new wording.
    selected = max(scored, key=lambda item: (item[0], item[1]))[2]

    # A majority of consumer-oriented transcript sources can agree only because
    # they all smooth away the same spoken hesitation. Prefer a more verbatim
    # source when removing its explicitly located disfluency spans leaves text
    # that is still semantically close to the majority candidate. Returning the
    # original source preserves punctuation and every restart; this block never
    # manufactures or cleans transcript text.
    selected_event_count = len(_verbatim_disfluency_events(selected))
    selected_skeleton = _fluent_skeleton_for_comparison(selected)
    verbatim_candidates: list[tuple[int, float, int, str]] = []
    for index, candidate in enumerate(present):
        event_count = len(_verbatim_disfluency_events(candidate))
        if event_count <= selected_event_count:
            continue
        skeleton_similarity = text_similarity(
            _fluent_skeleton_for_comparison(candidate),
            selected_skeleton,
        )
        if skeleton_similarity >= _MIN_FLUENT_SKELETON_SIMILARITY:
            verbatim_candidates.append(
                (event_count, skeleton_similarity, -index, candidate)
            )

    if verbatim_candidates:
        return max(verbatim_candidates)[3]
    return selected


def ensure_quality_target_text(turn: Turn) -> str:
    """Preserve the unedited transcript whose quality is being predicted.

    ``final_text`` is editable. Training directly against it after manual review
    would leak the Gold correction into the target and make nearly every reviewed
    turn look acceptable. New turns store the initial displayed candidate once.
    A new turn without a stored target uses the strongest current source candidate.
    """

    existing = turn.quality_target_text.strip()
    if existing:
        return existing

    selected = (
        choose_initial_text(turn).strip()
        or turn.model_text.strip()
        or turn.final_text.strip()
    )

    turn.quality_target_text = selected
    return selected


def _mark_overlaps(turns: list[Turn]) -> None:
    for turn in turns:
        turn.overlapping_speech = False

    timed_turns = sorted(
        (turn for turn in turns if turn.start is not None and turn.end is not None),
        key=lambda turn: (float(turn.start), float(turn.end)),
    )
    active: list[Turn] = []
    for turn in timed_turns:
        start = float(turn.start)
        active = [other for other in active if float(other.end) - start > 0.15]
        for other in active:
            overlap = min(float(turn.end), float(other.end)) - start
            if overlap > 0.15 and turn.speaker_raw != other.speaker_raw:
                turn.overlapping_speech = True
                other.overlapping_speech = True
        active.append(turn)


def analyze_turns(
    project: ProjectData,
    predictor: object | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> None:
    audio_path = Path(project.metadata.audio_file) if project.metadata.audio_file else None
    analyze_audio = bool(
        audio_path
        and audio_path.suffix.casefold() == ".wav"
        and audio_path.exists()
    )
    turn_count = len(project.turns)
    audio_signals: list[dict[str, float | None] | None] = [None] * turn_count
    if analyze_audio:
        try:
            audio_signals = analyze_wav_intervals(
                audio_path,
                ((turn.start, turn.end) for turn in project.turns),
            )
        except AudioFeatureError:
            pass
    for index, turn in enumerate(project.turns, start=1):
        if status_callback:
            status_callback(f"Analyzing turn {index} of {turn_count}...")
        ensure_quality_target_text(turn)
        signal = audio_signals[index - 1]
        if signal is not None:
            turn.volume_dbfs = signal["volume_dbfs"]
            turn.noise_snr_db = signal["noise_snr_db"]
        elif analyze_audio:
            turn.volume_dbfs = None
            turn.noise_snr_db = None
        update_turn_quality(turn, predictor)

    # Source voting can select grammar-normalized wording before review. Compare
    # the literal final text with every non-Gold source after detected turn flags
    # have been refreshed. This creates review evidence only; it never rewrites
    # the transcript or asserts that either wording is grammatical.
    refresh_grammar_preservation_events(project)
    turns_with_detected_pauses = {
        event.turn_id
        for event in project.speech_events
        if event.turn_id is not None
        and event.event_type == "silent_pause"
        and not event.reviewed
    }
    turns_with_unreviewed_grammar = {
        event.turn_id
        for event in project.speech_events
        if event.turn_id is not None
        and event.event_type == GRAMMAR_EVENT_TYPE
        and event.source == GRAMMAR_GUARD_SOURCE
        and not event.reviewed
    }
    turns_by_id = {turn.turn_id: turn for turn in project.turns}
    for turn_id in turns_with_detected_pauses:
        turn = turns_by_id.get(turn_id)
        if turn is not None:
            # Text-based feature extraction cannot see a silent pause. Retain
            # the detector's initial suggestion until a reviewer confirms the
            # event; energy alone cannot determine whether silence is hesitation.
            turn.hesitation_or_repetition = True
    for turn_id in turns_with_unreviewed_grammar:
        turn = turns_by_id.get(turn_id)
        if turn is not None and is_likely_learner_turn(project, turn):
            # A source disagreement is not a grammar diagnosis. It is enough to
            # prevent consensus scoring from silently accepting normalized
            # learner wording before a person checks the audio.
            turn.manual_review = True
    project.metrics = evaluate_turns(project.turns)
    project.metrics["source_comparison"] = per_source_metrics(project.turns)
    grammar_events = [
        event
        for event in project.speech_events
        if event.event_type == GRAMMAR_EVENT_TYPE
        and event.source == GRAMMAR_GUARD_SOURCE
    ]
    project.metrics["grammar_preservation_candidates"] = len(grammar_events)
    project.metrics["grammar_preservation_candidates_unreviewed"] = sum(
        not event.reviewed for event in grammar_events
    )


def apply_speaker_mapping(project: ProjectData, mapping: dict[str, str]) -> None:
    cleaned_mapping = {
        " ".join(label.split()).strip(): identity
        for label, identity in mapping.items()
        if " ".join(label.split()).strip()
    }
    project.speaker_mapping.update(cleaned_mapping)
    for turn in project.turns:
        turn.speaker = resolve_turn_speaker_identity(project, turn)

    propagate_detected_learner_identity(project)

    for turn in project.turns:
        if turn.gold_speaker:
            turn.gold_speaker = (
                _speaker_mapping_lookup(project, turn.gold_speaker)
                or turn.gold_speaker
            )
    # Reapply Gold Standard mapping from the original source labels when available.
    gold_segments = project.source_transcripts.get("gold", [])
    if gold_segments and project.turns:
        matched_segments = align_source_segments_to_turns(project.turns, gold_segments)
        for turn, segment in zip(project.turns, matched_segments):
            if segment:
                turn.gold_speaker = (
                    _speaker_mapping_lookup(project, segment.speaker)
                    or infer_role_from_name(
                        segment.speaker,
                        project.metadata.conversation_type,
                    )
                )

    # A corrected speaker boundary can change a same-speaker pause into an
    # inter-speaker response gap (or the reverse). Keep the acoustic event view
    # synchronized whenever speaker mapping is applied outside the turn editor.
    reassociate_automatic_delay_events(project)


def _quality_label_from_wer(error: float) -> int:
    if error <= 0.10:
        return 0
    if error <= 0.30:
        return 1
    return 2


def _quality_example_id(
    project: ProjectData,
    turn: Turn,
    target_text: str,
) -> str:
    """Return a stable ID that prevents duplicate clicks, not duplicate features."""
    payload = {
        "learner_id": project.metadata.learner_id,
        "session_number": project.metadata.session_number,
        "project_title": project.metadata.title,
        "turn_id": turn.turn_id,
        "start": turn.start,
        "end": turn.end,
        "speaker_raw": turn.speaker_raw,
        "zoom_text": turn.zoom_text,
        "chatgpt_text": turn.chatgpt_text,
        "model_text": turn.model_text,
        "quality_target_text": target_text,
        "gold_text": turn.gold_text,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _training_records_from_project(
    project: ProjectData,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for turn in project.turns:
        target_text = ensure_quality_target_text(turn)
        if not turn.gold_text.strip() or not target_text:
            continue
        error = word_error_rate(turn.gold_text, target_text)
        features = extract_features(turn)
        records.append(
            {
                "schema_version": QUALITY_TRAINING_SCHEMA_VERSION,
                "label_target": QUALITY_LABEL_TARGET,
                "feature_names": list(FEATURE_NAMES),
                "features": [features[name] for name in FEATURE_NAMES],
                "label": _quality_label_from_wer(error),
                "target_wer": error,
                "example_id": _quality_example_id(
                    project,
                    turn,
                    target_text,
                ),
            }
        )
    return records


def training_examples_from_project(
    project: ProjectData,
) -> tuple[list[list[float]], list[int]]:
    """Build labels for the transcript candidate actually shown for review."""
    records = _training_records_from_project(project)
    rows = [
        [float(value) for value in record["features"]]
        for record in records
    ]
    labels = [int(record["label"]) for record in records]
    return rows, labels


def _valid_quality_training_record(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("schema_version") != QUALITY_TRAINING_SCHEMA_VERSION:
        return False
    if item.get("label_target") != QUALITY_LABEL_TARGET:
        return False
    example_id = item.get("example_id")
    if not isinstance(example_id, str) or len(example_id) != 64:
        return False
    features = item.get("features")
    if not isinstance(features, list) or len(features) != len(FEATURE_NAMES):
        return False
    try:
        label = int(item["label"])
        numeric_features = [float(value) for value in features]
    except (KeyError, TypeError, ValueError):
        return False
    return label in (0, 1, 2) and all(math.isfinite(value) for value in numeric_features)


def append_training_examples(project: ProjectData, path: str | Path) -> int:
    new_records = _training_records_from_project(project)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, object]] = []
    if target.exists():
        raw_existing = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(raw_existing, list) or not all(
            _valid_quality_training_record(item) for item in raw_existing
        ):
            raise ValueError(
                "The quality training file does not match the current schema."
            )
        existing = list(raw_existing)

    existing_ids = {
        str(item["example_id"])
        for item in existing
    }
    added = 0
    for record in new_records:
        example_id = str(record["example_id"])
        if example_id in existing_ids:
            continue
        existing.append(record)
        existing_ids.add(example_id)
        added += 1

    payload = json.dumps(existing, indent=2)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{target.stem}_",
        suffix=".json.tmp",
        dir=target.parent,
        delete=False,
    ) as temporary_handle:
        temporary_handle.write(payload)
        temporary = Path(temporary_handle.name)
    try:
        temporary.replace(target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return added


def _quality_model_metadata() -> dict[str, object]:
    return {
        "training_schema_version": QUALITY_TRAINING_SCHEMA_VERSION,
        "label_target": QUALITY_LABEL_TARGET,
        "feature_names": list(FEATURE_NAMES),
    }


def train_quality_model(
    training_path: str | Path,
    model_path: str | Path,
) -> tuple[object, list[dict[str, float | str | int]]]:
    raw = json.loads(Path(training_path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("The quality training file must contain a JSON list.")
    data = [item for item in raw if _valid_quality_training_record(item)]
    if not data:
        raise ValueError(
            "No current quality-training records were found. "
            "Use Train and Compare ML Models to collect current Gold labels."
        )
    rows = [list(map(float, item["features"])) for item in data]
    labels = [int(item["label"]) for item in data]
    model, comparison = train_and_compare(rows, labels)
    save_model(model, model_path, metadata=_quality_model_metadata())
    return model, comparison


def load_quality_model_if_available(path: str | Path) -> object | None:
    target = Path(path)
    if not target.exists():
        return None
    return load_model(target, expected_metadata=_quality_model_metadata())
