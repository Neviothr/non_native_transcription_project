"""Association and rendering helpers for structured speech events.

Acoustic detections remain separate from literal transcript text.  This module
maps absolute-time detections to review turns and creates a deterministic
display/export view with pause markers without mutating ``Turn.final_text``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .models import ProjectData, SpeechEvent, Turn
from .text_utils import normalize_for_comparison


AUTOMATIC_DELAY_SOURCE = "audio_energy_vad"
_DISPLAY_TOKEN_RE = re.compile(r"\S+")
_UNKNOWN_SPEAKERS = {"", "unknown", "unmapped", "speaker", "none"}


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _known_speaker(turn: Turn | None) -> str:
    if turn is None:
        return ""
    normalized = normalize_for_comparison(turn.speaker or turn.speaker_raw)
    return "" if normalized in _UNKNOWN_SPEAKERS else normalized


def _timed_turns(turns: Iterable[Turn]) -> list[Turn]:
    return sorted(
        (
            turn
            for turn in turns
            if turn.start is not None
            and turn.end is not None
            and turn.end >= turn.start
        ),
        key=lambda turn: (float(turn.start), float(turn.end), turn.turn_id),
    )


def _event_owner(
    turns: list[Turn],
    start: float,
    end: float,
) -> tuple[Turn | None, str, dict[str, Any]]:
    """Return an owning turn, event type, and association details."""

    midpoint = (start + end) / 2.0
    for previous, following in zip(turns, turns[1:]):
        if previous.end is None or following.start is None:
            continue
        previous_end = float(previous.end)
        following_start = float(following.start)
        if previous_end > following_start:
            # Overlapping turns do not define a chronological response
            # boundary. Let the containing-turn logic below own the pause.
            continue
        boundary = (previous_end + following_start) / 2.0
        if not start <= boundary <= end:
            continue
        previous_speaker = _known_speaker(previous)
        following_speaker = _known_speaker(following)
        if (
            previous_speaker
            and following_speaker
            and previous_speaker != following_speaker
        ):
            return None, "response_gap", {
                "association": "speaker_boundary",
                "previous_turn_id": previous.turn_id,
                "following_turn_id": following.turn_id,
            }

    containing = [
        turn
        for turn in turns
        if turn.start is not None
        and turn.end is not None
        and float(turn.start) <= midpoint <= float(turn.end)
    ]
    if containing:
        owner = min(
            containing,
            key=lambda turn: (
                float(turn.end) - float(turn.start),  # type: ignore[arg-type]
                abs(
                    midpoint
                    - (float(turn.start) + float(turn.end)) / 2.0  # type: ignore[arg-type]
                ),
            ),
        )
        return owner, "silent_pause", {"association": "inside_turn"}

    previous = next(
        (
            turn
            for turn in reversed(turns)
            if turn.end is not None and float(turn.end) <= midpoint
        ),
        None,
    )
    following = next(
        (
            turn
            for turn in turns
            if turn.start is not None and float(turn.start) >= midpoint
        ),
        None,
    )
    previous_speaker = _known_speaker(previous)
    following_speaker = _known_speaker(following)
    different_known_speakers = bool(
        previous_speaker
        and following_speaker
        and previous_speaker != following_speaker
    )
    details: dict[str, Any] = {
        "association": "between_turns",
        "previous_turn_id": previous.turn_id if previous else None,
        "following_turn_id": following.turn_id if following else None,
    }
    if different_known_speakers or previous is None:
        return None, "response_gap", details

    # Without evidence of a speaker change, preserve the gap as a pause after
    # the preceding utterance. This is especially important when Whisper turns
    # all carry the placeholder speaker "Unknown".
    return previous, "silent_pause", details


def _estimated_token_index(turn: Turn, event_midpoint: float) -> int:
    text = turn.final_text or turn.model_text
    token_count = len(_DISPLAY_TOKEN_RE.findall(text))
    if token_count <= 0:
        return 0
    if turn.start is None or turn.end is None or turn.end <= turn.start:
        return token_count
    fraction = (event_midpoint - turn.start) / (turn.end - turn.start)
    return min(token_count, max(0, round(fraction * token_count)))


def replace_detected_delay_events(
    project: ProjectData,
    detections: Iterable[Mapping[str, object]],
) -> list[SpeechEvent]:
    """Replace prior automatic pause detections and associate new ones.

    Manual and text-derived events are retained. Invalid or zero-duration raw
    detections are ignored rather than corrupting the project timeline.
    """

    retained = [
        event
        for event in project.speech_events
        if event.source != AUTOMATIC_DELAY_SOURCE
    ]
    next_event_id = max((event.event_id for event in retained), default=0) + 1
    turns = _timed_turns(project.turns)
    added: list[SpeechEvent] = []

    for raw in detections:
        if not isinstance(raw, Mapping):
            continue
        start = _finite_float(raw.get("start_seconds"))
        end = _finite_float(raw.get("end_seconds"))
        if start is None or end is None or end <= start:
            continue

        owner, event_type, association_details = _event_owner(turns, start, end)
        midpoint = (start + end) / 2.0
        token_index = (
            _estimated_token_index(owner, midpoint)
            if owner is not None and event_type == "silent_pause"
            else None
        )
        details = {
            "detector": "frame_energy_vad",
            "interval_index": raw.get("interval_index"),
            "interval_start_seconds": raw.get("interval_start_seconds"),
            "interval_end_seconds": raw.get("interval_end_seconds"),
            "loudest_frame_dbfs": raw.get("loudest_frame_dbfs"),
            "audio_source_path": raw.get("audio_source_path"),
            "audio_source_size_bytes": raw.get("audio_source_size_bytes"),
            "audio_source_modified_time_ns": raw.get(
                "audio_source_modified_time_ns"
            ),
            "token_position_estimated": token_index is not None,
            **association_details,
        }
        event = SpeechEvent(
            event_id=next_event_id,
            turn_id=owner.turn_id if owner is not None else None,
            event_type=event_type,
            start=start,
            end=end,
            source=AUTOMATIC_DELAY_SOURCE,
            token_start=token_index,
            token_end=token_index,
            reviewed=False,
            details=details,
        )
        added.append(event)
        next_event_id += 1

    project.speech_events = sorted(
        [*retained, *added],
        key=lambda event: (
            float("inf") if event.start is None else event.start,
            event.event_id,
        ),
    )
    return added


def events_for_turn(
    project: ProjectData,
    turn_id: int,
    *,
    event_type: str | None = None,
) -> list[SpeechEvent]:
    return sorted(
        (
            event
            for event in project.speech_events
            if event.turn_id == turn_id
            and (event_type is None or event.event_type == event_type)
        ),
        key=lambda event: (
            float("inf") if event.start is None else event.start,
            event.event_id,
        ),
    )


def automatic_delay_events_match_audio(
    project: ProjectData,
    *,
    source_path: object,
    source_size_bytes: object,
    source_modified_time_ns: object,
) -> bool:
    """Return whether retained automatic events came from this exact source."""

    retained = [
        event
        for event in project.speech_events
        if event.source == AUTOMATIC_DELAY_SOURCE
    ]
    if not retained:
        return True
    try:
        expected_path = str(Path(str(source_path)).expanduser().resolve()).casefold()
        expected_size = int(source_size_bytes)
        expected_modified = int(source_modified_time_ns)
    except (TypeError, ValueError, OSError):
        return False
    for event in retained:
        raw_path = event.details.get("audio_source_path")
        try:
            event_path = str(Path(str(raw_path)).expanduser().resolve()).casefold()
            event_size = int(event.details["audio_source_size_bytes"])
            event_modified = int(event.details["audio_source_modified_time_ns"])
        except (KeyError, TypeError, ValueError, OSError):
            return False
        if (
            event_path != expected_path
            or event_size != expected_size
            or event_modified != expected_modified
        ):
            return False
    return True


def remap_nonautomatic_event_turn_ids(
    project: ProjectData,
    turn_id_mapping: Mapping[int, int],
) -> None:
    """Keep retained manual/text events attached when turn IDs are rewritten."""

    for event in project.speech_events:
        if event.source == AUTOMATIC_DELAY_SOURCE:
            continue
        if event.turn_id in turn_id_mapping:
            event.turn_id = turn_id_mapping[event.turn_id]
        for detail_key in ("previous_turn_id", "following_turn_id"):
            detail_turn_id = event.details.get(detail_key)
            if detail_turn_id in turn_id_mapping:
                event.details[detail_key] = turn_id_mapping[detail_turn_id]


def reassociate_automatic_delay_events(project: ProjectData) -> set[int]:
    """Refresh event ownership and return IDs whose review was invalidated."""

    turns = _timed_turns(project.turns)
    newly_flagged_pause_owners: set[int] = set()
    invalidated_event_ids: set[int] = set()
    for event in project.speech_events:
        if (
            event.source != AUTOMATIC_DELAY_SOURCE
            or event.start is None
            or event.end is None
            or event.end <= event.start
        ):
            continue
        previous_type = event.event_type
        previous_turn_id = event.turn_id
        previous_review_target_id = review_target_turn_id(event)
        owner, event_type, association_details = _event_owner(
            turns,
            event.start,
            event.end,
        )
        event.turn_id = owner.turn_id if owner is not None else None
        event.event_type = event_type
        event.token_start = (
            _estimated_token_index(owner, (event.start + event.end) / 2.0)
            if owner is not None and event_type == "silent_pause"
            else None
        )
        event.token_end = event.token_start
        for stale_key in (
            "association",
            "previous_turn_id",
            "following_turn_id",
        ):
            event.details.pop(stale_key, None)
        event.details.update(association_details)
        event.details["token_position_estimated"] = event.token_start is not None
        changed_target = (
            previous_type != event.event_type
            or previous_turn_id != event.turn_id
            or previous_review_target_id != review_target_turn_id(event)
        )
        if changed_target:
            # Confirmation applies to the old semantic event location. A
            # speaker/boundary correction creates a new review decision.
            event.reviewed = False
            invalidated_event_ids.add(event.event_id)
            if event.event_type == "silent_pause" and event.turn_id is not None:
                newly_flagged_pause_owners.add(event.turn_id)

    turns_by_id = {turn.turn_id: turn for turn in project.turns}
    for turn_id in newly_flagged_pause_owners:
        owner = turns_by_id.get(turn_id)
        if owner is not None:
            # A newly owned silent pause remains useful hesitation evidence,
            # but delay evidence alone does not alter the general review queue.
            owner.hesitation_or_repetition = True
    return invalidated_event_ids


def pause_marker(event: SpeechEvent) -> str:
    return f"[pause {event.duration():.2f}s]"


def response_gap_marker(event: SpeechEvent) -> str:
    return f"[response gap {event.duration():.2f}s]"


def review_target_turn_id(event: SpeechEvent) -> int | None:
    """Return the turn where a reviewer can inspect one delay event."""

    if event.event_type == "silent_pause":
        return event.turn_id
    if event.event_type != "response_gap":
        return None
    following_turn_id = event.details.get("following_turn_id")
    return following_turn_id if isinstance(following_turn_id, int) else None


def response_gaps_before_turn(
    project: ProjectData,
    turn_id: int,
) -> list[SpeechEvent]:
    """Return response gaps rendered/reviewed immediately before a turn."""

    return sorted(
        (
            event
            for event in project.speech_events
            if event.event_type == "response_gap"
            and review_target_turn_id(event) == turn_id
        ),
        key=lambda event: (
            float("inf") if event.start is None else event.start,
            event.event_id,
        ),
    )


def reviewable_delay_events_for_turn(
    project: ProjectData,
    turn_id: int,
) -> list[SpeechEvent]:
    """Return internal pauses plus response gaps assigned to one review row."""

    return sorted(
        [
            *events_for_turn(
                project,
                turn_id,
                event_type="silent_pause",
            ),
            *response_gaps_before_turn(project, turn_id),
        ],
        key=lambda event: (
            float("inf") if event.start is None else event.start,
            event.event_id,
        ),
    )


def render_text_with_speech_delays(
    text: str,
    events: Iterable[SpeechEvent],
) -> str:
    """Insert deterministic pause markers without changing the source text."""

    pause_events = [
        event
        for event in events
        if event.event_type == "silent_pause" and event.duration() > 0.0
    ]
    if not pause_events:
        return text

    token_matches = list(_DISPLAY_TOKEN_RE.finditer(text))
    insertions: dict[int, list[str]] = {}
    for event in sorted(
        pause_events,
        key=lambda item: (
            float("inf") if item.start is None else item.start,
            item.event_id,
        ),
    ):
        token_index = event.token_start
        if token_index is None:
            token_index = len(token_matches)
        token_index = max(0, min(len(token_matches), token_index))
        position = (
            len(text)
            if token_index == len(token_matches)
            else token_matches[token_index].start()
        )
        insertions.setdefault(position, []).append(pause_marker(event))

    rendered = text
    for position in sorted(insertions, reverse=True):
        marker_text = " ".join(insertions[position])
        before = rendered[:position].rstrip()
        after = rendered[position:].lstrip()
        rendered = " ".join(part for part in (before, marker_text, after) if part)
    return rendered


def render_turn_with_speech_delays(project: ProjectData, turn: Turn) -> str:
    rendered = render_text_with_speech_delays(
        turn.final_text,
        events_for_turn(project, turn.turn_id, event_type="silent_pause"),
    )
    response_markers = " ".join(
        response_gap_marker(event)
        for event in response_gaps_before_turn(project, turn.turn_id)
        if event.duration() > 0.0
    )
    return " ".join(part for part in (response_markers, rendered) if part)
