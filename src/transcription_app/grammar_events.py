"""Persist conservative grammar-preservation evidence for human review.

The guard compares the literal final transcript with every non-Gold source.
It never decides which form is grammatical and never rewrites transcript text.
Small, grammar-sensitive differences are stored as structured ``SpeechEvent``
records so a reviewer can resolve them against the audio.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from typing import Iterable

from .grammar_preservation import GrammarEditEvidence, review_grammar_edit_preservation
from .models import ProjectData, SpeechEvent, Turn
from .text_utils import detected_speech_events, normalize_for_comparison


GRAMMAR_EVENT_TYPE = "grammar_sensitive_difference"
GRAMMAR_GUARD_SOURCE = "grammar_preservation_guard"

_SOURCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("initial_baseline", "quality_target_text"),
    ("additional_model", "model_text"),
    ("chatgpt", "chatgpt_text"),
    ("zoom", "zoom_text"),
)
_NON_LEARNER_ROLES = {
    "ai",
    "assistant",
    "bot",
    "chat gpt",
    "chatgpt",
    "instructor",
    "monitor",
    "observer",
    "supervisor",
    "teacher",
    "tutor",
}
_UNKNOWN_ROLES = {"", "unknown", "unmapped", "none", "speaker"}
_LEARNER_ROLES = {"learner", "pupil", "student"}


def _source_texts(turn: Turn) -> Iterable[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for source_name, attribute in _SOURCE_FIELDS:
        text = getattr(turn, attribute)
        if not text.strip():
            continue
        key = (source_name, text)
        if key in seen:
            continue
        seen.add(key)
        yield source_name, text


def _turn_is_ambiguous(turn: Turn) -> bool:
    """Return whether automatic grammar evidence would be misleading."""

    explicitly_ambiguous = bool(
        turn.hebrew_switch
        or turn.unclear_speech
        or turn.overlapping_speech
        or turn.self_correction
    )
    if explicitly_ambiguous:
        return True
    return any(
        event.kind in {"partial_word", "repetition"}
        for text in (
            turn.final_text,
            turn.quality_target_text,
            turn.model_text,
            turn.chatgpt_text,
            turn.zoom_text,
        )
        if text
        for event in detected_speech_events(text)
    )


def _contains_role(value: str, roles: set[str]) -> bool:
    tokens = set(value.replace("_", " ").split())
    return value in roles or bool(tokens & roles) or any(
        value.startswith(role) and value[len(role):].isdigit()
        for role in roles
    )


def _contains_self_introduction(text: str, identity: str) -> bool:
    normalized = normalize_for_comparison(text)
    return any(
        phrase in normalized
        for phrase in (
            f"my name is {identity}",
            f"call me {identity}",
            f"i am {identity}",
            f"i'm {identity}",
        )
    )


def is_likely_learner_turn(project: ProjectData, turn: Turn) -> bool:
    """Return whether grammar evidence should enforce the review queue.

    Named identities require project evidence. Uploaded teacher names are
    intentionally preserved verbatim, so an arbitrary human-looking label is
    not enough to classify a speaker as the learner. Unknown speakers remain
    outside this grammar-specific hard gate; their unresolved identity already
    supplies an independent review reason.
    """

    identity = normalize_for_comparison(turn.speaker or turn.speaker_raw)
    if (
        identity in _UNKNOWN_ROLES
        or identity.startswith("speaker ")
        or _contains_role(identity, _NON_LEARNER_ROLES)
    ):
        return False
    if _contains_role(identity, _LEARNER_ROLES):
        return True

    learner_id = normalize_for_comparison(project.metadata.learner_id)
    if learner_id and (
        identity == learner_id
        or set(learner_id.split()).issubset(set(identity.split()))
    ):
        return True

    for candidate in project.turns:
        if normalize_for_comparison(candidate.speaker) != identity:
            continue
        if _contains_role(
            normalize_for_comparison(candidate.speaker_raw),
            _LEARNER_ROLES,
        ) or _contains_role(
            normalize_for_comparison(candidate.gold_speaker),
            _LEARNER_ROLES,
        ):
            return True
        if any(
            _contains_self_introduction(text, identity)
            for text in (
                candidate.final_text,
                candidate.model_text,
                candidate.zoom_text,
                candidate.chatgpt_text,
            )
            if text
        ):
            return True

    for raw_label, mapped_identity in project.speaker_mapping.items():
        if normalize_for_comparison(mapped_identity) != identity:
            continue
        raw = normalize_for_comparison(raw_label)
        if raw == identity:
            continue
        if _contains_role(raw, _LEARNER_ROLES):
            return True
    return False


def _evidence_key(evidence: GrammarEditEvidence) -> tuple[object, ...]:
    return (
        evidence.operation,
        evidence.pattern,
        normalize_for_comparison(evidence.baseline_fragment),
        normalize_for_comparison(evidence.final_fragment),
        evidence.baseline_char_start,
        evidence.baseline_char_end,
        evidence.baseline_token_start,
        evidence.baseline_token_end,
        normalize_for_comparison(evidence.left_anchor),
        normalize_for_comparison(evidence.right_anchor),
    )


def _fingerprint(turn_id: int, evidence: GrammarEditEvidence) -> str:
    payload = {
        "turn_id": turn_id,
        "operation": evidence.operation,
        "pattern": evidence.pattern,
        "observed_fragment": normalize_for_comparison(
            evidence.baseline_fragment
        ),
        "alternate_fragment": normalize_for_comparison(
            evidence.final_fragment
        ),
        "observed_token_span": [
            evidence.baseline_token_start,
            evidence.baseline_token_end,
        ],
        "anchors": [
            normalize_for_comparison(evidence.left_anchor),
            normalize_for_comparison(evidence.right_anchor),
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def grammar_events_for_turn(
    project: ProjectData,
    turn_id: int,
) -> list[SpeechEvent]:
    """Return deterministic grammar-preservation candidates for one turn."""

    return sorted(
        (
            event
            for event in project.speech_events
            if event.turn_id == turn_id
            and event.event_type == GRAMMAR_EVENT_TYPE
            and event.source == GRAMMAR_GUARD_SOURCE
        ),
        key=lambda event: (
            event.token_start if event.token_start is not None else -1,
            event.event_id,
        ),
    )


def _desired_evidence(
    turn: Turn,
) -> list[tuple[GrammarEditEvidence, list[dict[str, str]]]]:
    if not turn.final_text.strip() or _turn_is_ambiguous(turn):
        return []

    grouped: dict[
        tuple[object, ...],
        tuple[GrammarEditEvidence, dict[str, str]],
    ] = {}
    observed = turn.final_text
    for source_name, alternate in _source_texts(turn):
        review = review_grammar_edit_preservation(observed, alternate)
        for evidence in review.evidence:
            key = _evidence_key(evidence)
            if key not in grouped:
                grouped[key] = (evidence, {})
            grouped[key][1][source_name] = evidence.final_fragment

    return [
        (
            evidence,
            [
                {
                    "source": source,
                    "alternate_fragment": variants[source],
                }
                for source in sorted(variants)
            ],
        )
        for evidence, variants in sorted(
            grouped.values(),
            key=lambda item: (
                item[0].baseline_token_start,
                item[0].final_token_start,
                item[0].pattern,
                item[0].baseline_fragment,
                item[0].final_fragment,
            ),
        )
    ]


def refresh_grammar_preservation_events(
    project: ProjectData,
    turn_ids: Iterable[int] | None = None,
) -> set[int]:
    """Synchronize automatic grammar evidence and return invalidated IDs.

    Stable evidence retains its event ID, confirmation state, and reviewer
    decision. New or materially changed evidence starts unreviewed. Stale guard
    events are removed, while all other event sources remain untouched.
    """

    full_refresh = turn_ids is None
    selected_ids = (
        {turn.turn_id for turn in project.turns}
        if turn_ids is None
        else {int(turn_id) for turn_id in turn_ids}
    )
    turns_by_id = {
        turn.turn_id: turn
        for turn in project.turns
        if turn.turn_id in selected_ids
    }
    prior_by_fingerprint: dict[str, list[SpeechEvent]] = defaultdict(list)
    retained: list[SpeechEvent] = []
    invalidated_ids: set[int] = set()

    for event in project.speech_events:
        is_selected_guard_event = (
            (full_refresh or event.turn_id in selected_ids)
            and event.event_type == GRAMMAR_EVENT_TYPE
            and event.source == GRAMMAR_GUARD_SOURCE
        )
        if not is_selected_guard_event:
            retained.append(event)
            continue
        fingerprint = event.details.get("fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            prior_by_fingerprint[fingerprint].append(event)
        invalidated_ids.add(event.event_id)

    next_event_id = max(
        (event.event_id for event in project.speech_events),
        default=0,
    ) + 1
    synchronized: list[SpeechEvent] = []
    reused_ids: set[int] = set()

    for turn_id in sorted(turns_by_id):
        turn = turns_by_id[turn_id]
        for evidence, source_variants in _desired_evidence(turn):
            fingerprint = _fingerprint(turn_id, evidence)
            prior_candidates = prior_by_fingerprint.get(fingerprint, [])
            prior = next(
                (item for item in prior_candidates if item.event_id not in reused_ids),
                None,
            )
            if prior is None:
                event_id = next_event_id
                next_event_id += 1
                reviewed = False
                decision = "candidate"
            else:
                event_id = prior.event_id
                reused_ids.add(event_id)
                invalidated_ids.discard(event_id)
                reviewed = bool(prior.reviewed)
                decision = "confirmed_as_spoken" if reviewed else "candidate"

            evidence_data = asdict(evidence)
            details = {
                "detector": "conservative_source_difference_v1",
                "fingerprint": fingerprint,
                "decision": decision,
                "operation": evidence.operation,
                "pattern": evidence.pattern,
                "observed_fragment": evidence.baseline_fragment,
                "alternate_fragment": evidence.final_fragment,
                "observed_char_start": evidence.baseline_char_start,
                "observed_char_end": evidence.baseline_char_end,
                "alternate_char_start": evidence.final_char_start,
                "alternate_char_end": evidence.final_char_end,
                "alternate_token_start": evidence.final_token_start,
                "alternate_token_end": evidence.final_token_end,
                "left_anchor": evidence.left_anchor,
                "right_anchor": evidence.right_anchor,
                "rationale": evidence.rationale,
                "evidence_sources": [
                    variant["source"] for variant in source_variants
                ],
                "source_variants": source_variants,
                "evidence": evidence_data,
            }
            synchronized.append(
                SpeechEvent(
                    event_id=event_id,
                    turn_id=turn_id,
                    event_type=GRAMMAR_EVENT_TYPE,
                    start=None,
                    end=None,
                    text=evidence.baseline_fragment,
                    confidence=None,
                    source=GRAMMAR_GUARD_SOURCE,
                    token_start=evidence.baseline_token_start,
                    token_end=evidence.baseline_token_end,
                    reviewed=reviewed,
                    details=details,
                )
            )

    project.speech_events = sorted(
        [*retained, *synchronized],
        key=lambda event: (
            event.turn_id if event.turn_id is not None else -1,
            event.event_id,
        ),
    )
    return invalidated_ids | {
        event.event_id
        for event in synchronized
        if event.event_id not in reused_ids
    }


def set_grammar_events_confirmed(
    events: Iterable[SpeechEvent],
    confirmed: bool,
) -> None:
    """Apply the explicit reviewer decision without changing transcript text."""

    for event in events:
        if (
            event.event_type != GRAMMAR_EVENT_TYPE
            or event.source != GRAMMAR_GUARD_SOURCE
        ):
            continue
        event.reviewed = bool(confirmed)
        event.details["decision"] = (
            "confirmed_as_spoken" if confirmed else "candidate"
        )


def grammar_review_summary(project: ProjectData, turn_id: int) -> str:
    """Return a compact neutral summary for the review UI and exports."""

    events = grammar_events_for_turn(project, turn_id)
    if not events:
        return "No grammar-sensitive source differences"
    examples: list[str] = []
    for event in events[:3]:
        observed = str(event.details.get("observed_fragment", event.text)) or "∅"
        alternate = str(event.details.get("alternate_fragment", "")) or "∅"
        examples.append(f"{observed!r} vs {alternate!r}")
    state = "confirmed as spoken" if all(event.reviewed for event in events) else "audio review needed"
    suffix = "" if len(events) <= 3 else f"; +{len(events) - 3} more"
    return (
        f"{len(events)} grammar-sensitive difference(s): "
        f"{', '.join(examples)}{suffix} ({state}; not a grammar diagnosis)"
    )
