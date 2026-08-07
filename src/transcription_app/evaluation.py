"""Evaluation metrics against a manually corrected Gold Standard transcript.

The edit-distance implementation uses linear memory. Project-level metrics are
aggregated turn by turn so a long recording does not allocate one quadratic
matrix for the complete transcript.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from .grammar_evaluation import (
    aggregate_grammar_preservation,
    evaluate_grammar_preservation,
)
from .models import Turn
from .text_utils import (
    DetectedSpeechEvent,
    detected_speech_events,
    normalize_for_comparison,
    words,
)


@dataclass(slots=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    correct: int = 0
    approximate_segments: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def add(self, other: "EditCounts") -> None:
        self.substitutions += other.substitutions
        self.deletions += other.deletions
        self.insertions += other.insertions
        self.correct += other.correct
        self.approximate_segments += other.approximate_segments


@dataclass(frozen=True, slots=True)
class SpeechEventMatch:
    """One event pair supported by token alignment and local context."""

    reference: DetectedSpeechEvent
    hypothesis: DetectedSpeechEvent


@dataclass(frozen=True, slots=True)
class SpeechEventEvaluation:
    """Location-aware speech-event preservation result for one text pair."""

    reference_events: tuple[DetectedSpeechEvent, ...]
    hypothesis_events: tuple[DetectedSpeechEvent, ...]
    matches: tuple[SpeechEventMatch, ...]

    @property
    def reference_count(self) -> int:
        return len(self.reference_events)

    @property
    def hypothesis_count(self) -> int:
        return len(self.hypothesis_events)

    @property
    def matched_count(self) -> int:
        return len(self.matches)

    @property
    def precision(self) -> float | None:
        if not self.hypothesis_events:
            return None
        return self.matched_count / self.hypothesis_count

    @property
    def recall(self) -> float | None:
        if not self.reference_events:
            return None
        return self.matched_count / self.reference_count

    @property
    def f1(self) -> float | None:
        denominator = self.reference_count + self.hypothesis_count
        if denominator == 0:
            return None
        return 2.0 * self.matched_count / denominator


MAX_EXACT_EDIT_CELLS = 2_000_000
_GOLD_GRAMMAR_ANNOTATION_RE = re.compile(r"(?<=[\w'\u2019])@!")


def _metric_character_text(text: str) -> str:
    """Return a comparison-only character view without Gold annotations."""

    return (
        unicodedata.normalize("NFKC", _GOLD_GRAMMAR_ANNOTATION_RE.sub("", text))
        .replace("\u2019", "'")
        .casefold()
    )


def _sequence_matcher_edit_counts(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> EditCounts:
    """Return a bounded-memory fallback edit script for oversized sequences."""
    counts = EditCounts(approximate_segments=1)
    matcher = SequenceMatcher(
        None,
        reference,
        hypothesis,
        autojunk=False,
    )
    for tag, reference_start, reference_end, hypothesis_start, hypothesis_end in matcher.get_opcodes():
        reference_length = reference_end - reference_start
        hypothesis_length = hypothesis_end - hypothesis_start
        if tag == "equal":
            counts.correct += reference_length
        elif tag == "delete":
            counts.deletions += reference_length
        elif tag == "insert":
            counts.insertions += hypothesis_length
        else:
            substitutions = min(reference_length, hypothesis_length)
            counts.substitutions += substitutions
            counts.deletions += reference_length - substitutions
            counts.insertions += hypothesis_length - substitutions
    return counts


# Each dynamic-programming state stores:
# distance, substitutions, deletions, insertions, correct.
_EditState = tuple[int, int, int, int, int]


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Return exact Levenshtein operation counts using O(len(hypothesis)) memory.

    The previous implementation retained the complete rows-by-columns matrix and
    a second traceback matrix. Character-level evaluation of a long transcript
    could therefore consume hundreds of megabytes or more. Keeping only the
    previous and current row preserves the same substitution/deletion/insertion
    tie preference while making memory usage linear.
    """

    if len(reference) * len(hypothesis) > MAX_EXACT_EDIT_CELLS:
        return _sequence_matcher_edit_counts(reference, hypothesis)

    previous: list[_EditState] = [
        (column, 0, 0, column, 0)
        for column in range(len(hypothesis) + 1)
    ]

    for row_index, reference_item in enumerate(reference, start=1):
        current: list[_EditState] = [(row_index, 0, row_index, 0, 0)]
        for column_index, hypothesis_item in enumerate(hypothesis, start=1):
            diagonal = previous[column_index - 1]
            if reference_item == hypothesis_item:
                current.append(
                    (
                        diagonal[0],
                        diagonal[1],
                        diagonal[2],
                        diagonal[3],
                        diagonal[4] + 1,
                    )
                )
                continue

            substitution: _EditState = (
                diagonal[0] + 1,
                diagonal[1] + 1,
                diagonal[2],
                diagonal[3],
                diagonal[4],
            )
            above = previous[column_index]
            deletion: _EditState = (
                above[0] + 1,
                above[1],
                above[2] + 1,
                above[3],
                above[4],
            )
            left = current[column_index - 1]
            insertion: _EditState = (
                left[0] + 1,
                left[1],
                left[2],
                left[3] + 1,
                left[4],
            )

            # Preserve the original tie order: substitution, deletion, insertion.
            selected = substitution
            if deletion[0] < selected[0]:
                selected = deletion
            if insertion[0] < selected[0]:
                selected = insertion
            current.append(selected)
        previous = current

    final = previous[-1]
    return EditCounts(
        substitutions=final[1],
        deletions=final[2],
        insertions=final[3],
        correct=final[4],
    )


def _aggregate_word_edits(pairs: Sequence[tuple[str, str]]) -> tuple[EditCounts, int]:
    counts = EditCounts()
    denominator = 0
    for reference, hypothesis in pairs:
        reference_words = words(reference)
        hypothesis_words = words(hypothesis)
        denominator += len(reference_words)
        counts.add(edit_counts(reference_words, hypothesis_words))
    return counts, denominator


def _aggregate_character_edits(
    pairs: Sequence[tuple[str, str]],
) -> tuple[EditCounts, int]:
    counts = EditCounts()
    denominator = 0
    for reference, hypothesis in pairs:
        reference_characters = list(_metric_character_text(reference))
        hypothesis_characters = list(_metric_character_text(hypothesis))
        denominator += len(reference_characters)
        counts.add(edit_counts(reference_characters, hypothesis_characters))

    # The earlier implementation joined turns with one space. Account for those
    # matching boundaries without aligning unrelated neighboring turns together.
    boundaries = max(0, len(pairs) - 1)
    denominator += boundaries
    counts.correct += boundaries
    return counts, denominator


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = words(reference)
    hypothesis_words = words(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return edit_counts(reference_words, hypothesis_words).errors / len(reference_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference_chars = list(_metric_character_text(reference))
    hypothesis_chars = list(_metric_character_text(hypothesis))
    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0
    return edit_counts(reference_chars, hypothesis_chars).errors / len(reference_chars)


def _safe_rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _optional_rate(numerator: float, denominator: float) -> float | None:
    """Return None when a metric has no valid Gold Standard denominator."""
    return numerator / denominator if denominator else None


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

_SPEAKER_ROLE_ALIASES = {
    "learner": "student",
    "student": "student",
    "pupil": "student",
    "teacher": "teacher",
    "tutor": "teacher",
    "instructor": "teacher",
    "supervisor": "supervisor",
    "observer": "supervisor",
    "monitor": "supervisor",
    "ai": "ai",
    "assistant": "ai",
    "bot": "ai",
    "chatgpt": "ai",
}


def _canonical_speaker_label(value: str) -> str:
    """Normalize role aliases while preserving comparable named labels."""
    normalized = normalize_for_comparison(value)
    if normalized in _UNKNOWN_SPEAKER_LABELS:
        return ""
    tokens = set(words(normalized))
    for alias, canonical in _SPEAKER_ROLE_ALIASES.items():
        if alias in tokens:
            return canonical
    return normalized


def _equal_token_alignment(
    reference_tokens: Sequence[str],
    hypothesis_tokens: Sequence[str],
) -> dict[int, int]:
    """Map reference token indexes to equal hypothesis token indexes.

    Only exact, order-preserving matches are returned. Speech-event tokens can
    therefore be compared at their aligned occurrence instead of globally by
    value, which would incorrectly count an event moved elsewhere in a turn.
    """

    alignment: dict[int, int] = {}
    matcher = SequenceMatcher(
        None,
        reference_tokens,
        hypothesis_tokens,
        autojunk=False,
    )
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            alignment[block.a + offset] = block.b + offset
    return alignment


def _nearest_aligned_before(
    alignment: dict[int, int],
    token_index: int,
) -> int | None:
    candidates = [
        reference_index
        for reference_index in alignment
        if reference_index < token_index
    ]
    if not candidates:
        return None
    return alignment[max(candidates)]


def _nearest_aligned_after(
    alignment: dict[int, int],
    token_index: int,
) -> int | None:
    candidates = [
        reference_index
        for reference_index in alignment
        if reference_index >= token_index
    ]
    if not candidates:
        return None
    return alignment[min(candidates)]


def _event_location_score(
    reference: DetectedSpeechEvent,
    hypothesis: DetectedSpeechEvent,
    alignment: dict[int, int],
    reference_token_count: int,
    hypothesis_token_count: int,
) -> tuple[int, int, int] | None:
    """Score a same-key event pair when alignment supports its location."""

    if reference.key != hypothesis.key:
        return None

    mapped_inside = [
        alignment[index]
        for index in range(reference.token_start, reference.token_end)
        if index in alignment
        and hypothesis.token_start <= alignment[index] < hypothesis.token_end
    ]
    left = _nearest_aligned_before(alignment, reference.token_start)
    right = _nearest_aligned_after(alignment, reference.token_end)
    left_exact = left is not None and left == hypothesis.token_start - 1
    right_exact = right is not None and right == hypothesis.token_end
    exact_boundaries = int(left_exact) + int(right_exact)

    has_context_anchor = left is not None or right is not None
    reference_is_entire_text = (
        reference.token_start == 0
        and reference.token_end == reference_token_count
    )
    hypothesis_is_entire_text = (
        hypothesis.token_start == 0
        and hypothesis.token_end == hypothesis_token_count
    )

    if mapped_inside:
        # An aligned event token plus an adjacent aligned boundary is strong
        # occurrence evidence. When no other tokens align, the event itself is
        # the only available evidence and is accepted conservatively.
        if exact_boundaries == 0 and has_context_anchor:
            return None
    elif exact_boundaries == 0 and not (
        reference_is_entire_text and hypothesis_is_entire_text
    ):
        # Marker variants such as [unclear] / [inaudible] have no equal event
        # token, so at least one adjacent boundary must locate the occurrence.
        return None

    distance = abs(reference.token_start - hypothesis.token_start)
    return len(mapped_inside), exact_boundaries, -distance


def evaluate_speech_events(
    reference: str,
    hypothesis: str,
) -> SpeechEventEvaluation:
    """Evaluate transparent speech events at aligned transcript locations.

    Matching is one-to-one, requires the same normalized event key, and uses
    exact token alignment plus adjacent context. The input strings are not
    normalized in place or rewritten, so grammatical forms remain verbatim.
    """

    reference_events = tuple(detected_speech_events(reference))
    hypothesis_events = tuple(detected_speech_events(hypothesis))
    reference_tokens = words(reference)
    hypothesis_tokens = words(hypothesis)
    alignment = _equal_token_alignment(reference_tokens, hypothesis_tokens)

    unmatched_hypothesis = set(range(len(hypothesis_events)))
    matches: list[SpeechEventMatch] = []
    for reference_event in reference_events:
        candidates: list[tuple[tuple[int, int, int], int]] = []
        for hypothesis_index in unmatched_hypothesis:
            score = _event_location_score(
                reference_event,
                hypothesis_events[hypothesis_index],
                alignment,
                len(reference_tokens),
                len(hypothesis_tokens),
            )
            if score is not None:
                candidates.append((score, hypothesis_index))
        if not candidates:
            continue
        _, selected_index = max(candidates)
        unmatched_hypothesis.remove(selected_index)
        matches.append(
            SpeechEventMatch(
                reference=reference_event,
                hypothesis=hypothesis_events[selected_index],
            )
        )

    return SpeechEventEvaluation(
        reference_events=reference_events,
        hypothesis_events=hypothesis_events,
        matches=tuple(matches),
    )


def evaluate_turns(turns: list[Turn]) -> dict[str, float | int | None]:
    gold_turns = [turn for turn in turns if turn.gold_text.strip()]
    if not gold_turns:
        return {}

    text_pairs = [
        (turn.gold_text, turn.final_text)
        for turn in gold_turns
    ]
    word_edits, reference_word_count = _aggregate_word_edits(text_pairs)
    char_edits, reference_character_count = _aggregate_character_edits(text_pairs)

    speaker_evaluable: list[tuple[Turn, str]] = []
    for turn in gold_turns:
        gold_label = _canonical_speaker_label(turn.gold_speaker)
        if gold_label:
            speaker_evaluable.append((turn, gold_label))
    speaker_correct = sum(
        1
        for turn, gold_label in speaker_evaluable
        if _canonical_speaker_label(turn.speaker) == gold_label
    )

    speech_error_events_evaluated = 0
    speech_error_events_hypothesized = 0
    speech_error_events_preserved = 0
    for turn in gold_turns:
        event_evaluation = evaluate_speech_events(
            turn.gold_text,
            turn.final_text,
        )
        speech_error_events_evaluated += event_evaluation.reference_count
        speech_error_events_hypothesized += event_evaluation.hypothesis_count
        speech_error_events_preserved += event_evaluation.matched_count

    grammar_preservation = aggregate_grammar_preservation(
        evaluate_grammar_preservation(
            turn.gold_text,
            turn.final_text,
        )
        for turn in gold_turns
    )

    return {
        "turns_evaluated": len(gold_turns),
        "word_error_rate": _safe_rate(word_edits.errors, reference_word_count),
        "character_error_rate": _safe_rate(
            char_edits.errors,
            reference_character_count,
        ),
        "substitutions": word_edits.substitutions,
        "deletions": word_edits.deletions,
        "insertions": word_edits.insertions,
        "word_alignment_approximations": word_edits.approximate_segments,
        "character_alignment_approximations": char_edits.approximate_segments,
        "speaker_labels_evaluated": len(speaker_evaluable),
        "speaker_labels_correct": speaker_correct,
        "speaker_accuracy": _optional_rate(speaker_correct, len(speaker_evaluable)),
        "speech_error_events_evaluated": speech_error_events_evaluated,
        "speech_error_events_hypothesized": speech_error_events_hypothesized,
        "speech_error_events_preserved": speech_error_events_preserved,
        "speech_error_event_precision": _optional_rate(
            speech_error_events_preserved,
            speech_error_events_hypothesized,
        ),
        "speech_error_preservation_rate": _optional_rate(
            speech_error_events_preserved,
            speech_error_events_evaluated,
        ),
        "speech_error_event_f1": _optional_rate(
            2 * speech_error_events_preserved,
            speech_error_events_evaluated
            + speech_error_events_hypothesized,
        ),
        **grammar_preservation.to_metrics(),
        "manual_review_rate": _safe_rate(sum(turn.manual_review for turn in turns), len(turns)),
    }


def per_source_metrics(
    turns: list[Turn],
) -> list[dict[str, float | str | int]]:
    sources = {
        "Zoom": "zoom_text",
        "ChatGPT": "chatgpt_text",
        "Additional model": "model_text",
        "Final": "final_text",
    }
    results: list[dict[str, float | str | int]] = []
    for name, attribute in sources.items():
        valid = [
            turn
            for turn in turns
            if turn.gold_text.strip() and getattr(turn, attribute).strip()
        ]
        if not valid:
            continue

        pairs = [
            (turn.gold_text, getattr(turn, attribute))
            for turn in valid
        ]
        word_edits, reference_word_count = _aggregate_word_edits(pairs)
        char_edits, reference_character_count = _aggregate_character_edits(pairs)
        results.append(
            {
                "source": name,
                "wer": _safe_rate(word_edits.errors, reference_word_count),
                "cer": _safe_rate(char_edits.errors, reference_character_count),
                "alignment_approximations": (
                    word_edits.approximate_segments
                    + char_edits.approximate_segments
                ),
            }
        )
    return results
