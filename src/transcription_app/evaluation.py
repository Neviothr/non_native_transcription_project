"""Evaluation metrics against a manually corrected Gold Standard transcript.

The edit-distance implementation uses linear memory. Project-level metrics are
aggregated turn by turn so a long recording does not allocate one quadratic
matrix for the complete transcript.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from .models import Turn
from .text_utils import normalize_for_comparison, speech_error_events, words


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


MAX_EXACT_EDIT_CELLS = 2_000_000


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
        reference_characters = list(reference.casefold())
        hypothesis_characters = list(hypothesis.casefold())
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
    reference_chars = list(reference.casefold())
    hypothesis_chars = list(hypothesis.casefold())
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


def evaluate_turns(turns: list[Turn]) -> dict[str, float | int | None]:
    gold_turns = [turn for turn in turns if turn.gold_text.strip()]
    if not gold_turns:
        return {}

    text_pairs = [
        (turn.gold_text, turn.final_text or turn.model_text)
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
    speech_error_events_preserved = 0
    for turn in gold_turns:
        gold_events = speech_error_events(turn.gold_text)
        hypothesis_events = speech_error_events(turn.final_text or turn.model_text)
        speech_error_events_evaluated += sum(gold_events.values())
        speech_error_events_preserved += sum((gold_events & hypothesis_events).values())

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
        "speech_error_events_preserved": speech_error_events_preserved,
        "speech_error_preservation_rate": _optional_rate(
            speech_error_events_preserved,
            speech_error_events_evaluated,
        ),
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
