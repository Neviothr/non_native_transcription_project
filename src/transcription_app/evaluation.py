"""Evaluation metrics against a manually corrected Gold Standard transcript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import Turn
from .text_utils import contains_hesitation_or_repetition, contains_self_correction, words


@dataclass(slots=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    correct: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    distance = [[0] * cols for _ in range(rows)]
    operation = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        distance[i][0] = i
        operation[i][0] = "D"
    for j in range(1, cols):
        distance[0][j] = j
        operation[0][j] = "I"

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
                operation[i][j] = "C"
                continue
            candidates = {
                "S": distance[i - 1][j - 1] + 1,
                "D": distance[i - 1][j] + 1,
                "I": distance[i][j - 1] + 1,
            }
            op = min(candidates, key=candidates.get)
            distance[i][j] = candidates[op]
            operation[i][j] = op

    counts = EditCounts()
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        op = operation[i][j]
        if op == "C":
            counts.correct += 1
            i -= 1
            j -= 1
        elif op == "S":
            counts.substitutions += 1
            i -= 1
            j -= 1
        elif op == "D":
            counts.deletions += 1
            i -= 1
        elif op == "I":
            counts.insertions += 1
            j -= 1
        elif i > 0:
            counts.deletions += 1
            i -= 1
        else:
            counts.insertions += 1
            j -= 1
    return counts


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


def evaluate_turns(turns: list[Turn]) -> dict[str, float | int]:
    gold_turns = [turn for turn in turns if turn.gold_text.strip()]
    if not gold_turns:
        return {}
    reference = " ".join(turn.gold_text for turn in gold_turns)
    hypothesis = " ".join((turn.final_text or turn.model_text) for turn in gold_turns)
    word_edits = edit_counts(words(reference), words(hypothesis))
    char_edits = edit_counts(list(reference.casefold()), list(hypothesis.casefold()))

    speaker_known = [turn for turn in gold_turns if turn.gold_speaker.strip() and turn.speaker.strip()]
    speaker_correct = sum(
        1 for turn in speaker_known if turn.speaker.casefold() == turn.gold_speaker.casefold()
    )

    gold_disfluencies = sum(
        int(contains_hesitation_or_repetition(turn.gold_text))
        + int(contains_self_correction(turn.gold_text))
        for turn in gold_turns
    )
    preserved_disfluencies = sum(
        int(
            contains_hesitation_or_repetition(turn.gold_text)
            and contains_hesitation_or_repetition(turn.final_text or turn.model_text)
        )
        + int(
            contains_self_correction(turn.gold_text)
            and contains_self_correction(turn.final_text or turn.model_text)
        )
        for turn in gold_turns
    )

    return {
        "turns_evaluated": len(gold_turns),
        "word_error_rate": _safe_rate(word_edits.errors, len(words(reference))),
        "character_error_rate": _safe_rate(char_edits.errors, len(reference)),
        "substitutions": word_edits.substitutions,
        "deletions": word_edits.deletions,
        "insertions": word_edits.insertions,
        "speaker_accuracy": _safe_rate(speaker_correct, len(speaker_known)),
        "speech_error_preservation_rate": _safe_rate(preserved_disfluencies, gold_disfluencies),
        "manual_review_rate": _safe_rate(sum(turn.manual_review for turn in turns), len(turns)),
    }


def per_source_metrics(turns: list[Turn]) -> list[dict[str, float | str]]:
    sources = {
        "Zoom": "zoom_text",
        "ChatGPT": "chatgpt_text",
        "Additional model": "model_text",
        "Final": "final_text",
    }
    results: list[dict[str, float | str]] = []
    for name, attribute in sources.items():
        valid = [turn for turn in turns if turn.gold_text.strip() and getattr(turn, attribute).strip()]
        if not valid:
            continue
        reference = " ".join(turn.gold_text for turn in valid)
        hypothesis = " ".join(getattr(turn, attribute) for turn in valid)
        results.append(
            {
                "source": name,
                "wer": word_error_rate(reference, hypothesis),
                "cer": character_error_rate(reference, hypothesis),
            }
        )
    return results