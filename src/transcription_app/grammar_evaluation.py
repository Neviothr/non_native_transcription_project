"""Evaluate preservation of explicitly annotated grammar-error tokens.

Gold transcripts may mark a token with the ``@!`` suffix (for example,
``have@!`` or ``it’s@!``).  The marker says only that the exact spoken token
should be checked for preservation; this module does not infer grammaticality
from unannotated text.

Evaluation is occurrence-aware.  Tokens are aligned with a deterministic
Levenshtein traceback before annotated reference tokens are classified as
preserved, substituted, or deleted.  Insertions in the final transcript do not
penalize preservation of an aligned annotated token. Case and straight/curly
apostrophe typography are normalized for comparison only; grammatical word
forms and the stored strings are not transformed.
"""

from __future__ import annotations

from dataclasses import dataclass
from re import UNICODE, compile as compile_pattern
from typing import Iterable, Literal


GRAMMAR_ERROR_ANNOTATION_SUFFIX = "@!"

# ``[^\W_]`` is a Unicode-aware alphanumeric character.  Keeping apostrophes
# inside the token makes both straight and curly contractions a single unit.
# The annotation is deliberately valid only as a directly attached suffix.
_TOKEN_RE = compile_pattern(
    r"(?P<surface>[^\W_]+(?:['’][^\W_]+)*)(?P<annotation>@!)?",
    UNICODE,
)


@dataclass(frozen=True, slots=True)
class _ReferenceToken:
    surface: str
    annotated: bool


@dataclass(frozen=True, slots=True)
class GrammarPreservationEvaluation:
    """Aggregatable counts and rates for one or more text pairs."""

    grammar_error_tokens_evaluated: int = 0
    grammar_error_tokens_preserved: int = 0
    grammar_error_token_substitutions: int = 0
    grammar_error_token_deletions: int = 0

    def __post_init__(self) -> None:
        counts = (
            self.grammar_error_tokens_evaluated,
            self.grammar_error_tokens_preserved,
            self.grammar_error_token_substitutions,
            self.grammar_error_token_deletions,
        )
        if any(value < 0 for value in counts):
            raise ValueError("Grammar-preservation counts cannot be negative.")
        outcomes = (
            self.grammar_error_tokens_preserved
            + self.grammar_error_token_substitutions
            + self.grammar_error_token_deletions
        )
        if outcomes != self.grammar_error_tokens_evaluated:
            raise ValueError(
                "Preserved, substituted, and deleted token counts must add up "
                "to the evaluated token count."
            )

    @property
    def grammar_error_preservation_rate(self) -> float | None:
        """Annotated-token preservation rate; higher is better."""
        if self.grammar_error_tokens_evaluated == 0:
            return None
        return (
            self.grammar_error_tokens_preserved
            / self.grammar_error_tokens_evaluated
        )

    @property
    def grammar_error_token_loss_rate(self) -> float | None:
        """Annotated-token substitution/deletion rate; lower is better."""
        if self.grammar_error_tokens_evaluated == 0:
            return None
        losses = (
            self.grammar_error_token_substitutions
            + self.grammar_error_token_deletions
        )
        return losses / self.grammar_error_tokens_evaluated

    @property
    def unwanted_grammar_correction_rate(self) -> float | None:
        """Alias for annotated-token loss rate; lower is better."""
        return self.grammar_error_token_loss_rate

    def to_metrics(self) -> dict[str, int | float | None]:
        """Return stable metric names for project-level reporting/export."""
        return {
            "grammar_error_tokens_evaluated": self.grammar_error_tokens_evaluated,
            "grammar_error_tokens_preserved": self.grammar_error_tokens_preserved,
            "grammar_error_token_substitutions": self.grammar_error_token_substitutions,
            "grammar_error_token_deletions": self.grammar_error_token_deletions,
            "grammar_error_preservation_rate": self.grammar_error_preservation_rate,
            "unwanted_grammar_correction_rate": self.unwanted_grammar_correction_rate,
            "grammar_error_token_loss_rate": self.grammar_error_token_loss_rate,
        }


def _reference_tokens(text: str) -> list[_ReferenceToken]:
    return [
        _ReferenceToken(
            surface=_normalize_surface(match.group("surface")),
            annotated=match.group("annotation") is not None,
        )
        for match in _TOKEN_RE.finditer(text)
    ]


def _surface_tokens(text: str) -> list[str]:
    return [
        _normalize_surface(match.group("surface"))
        for match in _TOKEN_RE.finditer(text)
    ]


def _normalize_surface(value: str) -> str:
    """Normalize typography only, without changing grammatical word forms."""

    return value.replace("\u2019", "'").casefold()


_Operation = Literal["match", "substitute", "delete", "insert"]


def _alignment_operations(
    reference: list[str],
    hypothesis: list[str],
) -> list[tuple[_Operation, int | None]]:
    """Return one operation per traceback step and its reference token index.

    Ties prefer substitution, then deletion, then insertion, matching the main
    evaluator's edit-count convention.  Exact diagonal matches are preferred
    when tied.  This produces stable occurrence-level results for repeated
    words instead of testing whether an annotated surface appears anywhere.
    """
    reference_size = len(reference)
    hypothesis_size = len(hypothesis)
    distances = [
        [0] * (hypothesis_size + 1)
        for _ in range(reference_size + 1)
    ]
    traceback: list[list[_Operation | None]] = [
        [None] * (hypothesis_size + 1)
        for _ in range(reference_size + 1)
    ]

    for reference_index in range(1, reference_size + 1):
        distances[reference_index][0] = reference_index
        traceback[reference_index][0] = "delete"
    for hypothesis_index in range(1, hypothesis_size + 1):
        distances[0][hypothesis_index] = hypothesis_index
        traceback[0][hypothesis_index] = "insert"

    for reference_index in range(1, reference_size + 1):
        for hypothesis_index in range(1, hypothesis_size + 1):
            if reference[reference_index - 1] == hypothesis[hypothesis_index - 1]:
                match_cost = distances[reference_index - 1][hypothesis_index - 1]
                deletion_cost = distances[reference_index - 1][hypothesis_index] + 1
                insertion_cost = distances[reference_index][hypothesis_index - 1] + 1
                if match_cost <= deletion_cost and match_cost <= insertion_cost:
                    distances[reference_index][hypothesis_index] = match_cost
                    traceback[reference_index][hypothesis_index] = "match"
                    continue

            candidates: tuple[tuple[int, _Operation], ...] = (
                (
                    distances[reference_index - 1][hypothesis_index - 1] + 1,
                    "substitute",
                ),
                (distances[reference_index - 1][hypothesis_index] + 1, "delete"),
                (distances[reference_index][hypothesis_index - 1] + 1, "insert"),
            )
            cost, operation = min(candidates, key=lambda candidate: candidate[0])
            distances[reference_index][hypothesis_index] = cost
            traceback[reference_index][hypothesis_index] = operation

    operations: list[tuple[_Operation, int | None]] = []
    reference_index = reference_size
    hypothesis_index = hypothesis_size
    while reference_index or hypothesis_index:
        operation = traceback[reference_index][hypothesis_index]
        if operation is None:  # pragma: no cover - defensive invariant check
            raise RuntimeError("Grammar token alignment traceback is incomplete.")
        if operation in {"match", "substitute"}:
            reference_index -= 1
            hypothesis_index -= 1
            operations.append((operation, reference_index))
        elif operation == "delete":
            reference_index -= 1
            operations.append((operation, reference_index))
        else:
            hypothesis_index -= 1
            operations.append((operation, None))

    operations.reverse()
    return operations


def evaluate_grammar_preservation(
    gold_text: str,
    final_text: str,
) -> GrammarPreservationEvaluation:
    """Evaluate exact preservation of Gold tokens explicitly suffixed ``@!``.

    Unannotated Gold tokens contribute alignment context only.  Their wording
    never causes a token to be treated as a grammar error.
    """
    if not isinstance(gold_text, str) or not isinstance(final_text, str):
        raise TypeError("gold_text and final_text must both be strings")

    reference_tokens = _reference_tokens(gold_text)
    annotated_count = sum(token.annotated for token in reference_tokens)
    if annotated_count == 0:
        return GrammarPreservationEvaluation()

    operations = _alignment_operations(
        [token.surface for token in reference_tokens],
        _surface_tokens(final_text),
    )
    preserved = 0
    substitutions = 0
    deletions = 0
    for operation, reference_index in operations:
        if reference_index is None or not reference_tokens[reference_index].annotated:
            continue
        if operation == "match":
            preserved += 1
        elif operation == "substitute":
            substitutions += 1
        elif operation == "delete":
            deletions += 1

    return GrammarPreservationEvaluation(
        grammar_error_tokens_evaluated=annotated_count,
        grammar_error_tokens_preserved=preserved,
        grammar_error_token_substitutions=substitutions,
        grammar_error_token_deletions=deletions,
    )


def aggregate_grammar_preservation(
    evaluations: Iterable[GrammarPreservationEvaluation],
) -> GrammarPreservationEvaluation:
    """Combine per-turn counts and recompute rates from the total denominator."""
    evaluated = 0
    preserved = 0
    substitutions = 0
    deletions = 0
    for evaluation in evaluations:
        if not isinstance(evaluation, GrammarPreservationEvaluation):
            raise TypeError(
                "evaluations must contain GrammarPreservationEvaluation values"
            )
        evaluated += evaluation.grammar_error_tokens_evaluated
        preserved += evaluation.grammar_error_tokens_preserved
        substitutions += evaluation.grammar_error_token_substitutions
        deletions += evaluation.grammar_error_token_deletions
    return GrammarPreservationEvaluation(
        grammar_error_tokens_evaluated=evaluated,
        grammar_error_tokens_preserved=preserved,
        grammar_error_token_substitutions=substitutions,
        grammar_error_token_deletions=deletions,
    )


__all__ = [
    "GRAMMAR_ERROR_ANNOTATION_SUFFIX",
    "GrammarPreservationEvaluation",
    "aggregate_grammar_preservation",
    "evaluate_grammar_preservation",
]
