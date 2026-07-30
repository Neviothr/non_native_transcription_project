"""Text normalization and speech-feature detection."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
_UNCLEAR_RE = re.compile(
    r"\b(?:inaudible|unclear|unintelligible|unknown)\b|\[.*?(?:inaudible|unclear).*?\]|\?{2,}",
    re.IGNORECASE,
)
_HESITATION_RE = re.compile(r"\b(?:uh+|um+|erm+|hmm+|mm+)\b", re.IGNORECASE)
_SELF_CORRECTION_RE = re.compile(
    r"\b(?:i mean|sorry|rather|actually|no[, ]|let me (?:start|say|try) again|what i meant)\b",
    re.IGNORECASE,
)


def words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text)]


def normalize_for_comparison(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(words(normalized))


def contains_hebrew(text: str) -> bool:
    return bool(_HEBREW_RE.search(text))


def contains_unclear_marker(text: str) -> bool:
    return bool(_UNCLEAR_RE.search(text))


def contains_hesitation_or_repetition(text: str) -> bool:
    if _HESITATION_RE.search(text):
        return True
    tokens = words(text)
    return any(a == b for a, b in zip(tokens, tokens[1:]))


def contains_self_correction(text: str) -> bool:
    if _SELF_CORRECTION_RE.search(text):
        return True
    # A repeated partial restart such as "I was- I am" is treated as a likely self-correction.
    return bool(re.search(r"\b\w+[—-]\s+\w+", text))


def repetition_rate(text: str) -> float:
    tokens = words(text)
    if len(tokens) < 2:
        return 0.0
    repeated = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    return repeated / max(1, len(tokens) - 1)


def token_overlap(a: str, b: str) -> float:
    a_tokens = Counter(words(a))
    b_tokens = Counter(words(b))
    if not a_tokens or not b_tokens:
        return 0.0
    common = sum((a_tokens & b_tokens).values())
    total = sum((a_tokens | b_tokens).values())
    return common / total if total else 0.0


def word_difference_rate(a: str, b: str) -> float:
    """Normalized count of words that differ between two transcript versions."""
    a_tokens = Counter(words(a))
    b_tokens = Counter(words(b))
    if not a_tokens and not b_tokens:
        return 0.0
    differing = sum((a_tokens - b_tokens).values()) + sum((b_tokens - a_tokens).values())
    denominator = max(1, sum(a_tokens.values()), sum(b_tokens.values()))
    return min(1.0, differing / denominator)