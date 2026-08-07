"""Text normalization and speech-feature detection."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

_WORD_RE = re.compile(r"[\w'\u2019]+", re.UNICODE)
_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
_UNCLEAR_RE = re.compile(
    r"\b(?:inaudible|unclear|unintelligible|unknown)\b|\[.*?(?:inaudible|unclear).*?\]|\?{2,}",
    re.IGNORECASE,
)
_FILLER_RE = re.compile(
    r"(?<![\w'])(?:uh+|um+|erm+|er+|ah+|eh+|hmm+|mm+|mhm+)(?![\w'])",
    re.IGNORECASE,
)
# A trailing dash separated from the following word is transparent evidence of
# a cut-off word (``I wan- I want``). Internal hyphens in compounds such as
# ``state-of-the-art`` are deliberately excluded.
_PARTIAL_WORD_RE = re.compile(
    r"(?<![\w'])[\w']+[-\u2010\u2011\u2012\u2013\u2014](?=\s|$|[.,!?;:)\]\}])",
    re.UNICODE,
)
_SELF_CORRECTION_RE = re.compile(
    r"\b(?:i mean|sorry|rather|actually|no[, ]|let me (?:start|say|try) again|what i meant)\b",
    re.IGNORECASE,
)

_PERSISTED_EVENT_TYPES = {
    "filler": "filled_pause",
    "partial_word": "partial_word",
    "repetition": "repetition",
    "self_correction": "revision",
    "unclear_marker": "unclear",
    "hebrew": "code_switch",
}


@dataclass(frozen=True, slots=True)
class DetectedSpeechEvent:
    """One transparently detected transcript event and its source location.

    Character and token offsets are half-open. ``value`` is normalized only
    for comparison; the input text is never rewritten. The object intentionally
    stays independent of the persisted project model so detection and evaluation
    can be tested without changing project-file compatibility.
    """

    kind: str
    value: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int

    @property
    def key(self) -> str:
        if self.kind == "unclear_marker":
            return self.kind
        # Keep the historical ``hesitation:...`` counter keys while exposing
        # the clearer ``filler`` event kind to new callers.
        prefix = "hesitation" if self.kind == "filler" else self.kind
        return f"{prefix}:{self.value}"

    def persistence_fields(self) -> dict[str, object]:
        """Return model-neutral fields for an optional persisted event hook."""

        return {
            "event_type": _PERSISTED_EVENT_TYPES.get(self.kind, self.kind),
            "text": self.value,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "details": {
                "char_start": self.char_start,
                "char_end": self.char_end,
                "detector": "transparent_text_v1",
            },
        }


@dataclass(frozen=True, slots=True)
class _WordSpan:
    value: str
    char_start: int
    char_end: int


def _word_spans(text: str) -> list[_WordSpan]:
    return [
        _WordSpan(
            value=match.group(0).casefold().replace("\u2019", "'"),
            char_start=match.start(),
            char_end=match.end(),
        )
        for match in _WORD_RE.finditer(text)
    ]


def _token_range_for_chars(
    tokens: list[_WordSpan],
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    overlapping = [
        index
        for index, token in enumerate(tokens)
        if token.char_start < char_end and token.char_end > char_start
    ]
    if overlapping:
        return overlapping[0], overlapping[-1] + 1

    # Punctuation-only markers such as ``??`` live at the boundary following
    # the preceding word. This keeps every event locatable on the token axis.
    boundary = sum(token.char_end <= char_start for token in tokens)
    return boundary, boundary


def _event_from_match(
    kind: str,
    value: str,
    match: re.Match[str],
    tokens: list[_WordSpan],
) -> DetectedSpeechEvent:
    token_start, token_end = _token_range_for_chars(
        tokens,
        match.start(),
        match.end(),
    )
    return DetectedSpeechEvent(
        kind=kind,
        value=value,
        char_start=match.start(),
        char_end=match.end(),
        token_start=token_start,
        token_end=token_end,
    )


def _repetition_events(
    tokens: list[_WordSpan],
    max_ngram: int,
) -> list[DetectedSpeechEvent]:
    """Detect consecutive exact repetitions, preferring informative n-grams."""

    events: list[DetectedSpeechEvent] = []
    values = [token.value for token in tokens]
    filler_values = {
        normalize_for_comparison(match.group(0))
        for match in _FILLER_RE.finditer("uh um erm er ah eh hmm mm mhm")
    }
    index = 1
    while index < len(tokens):
        limit = min(max_ngram, index, len(tokens) - index)
        candidates = [
            size
            for size in range(1, limit + 1)
            if values[index - size:index] == values[index:index + size]
        ]
        if not candidates:
            index += 1
            continue

        # A run such as ``I I I`` is most clearly described as repeated
        # unigrams. Otherwise prefer the longest exact phrase (``I want I
        # want``) so nested unigram events do not inflate the count.
        informative = [
            size
            for size in candidates
            if len(set(values[index:index + size])) > 1
        ]
        size = max(informative) if informative else min(candidates)
        repeated = values[index:index + size]
        if not all(value in filler_values for value in repeated):
            events.append(
                DetectedSpeechEvent(
                    kind="repetition",
                    value=" ".join(repeated),
                    char_start=tokens[index].char_start,
                    char_end=tokens[index + size - 1].char_end,
                    token_start=index,
                    token_end=index + size,
                )
            )
        index += size
    return events


def words(text: str) -> list[str]:
    return [
        match.group(0).casefold().replace("\u2019", "'")
        for match in _WORD_RE.finditer(text)
    ]


def normalize_for_comparison(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(words(normalized))


def contains_hebrew(text: str) -> bool:
    return bool(_HEBREW_RE.search(text))


def contains_unclear_marker(text: str) -> bool:
    return bool(_UNCLEAR_RE.search(text))


def contains_hesitation_or_repetition(text: str) -> bool:
    return any(
        event.kind in {"filler", "partial_word", "repetition"}
        for event in detected_speech_events(text)
    )


def contains_self_correction(text: str) -> bool:
    if _SELF_CORRECTION_RE.search(text):
        return True
    if any(
        event.kind == "partial_word" and text[event.char_end:].strip()
        for event in detected_speech_events(text)
    ):
        return True
    return False


def detected_speech_events(
    text: str,
    *,
    max_repetition_ngram: int = 4,
) -> list[DetectedSpeechEvent]:
    """Return located, transparently detectable speech events.

    Detection is deliberately lexical and conservative. It covers explicit
    fillers, cut-off words ending in a dash, consecutive exact n-gram
    repetitions, explicit self-correction markers, unclear markers, and Hebrew
    words. It neither infers nor changes grammatical errors.
    """

    if max_repetition_ngram < 1:
        raise ValueError("max_repetition_ngram must be at least 1")

    tokens = _word_spans(text)
    events: list[DetectedSpeechEvent] = []
    partial_matches = list(_PARTIAL_WORD_RE.finditer(text))
    partial_token_indexes: set[int] = set()
    for match in partial_matches:
        value = normalize_for_comparison(match.group(0)) or "fragment"
        event = _event_from_match("partial_word", value, match, tokens)
        events.append(event)
        partial_token_indexes.update(range(event.token_start, event.token_end))

    for match in _FILLER_RE.finditer(text):
        token_start, token_end = _token_range_for_chars(
            tokens,
            match.start(),
            match.end(),
        )
        if any(index in partial_token_indexes for index in range(token_start, token_end)):
            continue
        marker = normalize_for_comparison(match.group(0)) or "marker"
        events.append(_event_from_match("filler", marker, match, tokens))

    events.extend(_repetition_events(tokens, max_repetition_ngram))

    for match in _SELF_CORRECTION_RE.finditer(text):
        marker = normalize_for_comparison(match.group(0)) or "marker"
        events.append(
            _event_from_match("self_correction", marker, match, tokens)
        )

    for match in _UNCLEAR_RE.finditer(text):
        events.append(
            _event_from_match("unclear_marker", "marker", match, tokens)
        )

    for index, token in enumerate(tokens):
        if _HEBREW_RE.search(token.value):
            events.append(
                DetectedSpeechEvent(
                    kind="hebrew",
                    value=token.value,
                    char_start=token.char_start,
                    char_end=token.char_end,
                    token_start=index,
                    token_end=index + 1,
                )
            )

    return sorted(
        events,
        key=lambda event: (
            event.char_start,
            event.char_end,
            event.kind,
            event.value,
        ),
    )


def speech_error_events(text: str) -> Counter[str]:
    """Return countable, comparable speech-error and disfluency events.

    The detector intentionally covers only phenomena that can be identified
    transparently from transcript text: fillers, partial words, exact n-gram
    repetitions, explicit self-correction markers, unclear-speech markers, and
    Hebrew words. It does not infer grammar errors or rewrite the input.
    """
    return Counter(event.key for event in detected_speech_events(text))


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
