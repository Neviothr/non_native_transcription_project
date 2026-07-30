"""Timeline and transcript alignment logic."""

from __future__ import annotations

from difflib import SequenceMatcher
from statistics import mean

from .models import TranscriptSegment, Turn
from .text_utils import normalize_for_comparison, token_overlap


def text_similarity(a: str, b: str) -> float:
    normalized_a = normalize_for_comparison(a)
    normalized_b = normalize_for_comparison(b)
    if not normalized_a or not normalized_b:
        return 0.0
    sequence = SequenceMatcher(None, normalized_a, normalized_b, autojunk=False).ratio()
    overlap = token_overlap(normalized_a, normalized_b)
    return 0.65 * sequence + 0.35 * overlap


def time_overlap(start_a: float | None, end_a: float | None, start_b: float | None, end_b: float | None) -> float:
    if None in (start_a, end_a, start_b, end_b):
        return 0.0
    assert start_a is not None and end_a is not None and start_b is not None and end_b is not None
    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return overlap / union if union > 0 else 0.0


def segments_to_turns(segments: list[TranscriptSegment]) -> list[Turn]:
    turns: list[Turn] = []
    for index, segment in enumerate(segments, start=1):
        turns.append(
            Turn(
                turn_id=index,
                start=segment.start,
                end=segment.end,
                speaker_raw=segment.speaker or "Unknown",
                speaker=segment.speaker or "Unknown",
                model_text=segment.text,
                final_text=segment.text,
                model_confidence=segment.confidence,
            )
        )
    return turns


def _has_timing(segments: list[TranscriptSegment]) -> bool:
    return any(segment.start is not None and segment.end is not None for segment in segments)


def align_source_to_turns(turns: list[Turn], source: list[TranscriptSegment]) -> list[str]:
    if not turns:
        return []
    if not source:
        return [""] * len(turns)
    if _has_timing(source) and any(turn.start is not None and turn.end is not None for turn in turns):
        return _align_by_time(turns, source)
    return _align_monotonic(turns, source)


def _align_by_time(turns: list[Turn], source: list[TranscriptSegment]) -> list[str]:
    results: list[str] = []
    for turn in turns:
        matches: list[tuple[float, TranscriptSegment]] = []
        for segment in source:
            overlap = time_overlap(turn.start, turn.end, segment.start, segment.end)
            if overlap > 0:
                speaker_bonus = 0.08 if segment.speaker == turn.speaker_raw else 0.0
                matches.append((overlap + speaker_bonus, segment))
        if matches:
            matches.sort(key=lambda item: item[0], reverse=True)
            selected = [item[1] for item in matches if item[0] >= max(0.08, matches[0][0] * 0.35)]
            selected.sort(key=lambda segment: segment.start if segment.start is not None else 0.0)
            results.append(" ".join(segment.text for segment in selected).strip())
        else:
            # Use nearby text only when it is also lexically similar.
            candidates = sorted(
                source,
                key=lambda segment: abs((segment.start or 0.0) - (turn.start or 0.0)),
            )[:3]
            best = max(candidates, key=lambda segment: text_similarity(turn.model_text, segment.text))
            score = text_similarity(turn.model_text, best.text)
            results.append(best.text if score >= 0.35 else "")
    return results


def _align_monotonic(turns: list[Turn], source: list[TranscriptSegment]) -> list[str]:
    """Monotonic dynamic-programming alignment for untimed transcripts."""
    n = len(turns)
    m = len(source)
    gap = -0.22
    scores = [[0.0] * (m + 1) for _ in range(n + 1)]
    choices = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        scores[i][0] = i * gap
        choices[i][0] = "turn_gap"
    for j in range(1, m + 1):
        scores[0][j] = j * gap
        choices[0][j] = "source_gap"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = text_similarity(turns[i - 1].model_text, source[j - 1].text)
            options = {
                "match": scores[i - 1][j - 1] + match_score,
                "turn_gap": scores[i - 1][j] + gap,
                "source_gap": scores[i][j - 1] + gap,
            }
            choice = max(options, key=options.get)
            scores[i][j] = options[choice]
            choices[i][j] = choice

    results = [""] * n
    i, j = n, m
    while i > 0 or j > 0:
        choice = choices[i][j]
        if choice == "match":
            similarity = text_similarity(turns[i - 1].model_text, source[j - 1].text)
            if similarity >= 0.18:
                results[i - 1] = source[j - 1].text
            i -= 1
            j -= 1
        elif choice == "turn_gap":
            i -= 1
        else:
            j -= 1

    # Attach unmatched source fragments to the most similar neighboring turn.
    used = {text for text in results if text}
    for segment in source:
        if segment.text in used:
            continue
        best_index = max(range(n), key=lambda idx: text_similarity(turns[idx].model_text, segment.text))
        best_score = text_similarity(turns[best_index].model_text, segment.text)
        if best_score >= 0.45 and not results[best_index]:
            results[best_index] = segment.text
    return results


def agreement_score(texts: list[str]) -> float:
    present = [text for text in texts if text.strip()]
    if len(present) < 2:
        return 0.0
    pair_scores: list[float] = []
    for index, first in enumerate(present):
        for second in present[index + 1 :]:
            pair_scores.append(text_similarity(first, second))
    return mean(pair_scores) if pair_scores else 0.0


def align_source_segments_to_turns(turns: list[Turn], source: list[TranscriptSegment]) -> list[TranscriptSegment | None]:
    """Return the best source segment for each turn, including its speaker label."""
    texts = align_source_to_turns(turns, source)
    results: list[TranscriptSegment | None] = []
    used: set[int] = set()
    for turn, text in zip(turns, texts):
        if not text:
            results.append(None)
            continue
        candidates = [
            (index, segment)
            for index, segment in enumerate(source)
            if index not in used
        ]
        if not candidates:
            candidates = list(enumerate(source))
        best_index, best_segment = max(
            candidates,
            key=lambda item: 0.65 * text_similarity(text, item[1].text)
            + 0.35 * time_overlap(turn.start, turn.end, item[1].start, item[1].end),
        )
        used.add(best_index)
        results.append(best_segment)
    return results