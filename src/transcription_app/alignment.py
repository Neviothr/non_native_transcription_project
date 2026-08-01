"""Timeline and transcript alignment logic.

Imported transcript systems often use different sentence and caption boundaries.
Alignment therefore operates on ordered word chunks rather than assuming that
one imported segment corresponds to one review turn. Each source chunk is
assigned to exactly one turn, which allows source segments to be split across
turns and adjacent source segments to be combined into one turn without
duplicating text.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import mean

from .models import TranscriptSegment, Turn
from .text_utils import normalize_for_comparison, token_overlap, words


@dataclass(slots=True)
class _SourceChunk:
    text: str
    segment_index: int
    midpoint: float | None
    speaker: str
    normalized_words: tuple[str, ...]


@dataclass(slots=True)
class _AlignmentDetails:
    texts: list[str]
    source_weights: list[Counter[int]]


def text_similarity(a: str, b: str) -> float:
    normalized_a = normalize_for_comparison(a)
    normalized_b = normalize_for_comparison(b)
    if not normalized_a or not normalized_b:
        return 0.0
    sequence = SequenceMatcher(None, normalized_a, normalized_b, autojunk=False).ratio()
    overlap = token_overlap(normalized_a, normalized_b)
    return 0.65 * sequence + 0.35 * overlap


def time_overlap(
    start_a: float | None,
    end_a: float | None,
    start_b: float | None,
    end_b: float | None,
) -> float:
    if None in (start_a, end_a, start_b, end_b):
        return 0.0
    assert start_a is not None and end_a is not None
    assert start_b is not None and end_b is not None
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
    return any(
        segment.start is not None and segment.end is not None
        for segment in segments
    )


def _turn_alignment_text(turn: Turn) -> str:
    """Return the strongest available text anchor for an existing turn."""
    return (
        turn.model_text.strip()
        or turn.quality_target_text.strip()
        or turn.final_text.strip()
        or turn.zoom_text.strip()
        or turn.chatgpt_text.strip()
        or turn.gold_text.strip()
    )


def _source_chunks(source: list[TranscriptSegment]) -> list[_SourceChunk]:
    chunks: list[_SourceChunk] = []
    for segment_index, segment in enumerate(source):
        raw_chunks = segment.text.split()
        if not raw_chunks:
            continue
        has_valid_time = (
            segment.start is not None
            and segment.end is not None
            and segment.end >= segment.start
        )
        duration = (
            segment.end - segment.start
            if has_valid_time
            and segment.start is not None
            and segment.end is not None
            else 0.0
        )
        for chunk_index, text in enumerate(raw_chunks):
            midpoint = None
            if has_valid_time and segment.start is not None:
                midpoint = segment.start + duration * (
                    (chunk_index + 0.5) / len(raw_chunks)
                )
            chunks.append(
                _SourceChunk(
                    text=text,
                    segment_index=segment_index,
                    midpoint=midpoint,
                    speaker=segment.speaker,
                    normalized_words=tuple(words(text)),
                )
            )
    return chunks


def _nearest_target_owner(
    target_owners: list[int],
    target_position: int,
) -> int | None:
    if not target_owners:
        return None
    if target_position <= 0:
        return target_owners[0]
    if target_position >= len(target_owners):
        return target_owners[-1]
    before = target_owners[target_position - 1]
    after = target_owners[target_position]
    return before if before == after else after


def _lexical_word_assignments(
    source_words: list[str],
    target_words: list[str],
    target_owners: list[int],
) -> tuple[list[int | None], list[bool]]:
    """Map source words to target turns while allowing arbitrary boundaries."""
    assignments: list[int | None] = [None] * len(source_words)
    exact: list[bool] = [False] * len(source_words)
    if not source_words or not target_words:
        return assignments, exact

    matcher = SequenceMatcher(
        None,
        source_words,
        target_words,
        autojunk=False,
    )
    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        source_length = source_end - source_start
        target_length = target_end - target_start
        if tag == "equal":
            for offset in range(source_length):
                assignments[source_start + offset] = target_owners[target_start + offset]
                exact[source_start + offset] = True
            continue

        if source_length <= 0:
            continue
        if target_length <= 0:
            owner = _nearest_target_owner(target_owners, target_start)
            for source_index in range(source_start, source_end):
                assignments[source_index] = owner
            continue

        # A replace block may cross a sentence/turn boundary. Distribute its
        # words monotonically over the corresponding target block instead of
        # forcing the complete source segment onto one turn.
        for offset in range(source_length):
            relative = (offset + 0.5) / source_length
            target_offset = min(
                target_length - 1,
                int(relative * target_length),
            )
            assignments[source_start + offset] = target_owners[
                target_start + target_offset
            ]

    return assignments, exact


def _timed_turn_index(
    turns: list[Turn],
) -> tuple[list[float], list[tuple[int, float, float]]]:
    timed = [
        (index, float(turn.start), float(turn.end))
        for index, turn in enumerate(turns)
        if turn.start is not None and turn.end is not None
    ]
    timed.sort(key=lambda item: (item[1], item[2], item[0]))
    return [item[1] for item in timed], timed


def _turn_for_midpoint(
    midpoint: float | None,
    starts: list[float],
    timed_turns: list[tuple[int, float, float]],
) -> int | None:
    if midpoint is None or not timed_turns:
        return None

    position = bisect_right(starts, midpoint)
    candidate_positions = range(
        max(0, position - 3),
        min(len(timed_turns), position + 2),
    )
    containing: list[tuple[float, int]] = []
    nearest: list[tuple[float, int]] = []
    for candidate_position in candidate_positions:
        turn_index, start, end = timed_turns[candidate_position]
        center = (start + end) / 2.0
        if start <= midpoint <= end:
            containing.append((abs(midpoint - center), turn_index))
        distance = 0.0 if start <= midpoint <= end else min(
            abs(midpoint - start),
            abs(midpoint - end),
        )
        nearest.append((distance, turn_index))
    if containing:
        return min(containing)[1]
    return min(nearest)[1] if nearest else None


def _speaker_matches(chunk: _SourceChunk, turn: Turn) -> bool:
    source_speaker = normalize_for_comparison(chunk.speaker)
    turn_speaker = normalize_for_comparison(turn.speaker_raw)
    unknown = {"", "unknown", "speaker", "unmapped"}
    return (
        source_speaker not in unknown
        and turn_speaker not in unknown
        and source_speaker == turn_speaker
    )


def _chunk_owners(
    turns: list[Turn],
    chunks: list[_SourceChunk],
) -> list[int]:
    target_words: list[str] = []
    target_owners: list[int] = []
    for turn_index, turn in enumerate(turns):
        anchor_words = words(_turn_alignment_text(turn))
        target_words.extend(anchor_words)
        target_owners.extend([turn_index] * len(anchor_words))

    source_words: list[str] = []
    source_word_chunks: list[int] = []
    for chunk_index, chunk in enumerate(chunks):
        for word in chunk.normalized_words:
            source_words.append(word)
            source_word_chunks.append(chunk_index)

    word_assignments, exact_assignments = _lexical_word_assignments(
        source_words,
        target_words,
        target_owners,
    )
    lexical_votes: list[Counter[int]] = [Counter() for _ in chunks]
    for source_word_index, owner in enumerate(word_assignments):
        if owner is None:
            continue
        chunk_index = source_word_chunks[source_word_index]
        lexical_votes[chunk_index][owner] += (
            4 if exact_assignments[source_word_index] else 1
        )

    starts, timed_turns = _timed_turn_index(turns)
    owners: list[int] = []
    previous_owner = 0
    for chunk_index, chunk in enumerate(chunks):
        scores: Counter[int] = Counter(lexical_votes[chunk_index])
        timed_owner = _turn_for_midpoint(
            chunk.midpoint,
            starts,
            timed_turns,
        )
        if timed_owner is not None:
            scores[timed_owner] += 3

        # Speaker evidence refines the already plausible lexical/time owners.
        # Scanning every turn for every word chunk would make long projects
        # quadratic, so unrelated turns are intentionally not considered here.
        plausible_owners = set(scores)
        plausible_owners.add(previous_owner)
        for turn_index in plausible_owners:
            if _speaker_matches(chunk, turns[turn_index]):
                scores[turn_index] += 2

        if scores:
            selected = max(
                scores,
                key=lambda owner: (
                    scores[owner],
                    -abs(owner - previous_owner),
                    -owner,
                ),
            )
        elif owners:
            selected = previous_owner
        else:
            selected = 0

        # Transcript order is monotonic even when sentence boundaries differ.
        # This also prevents an isolated timestamp or speaker-label anomaly from
        # moving later words back into an earlier turn.
        selected = max(previous_owner, min(len(turns) - 1, selected))
        owners.append(selected)
        previous_owner = selected

    return owners


def _has_temporal_support(
    turns: list[Turn],
    source: list[TranscriptSegment],
) -> bool:
    turn_ranges = [
        (turn.start, turn.end)
        for turn in turns
        if turn.start is not None and turn.end is not None
    ]
    source_ranges = [
        (segment.start, segment.end)
        for segment in source
        if segment.start is not None and segment.end is not None
    ]
    if not turn_ranges or not source_ranges:
        return False
    turn_start = min(start for start, _end in turn_ranges)
    turn_end = max(end for _start, end in turn_ranges)
    source_start = min(start for start, _end in source_ranges)
    source_end = max(end for _start, end in source_ranges)
    return min(turn_end, source_end) > max(turn_start, source_start)


def _align_source_detailed(
    turns: list[Turn],
    source: list[TranscriptSegment],
) -> _AlignmentDetails:
    empty = _AlignmentDetails(
        texts=[""] * len(turns),
        source_weights=[Counter() for _ in turns],
    )
    if not turns or not source:
        return empty

    chunks = _source_chunks(source)
    if not chunks:
        return empty

    target_text = " ".join(
        text for turn in turns if (text := _turn_alignment_text(turn))
    )
    source_text = " ".join(segment.text for segment in source if segment.text.strip())
    if (
        not _has_temporal_support(turns, source)
        and text_similarity(target_text, source_text) < 0.08
    ):
        return empty

    owners = _chunk_owners(turns, chunks)
    rendered: list[list[str]] = [[] for _ in turns]
    weights: list[Counter[int]] = [Counter() for _ in turns]
    for chunk, owner in zip(chunks, owners):
        rendered[owner].append(chunk.text)
        weights[owner][chunk.segment_index] += max(
            1,
            len(chunk.normalized_words),
        )

    texts = [" ".join(parts).strip() for parts in rendered]
    return _AlignmentDetails(texts=texts, source_weights=weights)


def align_source_to_turns(
    turns: list[Turn],
    source: list[TranscriptSegment],
) -> list[str]:
    """Align source text after normalizing split and combined segment boundaries.

    Every source word chunk is assigned once and only once. Consequently, a
    long imported segment may be split between adjacent turns, while several
    short imported segments may be combined into one turn.
    """
    if not turns:
        return []
    return _align_source_detailed(turns, source).texts


def agreement_score(texts: list[str]) -> float:
    present = [text for text in texts if text.strip()]
    if len(present) < 2:
        return 0.0
    pair_scores: list[float] = []
    for index, first in enumerate(present):
        for second in present[index + 1 :]:
            pair_scores.append(text_similarity(first, second))
    return mean(pair_scores) if pair_scores else 0.0


def align_source_segments_to_turns(
    turns: list[Turn],
    source: list[TranscriptSegment],
) -> list[TranscriptSegment | None]:
    """Return the strongest contributing source segment for each aligned turn.

    A source segment may legitimately support more than one turn after its text
    is split, so segment reuse is intentional. This method is used for speaker
    labels and model confidence, while :func:`align_source_to_turns` supplies the
    segmentation-normalized text.
    """
    if not turns:
        return []
    if not source:
        return [None] * len(turns)

    details = _align_source_detailed(turns, source)
    results: list[TranscriptSegment | None] = []
    for turn, text, weights in zip(turns, details.texts, details.source_weights):
        if not text or not weights:
            results.append(None)
            continue
        best_index = max(
            weights,
            key=lambda index: (
                weights[index],
                time_overlap(
                    turn.start,
                    turn.end,
                    source[index].start,
                    source[index].end,
                ),
                text_similarity(text, source[index].text),
                -index,
            ),
        )
        results.append(source[best_index])
    return results
