"""Feature extraction and selective manual-review scoring."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from statistics import mean

from .alignment import agreement_score, text_similarity
from .models import Turn
from .text_utils import (
    contains_hebrew,
    contains_hesitation_or_repetition,
    contains_self_correction,
    contains_unclear_marker,
    normalize_for_comparison,
    repetition_rate,
    word_difference_rate,
    words,
)

FEATURE_NAMES = (
    "agreement",
    "model_confidence",
    "zoom_model_similarity",
    "chatgpt_model_similarity",
    "zoom_chatgpt_similarity",
    "word_disagreement_rate",
    "source_presence_ratio",
    "speech_rate_normality",
    "snr_normality",
    "volume_normality",
    "overlap_penalty",
    "unclear_penalty",
    "repetition_rate",
    "hebrew_switch",
    "self_correction",
)

QUALITY_LABELS = (
    "Transcript acceptable",
    "Needs minor correction",
    "Needs major correction",
)

# The quality label and the review decision are intentionally separate.
# A near-boundary "minor correction" can be auto-cleared when independent
# transcript sources strongly support the same wording and no hard-risk signal
# is present. This reduces low-value reviews without hiding uncertain turns.
ACCEPTABLE_SCORE = 0.72
MINOR_SCORE = 0.46
CONSENSUS_AUTO_CLEAR_SCORE = 0.64
ML_BOUNDARY_REVIEW_RISK = 0.30

_UNKNOWN_SPEAKERS = {
    "",
    "unknown",
    "unmapped",
    "none",
    "n a",
    "na",
    "not available",
    "speaker",
}


@dataclass(slots=True)
class QualityResult:
    score: float
    label: str
    manual_review: bool
    features: dict[str, float]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def extract_features(turn: Turn) -> dict[str, float]:
    texts = [turn.zoom_text, turn.chatgpt_text, turn.model_text]
    present = [text for text in texts if text.strip()]
    agreement = agreement_score(present)
    model_confidence = (
        turn.model_confidence
        if turn.model_confidence is not None
        else agreement
    )
    zoom_model = (
        text_similarity(turn.zoom_text, turn.model_text)
        if turn.zoom_text.strip()
        else 0.0
    )
    chatgpt_model = (
        text_similarity(turn.chatgpt_text, turn.model_text)
        if turn.chatgpt_text.strip()
        else 0.0
    )
    zoom_chatgpt = (
        text_similarity(turn.zoom_text, turn.chatgpt_text)
        if turn.zoom_text.strip() and turn.chatgpt_text.strip()
        else 0.0
    )
    source_presence_ratio = len(present) / 3.0
    pair_differences: list[float] = []
    for index, first in enumerate(present):
        for second in present[index + 1 :]:
            pair_differences.append(word_difference_rate(first, second))
    word_disagreement = mean(pair_differences) if pair_differences else 1.0

    if turn.speech_rate_wpm is None:
        speech_rate_normality = 0.55
    else:
        # Broad range because non-native speakers and disfluencies are expected.
        speech_rate_normality = _clamp(
            1.0 - abs(turn.speech_rate_wpm - 125.0) / 180.0
        )

    if turn.noise_snr_db is None:
        snr_normality = 0.55
    else:
        snr_normality = _clamp((turn.noise_snr_db - 3.0) / 27.0)

    if turn.volume_dbfs is None:
        volume_normality = 0.55
    else:
        volume_normality = _clamp(
            1.0 - abs(turn.volume_dbfs + 24.0) / 30.0
        )

    final_or_model = turn.final_text or turn.model_text
    features = {
        "agreement": agreement,
        "model_confidence": _clamp(model_confidence),
        "zoom_model_similarity": zoom_model,
        "chatgpt_model_similarity": chatgpt_model,
        "zoom_chatgpt_similarity": zoom_chatgpt,
        "word_disagreement_rate": word_disagreement,
        "source_presence_ratio": source_presence_ratio,
        "speech_rate_normality": speech_rate_normality,
        "snr_normality": snr_normality,
        "volume_normality": volume_normality,
        "overlap_penalty": 1.0 if turn.overlapping_speech else 0.0,
        "unclear_penalty": (
            1.0 if contains_unclear_marker(final_or_model) else 0.0
        ),
        "repetition_rate": repetition_rate(final_or_model),
        "hebrew_switch": 1.0 if contains_hebrew(final_or_model) else 0.0,
        "self_correction": (
            1.0 if contains_self_correction(final_or_model) else 0.0
        ),
    }
    return features


def _source_texts(turn: Turn) -> list[str]:
    return [
        text
        for text in (turn.zoom_text, turn.chatgpt_text, turn.model_text)
        if text.strip()
    ]


def _has_exact_majority(turn: Turn) -> bool:
    normalized = [
        normalize_for_comparison(text)
        for text in _source_texts(turn)
    ]
    counts = Counter(text for text in normalized if text)
    return bool(counts) and max(counts.values()) >= 2


def _has_strong_consensus(
    turn: Turn,
    features: dict[str, float],
) -> bool:
    """Return True only when multiple sources provide credible support.

    Exact agreement between any two independent source slots is treated as a
    majority vote. Near-agreement requires stricter similarity and word-level
    limits, especially when only two sources are present.
    """

    source_count = len(_source_texts(turn))
    if source_count < 2:
        return False
    if _has_exact_majority(turn):
        return True

    agreement = features["agreement"]
    disagreement = features["word_disagreement_rate"]
    if source_count == 2:
        return agreement >= 0.88 and disagreement <= 0.20
    return agreement >= 0.78 and disagreement <= 0.35


def _has_hard_review_reason(
    turn: Turn,
    features: dict[str, float],
) -> bool:
    text = (turn.final_text or turn.model_text).strip()
    if not text:
        return True
    if features["overlap_penalty"] > 0.0:
        return True
    if features["unclear_penalty"] > 0.0:
        return True
    speaker = normalize_for_comparison(turn.speaker)
    if speaker in _UNKNOWN_SPEAKERS:
        return True
    return False


def _rule_review_required(
    turn: Turn,
    score: float,
    features: dict[str, float],
) -> bool:
    if _has_hard_review_reason(turn, features):
        return True
    if score >= ACCEPTABLE_SCORE:
        return False
    return not (
        score >= CONSENSUS_AUTO_CLEAR_SCORE
        and _has_strong_consensus(turn, features)
    )


def rule_based_quality(turn: Turn) -> QualityResult:
    features = extract_features(turn)
    positive = (
        0.29 * features["agreement"]
        + 0.16 * features["model_confidence"]
        + 0.10 * features["source_presence_ratio"]
        + 0.10
        * max(
            features["zoom_model_similarity"],
            features["chatgpt_model_similarity"],
        )
        + 0.07 * (1.0 - features["word_disagreement_rate"])
        + 0.08 * features["speech_rate_normality"]
        + 0.07 * features["snr_normality"]
        + 0.05 * features["volume_normality"]
    )
    penalties = (
        0.12 * features["overlap_penalty"]
        + 0.18 * features["unclear_penalty"]
        + 0.04 * min(1.0, features["repetition_rate"] * 5.0)
    )
    score = _clamp(positive - penalties)
    if score >= ACCEPTABLE_SCORE:
        label = QUALITY_LABELS[0]
    elif score >= MINOR_SCORE:
        label = QUALITY_LABELS[1]
    else:
        label = QUALITY_LABELS[2]
    manual_review = _rule_review_required(turn, score, features)
    return QualityResult(score, label, manual_review, features)


def apply_detected_features(turn: Turn) -> None:
    text = turn.final_text or turn.model_text
    turn.hebrew_switch = contains_hebrew(text)
    turn.hesitation_or_repetition = contains_hesitation_or_repetition(text)
    turn.self_correction = contains_self_correction(text)
    turn.unclear_speech = contains_unclear_marker(text)
    duration = turn.duration()
    turn.speech_rate_wpm = (
        len(words(text)) / duration * 60.0 if duration > 0 else None
    )


def _validated_probabilities(raw: object) -> list[float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    probabilities: list[float] = []
    for value in raw:
        try:
            probability = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(probability) or probability < 0.0:
            return None
        probabilities.append(probability)
    total = sum(probabilities)
    if total <= 0.0:
        return None
    return [value / total for value in probabilities]


def _ml_review_required(
    turn: Turn,
    features: dict[str, float],
    probabilities: list[float],
    predicted: int,
    score: float,
) -> bool:
    if _has_hard_review_reason(turn, features):
        return True
    if predicted == 0:
        return False
    if predicted == 2:
        return True

    # Class 1 is auto-cleared only at the boundary with class 0. A confident
    # minor-correction prediction still goes to review.
    review_risk = probabilities[2] + 0.5 * probabilities[1]
    return not (
        score >= CONSENSUS_AUTO_CLEAR_SCORE
        and review_risk <= ML_BOUNDARY_REVIEW_RISK
        and _has_strong_consensus(turn, features)
    )


def score_turn(
    turn: Turn,
    predictor: object | None = None,
) -> QualityResult:
    apply_detected_features(turn)
    features = extract_features(turn)
    if predictor is not None and hasattr(predictor, "predict_proba"):
        vector = [features[name] for name in FEATURE_NAMES]
        probabilities = _validated_probabilities(
            predictor.predict_proba(vector)
        )
        if probabilities is not None:
            predicted = max(
                range(3),
                key=probabilities.__getitem__,
            )
            label = QUALITY_LABELS[predicted]
            # Quality score is high when the model predicts the acceptable class.
            score = probabilities[0] + 0.5 * probabilities[1]
            manual_review = _ml_review_required(
                turn,
                features,
                probabilities,
                predicted,
                score,
            )
            return QualityResult(
                score,
                label,
                manual_review,
                features,
            )
    return rule_based_quality(turn)


def update_turn_quality(
    turn: Turn,
    predictor: object | None = None,
    preserve_manual_choice: bool = False,
) -> None:
    previous_review = turn.manual_review
    result = score_turn(turn, predictor)
    turn.agreement_score = result.features["agreement"]
    turn.quality_score = result.score
    turn.quality_label = result.label
    if not preserve_manual_choice:
        turn.manual_review = result.manual_review
    else:
        turn.manual_review = previous_review
