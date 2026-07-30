"""Feature extraction and review-priority scoring."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .alignment import agreement_score, text_similarity
from .models import Turn
from .text_utils import (
    contains_hebrew,
    contains_hesitation_or_repetition,
    contains_self_correction,
    contains_unclear_marker,
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

QUALITY_LABELS = ("Transcript acceptable", "Needs minor correction", "Needs major correction")


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
    model_confidence = turn.model_confidence if turn.model_confidence is not None else agreement
    zoom_model = text_similarity(turn.zoom_text, turn.model_text) if turn.zoom_text.strip() else 0.0
    chatgpt_model = text_similarity(turn.chatgpt_text, turn.model_text) if turn.chatgpt_text.strip() else 0.0
    zoom_chatgpt = text_similarity(turn.zoom_text, turn.chatgpt_text) if turn.zoom_text.strip() and turn.chatgpt_text.strip() else 0.0
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
        speech_rate_normality = _clamp(1.0 - abs(turn.speech_rate_wpm - 125.0) / 180.0)

    if turn.noise_snr_db is None:
        snr_normality = 0.55
    else:
        snr_normality = _clamp((turn.noise_snr_db - 3.0) / 27.0)

    if turn.volume_dbfs is None:
        volume_normality = 0.55
    else:
        volume_normality = _clamp(1.0 - abs(turn.volume_dbfs + 24.0) / 30.0)

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
        "unclear_penalty": 1.0 if contains_unclear_marker(final_or_model) else 0.0,
        "repetition_rate": repetition_rate(final_or_model),
        "hebrew_switch": 1.0 if contains_hebrew(final_or_model) else 0.0,
        "self_correction": 1.0 if contains_self_correction(final_or_model) else 0.0,
    }
    return features


def rule_based_quality(turn: Turn) -> QualityResult:
    features = extract_features(turn)
    positive = (
        0.29 * features["agreement"]
        + 0.16 * features["model_confidence"]
        + 0.10 * features["source_presence_ratio"]
        + 0.10 * max(features["zoom_model_similarity"], features["chatgpt_model_similarity"])
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
    if score >= 0.72:
        label = QUALITY_LABELS[0]
        manual_review = False
    elif score >= 0.46:
        label = QUALITY_LABELS[1]
        manual_review = True
    else:
        label = QUALITY_LABELS[2]
        manual_review = True
    return QualityResult(score, label, manual_review, features)


def apply_detected_features(turn: Turn) -> None:
    text = turn.final_text or turn.model_text
    turn.hebrew_switch = contains_hebrew(text)
    turn.hesitation_or_repetition = contains_hesitation_or_repetition(text)
    turn.self_correction = contains_self_correction(text)
    turn.unclear_speech = contains_unclear_marker(text)
    duration = turn.duration()
    turn.speech_rate_wpm = len(words(text)) / duration * 60.0 if duration > 0 else None


def score_turn(turn: Turn, predictor: object | None = None) -> QualityResult:
    apply_detected_features(turn)
    features = extract_features(turn)
    if predictor is not None and hasattr(predictor, "predict_proba"):
        vector = [features[name] for name in FEATURE_NAMES]
        probabilities = predictor.predict_proba(vector)
        if len(probabilities) == 3:
            predicted = max(range(3), key=probabilities.__getitem__)
            label = QUALITY_LABELS[predicted]
            # Quality score is high when the model predicts the acceptable class.
            score = probabilities[0] + 0.5 * probabilities[1]
            return QualityResult(score, label, predicted != 0, features)
    return rule_based_quality(turn)


def update_turn_quality(turn: Turn, predictor: object | None = None, preserve_manual_choice: bool = False) -> None:
    previous_review = turn.manual_review
    result = score_turn(turn, predictor)
    turn.agreement_score = result.features["agreement"]
    turn.quality_score = result.score
    turn.quality_label = result.label
    if not preserve_manual_choice:
        turn.manual_review = result.manual_review
    else:
        turn.manual_review = previous_review