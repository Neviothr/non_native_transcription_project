"""Machine-learned, conservative transcript source selection.

The enhancer does not generate or grammar-correct text.  It learns, from aligned
Gold Standard examples, which available transcript source is most likely to be
closest to the spoken turn.  At inference time it selects one of the aligned
Whisper, ChatGPT, or Zoom candidates, preserving the selected source verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .alignment import text_similarity
from .models import Turn
from .text_utils import (
    contains_hebrew,
    contains_self_correction,
    contains_unclear_marker,
    repetition_rate,
    words,
)

SOURCE_KEYS = ("model", "chatgpt", "zoom")
SOURCE_ATTRIBUTES = {
    "model": "model_text",
    "chatgpt": "chatgpt_text",
    "zoom": "zoom_text",
}
SOURCE_DISPLAY_NAMES = {
    "model": "Whisper",
    "chatgpt": "ChatGPT",
    "zoom": "Zoom",
}

ENHANCEMENT_FEATURE_NAMES = (
    "model_present",
    "chatgpt_present",
    "zoom_present",
    "source_presence_ratio",
    "model_confidence",
    "model_chatgpt_similarity",
    "model_zoom_similarity",
    "chatgpt_zoom_similarity",
    "model_consensus",
    "chatgpt_consensus",
    "zoom_consensus",
    "model_length_ratio",
    "chatgpt_length_ratio",
    "zoom_length_ratio",
    "model_median_length_ratio",
    "chatgpt_median_length_ratio",
    "zoom_median_length_ratio",
    "model_repetition_rate",
    "chatgpt_repetition_rate",
    "zoom_repetition_rate",
    "model_unclear_marker",
    "chatgpt_unclear_marker",
    "zoom_unclear_marker",
    "model_hebrew_switch",
    "chatgpt_hebrew_switch",
    "zoom_hebrew_switch",
    "model_self_correction",
    "chatgpt_self_correction",
    "zoom_self_correction",
)


@dataclass(slots=True)
class EnhancementSelection:
    text: str
    source_key: str
    source_name: str
    confidence: float | None
    method: str


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def source_texts(turn: Turn) -> dict[str, str]:
    return {
        key: " ".join(str(getattr(turn, attribute, "")).split()).strip()
        for key, attribute in SOURCE_ATTRIBUTES.items()
    }


def available_source_keys(turn: Turn) -> tuple[str, ...]:
    texts = source_texts(turn)
    return tuple(key for key in SOURCE_KEYS if texts[key])


def _pair_similarity(texts: dict[str, str], first: str, second: str) -> float:
    if not texts[first] or not texts[second]:
        return 0.0
    return text_similarity(texts[first], texts[second])


def _consensus_score(texts: dict[str, str], source_key: str) -> float:
    text = texts[source_key]
    if not text:
        return 0.0
    comparisons = [
        text_similarity(text, texts[other])
        for other in SOURCE_KEYS
        if other != source_key and texts[other]
    ]
    return sum(comparisons) / len(comparisons) if comparisons else 0.5


def extract_enhancement_features(turn: Turn) -> dict[str, float]:
    """Extract fixed-size features without using Gold or editable final text."""
    texts = source_texts(turn)
    present = [key for key in SOURCE_KEYS if texts[key]]
    word_counts = {key: len(words(texts[key])) for key in SOURCE_KEYS}
    positive_counts = [word_counts[key] for key in present]
    maximum_count = max(positive_counts, default=1)
    median_count = float(median(positive_counts)) if positive_counts else 1.0

    features: dict[str, float] = {
        "model_present": 1.0 if texts["model"] else 0.0,
        "chatgpt_present": 1.0 if texts["chatgpt"] else 0.0,
        "zoom_present": 1.0 if texts["zoom"] else 0.0,
        "source_presence_ratio": len(present) / len(SOURCE_KEYS),
        "model_confidence": _clamp(
            turn.model_confidence if turn.model_confidence is not None else 0.5
        ),
        "model_chatgpt_similarity": _pair_similarity(texts, "model", "chatgpt"),
        "model_zoom_similarity": _pair_similarity(texts, "model", "zoom"),
        "chatgpt_zoom_similarity": _pair_similarity(texts, "chatgpt", "zoom"),
    }

    for key in SOURCE_KEYS:
        text = texts[key]
        count = word_counts[key]
        features[f"{key}_consensus"] = _consensus_score(texts, key)
        features[f"{key}_length_ratio"] = count / maximum_count if text else 0.0
        features[f"{key}_median_length_ratio"] = (
            _clamp(count / max(1.0, median_count), 0.0, 2.0) / 2.0 if text else 0.0
        )
        features[f"{key}_repetition_rate"] = _clamp(repetition_rate(text) * 5.0) if text else 0.0
        features[f"{key}_unclear_marker"] = 1.0 if text and contains_unclear_marker(text) else 0.0
        features[f"{key}_hebrew_switch"] = 1.0 if text and contains_hebrew(text) else 0.0
        features[f"{key}_self_correction"] = 1.0 if text and contains_self_correction(text) else 0.0

    return features


def enhancement_feature_vector(turn: Turn) -> list[float]:
    features = extract_enhancement_features(turn)
    return [features[name] for name in ENHANCEMENT_FEATURE_NAMES]


def _consensus_fallback(turn: Turn) -> EnhancementSelection:
    texts = source_texts(turn)
    available = [key for key in SOURCE_KEYS if texts[key]]
    if not available:
        return EnhancementSelection("", "", "", None, "none")
    if len(available) == 1:
        key = available[0]
        return EnhancementSelection(
            texts[key], key, SOURCE_DISPLAY_NAMES[key], 1.0, "single_source"
        )

    scores = {key: _consensus_score(texts, key) for key in available}
    # Prefer Whisper when candidates are effectively tied, preserving the raw
    # local-model wording unless another source has stronger consensus support.
    priority = {"model": 2, "chatgpt": 1, "zoom": 0}
    selected = max(available, key=lambda key: (scores[key], priority[key]))
    ordered = sorted((scores[key] for key in available), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    confidence = _clamp(0.5 + margin / 2.0)
    return EnhancementSelection(
        texts[selected],
        selected,
        SOURCE_DISPLAY_NAMES[selected],
        confidence,
        "consensus_fallback",
    )


def select_enhanced_transcript(
    turn: Turn,
    predictor: object | None = None,
) -> EnhancementSelection:
    """Select one aligned source, using ML when a compatible model is available."""
    texts = source_texts(turn)
    available = [key for key in SOURCE_KEYS if texts[key]]
    if len(available) <= 1 or predictor is None or not hasattr(predictor, "predict_proba"):
        return _consensus_fallback(turn)

    try:
        raw_probabilities = list(predictor.predict_proba(enhancement_feature_vector(turn)))
    except (TypeError, ValueError, IndexError, AttributeError):
        return _consensus_fallback(turn)
    if len(raw_probabilities) != len(SOURCE_KEYS):
        return _consensus_fallback(turn)

    available_indexes = [SOURCE_KEYS.index(key) for key in available]
    probability_total = sum(max(0.0, raw_probabilities[index]) for index in available_indexes)
    if probability_total <= 0:
        return _consensus_fallback(turn)
    normalized = {
        key: max(0.0, raw_probabilities[SOURCE_KEYS.index(key)]) / probability_total
        for key in available
    }
    priority = {"model": 2, "chatgpt": 1, "zoom": 0}
    selected = max(available, key=lambda key: (normalized[key], priority[key]))
    return EnhancementSelection(
        texts[selected],
        selected,
        SOURCE_DISPLAY_NAMES[selected],
        normalized[selected],
        "machine_learning",
    )
