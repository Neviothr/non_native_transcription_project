"""Small pure-Python classifiers for transcript quality comparison.

The implementations are intentionally compact and dependency-free. They are suitable for
course-project datasets, not for large-scale production training.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EPSILON = 1e-12
MAX_MODEL_TRAINING_ROWS = 5_000
MAX_TREE_BOOTSTRAP_ROWS = 2_500
MIN_ADAPTIVE_EPOCHS = 80


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _balanced_class_weights(labels: list[int], classes: int) -> list[float]:
    """Return inverse-frequency weights without amplifying tiny classes excessively."""
    counts = [labels.count(label) for label in range(classes)]
    present = sum(count > 0 for count in counts)
    if present == 0:
        return [1.0] * classes
    total = len(labels)
    weights: list[float] = []
    for count in counts:
        if count == 0:
            weights.append(0.0)
        else:
            raw = total / (present * count)
            weights.append(max(0.35, min(4.0, raw)))
    return weights


@dataclass(slots=True)
class Standardizer:
    means: list[float]
    scales: list[float]

    @classmethod
    def fit(cls, rows: list[list[float]]) -> "Standardizer":
        if not rows:
            raise ValueError("Cannot fit a standardizer without rows.")
        columns = len(rows[0])
        means = [sum(row[index] for row in rows) / len(rows) for index in range(columns)]
        scales: list[float] = []
        for index in range(columns):
            variance = sum((row[index] - means[index]) ** 2 for row in rows) / len(rows)
            scales.append(max(math.sqrt(variance), 1e-6))
        return cls(means, scales)

    def transform(self, row: list[float]) -> list[float]:
        return [(value - mean) / scale for value, mean, scale in zip(row, self.means, self.scales)]


class SoftmaxLogisticRegression:
    name = "Logistic Regression"

    def __init__(self, learning_rate: float = 0.05, epochs: int = 900, l2: float = 0.005) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.standardizer: Standardizer | None = None
        self.weights: list[list[float]] = []
        self.biases: list[float] = []

    def fit(self, rows: list[list[float]], labels: list[int], classes: int = 3) -> None:
        if len(rows) != len(labels) or not rows:
            raise ValueError("Training rows and labels must be non-empty and have matching lengths.")
        self.standardizer = Standardizer.fit(rows)
        transformed = [self.standardizer.transform(row) for row in rows]
        features = len(rows[0])
        self.weights = [[0.0] * features for _ in range(classes)]
        self.biases = [0.0] * classes
        epoch_budget = min(
            self.epochs,
            max(MIN_ADAPTIVE_EPOCHS, 180_000 // max(1, len(rows))),
        )
        class_weights = _balanced_class_weights(labels, classes)
        for _ in range(epoch_budget):
            weight_gradient = [[0.0] * features for _ in range(classes)]
            bias_gradient = [0.0] * classes
            for row, label in zip(transformed, labels):
                sample_weight = class_weights[label]
                logits = [sum(weight * value for weight, value in zip(class_weights, row)) + bias for class_weights, bias in zip(self.weights, self.biases)]
                probabilities = _softmax(logits)
                for class_index in range(classes):
                    error = sample_weight * (
                        probabilities[class_index]
                        - (1.0 if label == class_index else 0.0)
                    )
                    bias_gradient[class_index] += error
                    for feature_index, value in enumerate(row):
                        weight_gradient[class_index][feature_index] += error * value
            count = len(rows)
            for class_index in range(classes):
                self.biases[class_index] -= self.learning_rate * bias_gradient[class_index] / count
                for feature_index in range(features):
                    gradient = weight_gradient[class_index][feature_index] / count + self.l2 * self.weights[class_index][feature_index]
                    self.weights[class_index][feature_index] -= self.learning_rate * gradient

    def predict_proba(self, row: list[float]) -> list[float]:
        if self.standardizer is None or not self.weights:
            raise ValueError("Model has not been trained.")
        transformed = self.standardizer.transform(row)
        logits = [sum(weight * value for weight, value in zip(class_weights, transformed)) + bias for class_weights, bias in zip(self.weights, self.biases)]
        return _softmax(logits)

    def to_dict(self) -> dict[str, Any]:
        assert self.standardizer is not None
        return {
            "type": "logistic",
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "means": self.standardizer.means,
            "scales": self.standardizer.scales,
            "weights": self.weights,
            "biases": self.biases,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SoftmaxLogisticRegression":
        model = cls(float(data.get("learning_rate", 0.05)), int(data.get("epochs", 900)), float(data.get("l2", 0.005)))
        model.standardizer = Standardizer(list(data["means"]), list(data["scales"]))
        model.weights = [list(row) for row in data["weights"]]
        model.biases = list(data["biases"])
        return model


class LinearSVMOVR:
    name = "Linear SVM"

    def __init__(self, learning_rate: float = 0.025, epochs: int = 700, regularization: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.standardizer: Standardizer | None = None
        self.weights: list[list[float]] = []
        self.biases: list[float] = []

    def fit(self, rows: list[list[float]], labels: list[int], classes: int = 3) -> None:
        self.standardizer = Standardizer.fit(rows)
        transformed = [self.standardizer.transform(row) for row in rows]
        features = len(rows[0])
        self.weights = [[0.0] * features for _ in range(classes)]
        self.biases = [0.0] * classes
        order = list(range(len(rows)))
        rng = random.Random(417)
        class_weights = _balanced_class_weights(labels, classes)
        epoch_budget = min(
            self.epochs,
            max(MIN_ADAPTIVE_EPOCHS, 140_000 // max(1, len(rows))),
        )
        for epoch in range(epoch_budget):
            rng.shuffle(order)
            rate = self.learning_rate / (1.0 + epoch * 0.003)
            for row_index in order:
                row = transformed[row_index]
                label = labels[row_index]
                for class_index in range(classes):
                    target = 1.0 if label == class_index else -1.0
                    margin = target * (sum(weight * value for weight, value in zip(self.weights[class_index], row)) + self.biases[class_index])
                    for feature_index in range(features):
                        self.weights[class_index][feature_index] *= 1.0 - rate * self.regularization
                    if margin < 1.0:
                        weighted_rate = rate * class_weights[label]
                        for feature_index, value in enumerate(row):
                            self.weights[class_index][feature_index] += weighted_rate * target * value
                        self.biases[class_index] += weighted_rate * target

    def predict_proba(self, row: list[float]) -> list[float]:
        if self.standardizer is None:
            raise ValueError("Model has not been trained.")
        transformed = self.standardizer.transform(row)
        margins = [sum(weight * value for weight, value in zip(class_weights, transformed)) + bias for class_weights, bias in zip(self.weights, self.biases)]
        return _softmax(margins)

    def to_dict(self) -> dict[str, Any]:
        assert self.standardizer is not None
        return {
            "type": "svm",
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "regularization": self.regularization,
            "means": self.standardizer.means,
            "scales": self.standardizer.scales,
            "weights": self.weights,
            "biases": self.biases,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinearSVMOVR":
        model = cls(float(data.get("learning_rate", 0.025)), int(data.get("epochs", 700)), float(data.get("regularization", 0.01)))
        model.standardizer = Standardizer(list(data["means"]), list(data["scales"]))
        model.weights = [list(row) for row in data["weights"]]
        model.biases = list(data["biases"])
        return model


@dataclass(slots=True)
class TreeNode:
    probabilities: list[float]
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    def predict(self, row: list[float]) -> list[float]:
        if self.feature is None or self.left is None or self.right is None or self.threshold is None:
            return self.probabilities
        return self.left.predict(row) if row[self.feature] <= self.threshold else self.right.predict(row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probabilities": self.probabilities,
            "feature": self.feature,
            "threshold": self.threshold,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreeNode":
        node = cls(list(data["probabilities"]), data.get("feature"), data.get("threshold"))
        if data.get("left"):
            node.left = cls.from_dict(data["left"])
        if data.get("right"):
            node.right = cls.from_dict(data["right"])
        return node


class RandomForestClassifier:
    name = "Random Forest"

    def __init__(
        self,
        trees: int = 31,
        max_depth: int = 5,
        min_samples: int = 3,
        feature_fraction: float = 0.6,
    ) -> None:
        self.trees = trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.feature_fraction = feature_fraction
        self.forest: list[TreeNode] = []
        self.classes = 3
        self.class_weights: list[float] = [1.0, 1.0, 1.0]

    def fit(
        self,
        rows: list[list[float]],
        labels: list[int],
        classes: int = 3,
    ) -> None:
        if not rows:
            raise ValueError("No training rows supplied.")
        self.classes = classes
        self.class_weights = _balanced_class_weights(labels, classes)
        self.forest = []
        rng = random.Random(9127)
        bootstrap_size = min(len(rows), MAX_TREE_BOOTSTRAP_ROWS)
        for _ in range(self.trees):
            indexes = [rng.randrange(len(rows)) for _ in range(bootstrap_size)]
            sample_rows = [rows[index] for index in indexes]
            sample_labels = [labels[index] for index in indexes]
            self.forest.append(
                self._build_tree(sample_rows, sample_labels, 0, rng)
            )

    def _distribution(self, labels: list[int]) -> list[float]:
        counts = [0.5] * self.classes
        for label in labels:
            counts[label] += self.class_weights[label]
        total = sum(counts)
        return [count / total for count in counts]

    def _weighted_mass(self, labels: list[int]) -> float:
        return sum(self.class_weights[label] for label in labels)

    def _gini(self, labels: list[int]) -> float:
        probabilities = self._distribution(labels)
        return 1.0 - sum(
            probability * probability for probability in probabilities
        )

    def _build_tree(
        self,
        rows: list[list[float]],
        labels: list[int],
        depth: int,
        rng: random.Random,
    ) -> TreeNode:
        node = TreeNode(self._distribution(labels))
        if (
            depth >= self.max_depth
            or len(rows) < self.min_samples * 2
            or len(set(labels)) <= 1
        ):
            return node

        feature_count = len(rows[0])
        selected_count = max(1, int(feature_count * self.feature_fraction))
        selected_features = rng.sample(range(feature_count), selected_count)
        best_gain = 0.0
        best_split: tuple[int, float, list[int], list[int]] | None = None
        parent_gini = self._gini(labels)
        parent_mass = max(self._weighted_mass(labels), EPSILON)

        for feature in selected_features:
            values = sorted(set(row[feature] for row in rows))
            if len(values) < 2:
                continue
            thresholds = [
                (first + second) / 2.0
                for first, second in zip(values, values[1:])
            ]
            if len(thresholds) > 12:
                step = max(1, len(thresholds) // 12)
                thresholds = thresholds[::step]

            for threshold in thresholds:
                left = [
                    index
                    for index, row in enumerate(rows)
                    if row[feature] <= threshold
                ]
                right = [
                    index
                    for index, row in enumerate(rows)
                    if row[feature] > threshold
                ]
                if (
                    len(left) < self.min_samples
                    or len(right) < self.min_samples
                ):
                    continue
                left_labels = [labels[index] for index in left]
                right_labels = [labels[index] for index in right]
                weighted = (
                    self._weighted_mass(left_labels) * self._gini(left_labels)
                    + self._weighted_mass(right_labels)
                    * self._gini(right_labels)
                ) / parent_mass
                gain = parent_gini - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold, left, right)

        if best_split is None:
            return node

        feature, threshold, left_indexes, right_indexes = best_split
        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(
            [rows[index] for index in left_indexes],
            [labels[index] for index in left_indexes],
            depth + 1,
            rng,
        )
        node.right = self._build_tree(
            [rows[index] for index in right_indexes],
            [labels[index] for index in right_indexes],
            depth + 1,
            rng,
        )
        return node

    def predict_proba(self, row: list[float]) -> list[float]:
        if not self.forest:
            raise ValueError("Model has not been trained.")
        totals = [0.0] * self.classes
        for tree in self.forest:
            probabilities = tree.predict(row)
            for index, probability in enumerate(probabilities):
                totals[index] += probability
        return [value / len(self.forest) for value in totals]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "random_forest",
            "trees": self.trees,
            "max_depth": self.max_depth,
            "min_samples": self.min_samples,
            "feature_fraction": self.feature_fraction,
            "classes": self.classes,
            "class_weights": self.class_weights,
            "forest": [tree.to_dict() for tree in self.forest],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "RandomForestClassifier":
        model = cls(
            int(data.get("trees", 31)),
            int(data.get("max_depth", 5)),
            int(data.get("min_samples", 3)),
            float(data.get("feature_fraction", 0.6)),
        )
        model.classes = int(data.get("classes", 3))
        model.class_weights = [
            float(value)
            for value in data.get(
                "class_weights",
                [1.0] * model.classes,
            )
        ]
        model.forest = [
            TreeNode.from_dict(item) for item in data.get("forest", [])
        ]
        return model


class WeightedEnsembleClassifier:
    """Soft-voting ensemble weighted by repeated validation performance."""

    name = "Weighted Ensemble"

    def __init__(
        self,
        models: list[object],
        weights: list[float],
    ) -> None:
        if not models or len(models) != len(weights):
            raise ValueError(
                "Ensemble models and weights must be non-empty and aligned."
            )
        cleaned = [max(0.0, float(weight)) for weight in weights]
        total = sum(cleaned)
        if total <= 0.0:
            cleaned = [1.0] * len(models)
            total = float(len(models))
        self.models = models
        self.weights = [weight / total for weight in cleaned]

    def predict_proba(self, row: list[float]) -> list[float]:
        totals = [0.0, 0.0, 0.0]
        for model, weight in zip(self.models, self.weights):
            probabilities = model.predict_proba(row)  # type: ignore[attr-defined]
            if len(probabilities) != 3:
                raise ValueError(
                    "An ensemble member returned an invalid probability vector."
                )
            for index, probability in enumerate(probabilities):
                totals[index] += weight * float(probability)
        total = sum(totals)
        if total <= 0.0:
            return [1.0 / 3.0] * 3
        return [value / total for value in totals]

    def to_dict(self) -> dict[str, Any]:
        payload: list[dict[str, Any]] = []
        for model in self.models:
            if not hasattr(model, "to_dict"):
                raise TypeError("Unsupported ensemble member.")
            payload.append(model.to_dict())  # type: ignore[attr-defined]
        return {
            "type": "weighted_ensemble",
            "weights": self.weights,
            "models": payload,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "WeightedEnsembleClassifier":
        raw_models = data.get("models", [])
        if not isinstance(raw_models, list):
            raise ValueError("Invalid ensemble model list.")
        models = [_model_from_dict(item) for item in raw_models]
        weights = [float(value) for value in data.get("weights", [])]
        return cls(models, weights)


def _labels_present(actual: list[int], classes: int) -> list[int]:
    return [
        label
        for label in range(classes)
        if any(value == label for value in actual)
    ]


def _macro_f1(
    actual: list[int],
    predicted: list[int],
    classes: int = 3,
) -> float:
    scores: list[float] = []
    for label in _labels_present(actual, classes):
        true_positive = sum(
            1
            for actual_label, predicted_label in zip(actual, predicted)
            if actual_label == label and predicted_label == label
        )
        false_positive = sum(
            1
            for actual_label, predicted_label in zip(actual, predicted)
            if actual_label != label and predicted_label == label
        )
        false_negative = sum(
            1
            for actual_label, predicted_label in zip(actual, predicted)
            if actual_label == label and predicted_label != label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def _balanced_accuracy(
    actual: list[int],
    predicted: list[int],
    classes: int = 3,
) -> float:
    recalls: list[float] = []
    for label in _labels_present(actual, classes):
        positives = sum(value == label for value in actual)
        correct = sum(
            actual_label == label and predicted_label == label
            for actual_label, predicted_label in zip(actual, predicted)
        )
        recalls.append(correct / positives if positives else 0.0)
    return sum(recalls) / len(recalls) if recalls else 0.0


def _selection_score(
    accuracy: float,
    macro_f1: float,
    balanced_accuracy: float,
) -> float:
    return (
        0.70 * macro_f1
        + 0.20 * balanced_accuracy
        + 0.10 * accuracy
    )


def stratified_split(
    rows: list[list[float]],
    labels: list[int],
    test_ratio: float = 0.25,
) -> tuple[list[list[float]], list[int], list[list[float]], list[int]]:
    """Compatibility wrapper returning the first deterministic validation split."""
    train_indexes, test_indexes = _repeated_stratified_indexes(
        labels,
        test_ratio=test_ratio,
        repeats=1,
    )[0]
    return (
        [rows[index] for index in train_indexes],
        [labels[index] for index in train_indexes],
        [rows[index] for index in test_indexes],
        [labels[index] for index in test_indexes],
    )


def _repeated_stratified_indexes(
    labels: list[int],
    *,
    test_ratio: float,
    repeats: int,
) -> list[tuple[list[int], list[int]]]:
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(label, []).append(index)

    splits: list[tuple[list[int], list[int]]] = []
    for repeat in range(repeats):
        rng = random.Random(2026 + repeat * 977)
        train_indexes: list[int] = []
        test_indexes: list[int] = []
        for label in sorted(grouped):
            indexes = list(grouped[label])
            rng.shuffle(indexes)
            if len(indexes) >= 2:
                test_count = max(1, round(len(indexes) * test_ratio))
                test_count = min(len(indexes) - 1, test_count)
            else:
                test_count = 0
            test_indexes.extend(indexes[:test_count])
            train_indexes.extend(indexes[test_count:])

        if not test_indexes:
            test_indexes = list(range(len(labels)))
            train_indexes = list(range(len(labels)))
        rng.shuffle(train_indexes)
        rng.shuffle(test_indexes)
        splits.append((train_indexes, test_indexes))
    return splits


def _balanced_training_sample(
    rows: list[list[float]],
    labels: list[int],
    maximum: int = MAX_MODEL_TRAINING_ROWS,
) -> tuple[list[list[float]], list[int]]:
    """Keep a deterministic, class-balanced subset for very large datasets."""
    if len(rows) <= maximum:
        return rows, labels

    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(label, []).append(index)

    rng = random.Random(7319)
    for indexes in grouped.values():
        rng.shuffle(indexes)

    selected: list[int] = []
    active = sorted(grouped)
    while len(selected) < maximum and active:
        next_active: list[int] = []
        for label in active:
            indexes = grouped[label]
            if indexes and len(selected) < maximum:
                selected.append(indexes.pop())
            if indexes:
                next_active.append(label)
        active = next_active

    rng.shuffle(selected)
    return (
        [rows[index] for index in selected],
        [labels[index] for index in selected],
    )


def _validate_training_data(
    rows: list[list[float]],
    labels: list[int],
) -> None:
    if len(rows) != len(labels):
        raise ValueError("Training rows and labels must have matching lengths.")
    if len(rows) < 9 or len(set(labels)) < 2:
        raise ValueError(
            "At least 9 labeled examples across at least 2 quality classes "
            "are required."
        )
    feature_count = len(rows[0])
    if feature_count == 0:
        raise ValueError("Training rows must contain at least one feature.")
    for row in rows:
        if len(row) != feature_count:
            raise ValueError("All training rows must have the same feature count.")
        if not all(math.isfinite(float(value)) for value in row):
            raise ValueError("Training features must be finite numbers.")
    if any(label not in (0, 1, 2) for label in labels):
        raise ValueError("Quality labels must be 0, 1, or 2.")


def _new_candidate_models(*, validation: bool) -> list[object]:
    return [
        SoftmaxLogisticRegression(),
        LinearSVMOVR(),
        RandomForestClassifier(trees=21 if validation else 31),
    ]


def _evaluate_candidate(
    model_index: int,
    rows: list[list[float]],
    labels: list[int],
    splits: list[tuple[list[int], list[int]]],
) -> tuple[dict[str, float | str | int], list[list[float]]]:
    actual: list[int] = []
    probabilities_by_event: list[list[float]] = []

    for train_indexes, test_indexes in splits:
        model = _new_candidate_models(validation=True)[model_index]
        train_rows = [rows[index] for index in train_indexes]
        train_labels = [labels[index] for index in train_indexes]
        model.fit(train_rows, train_labels, classes=3)  # type: ignore[attr-defined]
        for index in test_indexes:
            probabilities = [
                float(value)
                for value in model.predict_proba(rows[index])  # type: ignore[attr-defined]
            ]
            probabilities_by_event.append(probabilities)
            actual.append(labels[index])

    predicted = [
        max(range(3), key=probabilities.__getitem__)
        for probabilities in probabilities_by_event
    ]
    accuracy = (
        sum(a == p for a, p in zip(actual, predicted)) / len(actual)
    )
    macro_f1 = _macro_f1(actual, predicted)
    balanced_accuracy = _balanced_accuracy(actual, predicted)
    model_name = _new_candidate_models(validation=True)[model_index].name  # type: ignore[attr-defined]
    return (
        {
            "model": model_name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_accuracy,
            "selection_score": _selection_score(
                accuracy,
                macro_f1,
                balanced_accuracy,
            ),
            "validation_repeats": len(splits),
            "validation_predictions": len(actual),
        },
        probabilities_by_event,
    )


def train_and_compare(
    rows: list[list[float]],
    labels: list[int],
) -> tuple[object, list[dict[str, float | str | int]]]:
    """Select a quality model with repeated stratified validation.

    Repeated validation reduces the chance that one lucky split determines the
    active model. Class-weighted learners improve minority-class recall, and a
    soft-voting ensemble is evaluated alongside the three required classifiers.
    """

    _validate_training_data(rows, labels)
    available_rows = len(rows)
    rows, labels = _balanced_training_sample(rows, labels)
    repeats = 3 if len(rows) <= 600 else 2
    splits = _repeated_stratified_indexes(
        labels,
        test_ratio=0.25,
        repeats=repeats,
    )

    results: list[dict[str, float | str | int]] = []
    candidate_probabilities: list[list[list[float]]] = []
    for model_index in range(3):
        result, event_probabilities = _evaluate_candidate(
            model_index,
            rows,
            labels,
            splits,
        )
        result["training_rows"] = len(rows)
        result["available_rows"] = available_rows
        results.append(result)
        candidate_probabilities.append(event_probabilities)

    validation_weights = [
        max(0.05, float(result["selection_score"])) ** 2
        for result in results
    ]
    weight_total = sum(validation_weights)
    validation_weights = [
        weight / weight_total for weight in validation_weights
    ]

    actual_events: list[int] = []
    for _train_indexes, test_indexes in splits:
        actual_events.extend(labels[index] for index in test_indexes)

    ensemble_probabilities: list[list[float]] = []
    event_count = len(actual_events)
    for event_index in range(event_count):
        combined = [0.0, 0.0, 0.0]
        for candidate_index, weight in enumerate(validation_weights):
            probabilities = candidate_probabilities[candidate_index][event_index]
            for class_index in range(3):
                combined[class_index] += weight * probabilities[class_index]
        total = sum(combined)
        ensemble_probabilities.append(
            [value / total for value in combined]
            if total > 0.0
            else [1.0 / 3.0] * 3
        )

    ensemble_predictions = [
        max(range(3), key=probabilities.__getitem__)
        for probabilities in ensemble_probabilities
    ]
    ensemble_accuracy = sum(
        actual == predicted
        for actual, predicted in zip(
            actual_events,
            ensemble_predictions,
        )
    ) / len(actual_events)
    ensemble_macro_f1 = _macro_f1(
        actual_events,
        ensemble_predictions,
    )
    ensemble_balanced_accuracy = _balanced_accuracy(
        actual_events,
        ensemble_predictions,
    )
    results.append(
        {
            "model": WeightedEnsembleClassifier.name,
            "accuracy": ensemble_accuracy,
            "macro_f1": ensemble_macro_f1,
            "balanced_accuracy": ensemble_balanced_accuracy,
            "selection_score": _selection_score(
                ensemble_accuracy,
                ensemble_macro_f1,
                ensemble_balanced_accuracy,
            ),
            "validation_repeats": repeats,
            "validation_predictions": len(actual_events),
            "training_rows": len(rows),
            "available_rows": available_rows,
        }
    )

    final_models = _new_candidate_models(validation=False)
    for model in final_models:
        model.fit(rows, labels, classes=3)  # type: ignore[attr-defined]

    best_result = max(
        results,
        key=lambda item: (
            float(item["selection_score"]),
            float(item["macro_f1"]),
            float(item["accuracy"]),
        ),
    )
    if best_result["model"] == WeightedEnsembleClassifier.name:
        best_model: object = WeightedEnsembleClassifier(
            final_models,
            validation_weights,
        )
    else:
        by_name = {
            model.name: model  # type: ignore[attr-defined]
            for model in final_models
        }
        best_model = by_name[str(best_result["model"])]

    for result in results:
        result["selected"] = int(
            result["model"] == getattr(best_model, "name", "")
        )
    return best_model, results


def save_model(
    model: object,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not hasattr(model, "to_dict"):
        raise TypeError("Unsupported model type.")
    payload = model.to_dict()  # type: ignore[attr-defined]
    if metadata:
        payload["_metadata"] = metadata
    Path(path).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _model_from_dict(data: dict[str, Any]) -> object:
    model_type = data.get("type")
    if model_type == "logistic":
        return SoftmaxLogisticRegression.from_dict(data)
    if model_type == "svm":
        return LinearSVMOVR.from_dict(data)
    if model_type == "random_forest":
        return RandomForestClassifier.from_dict(data)
    if model_type == "weighted_ensemble":
        return WeightedEnsembleClassifier.from_dict(data)
    raise ValueError(f"Unknown model type: {model_type}")


def load_model(
    path: str | Path,
    *,
    expected_metadata: dict[str, Any] | None = None,
) -> object:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The saved model must be a JSON object.")
    if expected_metadata is not None:
        metadata = data.get("_metadata")
        if not isinstance(metadata, dict):
            raise ValueError(
                "The saved quality model predates the current label target. "
                "Retrain it from current Gold examples."
            )
        mismatched = [
            key
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        ]
        if mismatched:
            fields = ", ".join(mismatched)
            raise ValueError(
                "The saved quality model is incompatible with the current "
                f"training schema ({fields}). Retrain it."
            )
    return _model_from_dict(data)
