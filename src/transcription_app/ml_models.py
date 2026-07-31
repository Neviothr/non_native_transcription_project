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
        for _ in range(epoch_budget):
            weight_gradient = [[0.0] * features for _ in range(classes)]
            bias_gradient = [0.0] * classes
            for row, label in zip(transformed, labels):
                logits = [sum(weight * value for weight, value in zip(class_weights, row)) + bias for class_weights, bias in zip(self.weights, self.biases)]
                probabilities = _softmax(logits)
                for class_index in range(classes):
                    error = probabilities[class_index] - (1.0 if label == class_index else 0.0)
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
                        for feature_index, value in enumerate(row):
                            self.weights[class_index][feature_index] += rate * target * value
                        self.biases[class_index] += rate * target

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

    def __init__(self, trees: int = 31, max_depth: int = 5, min_samples: int = 3, feature_fraction: float = 0.6) -> None:
        self.trees = trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.feature_fraction = feature_fraction
        self.forest: list[TreeNode] = []
        self.classes = 3

    def fit(self, rows: list[list[float]], labels: list[int], classes: int = 3) -> None:
        if not rows:
            raise ValueError("No training rows supplied.")
        self.classes = classes
        self.forest = []
        rng = random.Random(9127)
        bootstrap_size = min(len(rows), MAX_TREE_BOOTSTRAP_ROWS)
        for _ in range(self.trees):
            indexes = [rng.randrange(len(rows)) for _ in range(bootstrap_size)]
            sample_rows = [rows[index] for index in indexes]
            sample_labels = [labels[index] for index in indexes]
            self.forest.append(self._build_tree(sample_rows, sample_labels, 0, rng))

    def _distribution(self, labels: list[int]) -> list[float]:
        counts = [1.0] * self.classes
        for label in labels:
            counts[label] += 1.0
        total = sum(counts)
        return [count / total for count in counts]

    def _gini(self, labels: list[int]) -> float:
        probabilities = self._distribution(labels)
        return 1.0 - sum(probability * probability for probability in probabilities)

    def _build_tree(self, rows: list[list[float]], labels: list[int], depth: int, rng: random.Random) -> TreeNode:
        node = TreeNode(self._distribution(labels))
        if depth >= self.max_depth or len(rows) < self.min_samples * 2 or len(set(labels)) <= 1:
            return node
        feature_count = len(rows[0])
        selected_count = max(1, int(feature_count * self.feature_fraction))
        selected_features = rng.sample(range(feature_count), selected_count)
        best_gain = 0.0
        best_split: tuple[int, float, list[int], list[int]] | None = None
        parent_gini = self._gini(labels)
        for feature in selected_features:
            values = sorted(set(row[feature] for row in rows))
            if len(values) < 2:
                continue
            thresholds = [(a + b) / 2.0 for a, b in zip(values, values[1:])]
            if len(thresholds) > 12:
                step = max(1, len(thresholds) // 12)
                thresholds = thresholds[::step]
            for threshold in thresholds:
                left = [index for index, row in enumerate(rows) if row[feature] <= threshold]
                right = [index for index, row in enumerate(rows) if row[feature] > threshold]
                if len(left) < self.min_samples or len(right) < self.min_samples:
                    continue
                weighted = (len(left) * self._gini([labels[index] for index in left]) + len(right) * self._gini([labels[index] for index in right])) / len(rows)
                gain = parent_gini - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold, left, right)
        if best_split is None:
            return node
        feature, threshold, left_indexes, right_indexes = best_split
        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree([rows[index] for index in left_indexes], [labels[index] for index in left_indexes], depth + 1, rng)
        node.right = self._build_tree([rows[index] for index in right_indexes], [labels[index] for index in right_indexes], depth + 1, rng)
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
            "forest": [tree.to_dict() for tree in self.forest],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RandomForestClassifier":
        model = cls(int(data.get("trees", 31)), int(data.get("max_depth", 5)), int(data.get("min_samples", 3)), float(data.get("feature_fraction", 0.6)))
        model.classes = int(data.get("classes", 3))
        model.forest = [TreeNode.from_dict(item) for item in data.get("forest", [])]
        return model


def _macro_f1(actual: list[int], predicted: list[int], classes: int = 3) -> float:
    scores: list[float] = []
    for label in range(classes):
        true_positive = sum(1 for a, p in zip(actual, predicted) if a == label and p == label)
        false_positive = sum(1 for a, p in zip(actual, predicted) if a != label and p == label)
        false_negative = sum(1 for a, p in zip(actual, predicted) if a == label and p != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(score)
    return sum(scores) / len(scores)


def stratified_split(rows: list[list[float]], labels: list[int], test_ratio: float = 0.25) -> tuple[list[list[float]], list[int], list[list[float]], list[int]]:
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(label, []).append(index)
    rng = random.Random(2026)
    train_indexes: list[int] = []
    test_indexes: list[int] = []
    for indexes in grouped.values():
        rng.shuffle(indexes)
        test_count = max(1, round(len(indexes) * test_ratio)) if len(indexes) >= 4 else 0
        test_indexes.extend(indexes[:test_count])
        train_indexes.extend(indexes[test_count:])
    if not test_indexes:
        return rows, labels, rows, labels
    return (
        [rows[index] for index in train_indexes],
        [labels[index] for index in train_indexes],
        [rows[index] for index in test_indexes],
        [labels[index] for index in test_indexes],
    )


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


def train_and_compare(
    rows: list[list[float]],
    labels: list[int],
) -> tuple[object, list[dict[str, float | str | int]]]:
    if len(rows) < 9 or len(set(labels)) < 2:
        raise ValueError("At least 9 labeled examples across at least 2 quality classes are required.")
    available_rows = len(rows)
    rows, labels = _balanced_training_sample(rows, labels)
    train_rows, train_labels, test_rows, test_labels = stratified_split(rows, labels)
    candidates: list[object] = [
        SoftmaxLogisticRegression(),
        LinearSVMOVR(),
        RandomForestClassifier(),
    ]
    results: list[dict[str, float | str | int]] = []
    best_model: object | None = None
    best_score = -1.0
    for model in candidates:
        model.fit(train_rows, train_labels, classes=3)  # type: ignore[attr-defined]
        predictions = [max(range(3), key=model.predict_proba(row).__getitem__) for row in test_rows]  # type: ignore[attr-defined]
        accuracy = sum(a == p for a, p in zip(test_labels, predictions)) / len(test_labels)
        macro_f1 = _macro_f1(test_labels, predictions)
        results.append(
            {
                "model": model.name,  # type: ignore[attr-defined]
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "training_rows": len(rows),
                "available_rows": available_rows,
            }
        )
        selection_score = macro_f1 + 0.2 * accuracy
        if selection_score > best_score:
            best_score = selection_score
            best_model = model
    assert best_model is not None
    # Refit the selected model using the bounded, balanced training set.
    best_model.fit(rows, labels, classes=3)  # type: ignore[attr-defined]
    return best_model, results


def save_model(model: object, path: str | Path) -> None:
    if not hasattr(model, "to_dict"):
        raise TypeError("Unsupported model type.")
    Path(path).write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")  # type: ignore[attr-defined]


def load_model(path: str | Path) -> object:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    model_type = data.get("type")
    if model_type == "logistic":
        return SoftmaxLogisticRegression.from_dict(data)
    if model_type == "svm":
        return LinearSVMOVR.from_dict(data)
    if model_type == "random_forest":
        return RandomForestClassifier.from_dict(data)
    raise ValueError(f"Unknown model type: {model_type}")
