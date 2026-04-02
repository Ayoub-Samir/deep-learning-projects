from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion_matrix: np.ndarray
    classification_report: dict[str, dict[str, float] | float]

    def to_dict(self) -> dict[str, object]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "classification_report": self.classification_report,
        }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_labels: list[int],
) -> EvaluationResult:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    return EvaluationResult(
        loss=float(log_loss(y_true, y_proba, labels=class_labels)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(precision_weighted),
        recall_weighted=float(recall_weighted),
        f1_weighted=float(f1_weighted),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=class_labels),
        classification_report=classification_report(
            y_true,
            y_pred,
            labels=class_labels,
            output_dict=True,
            zero_division=0,
        ),
    )
