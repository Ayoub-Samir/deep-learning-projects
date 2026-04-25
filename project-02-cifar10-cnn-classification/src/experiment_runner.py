from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    EXPERIMENT_SPECS,
    FEATURES_DIR,
    FIGURES_DIR,
    NUM_CLASSES,
    REPORTS_DIR,
    SEED,
)
from src.data.cifar10 import CIFAR10Context, build_cifar10_loaders, prepare_cifar10_context
from src.models.hybrid_ml import extract_features, save_feature_arrays, train_linear_svm
from src.models.improved_cnn import ImprovedCNN
from src.models.lenet_like import LeNetLikeCNN
from src.models.pretrained_cnn import ResNet18Classifier
from src.training.engine import evaluate_torch_model, train_torch_classifier
from src.training.evaluation import evaluate_predictions
from src.training.reporting import (
    plot_confusion_matrix,
    plot_model_metric_comparison,
    plot_training_history,
    write_reports,
)
from src.utils.common import ensure_directory, get_device


@dataclass
class ExperimentArtifact:
    model: object | None
    history: dict[str, list[float]] | None
    loader_key: str
    batch_size: int


@dataclass
class ExperimentContext:
    dataset_context: CIFAR10Context
    device: object
    loader_cache: dict[str, dict[str, object]]
    artifacts: dict[str, ExperimentArtifact]


def prepare_experiment_context() -> ExperimentContext:
    return ExperimentContext(
        dataset_context=prepare_cifar10_context(),
        device=get_device(),
        loader_cache={},
        artifacts={},
    )


def get_experiment_spec(name: str):
    for spec in EXPERIMENT_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"Experiment spec not found: {name}")


def get_loaders(context: ExperimentContext, loader_key: str, batch_size: int):
    cache_key = f"{loader_key}:{batch_size}"
    if cache_key not in context.loader_cache:
        context.loader_cache[cache_key] = build_cifar10_loaders(
            context.dataset_context,
            variant=loader_key,
            batch_size=batch_size,
        )
    return context.loader_cache[cache_key]


def build_torch_model(model_key: str):
    if model_key == "lenet_like":
        return LeNetLikeCNN(num_classes=NUM_CLASSES)
    if model_key == "improved_cnn":
        return ImprovedCNN(num_classes=NUM_CLASSES)
    if model_key == "resnet18":
        return ResNet18Classifier(num_classes=NUM_CLASSES, pretrained=True, unfreeze_layer4=True)
    raise ValueError(f"Unsupported torch model key: {model_key}")


def _run_torch_experiment(spec, context: ExperimentContext) -> dict[str, object]:
    ensure_directory(FEATURES_DIR)
    ensure_directory(FIGURES_DIR)
    ensure_directory(REPORTS_DIR)

    model = build_torch_model(spec.model_key)
    loaders = get_loaders(context, spec.loader_key, spec.batch_size)
    training_result = train_torch_classifier(
        model=model,
        loaders=loaders,
        device=context.device,
        class_labels=list(range(NUM_CLASSES)),
        epochs=spec.epochs,
        optimizer_name=spec.optimizer,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        momentum=spec.momentum,
    )

    validation_result = evaluate_torch_model(
        training_result.model,
        loaders["val"],
        training_result.device,
        class_labels=list(range(NUM_CLASSES)),
    )
    test_result = evaluate_torch_model(
        training_result.model,
        loaders["test"],
        training_result.device,
        class_labels=list(range(NUM_CLASSES)),
    )

    history_path = FIGURES_DIR / f"{spec.name}_history.png"
    confusion_matrix_path = FIGURES_DIR / f"{spec.name}_confusion_matrix.png"
    plot_training_history(training_result.history, history_path, spec.name)
    plot_confusion_matrix(
        test_result.confusion_matrix,
        context.dataset_context.class_names,
        confusion_matrix_path,
        f"{spec.name} Test Confusion Matrix",
    )

    context.artifacts[spec.name] = ExperimentArtifact(
        model=training_result.model,
        history=training_result.history,
        loader_key=spec.loader_key,
        batch_size=spec.batch_size,
    )

    return {
        **spec.to_dict(),
        "device": str(context.device),
        "n_steps": training_result.n_steps,
        "val_loss": validation_result.loss,
        "val_accuracy": validation_result.accuracy,
        "val_precision_macro": validation_result.precision_macro,
        "val_recall_macro": validation_result.recall_macro,
        "val_f1_macro": validation_result.f1_macro,
        "test_loss": test_result.loss,
        "test_accuracy": test_result.accuracy,
        "test_precision_macro": test_result.precision_macro,
        "test_recall_macro": test_result.recall_macro,
        "test_f1_macro": test_result.f1_macro,
        "final_train_accuracy": training_result.history["train_accuracy"][-1],
        "final_val_accuracy": training_result.history["val_accuracy"][-1],
        "history_path": history_path,
        "confusion_matrix_path": confusion_matrix_path,
        "validation_details": validation_result.to_dict(),
        "test_details": test_result.to_dict(),
    }


def _run_hybrid_experiment(spec, context: ExperimentContext) -> dict[str, object]:
    if spec.feature_reference is None:
        raise ValueError("Hybrid experiment requires a feature_reference.")
    if spec.feature_reference not in context.artifacts:
        raise RuntimeError(
            f"Reference model '{spec.feature_reference}' has not been trained yet."
        )

    reference_artifact = context.artifacts[spec.feature_reference]
    if reference_artifact.model is None:
        raise RuntimeError(f"Reference artifact '{spec.feature_reference}' has no model.")

    loaders = get_loaders(context, reference_artifact.loader_key, reference_artifact.batch_size)
    feature_model = reference_artifact.model

    X_train, y_train = extract_features(feature_model, loaders["train_eval"], context.device)
    X_val, y_val = extract_features(feature_model, loaders["val"], context.device)
    X_test, y_test = extract_features(feature_model, loaders["test"], context.device)

    train_feature_path, train_label_path = save_feature_arrays(
        FEATURES_DIR,
        "train",
        X_train,
        y_train,
    )
    val_feature_path, val_label_path = save_feature_arrays(
        FEATURES_DIR,
        "val",
        X_val,
        y_val,
    )
    test_feature_path, test_label_path = save_feature_arrays(
        FEATURES_DIR,
        "test",
        X_test,
        y_test,
    )

    classifier = train_linear_svm(X_train, y_train, c_value=spec.svm_c or 1.0)
    val_predictions = classifier.predict(X_val)
    test_predictions = classifier.predict(X_test)

    validation_result = evaluate_predictions(
        y_val,
        val_predictions,
        class_labels=list(range(NUM_CLASSES)),
    )
    test_result = evaluate_predictions(
        y_test,
        test_predictions,
        class_labels=list(range(NUM_CLASSES)),
    )

    confusion_matrix_path = FIGURES_DIR / f"{spec.name}_confusion_matrix.png"
    plot_confusion_matrix(
        test_result.confusion_matrix,
        context.dataset_context.class_names,
        confusion_matrix_path,
        f"{spec.name} Test Confusion Matrix",
    )

    context.artifacts[spec.name] = ExperimentArtifact(
        model=classifier,
        history=None,
        loader_key=spec.loader_key,
        batch_size=spec.batch_size,
    )

    return {
        **spec.to_dict(),
        "device": str(context.device),
        "n_steps": 1,
        "val_loss": None,
        "val_accuracy": validation_result.accuracy,
        "val_precision_macro": validation_result.precision_macro,
        "val_recall_macro": validation_result.recall_macro,
        "val_f1_macro": validation_result.f1_macro,
        "test_loss": None,
        "test_accuracy": test_result.accuracy,
        "test_precision_macro": test_result.precision_macro,
        "test_recall_macro": test_result.recall_macro,
        "test_f1_macro": test_result.f1_macro,
        "final_train_accuracy": None,
        "final_val_accuracy": validation_result.accuracy,
        "history_path": None,
        "confusion_matrix_path": confusion_matrix_path,
        "feature_train_shape": list(X_train.shape),
        "feature_val_shape": list(X_val.shape),
        "feature_test_shape": list(X_test.shape),
        "train_feature_path": train_feature_path,
        "train_label_path": train_label_path,
        "val_feature_path": val_feature_path,
        "val_label_path": val_label_path,
        "test_feature_path": test_feature_path,
        "test_label_path": test_label_path,
        "validation_details": validation_result.to_dict(),
        "test_details": test_result.to_dict(),
    }


def run_experiment(spec, context: ExperimentContext) -> dict[str, object]:
    if spec.family == "torch":
        return _run_torch_experiment(spec, context)
    if spec.family == "hybrid":
        return _run_hybrid_experiment(spec, context)
    raise ValueError(f"Unsupported experiment family: {spec.family}")


def upsert_result(results: list[dict[str, object]], new_result: dict[str, object]) -> list[dict[str, object]]:
    filtered_results = [result for result in results if result["name"] != new_result["name"]]
    filtered_results.append(new_result)
    return sorted(filtered_results, key=lambda result: result["name"])


def summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        return {
            "selection_rule": "Highest validation accuracy wins; ties are resolved with lower n_steps.",
            "best_overall_model": None,
            "best_cnn_model": None,
            "hybrid_reference_model": None,
        }

    best_overall_model = sorted(
        results,
        key=lambda result: (-result["val_accuracy"], result["n_steps"]),
    )[0]
    torch_results = [result for result in results if result["family"] == "torch"]
    best_cnn_model = None
    if torch_results:
        best_cnn_model = sorted(
            torch_results,
            key=lambda result: (-result["val_accuracy"], result["n_steps"]),
        )[0]

    hybrid_reference_model = None
    for result in results:
        if result["family"] == "hybrid":
            hybrid_reference_model = result.get("feature_reference")
            break

    return {
        "selection_rule": "Highest validation accuracy wins; ties are resolved with lower n_steps.",
        "best_overall_model": best_overall_model,
        "best_cnn_model": best_cnn_model,
        "hybrid_reference_model": hybrid_reference_model,
    }


def write_experiment_reports(results: list[dict[str, object]], output_dir: Path = REPORTS_DIR) -> dict[str, object]:
    selection_summary = summarize_results(results)
    write_reports(results, selection_summary, output_dir)
    plot_model_metric_comparison(results, FIGURES_DIR / "model_metric_comparison.png")
    return selection_summary


def results_summary_frame(results: list[dict[str, object]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()

    columns = [
        "name",
        "family",
        "model_key",
        "loader_key",
        "n_steps",
        "val_accuracy",
        "test_accuracy",
        "test_f1_macro",
    ]
    frame = pd.DataFrame(results)
    return frame[columns].sort_values("name").reset_index(drop=True)
