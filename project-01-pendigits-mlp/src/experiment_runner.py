from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, FIGURES_DIR, REPORTS_DIR, SEED
from src.data.pendigits import PendigitsDataset, load_pendigits_dataset
from src.models.numpy_mlp import NumpyMLPClassifier
from src.models.pytorch_mlp import PyTorchMLPClassifier
from src.models.sklearn_mlp import train_sklearn_mlp
from src.training.metrics import evaluate_predictions
from src.training.preprocessing import Standardizer
from src.training.reporting import (
    plot_confusion_matrix,
    plot_model_metric_comparison,
    plot_training_history,
    write_reports,
)
from src.utils.common import ParameterBundle, ensure_directory, initialize_parameters


@dataclass
class ExperimentContext:
    dataset: PendigitsDataset
    feature_views: dict[str, dict[str, object]]
    initializations: dict[tuple[int, ...], ParameterBundle]


def build_feature_views(dataset: PendigitsDataset) -> dict[str, dict[str, object]]:
    standardizer = Standardizer().fit(dataset.X_train)
    return {
        "raw": {
            "X_train": dataset.X_train,
            "X_val": dataset.X_val,
            "X_test": dataset.X_test,
        },
        "standardized": {
            "X_train": standardizer.transform(dataset.X_train),
            "X_val": standardizer.transform(dataset.X_val),
            "X_test": standardizer.transform(dataset.X_test),
        },
    }


def prepare_experiment_context(data_dir: Path = DATA_DIR) -> ExperimentContext:
    dataset = load_pendigits_dataset(data_dir)
    feature_views = build_feature_views(dataset)

    input_dim = dataset.X_train.shape[1]
    output_dim = len(dataset.class_labels)
    initializations = {
        (32,): initialize_parameters((input_dim, 32, output_dim), SEED),
        (64, 32): initialize_parameters((input_dim, 64, 32, output_dim), SEED),
    }

    return ExperimentContext(
        dataset=dataset,
        feature_views=feature_views,
        initializations=initializations,
    )


def evaluate_model(model, features, labels, class_labels):
    probabilities = model.predict_proba(features)
    predictions = model.predict(features)
    return evaluate_predictions(labels, predictions, probabilities, class_labels)


def run_experiment(spec, context: ExperimentContext) -> dict[str, object]:
    ensure_directory(FIGURES_DIR)
    ensure_directory(REPORTS_DIR)

    dataset = context.dataset
    input_dim = dataset.X_train.shape[1]
    output_dim = len(dataset.class_labels)

    feature_key = "standardized" if spec.standardize else "raw"
    current_features = context.feature_views[feature_key]
    layer_sizes = (input_dim, *spec.hidden_layers, output_dim)
    initial_parameters = context.initializations[spec.hidden_layers].copy()

    if spec.family == "numpy":
        model = NumpyMLPClassifier(
            layer_sizes=layer_sizes,
            learning_rate=spec.learning_rate,
            batch_size=spec.batch_size,
            max_epochs=spec.max_epochs,
            l2_lambda=spec.l2_lambda,
            random_state=SEED,
            initial_parameters=initial_parameters,
        ).fit(
            current_features["X_train"],
            dataset.y_train,
            current_features["X_val"],
            dataset.y_val,
        )
        history = model.history_
        n_steps = model.n_steps_
    elif spec.family == "sklearn":
        training_result = train_sklearn_mlp(
            X_train=current_features["X_train"],
            y_train=dataset.y_train,
            X_val=current_features["X_val"],
            y_val=dataset.y_val,
            hidden_layers=spec.hidden_layers,
            learning_rate=spec.learning_rate,
            batch_size=spec.batch_size,
            max_epochs=spec.max_epochs,
            l2_lambda=spec.l2_lambda,
            random_state=SEED,
            initial_parameters=initial_parameters,
        )
        model = training_result.model
        history = training_result.history
        n_steps = training_result.n_steps
    elif spec.family == "pytorch":
        model = PyTorchMLPClassifier(
            layer_sizes=layer_sizes,
            learning_rate=spec.learning_rate,
            batch_size=spec.batch_size,
            max_epochs=spec.max_epochs,
            l2_lambda=spec.l2_lambda,
            random_state=SEED,
            initial_parameters=initial_parameters,
        ).fit(
            current_features["X_train"],
            dataset.y_train,
            current_features["X_val"],
            dataset.y_val,
        )
        history = model.history_
        n_steps = model.n_steps_
    else:
        raise ValueError(f"Unsupported experiment family: {spec.family}")

    validation_result = evaluate_model(
        model,
        current_features["X_val"],
        dataset.y_val,
        dataset.class_labels,
    )
    test_result = evaluate_model(
        model,
        current_features["X_test"],
        dataset.y_test,
        dataset.class_labels,
    )

    history_path = FIGURES_DIR / f"{spec.name}_history.png"
    confusion_matrix_path = FIGURES_DIR / f"{spec.name}_confusion_matrix.png"

    plot_training_history(history, history_path, spec.name)
    plot_confusion_matrix(
        test_result.confusion_matrix.tolist(),
        dataset.class_labels,
        confusion_matrix_path,
        f"{spec.name} Test Confusion Matrix",
    )

    return {
        **spec.to_dict(),
        "n_steps": n_steps,
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
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_val_accuracy": history["val_accuracy"][-1],
        "history_path": history_path,
        "confusion_matrix_path": confusion_matrix_path,
        "validation_details": validation_result.to_dict(),
        "test_details": test_result.to_dict(),
    }


def upsert_result(
    results: list[dict[str, object]],
    new_result: dict[str, object],
) -> list[dict[str, object]]:
    filtered_results = [result for result in results if result["name"] != new_result["name"]]
    filtered_results.append(new_result)
    return sorted(filtered_results, key=lambda result: result["name"])


def summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        return {
            "selection_rule": (
                "Highest validation accuracy wins; ties are resolved with lower n_steps."
            ),
            "best_manual_model": None,
            "best_overall_model": None,
        }

    manual_results = [result for result in results if result["family"] == "numpy"]
    best_manual_model = None
    if manual_results:
        best_manual_model = sorted(
            manual_results,
            key=lambda result: (-result["val_accuracy"], result["n_steps"]),
        )[0]

    best_overall_model = sorted(
        results,
        key=lambda result: (-result["val_accuracy"], result["n_steps"]),
    )[0]

    return {
        "selection_rule": (
            "Highest validation accuracy wins; ties are resolved with lower n_steps."
        ),
        "best_manual_model": best_manual_model,
        "best_overall_model": best_overall_model,
    }


def write_experiment_reports(
    results: list[dict[str, object]],
    output_dir: Path = REPORTS_DIR,
) -> dict[str, object]:
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
        "hidden_layers",
        "standardize",
        "l2_lambda",
        "n_steps",
        "val_accuracy",
        "test_accuracy",
        "test_f1_macro",
    ]
    frame = pd.DataFrame(results)
    return frame[columns].sort_values("name").reset_index(drop=True)
