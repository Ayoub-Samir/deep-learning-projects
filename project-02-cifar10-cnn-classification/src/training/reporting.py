from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.common import save_json


def plot_training_history(
    history: dict[str, list[float]],
    output_path: Path,
    title: str,
) -> None:
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation Loss")
    axes[0].set_title(f"{title} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="Train Accuracy")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="Validation Accuracy")
    axes[1].set_title(f"{title} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(
    matrix,
    class_labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    sns.set_theme(style="white")
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_model_metric_comparison(
    results: list[dict[str, object]],
    output_path: Path,
) -> None:
    if not results:
        return

    frame = pd.DataFrame(results).sort_values("name").copy()
    frame["label"] = frame["name"].map(
        {
            "m1_lenet_like": "M1",
            "m2_improved_cnn": "M2",
            "m3_resnet18": "M3",
            "m4_resnet18_linear_svm_hybrid": "M4",
        }
    )

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_specs = [
        ("val_accuracy", "Validation Accuracy", "#1f77b4"),
        ("test_accuracy", "Test Accuracy", "#2ca02c"),
        ("test_f1_macro", "Test Macro F1", "#d62728"),
    ]

    for axis, (metric_name, title, color) in zip(axes, metric_specs):
        sns.barplot(data=frame, x="label", y=metric_name, color=color, ax=axis)
        axis.set_title(title)
        axis.set_xlabel("Model")
        axis.set_ylabel("Score")
        axis.set_ylim(0.0, 1.0)
        for container in axis.containers:
            axis.bar_label(container, fmt="%.3f", padding=2, fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_reports(
    results: list[dict[str, object]],
    selection_summary: dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        {
            key: value
            for key, value in result.items()
            if key not in {"validation_details", "test_details"}
        }
        for result in results
    ]
    pd.DataFrame(summary_rows).to_csv(output_dir / "metrics_summary.csv", index=False)
    save_json(results, output_dir / "detailed_results.json")
    save_json(selection_summary, output_dir / "model_selection.json")
