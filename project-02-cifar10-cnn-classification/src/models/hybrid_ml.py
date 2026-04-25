from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def extract_features(
    model,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            features = model.forward_features(inputs).cpu().numpy()
            all_features.append(features)
            all_labels.append(labels.numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def save_feature_arrays(
    output_dir: Path,
    prefix: str,
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / f"{prefix}_features.npy"
    label_path = output_dir / f"{prefix}_labels.npy"
    np.save(feature_path, features)
    np.save(label_path, labels)
    return feature_path, label_path


def train_linear_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
) -> object:
    classifier = make_pipeline(
        StandardScaler(),
        LinearSVC(C=c_value, dual=False, max_iter=5000, random_state=42),
    )
    classifier.fit(X_train, y_train)
    return classifier
