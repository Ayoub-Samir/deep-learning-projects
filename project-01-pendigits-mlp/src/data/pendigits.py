from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PENDIGITS_URLS, SEED, VALIDATION_SIZE


@dataclass(frozen=True)
class PendigitsDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_labels: list[int]


def ensure_pendigits_files(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    target_paths = {
        split_name: data_dir / f"pendigits.{split_name}"
        for split_name in ("train", "test", "names")
    }

    for split_name, file_path in target_paths.items():
        if not file_path.exists():
            urlretrieve(PENDIGITS_URLS[split_name], file_path)

    return target_paths


def _read_pendigits_split(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    dataframe = pd.read_csv(file_path, header=None, skipinitialspace=True)
    features = dataframe.iloc[:, :-1].to_numpy(dtype=np.float64)
    labels = dataframe.iloc[:, -1].to_numpy(dtype=np.int64)
    return features, labels


def load_pendigits_dataset(
    data_dir: Path,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = SEED,
) -> PendigitsDataset:
    file_paths = ensure_pendigits_files(data_dir)
    X_train_pool, y_train_pool = _read_pendigits_split(file_paths["train"])
    X_test, y_test = _read_pendigits_split(file_paths["test"])

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool,
        y_train_pool,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train_pool,
    )

    class_labels = sorted(np.unique(y_train_pool).tolist())

    return PendigitsDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_labels=class_labels,
    )
