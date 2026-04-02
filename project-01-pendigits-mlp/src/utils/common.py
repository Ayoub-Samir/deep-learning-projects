from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class ParameterBundle:
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    def copy(self) -> "ParameterBundle":
        return ParameterBundle(
            weights=[weights.copy() for weights in self.weights],
            biases=[biases.copy() for biases in self.biases],
        )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def initialize_parameters(
    layer_sizes: tuple[int, ...],
    seed: int,
) -> ParameterBundle:
    generator = np.random.default_rng(seed)
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []

    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        std = np.sqrt(2.0 / fan_in)
        weights.append(
            generator.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)
        )
        biases.append(np.zeros((1, fan_out), dtype=np.float64))

    return ParameterBundle(weights=weights, biases=biases)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(_to_serializable(data), file_handle, indent=2, ensure_ascii=False)
