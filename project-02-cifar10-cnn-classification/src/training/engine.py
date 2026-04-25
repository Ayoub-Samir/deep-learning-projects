from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.training.evaluation import EvaluationResult, evaluate_predictions


@dataclass
class TorchTrainingResult:
    model: nn.Module
    history: dict[str, list[float]]
    n_steps: int
    device: torch.device


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            trainable_parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def evaluate_torch_model(
    model: nn.Module,
    loader,
    device: torch.device,
    class_labels: list[int],
    criterion: nn.Module | None = None,
) -> EvaluationResult:
    model.eval()
    losses: list[float] = []
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)

            if criterion is not None:
                losses.append(float(criterion(logits, targets).item()))

            predictions = torch.argmax(logits, dim=1)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    loss = float(np.mean(losses)) if losses else None
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)
    return evaluate_predictions(y_true, y_pred, class_labels, loss=loss)


def train_torch_classifier(
    model: nn.Module,
    loaders: dict[str, object],
    device: torch.device,
    class_labels: list[int],
    epochs: int,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
) -> TorchTrainingResult:
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    model.to(device)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = -float("inf")
    n_steps = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets in loaders["train"]:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            n_steps += 1

        train_result = evaluate_torch_model(
            model,
            loaders["train_eval"],
            device,
            class_labels,
            criterion=criterion,
        )
        val_result = evaluate_torch_model(
            model,
            loaders["val"],
            device,
            class_labels,
            criterion=criterion,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_result.loss if train_result.loss is not None else np.nan)
        history["val_loss"].append(val_result.loss if val_result.loss is not None else np.nan)
        history["train_accuracy"].append(train_result.accuracy)
        history["val_accuracy"].append(val_result.accuracy)

        if val_result.accuracy > best_val_accuracy:
            best_val_accuracy = val_result.accuracy
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return TorchTrainingResult(
        model=model,
        history=history,
        n_steps=n_steps,
        device=device,
    )
