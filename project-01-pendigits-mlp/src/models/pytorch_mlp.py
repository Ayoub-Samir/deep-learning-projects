from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.common import ParameterBundle


class TorchMLP(nn.Module):
    def __init__(self, layer_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        for layer_index in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[layer_index], layer_sizes[layer_index + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass
class PyTorchMLPClassifier:
    layer_sizes: tuple[int, ...]
    learning_rate: float
    batch_size: int
    max_epochs: int
    l2_lambda: float = 0.0
    random_state: int = 42
    initial_parameters: ParameterBundle | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.device_ = torch.device(self.device)
        self.model_ = TorchMLP(self.layer_sizes).to(self.device_)
        self.n_steps_ = 0
        self.history_ = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }
        self._load_initial_parameters()

    def _linear_layers(self) -> list[nn.Linear]:
        return [module for module in self.model_.modules() if isinstance(module, nn.Linear)]

    def _load_initial_parameters(self) -> None:
        if self.initial_parameters is None:
            return

        linear_layers = self._linear_layers()
        for layer, weights, biases in zip(
            linear_layers,
            self.initial_parameters.weights,
            self.initial_parameters.biases,
        ):
            layer.weight.data = torch.tensor(weights.T, dtype=torch.float32, device=self.device_)
            layer.bias.data = torch.tensor(
                biases.reshape(-1),
                dtype=torch.float32,
                device=self.device_,
            )

    def _evaluate_dataset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        loss_function: nn.Module,
    ) -> tuple[float, float]:
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(
                torch.tensor(features, dtype=torch.float32, device=self.device_)
            )
            targets = torch.tensor(labels, dtype=torch.long, device=self.device_)
            loss = float(loss_function(logits, targets).item())
            predictions = torch.argmax(logits, dim=1)
            accuracy = float((predictions == targets).float().mean().item())
        return loss, accuracy

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "PyTorchMLPClassifier":
        torch.manual_seed(self.random_state)

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        data_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()

            for batch_features, batch_labels in data_loader:
                batch_features = batch_features.to(self.device_)
                batch_labels = batch_labels.to(self.device_)

                optimizer.zero_grad()
                logits = self.model_(batch_features)
                loss = loss_function(logits, batch_labels)
                loss.backward()
                optimizer.step()
                self.n_steps_ += 1

            train_loss, train_accuracy = self._evaluate_dataset(
                X_train,
                y_train,
                loss_function,
            )
            val_loss, val_accuracy = self._evaluate_dataset(
                X_val,
                y_val,
                loss_function,
            )

            self.history_["epoch"].append(epoch)
            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            self.history_["train_accuracy"].append(train_accuracy)
            self.history_["val_accuracy"].append(val_accuracy)

        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(
                torch.tensor(features, dtype=torch.float32, device=self.device_)
            )
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.cpu().numpy()

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(features), axis=1)
