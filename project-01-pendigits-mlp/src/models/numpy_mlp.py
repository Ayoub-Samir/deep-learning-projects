from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.common import ParameterBundle, initialize_parameters


@dataclass
class NumpyMLPClassifier:
    layer_sizes: tuple[int, ...]
    learning_rate: float
    batch_size: int
    max_epochs: int
    l2_lambda: float = 0.0
    random_state: int = 42
    shuffle: bool = True
    initial_parameters: ParameterBundle | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_state)
        self.n_steps_ = 0
        self.history_ = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        bundle = self.initial_parameters or initialize_parameters(
            self.layer_sizes,
            self.random_state,
        )
        self.weights_ = [weight.copy() for weight in bundle.weights]
        self.biases_ = [bias.copy() for bias in bundle.biases]

    def _relu(self, values: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, values)

    def _relu_grad(self, values: np.ndarray) -> np.ndarray:
        return (values > 0.0).astype(np.float64)

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def _one_hot(self, labels: np.ndarray) -> np.ndarray:
        encoded = np.zeros((labels.shape[0], self.layer_sizes[-1]), dtype=np.float64)
        encoded[np.arange(labels.shape[0]), labels] = 1.0
        return encoded

    def _forward(
        self,
        features: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [features]
        pre_activations = []

        for layer_index, (weights, biases) in enumerate(zip(self.weights_, self.biases_)):
            z_values = activations[-1] @ weights + biases
            pre_activations.append(z_values)
            if layer_index == len(self.weights_) - 1:
                activations.append(self._softmax(z_values))
            else:
                activations.append(self._relu(z_values))

        return activations, pre_activations

    def _compute_loss(self, probabilities: np.ndarray, targets: np.ndarray) -> float:
        epsilon = 1e-12
        clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
        cross_entropy = -np.mean(np.sum(targets * np.log(clipped), axis=1))
        l2_penalty = 0.5 * self.l2_lambda * sum(
            np.sum(weights * weights) for weights in self.weights_
        )
        return float(cross_entropy + l2_penalty)

    def _compute_accuracy(self, labels: np.ndarray, probabilities: np.ndarray) -> float:
        predictions = np.argmax(probabilities, axis=1)
        return float(np.mean(predictions == labels))

    def _backward(
        self,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        targets: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        gradients_w = [np.zeros_like(weights) for weights in self.weights_]
        gradients_b = [np.zeros_like(biases) for biases in self.biases_]

        batch_size = targets.shape[0]
        delta = activations[-1] - targets

        for layer_index in reversed(range(len(self.weights_))):
            gradients_w[layer_index] = (
                activations[layer_index].T @ delta / batch_size
                + self.l2_lambda * self.weights_[layer_index]
            )
            gradients_b[layer_index] = np.mean(delta, axis=0, keepdims=True)

            if layer_index > 0:
                delta = (delta @ self.weights_[layer_index].T) * self._relu_grad(
                    pre_activations[layer_index - 1]
                )

        return gradients_w, gradients_b

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "NumpyMLPClassifier":
        self.classes_ = np.unique(y_train)
        if len(self.classes_) != self.layer_sizes[-1]:
            raise ValueError("Output dimension must match the number of classes.")

        train_targets = self._one_hot(y_train)
        validation_targets = self._one_hot(y_val)
        indices = np.arange(X_train.shape[0])

        for epoch in range(1, self.max_epochs + 1):
            if self.shuffle:
                self._rng.shuffle(indices)

            for start in range(0, X_train.shape[0], self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                batch_features = X_train[batch_indices]
                batch_targets = train_targets[batch_indices]

                activations, pre_activations = self._forward(batch_features)
                gradients_w, gradients_b = self._backward(
                    activations,
                    pre_activations,
                    batch_targets,
                )

                for layer_index in range(len(self.weights_)):
                    self.weights_[layer_index] -= self.learning_rate * gradients_w[layer_index]
                    self.biases_[layer_index] -= self.learning_rate * gradients_b[layer_index]

                self.n_steps_ += 1

            train_probabilities = self.predict_proba(X_train)
            val_probabilities = self.predict_proba(X_val)

            self.history_["epoch"].append(epoch)
            self.history_["train_loss"].append(
                self._compute_loss(train_probabilities, train_targets)
            )
            self.history_["val_loss"].append(
                self._compute_loss(val_probabilities, validation_targets)
            )
            self.history_["train_accuracy"].append(
                self._compute_accuracy(y_train, train_probabilities)
            )
            self.history_["val_accuracy"].append(
                self._compute_accuracy(y_val, val_probabilities)
            )

        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        activations, _ = self._forward(features)
        return activations[-1]

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(features), axis=1)

    def get_parameters(self) -> ParameterBundle:
        return ParameterBundle(
            weights=[weights.copy() for weights in self.weights_],
            biases=[biases.copy() for biases in self.biases_],
        )
