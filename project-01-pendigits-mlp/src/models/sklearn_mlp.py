from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

from src.utils.common import ParameterBundle


class InitializedMLPClassifier(MLPClassifier):
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        *,
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=0.0001,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        n_iter_no_change=10,
        max_fun=15000,
        initial_parameters: ParameterBundle | None = None,
    ) -> None:
        self.initial_parameters = initial_parameters.copy() if initial_parameters else None
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )

    def _initialize(self, y, layer_units, dtype):  # type: ignore[override]
        super()._initialize(y, layer_units, dtype)
        if self.initial_parameters is None:
            return

        if len(self.initial_parameters.weights) != len(self.coefs_):
            raise ValueError("Initial parameter count does not match the network depth.")

        self.coefs_ = [
            weights.astype(dtype, copy=True)
            for weights in self.initial_parameters.weights
        ]
        self.intercepts_ = [
            biases.reshape(-1).astype(dtype, copy=True)
            for biases in self.initial_parameters.biases
        ]
        self._best_coefs = [weights.copy() for weights in self.coefs_]
        self._best_intercepts = [biases.copy() for biases in self.intercepts_]


@dataclass
class SklearnTrainingResult:
    model: InitializedMLPClassifier
    history: dict[str, list[float]]
    n_steps: int


def train_sklearn_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layers: tuple[int, ...],
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    l2_lambda: float,
    random_state: int,
    initial_parameters: ParameterBundle,
) -> SklearnTrainingResult:
    model = InitializedMLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="sgd",
        alpha=l2_lambda,
        batch_size=batch_size,
        learning_rate="constant",
        learning_rate_init=learning_rate,
        max_iter=1,
        shuffle=True,
        random_state=random_state,
        tol=0.0,
        warm_start=True,
        momentum=0.0,
        nesterovs_momentum=False,
        early_stopping=False,
        n_iter_no_change=max_epochs + 1,
        initial_parameters=initial_parameters,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    batches_per_epoch = math.ceil(X_train.shape[0] / batch_size)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for epoch in range(1, max_epochs + 1):
            model.fit(X_train, y_train)

            train_probabilities = model.predict_proba(X_train)
            val_probabilities = model.predict_proba(X_val)
            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)

            history["epoch"].append(epoch)
            history["train_loss"].append(
                float(log_loss(y_train, train_probabilities, labels=model.classes_))
            )
            history["val_loss"].append(
                float(log_loss(y_val, val_probabilities, labels=model.classes_))
            )
            history["train_accuracy"].append(
                float(np.mean(train_predictions == y_train))
            )
            history["val_accuracy"].append(float(np.mean(val_predictions == y_val)))

    return SklearnTrainingResult(
        model=model,
        history=history,
        n_steps=batches_per_epoch * max_epochs,
    )
