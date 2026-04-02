from __future__ import annotations

import numpy as np


class Standardizer:
    def fit(self, features: np.ndarray) -> "Standardizer":
        self.mean_ = features.mean(axis=0)
        self.std_ = features.std(axis=0)
        self.std_[self.std_ == 0.0] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean_) / self.std_

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)
