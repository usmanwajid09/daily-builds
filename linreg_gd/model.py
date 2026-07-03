"""Simple linear regression trained with batch gradient descent."""
from __future__ import annotations

import numpy as np


class LinearRegressionGD:
    """Fits y = w*x + b using batch gradient descent on MSE loss."""

    def __init__(self, learning_rate: float = 0.05, n_iters: int = 1000):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if n_iters <= 0:
            raise ValueError("n_iters must be positive")
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w_: float | None = None
        self.b_: float | None = None
        self.loss_history_: list[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")

        n = x.shape[0]
        w, b = 0.0, 0.0
        self.loss_history_ = []

        for _ in range(self.n_iters):
            y_pred = w * x + b
            error = y_pred - y

            loss = float(np.mean(error ** 2))
            self.loss_history_.append(loss)

            dw = (2 / n) * np.dot(error, x)
            db = (2 / n) * np.sum(error)

            w -= self.learning_rate * dw
            b -= self.learning_rate * db

            if not np.isfinite(w) or not np.isfinite(b):
                raise FloatingPointError(
                    "Gradient descent diverged - try a smaller learning_rate"
                )

        self.w_, self.b_ = w, b
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w_ is None or self.b_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")
        x = np.asarray(x, dtype=float).reshape(-1)
        return self.w_ * x + self.b_
