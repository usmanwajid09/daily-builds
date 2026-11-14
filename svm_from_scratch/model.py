"""Linear support-vector classifier, from scratch with numpy only.

Trains by stochastic gradient descent on the primal objective::

    0.5 * ||w||^2 + C * sum(max(0, 1 - y_i (w.x_i + b)))

with labels mapped to -1 / +1. Each epoch shuffles the data, then applies
one sub-gradient step per sample: the regularisation term always shrinks
``w``, and the hinge term pushes back only for samples inside the margin.
"""
from __future__ import annotations

import numpy as np


class LinearSVC:
    """Linear SVM classifier trained with SGD on the hinge loss.

    Parameters
    ----------
    C : float
        Regularisation strength: larger values penalise margin violations
        more heavily. Must be positive.
    learning_rate : float
        Step size for the SGD updates. Must be positive.
    n_epochs : int
        Number of passes over the training data. Must be at least 1.
    random_state : int or None
        Seed for shuffling the data each epoch.
    """

    def __init__(self, C: float = 1.0, learning_rate: float = 0.01,
                 n_epochs: int = 1000, random_state: int | None = None):
        if not isinstance(C, (int, float)) or isinstance(C, bool) or C <= 0:
            raise ValueError("C must be a positive number")
        if (not isinstance(learning_rate, (int, float))
                or isinstance(learning_rate, bool) or learning_rate <= 0):
            raise ValueError("learning_rate must be a positive number")
        if not isinstance(n_epochs, int) or isinstance(n_epochs, bool) or n_epochs < 1:
            raise ValueError("n_epochs must be an integer of at least 1")
        if random_state is not None and (
                not isinstance(random_state, int) or isinstance(random_state, bool)):
            raise ValueError("random_state must be an integer or None")
        self.C = float(C)
        self.learning_rate = float(learning_rate)
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.classes_: np.ndarray | None = None

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearSVC":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                "LinearSVC is a binary classifier: y must contain exactly 2 classes, "
                f"got {len(self.classes_)}"
            )
        y_signed = np.where(y == self.classes_[0], -1.0, 1.0)
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = x.shape
        w = np.zeros(n_features)
        b = 0.0
        lr = self.learning_rate
        for _epoch in range(self.n_epochs):
            for i in rng.permutation(n_samples):
                margin = y_signed[i] * (x[i] @ w + b)
                if margin < 1.0:
                    w -= lr * (w - self.C * y_signed[i] * x[i])
                    b += lr * self.C * y_signed[i]
                else:
                    w -= lr * w
        self.coef_ = w
        self.intercept_ = float(b)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Signed distance to the separating hyperplane (times ||w||)."""
        self._check_fitted()
        x = self._check_x(x)
        return x @ self.coef_ + self.intercept_

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

    # -- internals ------------------------------------------------------
    def _check_fitted(self):
        if self.coef_ is None or self.classes_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    def _check_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                "x has %d features, expected %d" % (x.shape[1], self.coef_.shape[0])
            )
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        return x
