"""Binary logistic regression, from scratch with numpy only.

Trains by batch gradient descent on the log-loss, with an optional L2
penalty on the weights (the bias is never penalised). Features are
standardised inside ``fit`` using the training mean and standard
deviation, so no manual scaling is needed.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip to avoid overflow in exp for very large |z|.
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    """Binary logistic regression trained by gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent.
    n_epochs : int
        Number of full passes over the training data.
    l2 : float
        L2 penalty strength on the weights (0 means no regularisation).
    tol : float
        Stop early when the change in loss between epochs falls below this.
    """

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 1000,
                 l2: float = 0.0, tol: float = 1e-6):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(n_epochs, int) or isinstance(n_epochs, bool) or n_epochs < 1:
            raise ValueError("n_epochs must be a positive integer")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2 = l2
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_epochs_run_: int = 0
        self.loss_history_: list[float] = []
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegression":
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
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                "logistic regression needs exactly 2 classes, got %d" % len(classes)
            )
        y_bin = (y == classes[1]).astype(float)
        self._mean = x.mean(axis=0)
        self._scale = x.std(axis=0)
        self._scale[self._scale == 0] = 1.0  # constant feature: leave unscaled
        xs = (x - self._mean) / self._scale
        n_samples, n_features = xs.shape
        weights = np.zeros(n_features)
        bias = 0.0
        prev_loss = None
        for epoch in range(self.n_epochs):
            probs = _sigmoid(xs @ weights + bias)
            error = probs - y_bin
            grad_w = xs.T @ error / n_samples + (self.l2 / n_samples) * weights
            grad_b = float(error.mean())
            weights -= self.learning_rate * grad_w
            bias -= self.learning_rate * grad_b
            loss = self._loss(xs, y_bin, weights, bias)
            self.loss_history_.append(loss)
            self.n_epochs_run_ = epoch + 1
            if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        self.coef_ = weights
        self.intercept_ = bias
        return self

    def _loss(self, xs, y_bin, weights, bias) -> float:
        probs = np.clip(_sigmoid(xs @ weights + bias), 1e-15, 1 - 1e-15)
        log_loss = -np.mean(
            y_bin * np.log(probs) + (1 - y_bin) * np.log(1 - probs)
        )
        return float(log_loss + 0.5 * self.l2 * np.sum(weights ** 2) / len(y_bin))

    def _check_fitted(self):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    def _standardise(self, x: np.ndarray) -> np.ndarray:
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
        return (x - self._mean) / self._scale

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probability of the positive class for each sample."""
        self._check_fitted()
        xs = self._standardise(x)
        probs = _sigmoid(xs @ self.coef_ + self.intercept_)
        return np.column_stack([1 - probs, probs])

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)
