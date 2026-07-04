"""Logistic regression, implemented from scratch in numpy, for predicting
next-day price direction (up/down) from the features in `features.py`.

No scikit-learn / no external ML library -- consistent with the rest of
this repo (`linreg_gd`, `gpt_from_scratch`), and it keeps the whole
pipeline auditable: every number in a trading decision traces back to code
you can read line by line. This is a deliberately simple model: a linear
decision boundary over a handful of engineered features. It is not claimed
to be a good trading strategy -- see the "not financial advice" note in
README.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid: avoids overflow in exp(-z) for very
    # negative z and exp(z) for very positive z by branching per-element.
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


@dataclass
class StandardScaler:
    """Z-score feature scaling, fit on training data only.

    Fitting on the training set only (never on test data) matters here:
    fitting on the full dataset would leak test-set statistics (mean/std)
    into what's supposed to be an out-of-sample evaluation.
    """

    mean_: np.ndarray = field(default=None)
    std_: np.ndarray = field(default=None)

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0  # avoid divide-by-zero for a constant feature
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("StandardScaler.transform() called before fit()")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass
class LogisticRegressionGD:
    """Binary logistic regression trained with batch gradient descent on
    the average cross-entropy loss, with optional L2 regularization.
    """

    learning_rate: float = 0.1
    n_iters: int = 2000
    l2: float = 0.01
    weights_: np.ndarray = field(default=None)
    bias_: float = field(default=0.0)
    loss_history_: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.n_iters <= 0:
            raise ValueError("n_iters must be positive")
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(X) != len(y):
            raise ValueError(
                f"X has {len(X)} rows but y has {len(y)} labels"
            )
        if len(X) == 0:
            raise ValueError("cannot fit on an empty dataset")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("y must contain only 0/1 labels")

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iters):
            z = X @ self.weights_ + self.bias_
            p = _sigmoid(z)

            # Average cross-entropy loss + L2 penalty (bias not penalized).
            eps = 1e-12  # guards log(0) for a perfectly confident wrong prediction
            ce = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            l2_term = self.l2 * np.sum(self.weights_**2) / (2 * n_samples)
            self.loss_history_.append(ce + l2_term)

            error = p - y
            grad_w = (X.T @ error) / n_samples + (self.l2 / n_samples) * self.weights_
            grad_b = np.mean(error)

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

            if not (np.all(np.isfinite(self.weights_)) and np.isfinite(self.bias_)):
                raise FloatingPointError(
                    "gradient descent diverged (non-finite weights) -- try a "
                    "smaller learning_rate"
                )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("predict_proba() called before fit()")
        X = np.asarray(X, dtype=float)
        return _sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
