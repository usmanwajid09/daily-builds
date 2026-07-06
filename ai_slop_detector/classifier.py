"""From-scratch (numpy-only) logistic regression classifier + metrics.

No sklearn dependency by choice, consistent with the rest of this repo's
daily-builds (see gpt_from_scratch, ai_trading_bot): the point of these
projects is to implement the actual mechanics, not call a library.
"""
from __future__ import annotations

import numpy as np


class StandardScaler:
    """Standardizes features to zero mean / unit variance.

    Fit on training data only, then reused (not refit) on test data -- this
    is a common leakage bug so it's worth stating explicitly and testing.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # Guard against a constant feature column producing a divide-by-zero
        # (would otherwise silently yield inf/NaN for every future sample).
        std[std == 0] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler.transform() called before fit()")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip to avoid overflow in np.exp for very negative/positive z.
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    """Binary logistic regression trained with full-batch gradient descent
    and L2 regularization, implemented from scratch in numpy.

    Label convention: y=1 means "ai", y=0 means "human" (see train.py).
    """

    def __init__(self, lr: float = 0.1, epochs: int = 2000, l2: float = 0.01):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}")
        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError(f"y has {y.shape[0]} rows but X has {n_samples}")
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("y must be binary (0/1)")

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            preds = _sigmoid(z)
            error = preds - y

            grad_w = (X.T @ error) / n_samples + (self.l2 / n_samples) * self.weights
            grad_b = error.mean()

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            # Binary cross-entropy + L2 penalty, clipped to avoid log(0).
            eps = 1e-12
            bce = -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))
            reg = (self.l2 / (2 * n_samples)) * np.sum(self.weights ** 2)
            self.loss_history.append(bce + reg)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("LogisticRegression.predict_proba() called before fit()")
        return _sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **cm,
    }
