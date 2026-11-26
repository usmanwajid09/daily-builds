"""Passive-Aggressive online binary classifier, from scratch with numpy only.

On each example the PA algorithm stays passive (no update) when the
prediction is correct with enough margin, and otherwise aggressively
projects the weights onto the set of weights that classify the example
correctly. This implementation is PA-I: the update step is capped by
``C`` so a single noisy example cannot move the weights too far.
"""
from __future__ import annotations

import numpy as np


class PassiveAggressiveClassifier:
    """Passive-Aggressive (PA-I) online binary classifier.

    Parameters
    ----------
    C : float
        Aggressiveness cap on the update step (PA-I). Larger ``C`` lets
        single examples move the weights further.
    n_epochs : int
        Passes over the data in :meth:`fit`.
    random_state : int or None
        Seed for shuffling the examples each epoch in :meth:`fit`.
    """

    def __init__(self, C: float = 1.0, n_epochs: int = 5,
                 random_state: int | None = None):
        if not isinstance(C, (int, float)) or isinstance(C, bool):
            raise TypeError("C must be a number")
        if float(C) <= 0:
            raise ValueError("C must be positive, got %r" % (C,))
        if not isinstance(n_epochs, int) or isinstance(n_epochs, bool):
            raise ValueError("n_epochs must be an integer")
        if n_epochs < 1:
            raise ValueError("n_epochs must be at least 1")
        self.C = float(C)
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.classes_: np.ndarray | None = None
        self.n_updates_: int = 0
        self.n_seen_: int = 0

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray) -> "PassiveAggressiveClassifier":
        """Train over ``n_epochs`` passes; each pass shuffles the examples."""
        x, y_signed = self._check_xy(x, y)
        self._init_weights(x.shape[1])
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_epochs):
            order = rng.permutation(x.shape[0])
            for i in order:
                self._update(x[i], y_signed[i])
        return self

    def partial_fit(self, x: np.ndarray, y: np.ndarray,
                    classes: np.ndarray | None = None) -> "PassiveAggressiveClassifier":
        """Online update on one batch of examples (a single pass, no shuffle).

        The first call must pass ``classes`` (the two class labels) so the
        label mapping is known before any weight exists.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        if self.classes_ is None:
            if classes is None:
                raise ValueError(
                    "classes must be given on the first call to partial_fit"
                )
            classes = np.asarray(classes)
            if len(classes) != 2:
                raise ValueError("PassiveAggressiveClassifier is binary - need exactly 2 classes")
            self.classes_ = np.unique(classes)
            self._init_weights(x.shape[1])
        else:
            if x.shape[1] != self.coef_.shape[0]:
                raise ValueError(
                    "x has %d features, expected %d" % (x.shape[1], self.coef_.shape[0])
                )
        for xi, yi in zip(x, y):
            if yi not in self.classes_:
                raise ValueError("unknown class label %r" % (yi,))
            self._update(xi, self._to_signed(yi))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = self._check_x(x)
        scores = x @ self.coef_ + self.intercept_
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Signed distance to the decision boundary."""
        self._check_fitted()
        x = self._check_x(x)
        return x @ self.coef_ + self.intercept_

    # -- internals ------------------------------------------------------
    def _init_weights(self, n_features: int):
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

    def _to_signed(self, label) -> float:
        return 1.0 if label == self.classes_[1] else -1.0

    def _check_xy(self, x, y):
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
                "PassiveAggressiveClassifier is binary - got %d classes" % len(classes)
            )
        self.classes_ = classes
        return x, np.where(y == classes[1], 1.0, -1.0)

    def _check_fitted(self):
        if self.coef_ is None or self.classes_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() or partial_fit() first")

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

    def _update(self, xi: np.ndarray, y_signed: float):
        """One PA-I update for a single example."""
        margin = y_signed * (float(xi @ self.coef_) + self.intercept_)
        loss = max(0.0, 1.0 - margin)
        self.n_seen_ += 1
        if loss == 0.0:
            return
        norm_sq = float(xi @ xi)
        if norm_sq == 0.0:
            return
        tau = min(self.C, loss / norm_sq)
        self.coef_ += tau * y_signed * xi
        self.intercept_ += tau * y_signed
        self.n_updates_ += 1
