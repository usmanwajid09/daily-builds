"""Perceptron classifier from scratch.

The classic Rosenblatt perceptron learns a linear decision boundary
``sign(w . x + b)`` by updating the weights on every misclassified
sample: ``w += y * x``, ``b += y``. On linearly separable data it is
guaranteed to converge; on non-separable data it would keep cycling, so
this implementation tracks the *pocket*: the best weights seen so far
(fewest training errors), and returns those. Training also stops early
once an epoch makes no errors.
"""

from __future__ import annotations

import numpy as np


class Perceptron:
    """Binary linear classifier trained with the perceptron rule.

    Args:
        max_epochs: maximum number of passes over the training data.
        learning_rate: step size for each weight update, must be > 0.
        random_state: seed for shuffling the sample order each epoch
            (shuffling speeds up convergence). ``None`` uses the data
            order as given.

    Attributes:
        coef_: weight vector, set by :meth:`fit`.
        intercept_: bias term, set by :meth:`fit`.
        n_epochs_: number of epochs actually run, set by :meth:`fit`.
        n_errors_: training errors of the returned (pocket) weights.
    """

    def __init__(self, max_epochs: int = 1000, learning_rate: float = 1.0,
                 random_state: int | None = None):
        if isinstance(max_epochs, bool) or not isinstance(max_epochs, (int, np.integer)):
            raise ValueError("max_epochs must be an integer")
        if int(max_epochs) < 1:
            raise ValueError("max_epochs must be at least 1")
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, (int, float, np.floating)
        ):
            raise ValueError("learning_rate must be a number > 0")
        if not np.isfinite(learning_rate) or float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be > 0")
        self.max_epochs = int(max_epochs)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_epochs_: int = 0
        self.n_errors_: int | None = None

    def _validate_labels(self, y: np.ndarray) -> np.ndarray:
        labels = np.unique(y)
        if labels.size != 2:
            raise ValueError(
                f"perceptron needs exactly two distinct labels, got {labels.tolist()}"
            )
        return y

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Perceptron":
        """Fit the classifier to training data.

        Labels may be any two distinct values (e.g. 0/1 or -1/+1);
        internally they are mapped to -1/+1, and :meth:`predict`
        returns the original labels.
        """
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
            raise ValueError("x must contain only finite values")
        self._validate_labels(y)

        labels = np.unique(y)
        y_signed = np.where(y == labels[1], 1.0, -1.0)
        self._labels = labels

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = x.shape
        weights = np.zeros(n_features, dtype=float)
        bias = 0.0
        # Pocket: best weights seen so far (fewest training errors).
        best_weights = weights.copy()
        best_bias = 0.0
        best_errors = n_samples + 1

        for epoch in range(1, self.max_epochs + 1):
            errors = 0
            order = rng.permutation(n_samples) if self.random_state is not None \
                else np.arange(n_samples)
            for i in order:
                if y_signed[i] * (x[i] @ weights + bias) <= 0.0:
                    weights += self.learning_rate * y_signed[i] * x[i]
                    bias += self.learning_rate * y_signed[i]
                    errors += 1
            # Score the weights after this epoch and keep the best.
            train_errors = int(np.sum(
                y_signed * (x @ weights + bias) <= 0.0
            ))
            if train_errors < best_errors:
                best_errors = train_errors
                best_weights = weights.copy()
                best_bias = bias
            self.n_epochs_ = epoch
            if errors == 0:
                break

        self.coef_ = best_weights
        self.intercept_ = float(best_bias)
        self.n_errors_ = best_errors
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the original labels for ``x``."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                "x has %d features, expected %d"
                % (x.shape[1], self.coef_.shape[0])
            )
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        signed = np.where(x @ self.coef_ + self.intercept_ >= 0.0, 1, 0)
        return self._labels[signed]

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the fraction of correct predictions on ``(x, y)``."""
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        preds = self.predict(x)
        if y.shape[0] != preds.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        return float(np.mean(preds == y))
