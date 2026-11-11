"""K-nearest neighbours classifier, from scratch with numpy only."""
from __future__ import annotations

import numpy as np


class KNNClassifier:
    """Classifies by majority vote of the k nearest training points.

    Distance is Euclidean. Ties in the vote are broken by the label of
    the single nearest neighbour among the tied labels, so predictions
    are deterministic for a given training set.
    """

    def __init__(self, k: int = 5):
        if not isinstance(k, int) or isinstance(k, bool):
            raise ValueError("k must be an integer")
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier":
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
        if self.k > x.shape[0]:
            raise ValueError("k cannot be larger than the number of training samples")
        self._x_train = x
        self._y_train = y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._x_train is None or self._y_train is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self._x_train.shape[1]:
            raise ValueError(
                "x has %d features, expected %d"
                % (x.shape[1], self._x_train.shape[1])
            )
        # (n_query, n_train) squared Euclidean distances.
        diff = x[:, None, :] - self._x_train[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        nearest = np.argsort(dists, axis=1)[:, : self.k]
        preds = []
        for row, neighbours in zip(dists, nearest):
            labels = self._y_train[neighbours]
            values, counts = np.unique(labels, return_counts=True)
            best_count = counts.max()
            tied = values[counts == best_count]
            if len(tied) == 1:
                preds.append(tied[0])
            else:
                # Tie-break: label of the nearest neighbour among the tied.
                for idx in neighbours:
                    if self._y_train[idx] in tied:
                        preds.append(self._y_train[idx])
                        break
        return np.asarray(preds)
