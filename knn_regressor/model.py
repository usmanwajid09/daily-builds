"""K-nearest neighbours regressor, from scratch with numpy only."""
from __future__ import annotations

import numpy as np


class KNNRegressor:
    """Predicts a numeric target as a weighted mean of the k nearest targets.

    Distance is Euclidean. With ``weights="uniform"`` every neighbour
    counts equally; with ``weights="distance"`` each neighbour is
    weighted by the inverse of its distance, so nearer points matter
    more. A query point that exactly matches a training point (distance
    0) returns the mean of the matching training targets.
    """

    def __init__(self, k: int = 5, weights: str = "uniform"):
        if not isinstance(k, int) or isinstance(k, bool):
            raise ValueError("k must be an integer")
        if k < 1:
            raise ValueError("k must be at least 1")
        if weights not in ("uniform", "distance"):
            raise ValueError('weights must be "uniform" or "distance"')
        self.k = k
        self.weights = weights
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of targets")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        if not np.isfinite(y).all():
            raise ValueError("y must contain only finite values")
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
        if x.shape[0] == 0:
            raise ValueError("cannot predict on empty data")
        if x.shape[1] != self._x_train.shape[1]:
            raise ValueError(
                "x has %d features, expected %d"
                % (x.shape[1], self._x_train.shape[1])
            )
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        # (n_query, n_train) Euclidean distances.
        diff = x[:, None, :] - self._x_train[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        nearest = np.argsort(dists, axis=1)[:, : self.k]
        preds = []
        for row, neighbours in zip(dists, nearest):
            targets = self._y_train[neighbours]
            if self.weights == "uniform":
                preds.append(float(np.mean(targets)))
            else:
                neighbour_dists = row[neighbours]
                zero_mask = neighbour_dists == 0.0
                if zero_mask.any():
                    # Exact match: average only the zero-distance targets.
                    preds.append(float(np.mean(targets[zero_mask])))
                else:
                    weights = 1.0 / neighbour_dists
                    preds.append(float(np.sum(weights * targets) / np.sum(weights)))
        return np.asarray(preds, dtype=float)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the R^2 coefficient of determination on the given data."""
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of targets")
        preds = self.predict(x)
        if y.shape[0] != preds.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot
