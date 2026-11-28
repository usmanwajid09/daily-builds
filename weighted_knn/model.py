"""Weighted k-nearest neighbours classifier from scratch, numpy only."""

import numpy as np


class WeightedKNNClassifier:
    """k-nearest neighbours classifier with optional distance weighting.

    Each neighbour votes for its class. With ``weights="uniform"`` every
    neighbour counts equally; with ``weights="distance"`` each vote is
    weighted by 1 / (distance + epsilon), so closer neighbours matter
    more — this often fixes the boundary errors of plain k-NN when
    classes overlap.

    Parameters
    ----------
    n_neighbors : int, default 5
        Number of neighbours to consult. Must be positive.
    weights : {"uniform", "distance"}, default "distance"
        How neighbour votes are weighted.
    p : int, default 2
        Minkowski power: 1 = Manhattan, 2 = Euclidean. Must be >= 1.
    """

    def __init__(self, n_neighbors=5, weights="distance", p=2):
        if n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be positive, got {n_neighbors!r}")
        if weights not in ("uniform", "distance"):
            raise ValueError(
                f"weights must be 'uniform' or 'distance', got {weights!r}")
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p!r}")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        """Store the training data (k-NN is a lazy learner).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} != {y.shape[0]}")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or infinite values")
        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) exceeds the number of "
                f"training samples ({X.shape[0]})")
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError(
                f"Need at least 2 classes, got {len(self.classes_)}")
        # Map labels to 0..n_classes-1 for bincount-based voting.
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self.X_train_ = X
        self.y_idx_ = np.array([self._class_to_idx[v] for v in y])
        return self

    def _neighbours(self, X):
        if not hasattr(self, "X_train_"):
            raise RuntimeError(
                "WeightedKNNClassifier is not fitted yet; call fit() first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.X_train_.shape[1]}")
        # Minkowski distances, shape (n_query, n_train).
        diff = np.abs(X[:, None, :] - self.X_train_[None, :, :])
        dist = np.sum(diff ** self.p, axis=2) ** (1.0 / self.p)
        k = self.n_neighbors
        idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
        # Sort the k neighbours by distance for determinism.
        order = np.argsort(np.take_along_axis(dist, idx, axis=1), axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        dist_k = np.take_along_axis(dist, idx, axis=1)
        return idx, dist_k

    def _vote_weights(self, dist_k):
        if self.weights == "uniform":
            return np.ones_like(dist_k)
        return 1.0 / (dist_k + 1e-8)

    def predict_proba(self, X):
        """Class probabilities from the weighted neighbour votes."""
        idx, dist_k = self._neighbours(X)
        w = self._vote_weights(dist_k)
        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes))
        for j in range(self.n_neighbors):
            np.add.at(proba, (np.arange(X.shape[0]), self.y_idx_[idx[:, j]]),
                      w[:, j])
        row_sums = proba.sum(axis=1, keepdims=True)
        return proba / row_sums

    def predict(self, X):
        """Predict class labels (ties go to the smallest class label)."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        """Mean accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
