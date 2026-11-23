"""Nearest-centroid classifier with optional shrinkage, from scratch.

Each class is summarised by its centroid (mean feature vector); a new
sample is assigned to the class whose centroid is nearest in Euclidean
distance. Optionally, centroids can be shrunk toward the overall mean —
shrinking by a factor ``shrinkage`` in [0, 1] — which regularises the
classifier and can help when a class has few samples or noisy features.
"""
from __future__ import annotations

import numpy as np


class NearestCentroid:
    """Nearest-centroid classifier.

    Parameters
    ----------
    shrinkage : float
        Fraction by which each class centroid is pulled toward the
        overall mean: 0.0 keeps the raw class centroids, 1.0 collapses
        every centroid onto the overall mean (and the classifier becomes
        useless — ties break to the first class). Must lie in [0, 1].
    metric : str
        Distance used to compare a sample with the centroids:
        ``"euclidean"`` (default) or ``"manhattan"``.
    """

    def __init__(self, shrinkage: float = 0.0, metric: str = "euclidean"):
        if not (0.0 <= shrinkage <= 1.0):
            raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage!r}")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError(
                f"metric must be 'euclidean' or 'manhattan', got {metric!r}"
            )
        self.shrinkage = float(shrinkage)
        self.metric = metric
        self.centroids_ = None
        self.classes_ = None
        self.overall_mean_ = None

    # -- validation -----------------------------------------------------
    @staticmethod
    def _check_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must have at least one sample and one feature")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def _check_fitted(self):
        if self.centroids_ is None:
            raise RuntimeError("NearestCentroid is not fitted yet; call fit() first")

    # -- fitting --------------------------------------------------------
    def fit(self, X, y):
        """Compute the (optionally shrunk) class centroids."""
        X = self._check_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if len(y) != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {len(y)} labels"
            )
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError(
                f"need at least 2 classes to fit, got {len(self.classes_)}"
            )
        self.overall_mean_ = X.mean(axis=0)
        centroids = np.zeros((len(self.classes_), X.shape[1]))
        for k in range(len(self.classes_)):
            centroids[k] = X[y_idx == k].mean(axis=0)
        # Shrink each centroid toward the overall mean.
        self.centroids_ = (
            (1.0 - self.shrinkage) * centroids
            + self.shrinkage * self.overall_mean_
        )
        return self

    # -- prediction -----------------------------------------------------
    def _distances(self, X):
        """Distance from each sample to each centroid."""
        if self.metric == "euclidean":
            diff = X[:, None, :] - self.centroids_[None, :, :]
            return np.sqrt(np.sum(diff ** 2, axis=2))
        diff = np.abs(X[:, None, :] - self.centroids_[None, :, :])
        return np.sum(diff, axis=2)

    def predict(self, X):
        """Predicted class labels (nearest centroid wins)."""
        self._check_fitted()
        X = self._check_X(X)
        if X.shape[1] != self.centroids_.shape[1]:
            raise ValueError(
                f"expected {self.centroids_.shape[1]} features, got {X.shape[1]}"
            )
        return self.classes_[np.argmin(self._distances(X), axis=1)]

    def predict_proba(self, X):
        """Softmax over negative distances, shape (n_samples, n_classes)."""
        self._check_fitted()
        X = self._check_X(X)
        if X.shape[1] != self.centroids_.shape[1]:
            raise ValueError(
                f"expected {self.centroids_.shape[1]} features, got {X.shape[1]}"
            )
        neg_dist = -self._distances(X)
        neg_dist = neg_dist - neg_dist.max(axis=1, keepdims=True)
        exp_d = np.exp(neg_dist)
        return exp_d / exp_d.sum(axis=1, keepdims=True)

    def score(self, X, y):
        """Mean accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
