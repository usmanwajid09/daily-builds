"""K-means clustering, from scratch with numpy only.

Centroids are initialised with k-means++ (the first centroid is picked at
random, each next one with probability proportional to its squared
distance to the nearest existing centroid), then Lloyd's algorithm
alternates assignment and centroid updates until the assignments stop
changing or ``max_iter`` is reached. ``inertia_`` is the sum of squared
distances of samples to their assigned centroid.
"""
from __future__ import annotations

import numpy as np


class KMeans:
    """K-means clustering with k-means++ initialisation.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    max_iter : int
        Maximum number of Lloyd iterations per run.
    n_init : int
        Number of times to run with different centroid seeds; the run
        with the lowest inertia is kept.
    tol : float
        Stop when the change in inertia between iterations falls below this.
    random_state : int or None
        Seed for reproducible centroid initialisation.
    """

    def __init__(self, n_clusters: int = 3, max_iter: int = 300, n_init: int = 10,
                 tol: float = 1e-4, random_state: int | None = None):
        if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
            raise ValueError("n_clusters must be an integer")
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(n_init, int) or isinstance(n_init, bool) or n_init < 1:
            raise ValueError("n_init must be a positive integer")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int = 0

    def fit(self, x: np.ndarray) -> "KMeans":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        if self.n_clusters > x.shape[0]:
            raise ValueError(
                "n_clusters (%d) cannot exceed the number of samples (%d)"
                % (self.n_clusters, x.shape[0])
            )
        best = None
        for run in range(self.n_init):
            rng = np.random.default_rng(
                None if self.random_state is None else self.random_state + run
            )
            centers = self._init_centers(x, rng)
            centers, labels, inertia, n_iter = self._lloyd(x, centers)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia, n_iter)
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(
                "x has %d features, expected %d"
                % (x.shape[1], self.cluster_centers_.shape[1])
            )
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        return self._assign(x, self.cluster_centers_)

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.labels_

    # -- internals ------------------------------------------------------
    def _init_centers(self, x, rng) -> np.ndarray:
        n_samples = x.shape[0]
        centers = [x[rng.integers(n_samples)]]
        closest_sq = np.sum((x - centers[0]) ** 2, axis=1)
        for _ in range(1, self.n_clusters):
            total = closest_sq.sum()
            if total == 0:
                # All points coincide with an existing centroid.
                idx = rng.integers(n_samples)
            else:
                idx = rng.choice(n_samples, p=closest_sq / total)
            centers.append(x[idx])
            closest_sq = np.minimum(
                closest_sq, np.sum((x - centers[-1]) ** 2, axis=1)
            )
        return np.array(centers)

    @staticmethod
    def _assign(x, centers) -> np.ndarray:
        dists = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        return np.argmin(dists, axis=1)

    @staticmethod
    def _inertia(x, labels, centers) -> float:
        return float(np.sum((x - centers[labels]) ** 2))

    def _lloyd(self, x, centers):
        prev_inertia = None
        labels = None
        for iteration in range(self.max_iter):
            labels = self._assign(x, centers)
            new_centers = centers.copy()
            for k in range(self.n_clusters):
                members = x[labels == k]
                if len(members) > 0:
                    new_centers[k] = members.mean(axis=0)
                # Empty cluster: keep its previous centroid.
            centers = new_centers
            inertia = self._inertia(x, labels, centers)
            if prev_inertia is not None and abs(prev_inertia - inertia) < self.tol:
                return centers, labels, inertia, iteration + 1
            prev_inertia = inertia
        return centers, labels, self._inertia(x, labels, centers), self.max_iter
