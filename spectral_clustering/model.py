"""Spectral clustering, from scratch with numpy only.

Builds an RBF affinity graph over the data, forms the symmetric
normalized graph Laplacian, embeds the data in the space spanned by its
smallest eigenvectors, and clusters that embedding with k-means
(k-means++ init). Non-convex clusters such as concentric rings or moons,
which k-means on the raw features cannot split, become linearly
separable in the spectral embedding.
"""
from __future__ import annotations

import numpy as np


def _kmeans(x, n_clusters, *, random_state=None, max_iter=100, n_init=10):
    """Small k-means with k-means++ init; returns (labels, inertia)."""
    rng = np.random.default_rng(random_state)
    n = len(x)
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        centers = np.empty((n_clusters, x.shape[1]))
        first = int(rng.integers(n))
        centers[0] = x[first]
        closest_sq = np.sum((x - centers[0]) ** 2, axis=1)
        for c in range(1, n_clusters):
            total = closest_sq.sum()
            if total <= 0:
                centers[c] = x[int(rng.integers(n))]
            else:
                probs = closest_sq / total
                centers[c] = x[rng.choice(n, p=probs)]
            closest_sq = np.minimum(
                closest_sq, np.sum((x - centers[c]) ** 2, axis=1))
        for _ in range(max_iter):
            dists = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)
            new_centers = centers.copy()
            for c in range(n_clusters):
                if np.any(labels == c):
                    new_centers[c] = x[labels == c].mean(axis=0)
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers
        dists = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        inertia = float(np.sum(dists[np.arange(n), labels]))
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels, best_inertia


class SpectralClustering:
    """Spectral clustering with an RBF affinity graph.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to find.
    gamma : float or None
        RBF affinity coefficient: affinity = exp(-gamma * ||x - z||^2).
        ``None`` uses ``1 / n_features``.
    assign_labels : str
        Only ``"kmeans"`` is supported (k-means on the spectral embedding).
    random_state : int or None
        Seed for the k-means stage.
    """

    def __init__(self, *, n_clusters=2, gamma=None, assign_labels="kmeans",
                 random_state=None):
        if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 2:
            raise ValueError("n_clusters must be an integer >= 2, "
                             f"got {n_clusters!r}")
        if gamma is not None and (not np.isfinite(gamma) or gamma <= 0):
            raise ValueError(f"gamma must be positive, got {gamma!r}")
        if assign_labels != "kmeans":
            raise ValueError('assign_labels must be "kmeans", '
                             f"got {assign_labels!r}")
        self.n_clusters = int(n_clusters)
        self.gamma = gamma
        self.assign_labels = assign_labels
        self.random_state = random_state

    def _affinity(self, x):
        gamma = self.gamma if self.gamma is not None \
            else 1.0 / x.shape[1]
        sq = (np.sum(x ** 2, axis=1)[:, None]
              + np.sum(x ** 2, axis=1)[None, :]
              - 2.0 * x @ x.T)
        w = np.exp(-gamma * np.maximum(sq, 0.0))
        np.fill_diagonal(w, 0.0)
        return w

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {x.shape}")
        if len(x) == 0:
            raise ValueError("Cannot fit on an empty dataset")
        if len(x) <= self.n_clusters:
            raise ValueError("Need more samples than clusters: "
                             f"{len(x)} samples, {self.n_clusters} clusters")
        self.n_features_in_ = x.shape[1]

        w = self._affinity(x)
        degrees = w.sum(axis=1)
        if np.any(degrees <= 0):
            raise ValueError("Affinity graph has isolated points (zero "
                             "degree) — try a larger gamma or check the data")
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
        # Symmetric normalized Laplacian: L = I - D^-1/2 W D^-1/2.
        laplacian = np.eye(len(x)) \
            - (d_inv_sqrt[:, None] * w * d_inv_sqrt[None, :])
        eigvals, eigvecs = np.linalg.eigh(laplacian)
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        self.eigenvalues_ = eigvals[:self.n_clusters]
        embedding = eigvecs[:, :self.n_clusters]
        # Row-normalise the embedding (Ng-Jordan-Weiss step).
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embedding = embedding / norms
        self.embedding_ = embedding

        labels, inertia = _kmeans(embedding, self.n_clusters,
                                  random_state=self.random_state)
        self.labels_ = labels
        self.inertia_ = inertia
        return self

    def fit_predict(self, x):
        """Fit the model and return the cluster labels."""
        return self.fit(x).labels_
