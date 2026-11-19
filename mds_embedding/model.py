"""Classical multidimensional scaling (MDS), from scratch with numpy only.

Given an n x n matrix of pairwise distances, classical MDS recovers
coordinates in a low-dimensional space whose Euclidean distances match
the input as closely as possible. It double-centres the squared
distance matrix, B = -0.5 * J D^2 J, takes the eigendecomposition of B,
and keeps the top-k positive eigenvalues and their eigenvectors.
Kruskal's stress-1 measures how well the embedding reproduces the
input distances.
"""
from __future__ import annotations

import numpy as np


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix for rows of ``x``."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
    if x.shape[0] == 0:
        raise ValueError("cannot compute distances for empty data")
    if not np.isfinite(x).all():
        raise ValueError("x must not contain NaN or infinite values")
    sq = np.sum(x ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * x @ x.T
    np.maximum(d2, 0.0, out=d2)  # guard against tiny negative round-off
    return np.sqrt(d2)


def double_centre(d: np.ndarray) -> np.ndarray:
    """Return B = -0.5 * J D^2 J for a distance matrix ``d``."""
    d = np.asarray(d, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("d must be a square distance matrix")
    if d.shape[0] == 0:
        raise ValueError("cannot double-centre an empty matrix")
    if not np.isfinite(d).all():
        raise ValueError("d must not contain NaN or infinite values")
    if np.any(d < 0):
        raise ValueError("d must not contain negative distances")
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * j @ (d ** 2) @ j


def kruskal_stress(d: np.ndarray, embedding: np.ndarray) -> float:
    """Kruskal stress-1 between input distances and embedding distances.

    stress = sqrt(sum((d - d_hat)^2) / sum(d^2)) over the upper
    triangle. 0 means the distances are reproduced exactly.
    """
    d = np.asarray(d, dtype=float)
    embedding = np.asarray(embedding, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("d must be a square distance matrix")
    if embedding.ndim != 2 or embedding.shape[0] != d.shape[0]:
        raise ValueError("embedding must have one row per point in d")
    iu = np.triu_indices(d.shape[0], k=1)
    d_flat = d[iu]
    d_hat = pairwise_distances(embedding)[iu]
    denom = float(np.sum(d_flat ** 2))
    if denom == 0:
        raise ValueError("cannot compute stress: all input distances are zero")
    return float(np.sqrt(np.sum((d_flat - d_hat) ** 2) / denom))


class ClassicalMDS:
    """Classical (metric) multidimensional scaling.

    Parameters
    ----------
    n_components : int
        Number of dimensions of the embedding. Must be at least 1 and
        at most n_samples - 1.

    Attributes
    -----------
    embedding_ : ndarray, shape (n_samples, n_components)
        The recovered coordinates.
    eigenvalues_ : ndarray
        The top ``n_components`` eigenvalues of the double-centred
        matrix (largest first).
    stress_ : float
        Kruskal stress-1 of the embedding against the input distances.
    """

    def __init__(self, n_components: int = 2):
        if not isinstance(n_components, int) or isinstance(n_components, bool) \
                or n_components < 1:
            raise ValueError("n_components must be an integer of at least 1")
        self.n_components = n_components
        self.embedding_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.stress_: float | None = None

    def fit(self, d: np.ndarray) -> "ClassicalMDS":
        """Fit from a square distance matrix ``d``."""
        d = np.asarray(d, dtype=float)
        if d.ndim != 2 or d.shape[0] != d.shape[1]:
            raise ValueError("d must be a square distance matrix")
        n = d.shape[0]
        if n < 2:
            raise ValueError("need at least 2 points to embed")
        if self.n_components > n - 1:
            raise ValueError(
                f"n_components={self.n_components} exceeds n_samples - 1 = {n - 1}"
            )
        if not np.isfinite(d).all():
            raise ValueError("d must not contain NaN or infinite values")
        if np.any(d < 0):
            raise ValueError("d must not contain negative distances")
        if not np.allclose(np.diag(d), 0.0):
            raise ValueError("diagonal of d must be zero (distance to self)")
        if not np.allclose(d, d.T):
            raise ValueError("d must be symmetric")
        b = double_centre(d)
        # B is symmetric, so eigh is exact and returns sorted eigenvalues.
        values, vectors = np.linalg.eigh(b)
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = vectors[:, order]
        top_values = values[: self.n_components]
        if np.any(top_values <= 0):
            raise ValueError(
                "distance matrix is not Euclidean: the top eigenvalues "
                "of the double-centred matrix are not all positive"
            )
        self.eigenvalues_ = top_values
        self.embedding_ = vectors[:, : self.n_components] * np.sqrt(top_values)
        self.stress_ = kruskal_stress(d, self.embedding_)
        return self

    def fit_transform(self, d: np.ndarray) -> np.ndarray:
        """Fit from a distance matrix and return the embedding."""
        self.fit(d)
        return self.embedding_

    @classmethod
    def from_points(cls, x: np.ndarray, n_components: int = 2) -> "ClassicalMDS":
        """Fit from raw points: distances are computed first."""
        model = cls(n_components=n_components)
        model.fit(pairwise_distances(x))
        return model
