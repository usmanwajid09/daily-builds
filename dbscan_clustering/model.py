"""DBSCAN (Density-Based Spatial Clustering of Applications with Noise),
from scratch with numpy only.

A point is a *core point* if it has at least ``min_samples`` neighbours
within distance ``eps`` (counting itself). Clusters grow from core points
by density reachability: every point within ``eps`` of a core point joins
its cluster, and core points reached that way keep expanding it. A point
that is within ``eps`` of a core point but is not core itself is a *border
point* -- it joins the cluster but does not expand it. Any point that
belongs to no cluster is *noise*, labelled ``-1``.

Unlike k-means, DBSCAN needs no ``k`` up front, finds clusters of
arbitrary shape, and explicitly labels outliers as noise. The pairwise
distance matrix is precomputed with numpy broadcasting, which is fast
enough for a few hundred points.
"""
from __future__ import annotations

import numpy as np


class DBSCAN:
    """Density-based clustering with noise labelling.

    Parameters
    ----------
    eps : float
        Maximum distance between two points for them to count as
        neighbours. Must be a positive, finite number.
    min_samples : int
        Minimum number of neighbours within ``eps`` (including the
        point itself) for a point to be a core point. Must be a
        positive integer.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        if isinstance(eps, bool) or not isinstance(eps, (int, float, np.floating)):
            raise ValueError("eps must be a number")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a positive finite number")
        if isinstance(min_samples, bool) or not isinstance(min_samples, (int, np.integer)):
            raise ValueError("min_samples must be an integer")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.labels_: np.ndarray | None = None
        self.n_clusters_: int = 0
        self.core_sample_indices_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, x: np.ndarray) -> "DBSCAN":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")

        n_samples = x.shape[0]
        self.n_features_in_ = x.shape[1]

        # Pairwise Euclidean distances via broadcasting.
        dists = np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
        neighbours = dists <= self.eps
        is_core = neighbours.sum(axis=1) >= self.min_samples
        self.core_sample_indices_ = np.flatnonzero(is_core)

        labels = np.full(n_samples, -1, dtype=int)
        cluster_id = 0
        for i in range(n_samples):
            if labels[i] != -1 or not is_core[i]:
                continue
            # Start a new cluster from core point i and expand it.
            labels[i] = cluster_id
            queue = [i]
            head = 0
            while head < len(queue):
                point = queue[head]
                head += 1
                for j in np.flatnonzero(neighbours[point]):
                    if labels[j] == -1:
                        # Unlabelled point joins the cluster; only core
                        # points expand it further.
                        labels[j] = cluster_id
                        if is_core[j]:
                            queue.append(j)
            cluster_id += 1

        self.labels_ = labels
        self.n_clusters_ = cluster_id
        return self

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.labels_
