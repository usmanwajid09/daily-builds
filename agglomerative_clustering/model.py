"""Agglomerative (hierarchical) clustering from scratch, numpy only.

Starts with every point in its own cluster and repeatedly merges the two
closest clusters until ``n_clusters`` remain. Cluster distance is defined
by the chosen linkage rule, recomputed from member points at each merge
(simple and correct for small data):

- ``"single"``:   distance between the closest pair of points.
- ``"complete"``: distance between the farthest pair of points.
- ``"average"``:  mean distance over all cross pairs of points.
"""

import numpy as np


class AgglomerativeClustering:
    """Hierarchical agglomerative clustering.

    Parameters
    ----------
    n_clusters : int, default 2
        Number of clusters to stop at (1 <= n_clusters <= n_samples).
    linkage : str, default "complete"
        One of "single", "complete", "average".

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster label for each sample, 0 .. n_clusters - 1.
    merges_ : list of (cluster_a, cluster_b, distance, size)
        One entry per merge, in merge order. ``cluster_a``/``cluster_b``
        are the ids of the merged clusters (original point indices for
        singletons, merge index + n_samples for merged clusters) and
        ``size`` is the size of the newly formed cluster.
    children_ : list of (cluster_a, cluster_b)
        Merge pairs only, for inspection (same order as ``merges_``).
    n_clusters_ : int
        Number of clusters produced (== n_clusters).
    """

    _LINKAGES = ("single", "complete", "average")

    def __init__(self, n_clusters=2, linkage="complete"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        """Fit the clustering to X and store ``labels_`` etc.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. Must be 2D, non-empty, and finite.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D array")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be non-empty")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values (no NaN or inf)")
        n_samples = X.shape[0]
        if not isinstance(self.n_clusters, (int, np.integer)) or isinstance(
            self.n_clusters, bool
        ):
            raise ValueError(
                f"n_clusters must be an integer, got {self.n_clusters!r}"
            )
        if self.n_clusters < 1:
            raise ValueError(
                f"n_clusters must be >= 1, got {self.n_clusters}"
            )
        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) must be <= n_samples ({n_samples})"
            )
        if self.linkage not in self._LINKAGES:
            raise ValueError(
                f"linkage must be one of {list(self._LINKAGES)}, "
                f"got {self.linkage!r}"
            )

        # Pairwise Euclidean distances between points.
        diff = X[:, None, :] - X[None, :, :]
        point_dist = np.sqrt((diff ** 2).sum(axis=-1))

        # Each cluster is a list of point indices; id = point index for
        # singletons, n_samples + merge_index for merged clusters.
        clusters = {i: [i] for i in range(n_samples)}
        next_id = n_samples
        merges = []
        children = []

        def cluster_distance(members_a, members_b):
            d = point_dist[np.ix_(members_a, members_b)]
            if self.linkage == "single":
                return float(d.min())
            if self.linkage == "complete":
                return float(d.max())
            return float(d.mean())  # average

        while len(clusters) > self.n_clusters:
            ids = sorted(clusters)
            best = None  # (distance, id_a, id_b)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    d = cluster_distance(clusters[ids[i]], clusters[ids[j]])
                    if best is None or d < best[0]:
                        best = (d, ids[i], ids[j])
            dist, id_a, id_b = best
            merged = clusters.pop(id_a) + clusters.pop(id_b)
            clusters[next_id] = merged
            merges.append((id_a, id_b, dist, len(merged)))
            children.append((id_a, id_b))
            next_id += 1

        # Relabel clusters 0 .. n_clusters-1 (in sorted cluster-id order).
        labels = np.empty(n_samples, dtype=int)
        for label, cid in enumerate(sorted(clusters)):
            for idx in clusters[cid]:
                labels[idx] = label

        self.labels_ = labels
        self.merges_ = merges
        self.children_ = children
        self.n_clusters_ = self.n_clusters
        return self

    def fit_predict(self, X):
        """Fit to X and return the cluster labels."""
        self.fit(X)
        return self.labels_
