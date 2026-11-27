"""OPTICS (Ordering Points To Identify the Clustering Structure),
from scratch with numpy only.

OPTICS does not cluster the points directly. Instead it produces an
*ordering* of the points plus a *reachability distance* for each point
in that ordering, and clusters are read off the resulting reachability
plot: dense regions show up as valleys (low reachability), separated by
peaks where the ordering jumps from one dense region to another.

Three quantities drive it:

* **Core distance** of a point: the distance to its ``(min_samples - 1)``-th
  nearest *other* point. A point in a dense neighbourhood has a small
  core distance; a point in a sparse neighbourhood has a large one (or
  an infinite one if it has fewer than ``min_samples - 1`` neighbours
  within ``max_eps``). With ``min_samples == 1`` every point is trivially
  core, so the core distance is 0.
* **Reachability distance** of a point ``q`` with respect to a point
  ``p``: ``max(core_distance(p), distance(p, q))``. It is the smallest
  radius around ``p`` that both makes ``p`` core and reaches ``q``.
* **Ordering**: start from an arbitrary point, then repeatedly take the
  unprocessed point with the smallest reachability distance from any
  already-processed point (a priority queue of "seeds"). Points in the
  same dense region follow each other with small reachability distances,
  so the ordering walks through one cluster at a time.

Clusters are extracted from the reachability plot with the **xi-steepness
method**: a cluster boundary is a place where the reachability rises or
falls *steeply* -- by a factor of ``1 / (1 - xi)`` or more, i.e. a drop
or jump far larger than the ordinary point-to-point variation inside a
dense region. Concretely, the extraction looks for the largest
multiplicative gap in the sorted finite reachability values above the
typical (median) level; a gap counts as a separation only if it is at
least that steep, so ordinary within-cluster variation never splits a
cluster. Points on either side of such a gap belong to different
clusters, and the high-reachability "entry" point of a cluster (the
first point reached when the ordering jumps into a new dense region)
belongs to the cluster it enters. A run of points separated this way
only counts as a cluster if it holds at least ``min_samples`` points;
shorter runs -- isolated points and sparse stragglers -- are noise,
labelled ``-1``. If no steep gap exists, all connected points form a
single cluster. (The extraction assumes clusters of roughly similar
density: a much sparser cluster's own reachability level can sit above
the separation level of a denser one.)

Unlike k-means, OPTICS needs no ``k`` up front, finds clusters of
arbitrary shape, works when clusters have different densities (the
ordering, not a single global radius, decides), and explicitly labels
outliers as noise. The pairwise distance matrix is precomputed with
numpy broadcasting, which is fast enough for a few hundred points.
"""
from __future__ import annotations

import heapq

import numpy as np


class OPTICS:
    """Density-based clustering via an OPTICS ordering.

    Parameters
    ----------
    min_samples : int
        Number of neighbours that define a point's core distance: the
        core distance is the distance to the ``(min_samples - 1)``-th
        nearest other point. Also the minimum size of a cluster found
        by the xi extraction. Must be a positive integer.
    xi : float
        Steepness threshold for cluster extraction, in ``(0, 1)``.
        A rise/fall in reachability only counts as a cluster boundary
        if it is steep by a factor of at least ``1 / (1 - xi)`` (and at
        least a doubling, so ordinary within-cluster variation does not
        split a cluster). Smaller ``xi`` = less steep a jump needs to
        be to count as a boundary.
    max_eps : float or None
        Maximum radius considered when looking for neighbours. Points
        farther apart than ``max_eps`` are never directly reachable
        from each other. ``None`` (the default) means no limit. Must
        be a positive finite number if given.

    Attributes
    ----------
    ordering_ : np.ndarray of int, shape (n_samples,)
        Indices of the points in OPTICS order.
    reachability_ : np.ndarray of float, shape (n_samples,)
        Reachability distance of each point *in ordering order*;
        ``reachability_[0]`` is ``inf`` (the first point is reached
        from nowhere).
    core_distances_ : np.ndarray of float, shape (n_samples,)
        Core distance of each point, indexed by the *original* sample
        index (``inf`` where the point has too few neighbours to be
        core).
    labels_ : np.ndarray of int, shape (n_samples,)
        Cluster label per point in original order; ``-1`` is noise.
    n_clusters_ : int
        Number of clusters found (noise excluded).
    cluster_hierarchy_ : list of dict
        One entry per cluster: ``label``, ``start`` and ``end``
        (positions in the ordering), and ``size``.
    n_features_in_ : int
        Number of features seen at fit time.
    """

    def __init__(self, min_samples: int = 5, xi: float = 0.05,
                 max_eps: float | None = None):
        if isinstance(min_samples, bool) or not isinstance(
                min_samples, (int, np.integer)):
            raise ValueError("min_samples must be an integer")
        if min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        if isinstance(xi, bool) or not isinstance(
                xi, (int, float, np.floating)):
            raise ValueError("xi must be a number")
        if not np.isfinite(xi) or not 0 < xi < 1:
            raise ValueError("xi must be in the open interval (0, 1)")
        if max_eps is not None:
            if isinstance(max_eps, bool) or not isinstance(
                    max_eps, (int, float, np.floating)):
                raise ValueError("max_eps must be a positive number or None")
            if not np.isfinite(max_eps) or max_eps <= 0:
                raise ValueError(
                    "max_eps must be a positive finite number or None")
        self.min_samples = int(min_samples)
        self.xi = float(xi)
        self.max_eps = None if max_eps is None else float(max_eps)
        self.ordering_: np.ndarray | None = None
        self.reachability_: np.ndarray | None = None
        self.core_distances_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.n_clusters_: int = 0
        self.cluster_hierarchy_: list = []
        self.n_features_in_: int | None = None

    # ------------------------------------------------------------------
    def fit(self, X) -> "OPTICS":
        """Compute the OPTICS ordering and extract clusters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Must be a 2D numeric array with at least 2 samples and at
            least 1 feature, containing only finite values.
        """
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("X must be a 2D numeric array") from None
        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array of shape (n_samples, n_features)")
        if X.shape[0] < 2:
            raise ValueError("X must contain at least 2 samples")
        if X.shape[1] < 1:
            raise ValueError("X must contain at least 1 feature")
        if not np.isfinite(X).all():
            raise ValueError("X must not contain NaN or infinite values")

        n_samples = X.shape[0]
        self.n_features_in_ = X.shape[1]

        # Pairwise Euclidean distances via broadcasting.
        dists = np.sqrt(
            np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))

        self.core_distances_ = self._core_distances(dists)
        self.ordering_, self.reachability_ = self._ordering(dists)
        self.labels_ = self._extract_clusters()
        self.n_clusters_ = int(
            len(np.unique(self.labels_[self.labels_ >= 0]))
            if np.any(self.labels_ >= 0) else 0)
        return self

    # ------------------------------------------------------------------
    def _core_distances(self, dists: np.ndarray) -> np.ndarray:
        """Distance to the (min_samples - 1)-th nearest other point."""
        n = dists.shape[0]
        if self.min_samples == 1:
            # No other point needed: every point is trivially core.
            return np.zeros(n, dtype=float)
        if n - 1 < self.min_samples - 1:
            # Too few points for any of them to be core.
            return np.full(n, np.inf)
        # Distances to other points only (self-distance is 0, and
        # 0 * inf is nan, so mask the diagonal with where, not +inf).
        others = np.sort(
            np.where(np.eye(n, dtype=bool), np.inf, dists), axis=1)
        core = others[:, self.min_samples - 2].copy()
        if self.max_eps is not None:
            # Fewer than min_samples neighbours within max_eps means
            # the point is not core at this radius.
            core[core > self.max_eps] = np.inf
        return core

    # ------------------------------------------------------------------
    def _ordering(self, dists: np.ndarray):
        """OPTICS ordering via a min-heap of reachability seeds."""
        n = dists.shape[0]
        processed = np.zeros(n, dtype=bool)
        seeds = np.full(n, np.inf)
        heap: list = []
        ordering: list[int] = []
        reachability: list[float] = []

        def _process(idx: int, reach: float) -> None:
            processed[idx] = True
            ordering.append(idx)
            reachability.append(reach)
            core = self.core_distances_[idx]
            if not np.isfinite(core):
                return  # Not core: it reaches nothing.
            unproc = np.flatnonzero(~processed)
            if unproc.size == 0:
                return
            d = dists[idx, unproc]
            if self.max_eps is not None:
                keep = d <= self.max_eps
                unproc, d = unproc[keep], d[keep]
                if unproc.size == 0:
                    return
            new_reach = np.maximum(core, d)
            improved = new_reach < seeds[unproc]
            for q, r in zip(unproc[improved], new_reach[improved]):
                seeds[q] = r
                heapq.heappush(heap, (float(r), int(q)))

        while len(ordering) < n:
            # Take the unprocessed point with the smallest seed
            # reachability, if any seed is still live.
            nxt = None
            while heap:
                r, idx = heapq.heappop(heap)
                if processed[idx] or seeds[idx] != r:
                    continue  # Stale heap entry; skip it.
                nxt = (idx, r)
                break
            if nxt is not None:
                _process(nxt[0], nxt[1])
            else:
                # No live seeds: start a new component at the lowest
                # unprocessed index. Its reachability is undefined.
                idx = int(np.flatnonzero(~processed)[0])
                _process(idx, np.inf)

        return (np.array(ordering, dtype=int),
                np.array(reachability, dtype=float))

    # ------------------------------------------------------------------
    def _extract_clusters(self) -> np.ndarray:
        """Extract clusters from the reachability plot (xi method).

        A separation is a steep multiplicative gap in the sorted finite
        reachability values above the typical (median) level: the gap
        must be at least as steep as ``1 / (1 - xi)`` and at least a
        doubling, so ordinary within-cluster variation never splits a
        cluster. The *first* (lowest) such gap is the separation
        level -- a higher gap, e.g. the jump out to far noise points,
        must not raise the level and merge real clusters. Points above
        the gap level are separators; the entry point of a cluster (the
        separator just before a run of low reachability points) belongs
        to the cluster it enters. A run only counts as a cluster if,
        entry point included, it holds at least ``min_samples``
        points; anything shorter is noise.
        """
        n = len(self.ordering_)
        labels = np.full(n, -1, dtype=int)
        reach = self.reachability_
        finite = reach[np.isfinite(reach)]

        if finite.size == 0:
            return labels  # Nothing reachable: everything is noise.

        # --- find the separation level from the first steep gap -----
        level = float(np.median(finite))
        threshold = np.inf  # No separation: one connected region.
        if finite.size >= 2:
            s = np.sort(finite)
            # A gap only counts above the typical level, where the
            # lower side is a genuine within-cluster reachability.
            # Take the first (lowest) steep gap: a higher gap, such as
            # the jump out to far noise points, must not raise the
            # level and merge real clusters.
            above = s >= level
            ratios = np.where(
                s[:-1] > 0, s[1:] / np.maximum(s[:-1], 1e-300),
                np.inf)
            ratios[~above[:-1]] = 0.0
            steep = max(2.0, 1.0 / (1.0 - self.xi))
            steep_idx = np.flatnonzero(ratios >= steep)
            if steep_idx.size > 0:
                j = int(steep_idx[0])
                threshold = float(np.sqrt(s[j] * s[j + 1]))

        # --- split the ordering into runs of low reachability -------
        # A run is a maximal run of points with reachability <= the
        # threshold; the separator point just before a run is its
        # entry point and belongs to that run's cluster.
        hierarchy: list = []
        cluster_id = 0
        i = 0
        while i < n:
            if reach[i] > threshold:
                i += 1  # Separator (or ordering start): not in a run.
                continue
            j = i
            while j + 1 < n and reach[j + 1] <= threshold:
                j += 1
            # Run is ordering positions i..j; its entry point is the
            # separator at i - 1, if there is one.
            start = i - 1 if i > 0 else i
            size = j - start + 1
            if size >= self.min_samples:
                for pos in range(start, j + 1):
                    labels[self.ordering_[pos]] = cluster_id
                hierarchy.append({
                    "label": cluster_id,
                    "start": start,
                    "end": j,
                    "size": size,
                })
                cluster_id += 1
            # Short runs (and their entry points) stay labelled -1.
            i = j + 1

        self.cluster_hierarchy_ = hierarchy
        return labels

    # ------------------------------------------------------------------
    def reachability_plot_data(self):
        """Return ``(ordering, reachability)`` for the reachability plot.

        ``ordering`` holds the point indices in OPTICS order and
        ``reachability`` the reachability distance of each point in
        that order (first entry ``inf``). Plot reachability against
        ordering position: valleys are clusters, peaks are the jumps
        between them.
        """
        if self.ordering_ is None:
            raise ValueError("call fit(X) before reachability_plot_data()")
        return self.ordering_.copy(), self.reachability_.copy()

    def fit_predict(self, X) -> np.ndarray:
        """Fit and return ``labels_``."""
        return self.fit(X).labels_
