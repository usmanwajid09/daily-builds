"""BIRCH clustering from scratch (numpy only).

BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
compresses the data into a CF-tree of *clustering features* and then
clusters the resulting leaf subclusters.

A clustering feature (CF) summarises a set of points with three numbers::

    CF = (N, LS, SS)

where N is the point count, LS the linear (vector) sum of the points and
SS the sum of squared norms.  From a CF we recover the centroid (LS / N)
and the radius ``sqrt(SS / N - ||centroid||**2)`` without keeping the
points themselves.

The CF-tree is built by inserting points one by one.  A point joins the
closest leaf entry if the merged entry's radius stays within
``threshold``; otherwise it starts a new entry.  A leaf holding more
than ``branching_factor`` entries splits (two farthest entries become
seeds, the rest go to the nearest seed), and internal nodes split the
same way.  Once the tree is built, the leaf subcluster centroids are
grouped with a small deterministic k-means (unless ``n_clusters`` is
None, in which case each leaf subcluster is its own final cluster).
"""

import numpy as np


class _CF:
    """A clustering feature: (N, LS, SS) summary of a set of points."""

    def __init__(self, n, ls, ss):
        self.n = int(n)
        self.ls = np.asarray(ls, dtype=float)
        self.ss = float(ss)

    @classmethod
    def from_point(cls, x):
        x = np.asarray(x, dtype=float)
        return cls(1, x.copy(), float(x @ x))

    def centroid(self):
        return self.ls / self.n

    def radius(self):
        c = self.centroid()
        return float(np.sqrt(max(0.0, self.ss / self.n - float(c @ c))))

    def merged_with_point(self, x):
        """CF that would result from absorbing point ``x``."""
        x = np.asarray(x, dtype=float)
        return _CF(self.n + 1, self.ls + x, self.ss + float(x @ x))

    def merged_with(self, other):
        return _CF(self.n + other.n, self.ls + other.ls, self.ss + other.ss)

    def absorb(self, x):
        x = np.asarray(x, dtype=float)
        self.n += 1
        self.ls += x
        self.ss += float(x @ x)


class _Node:
    """One node of the CF-tree: a leaf holds CF entries, an internal
    node holds child nodes.  The node's CF is derived from its contents,
    so it is always up to date."""

    def __init__(self, leaf):
        self.leaf = leaf
        self.entries = [] if leaf else None   # list[_CF], leaves only
        self.children = [] if not leaf else None  # list[_Node], internal only

    def cf(self):
        """CF summarising everything below this node."""
        if self.leaf:
            items = self.entries
        else:
            items = [child.cf() for child in self.children]
        if not items:
            raise ValueError("cannot summarise an empty CF-tree node")
        out = items[0]
        for item in items[1:]:
            out = out.merged_with(item)
        return out

    def centroid(self):
        return self.cf().centroid()


def _farthest_pair(items):
    """Indices of the two items whose centroids are farthest apart."""
    cents = np.array([item.centroid() for item in items])
    best = (0, 1)
    best_d2 = -1.0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            d2 = float(((cents[i] - cents[j]) ** 2).sum())
            if d2 > best_d2:
                best_d2 = d2
                best = (i, j)
    return best


def _split_items(items):
    """Split ``items`` into two groups around the two farthest items.

    Returns (group_a, group_b); each remaining item joins the group
    whose seed centroid is nearer.
    """
    i, j = _farthest_pair(items)
    seed_a, seed_b = items[i], items[j]
    group_a = [seed_a]
    group_b = [seed_b]
    for k, item in enumerate(items):
        if k in (i, j):
            continue
        da = float(((item.centroid() - seed_a.centroid()) ** 2).sum())
        db = float(((item.centroid() - seed_b.centroid()) ** 2).sum())
        (group_a if da <= db else group_b).append(item)
    return group_a, group_b


def _kmeans(X, weights, k, seed=0, max_iter=100):
    """Deterministic weighted k-means with k-means++ style init.

    Returns (labels, centers).  ``X`` rows are the points to cluster,
    ``weights`` their weights.  The fixed seed makes results repeatable.
    """
    n = len(X)
    rng = np.random.default_rng(seed)
    centers = np.empty((k, X.shape[1]), dtype=float)

    # First centre: a random point (fixed seed -> deterministic).
    first = int(rng.integers(n))
    centers[0] = X[first]
    closest_d2 = ((X - centers[0]) ** 2).sum(axis=1)
    for c in range(1, k):
        total = float((weights * closest_d2).sum())
        if total <= 0.0:
            idx = int(rng.integers(n))
        else:
            probs = weights * closest_d2 / total
            idx = int(rng.choice(n, p=probs))
        centers[c] = X[idx]
        d2 = ((X - centers[c]) ** 2).sum(axis=1)
        closest_d2 = np.minimum(closest_d2, d2)

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        new_centers = centers.copy()
        for c in range(k):
            mask = new_labels == c
            if mask.any():
                new_centers[c] = np.average(X[mask], axis=0, weights=weights[mask])
            else:
                # Empty cluster: restart it at the worst-fitted point.
                worst = int(np.argmax(d2[np.arange(n), new_labels]))
                new_centers[c] = X[worst]
        shift = float(np.abs(new_centers - centers).max())
        centers, labels = new_centers, new_labels
        if shift < 1e-6:
            break
    return labels, centers


class BIRCH:
    """BIRCH clustering.

    Parameters
    ----------
    threshold : float, default 0.5
        Maximum radius of a subcluster stored in a leaf entry.  Smaller
        values give more (finer) leaf subclusters.
    branching_factor : int, default 50
        Maximum number of entries in a leaf, or children of an internal
        node, before it splits.
    n_clusters : int or None, default 3
        Number of final clusters.  If None, each leaf subcluster becomes
        its own final cluster.

    Attributes
    -----------
    labels_ : ndarray of shape (n_samples,)
        Final cluster label (0 .. n_clusters_ - 1) for each point.
    cluster_centers_ : ndarray of shape (n_clusters_, n_features)
        Centres of the final clusters.
    n_clusters_ : int
        Number of final clusters actually produced.
    n_subclusters_ : int
        Number of leaf subclusters in the CF-tree.
    subclusters_ : ndarray of shape (n_subclusters_, n_features)
        Centroids of the leaf subclusters.
    """

    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters

    # -- validation ----------------------------------------------------
    def _validate_params(self):
        t = self.threshold
        if isinstance(t, bool) or not isinstance(t, (int, float, np.floating)):
            raise ValueError(f"threshold must be a number, got {t!r}")
        if not np.isfinite(t) or t <= 0:
            raise ValueError(f"threshold must be finite and > 0, got {t!r}")
        b = self.branching_factor
        if isinstance(b, bool) or not isinstance(b, (int, np.integer)):
            raise ValueError(
                f"branching_factor must be an int >= 2, got {b!r}"
            )
        if b < 2:
            raise ValueError(
                f"branching_factor must be an int >= 2, got {b!r}"
            )
        k = self.n_clusters
        if k is not None:
            if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
                raise ValueError(f"n_clusters must be None or an int >= 1, got {k!r}")
            if k < 1:
                raise ValueError(f"n_clusters must be None or an int >= 1, got {k!r}")

    @staticmethod
    def _validate_X(X):
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("X must be a 2D numeric array") from None
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        if X.shape[0] < 2:
            raise ValueError(f"X must have at least 2 samples, got {X.shape[0]}")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        return X

    # -- CF-tree construction ------------------------------------------
    def _insert(self, node, x):
        """Insert point ``x`` into the subtree rooted at ``node``.

        Returns ``(left, right)`` if ``node`` split (the caller replaces
        ``node`` with ``left`` and adds ``right`` as a sibling), else None.
        """
        if node.leaf:
            if not node.entries:
                node.entries.append(_CF.from_point(x))
                return None
            cents = np.array([e.centroid() for e in node.entries])
            dists = ((cents - x) ** 2).sum(axis=1)
            nearest = int(np.argmin(dists))
            entry = node.entries[nearest]
            if entry.merged_with_point(x).radius() <= self.threshold:
                entry.absorb(x)
            else:
                node.entries.append(_CF.from_point(x))
            if len(node.entries) > self.branching_factor:
                group_a, group_b = _split_items(node.entries)
                left = _Node(leaf=True)
                left.entries = group_a
                right = _Node(leaf=True)
                right.entries = group_b
                return left, right
            return None

        # Internal node: descend to the closest child.
        cents = np.array([child.centroid() for child in node.children])
        dists = ((cents - x) ** 2).sum(axis=1)
        child = node.children[int(np.argmin(dists))]
        split = self._insert(child, x)
        if split is not None:
            left, right = split
            node.children[node.children.index(child)] = left
            node.children.append(right)
            if len(node.children) > self.branching_factor:
                group_a, group_b = _split_items(node.children)
                left_node = _Node(leaf=False)
                left_node.children = group_a
                right_node = _Node(leaf=False)
                right_node.children = group_b
                return left_node, right_node
        return None

    def _collect_leaves(self, node, out):
        if node.leaf:
            out.extend(node.entries)
        else:
            for child in node.children:
                self._collect_leaves(child, out)

    # -- fitting --------------------------------------------------------
    def fit(self, X):
        """Build the CF-tree from ``X`` and cluster the leaf subclusters.

        Returns self.
        """
        self._validate_params()
        X = self._validate_X(X)
        n_samples = X.shape[0]

        root = _Node(leaf=True)
        for x in X:
            split = self._insert(root, x)
            if split is not None:
                new_root = _Node(leaf=False)
                new_root.children = [split[0], split[1]]
                root = new_root
        self._root = root

        leaf_cfs = []
        self._collect_leaves(root, leaf_cfs)
        self.n_subclusters_ = len(leaf_cfs)
        self.subclusters_ = np.array([cf.centroid() for cf in leaf_cfs])
        weights = np.array([cf.n for cf in leaf_cfs], dtype=float)

        # Assign every point to its leaf subcluster.
        sub_labels = np.empty(n_samples, dtype=int)
        for i, x in enumerate(X):
            sub_labels[i] = self._walk_to_leaf_index(root, x)

        k = self.n_clusters
        if k is None or k >= self.n_subclusters_:
            # Each leaf subcluster is its own final cluster.
            self.n_clusters_ = self.n_subclusters_
            sub_to_cluster = np.arange(self.n_subclusters_)
        else:
            sub_to_cluster, _ = _kmeans(
                self.subclusters_, weights, k, seed=0
            )
            self.n_clusters_ = int(k)

        self.labels_ = sub_to_cluster[sub_labels]
        centers = []
        for c in range(self.n_clusters_):
            mask = sub_to_cluster == c
            if mask.any():
                centers.append(np.average(
                    self.subclusters_[mask], axis=0, weights=weights[mask]
                ))
            else:
                # Defensive: an empty final cluster falls back to the
                # overall data mean (should not happen in practice).
                centers.append(np.average(self.subclusters_, axis=0, weights=weights))
        self.cluster_centers_ = np.array(centers)
        return self

    def _walk_to_leaf_index(self, node, x):
        """Flat index of the leaf entry nearest to ``x``."""
        offset = 0
        while not node.leaf:
            cents = np.array([child.centroid() for child in node.children])
            child = node.children[int(np.argmin(((cents - x) ** 2).sum(axis=1)))]
            # Count the leaf entries skipped in earlier siblings.
            for sibling in node.children:
                if sibling is child:
                    break
                offset += self._count_leaf_entries(sibling)
            node = child
        cents = np.array([e.centroid() for e in node.entries])
        return offset + int(np.argmin(((cents - x) ** 2).sum(axis=1)))

    @staticmethod
    def _count_leaf_entries(node):
        if node.leaf:
            return len(node.entries)
        return sum(BIRCH._count_leaf_entries(c) for c in node.children)

    def fit_predict(self, X):
        """Fit the model to ``X`` and return the cluster labels."""
        return self.fit(X).labels_
