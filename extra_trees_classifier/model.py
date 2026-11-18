"""Extremely randomized trees classifier, from scratch with numpy only.

Like a random forest, but splits are chosen with extra randomness: at
each node, for each of ``max_features`` randomly chosen features a single
threshold is drawn uniformly at random between the feature's min and max
in that node, and the best of those random candidates (by Gini impurity)
is used. Full trees are grown on bootstrap samples (or the whole data),
and predictions are by majority vote across the forest.
"""
from __future__ import annotations

import numpy as np


def _gini(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - float(np.sum(p * p))


class _Node:
    """One node of an extra tree: a split or a leaf with class counts."""

    def __init__(self, *, feature=None, threshold=None, left=None,
                 right=None, counts=None, n_samples=0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.counts = counts  # leaf only: per-class sample counts
        self.n_samples = n_samples

    @property
    def is_leaf(self):
        return self.counts is not None


class _ExtraTree:
    """A single extremely randomized decision tree."""

    def __init__(self, *, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.root = None
        self.importances_ = None

    def fit(self, x, y, n_classes):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y).ravel()
        self.n_features_in_ = x.shape[1]
        self.n_classes_ = int(n_classes)
        self.importances_ = np.zeros(self.n_features_in_)
        self.root = self._grow(x, y, depth=0)
        total = self.importances_.sum()
        if total > 0:
            self.importances_ /= total
        return self

    def _grow(self, x, y, depth):
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        node = _Node(counts=counts, n_samples=len(y))
        if (len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)
                or len(np.unique(y)) == 1):
            return node
        split = self._best_random_split(x, y, counts)
        if split is None:
            return node
        feature, threshold, gain = split
        node.feature = feature
        node.threshold = threshold
        node.counts = None  # becomes an internal node
        self.importances_[feature] += gain
        left_mask = x[:, feature] <= threshold
        node.left = self._grow(x[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(x[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_random_split(self, x, y, counts):
        n, n_features = x.shape
        if self.max_features is None:
            k = n_features
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
        else:
            k = min(int(self.max_features), n_features)
        features = self.rng.choice(n_features, size=k, replace=False)
        parent_gini = _gini(counts)
        best = None
        for feature in features:
            col = x[:, feature]
            lo, hi = float(col.min()), float(col.max())
            if lo == hi:
                continue
            threshold = float(self.rng.uniform(lo, hi))
            left_mask = col <= threshold
            n_left = int(left_mask.sum())
            n_right = n - n_left
            if n_left < self.min_samples_leaf \
                    or n_right < self.min_samples_leaf:
                continue
            gini_left = _gini(np.bincount(y[left_mask],
                                          minlength=self.n_classes_))
            gini_right = _gini(np.bincount(y[~left_mask],
                                           minlength=self.n_classes_))
            weighted = (n_left * gini_left + n_right * gini_right) / n
            gain = parent_gini - weighted
            if best is None or gain > best[2]:
                best = (int(feature), threshold, float(gain))
        if best is None or best[2] <= 0:
            return None
        return best

    def predict_proba_row(self, row):
        node = self.root
        while not node.is_leaf:
            node = (node.left if row[node.feature] <= node.threshold
                    else node.right)
        total = node.counts.sum()
        return node.counts / total


class ExtraTreesClassifier:
    """Extremely randomized trees classifier.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum depth of each tree; ``None`` grows until leaves are pure.
    min_samples_split : int
        Minimum samples a node needs before it may be split.
    min_samples_leaf : int
        Minimum samples a leaf must contain.
    max_features : int, float or None
        Number of features considered at each split: an int count, a
        fraction of the features, or ``None`` (all features, like the
        original Extra-Trees paper's default of sqrt for classification
        is approximated by ``"sqrt"``).
    bootstrap : bool
        Whether each tree is trained on a bootstrap sample (default False,
        matching the Extra-Trees default of using the whole dataset).
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, *, n_estimators=100, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features="sqrt", bootstrap=False, random_state=None):
        if not isinstance(n_estimators, (int, np.integer)) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer, "
                             f"got {n_estimators!r}")
        if max_depth is not None and (not isinstance(max_depth, (int, np.integer))
                                      or max_depth < 1):
            raise ValueError("max_depth must be a positive integer or None, "
                             f"got {max_depth!r}")
        if not isinstance(min_samples_split, (int, np.integer)) \
                or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2, "
                             f"got {min_samples_split!r}")
        if not isinstance(min_samples_leaf, (int, np.integer)) \
                or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1, "
                             f"got {min_samples_leaf!r}")
        if isinstance(max_features, str) and max_features != "sqrt":
            raise ValueError('max_features string must be "sqrt", '
                             f"got {max_features!r}")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {x.shape}")
        if len(x) != len(y):
            raise ValueError("X and y have different lengths "
                             f"({len(x)} vs {len(y)})")
        if len(x) == 0:
            raise ValueError("Cannot fit on an empty dataset")
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("Need at least 2 classes to fit, "
                             f"got {len(classes)}")
        self.classes_ = classes
        self.n_classes_ = len(classes)
        self.n_features_in_ = x.shape[1]
        y_idx = np.searchsorted(classes, y)

        rng = np.random.default_rng(self.random_state)
        max_features = self.max_features
        if max_features == "sqrt":
            max_features = max(1, int(np.sqrt(self.n_features_in_)))

        self.trees_ = []
        self.feature_importances_ = np.zeros(self.n_features_in_)
        for _ in range(self.n_estimators):
            seed = int(rng.integers(0, 2 ** 31 - 1))
            if self.bootstrap:
                idx = rng.integers(0, len(x), size=len(x))
                x_tree, y_tree = x[idx], y_idx[idx]
            else:
                x_tree, y_tree = x, y_idx
            tree = _ExtraTree(max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split,
                              min_samples_leaf=self.min_samples_leaf,
                              max_features=max_features,
                              random_state=seed)
            tree.fit(x_tree, y_tree, self.n_classes_)
            self.feature_importances_ += tree.importances_
            self.trees_.append(tree)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def predict_proba(self, x):
        self._check_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {x.shape}")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {x.shape[1]} features, expected "
                             f"{self.n_features_in_}")
        proba = np.zeros((len(x), self.n_classes_))
        for tree in self.trees_:
            for i, row in enumerate(x):
                proba[i] += tree.predict_proba_row(row)
        return proba / len(self.trees_)

    def predict(self, x):
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, x, y):
        """Return classification accuracy on (x, y)."""
        return float(np.mean(self.predict(x) == np.asarray(y)))

    def _check_fitted(self):
        if not hasattr(self, "trees_"):
            raise ValueError("Model is not fitted yet — call fit() first")
