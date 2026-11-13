"""CART decision-tree classifier, from scratch with numpy only.

Splits are chosen greedily to minimise the weighted Gini impurity of the
two child nodes. Growth stops when a node is pure, when ``max_depth`` is
reached, or when a node has fewer than ``min_samples_split`` samples.
"""
from __future__ import annotations

import numpy as np


class _Node:
    """One node of the tree: either a split or a leaf prediction."""

    def __init__(self, *, feature=None, threshold=None, left=None, right=None,
                 prediction=None, n_samples=0, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
        self.n_samples = n_samples
        self.proba = proba  # class probabilities at this node (leaf only)

    @property
    def is_leaf(self) -> bool:
        return self.prediction is not None


class DecisionTreeClassifier:
    """CART decision-tree classifier using Gini impurity.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree. ``None`` grows until leaves are pure
        or ``min_samples_split`` stops growth.
    min_samples_split : int
        Minimum number of samples a node needs before it may be split.
    """

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2):
        if max_depth is not None:
            if not isinstance(max_depth, int) or isinstance(max_depth, bool):
                raise ValueError("max_depth must be an integer or None")
            if max_depth < 1:
                raise ValueError("max_depth must be at least 1")
        if not isinstance(min_samples_split, int) or isinstance(min_samples_split, bool):
            raise ValueError("min_samples_split must be an integer")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._root: _Node | None = None
        self.classes_: np.ndarray | None = None

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        self._root = self._grow(x, y_idx, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = self._check_x(x)
        preds = np.array([self.classes_[self._traverse(row, self._root)] for row in x])
        return preds

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities from the leaf's class frequencies."""
        self._check_fitted()
        x = self._check_x(x)
        rows = []
        for row in x:
            node = self._root
            while not node.is_leaf:
                node = node.left if row[node.feature] <= node.threshold else node.right
            rows.append(node.proba)
        return np.array(rows)

    @property
    def depth(self) -> int:
        """Depth of the fitted tree (0 for a single leaf)."""
        self._check_fitted()
        return self._depth(self._root)

    # -- internals ------------------------------------------------------
    def _check_fitted(self):
        if self._root is None or self.classes_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    def _check_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self._root.n_features:
            raise ValueError(
                "x has %d features, expected %d" % (x.shape[1], self._root.n_features)
            )
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        return x

    @staticmethod
    def _gini(counts: np.ndarray) -> float:
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        return float(1.0 - np.sum(p ** 2))

    def _best_split(self, x, y_idx):
        """Return (feature, threshold, gain) of the best split, or None."""
        n_samples, n_features = x.shape
        n_classes = len(self.classes_)
        parent_counts = np.bincount(y_idx, minlength=n_classes)
        parent_gini = self._gini(parent_counts)
        best = None
        for feature in range(n_features):
            order = np.argsort(x[:, feature], kind="stable")
            xs = x[order, feature]
            ys = y_idx[order]
            # Candidate thresholds are midpoints between distinct values.
            for i in range(1, n_samples):
                if xs[i] == xs[i - 1]:
                    continue
                left_counts = np.bincount(ys[:i], minlength=n_classes)
                right_counts = np.bincount(ys[i:], minlength=n_classes)
                weighted = (
                    i * self._gini(left_counts)
                    + (n_samples - i) * self._gini(right_counts)
                ) / n_samples
                gain = parent_gini - weighted
                if best is None or gain > best[2]:
                    best = (feature, float((xs[i] + xs[i - 1]) / 2.0), gain)
        if best is None or best[2] <= 0.0:
            return None
        return best

    def _grow(self, x, y_idx, depth):
        n_samples = x.shape[0]
        counts = np.bincount(y_idx, minlength=len(self.classes_))
        node = _Node(n_samples=n_samples)
        node.n_features = x.shape[1]
        prediction = int(np.argmax(counts))
        proba = counts / counts.sum()
        if (
            len(counts[counts > 0]) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
        ):
            node.prediction = prediction
            node.proba = proba
            return node
        split = self._best_split(x, y_idx)
        if split is None:
            node.prediction = prediction
            node.proba = proba
            return node
        feature, threshold, _gain = split
        mask = x[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(x[mask], y_idx[mask], depth + 1)
        node.right = self._grow(x[~mask], y_idx[~mask], depth + 1)
        return node

    @staticmethod
    def _traverse(row, node):
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.prediction

    def _depth(self, node) -> int:
        if node.is_leaf:
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))
