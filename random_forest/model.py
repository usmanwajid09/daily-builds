"""Random-forest classifier, from scratch with numpy only.

Each tree is a small CART classifier (Gini impurity, greedy splits) grown
on a bootstrap resample of the data, and at every split only a random
subset of the features is considered. Predictions are by majority vote
across the trees, which reduces the variance of a single tree without
increasing its bias much.
"""
from __future__ import annotations

import numpy as np


class _Node:
    """One node of a tree: either a split or a leaf prediction."""

    def __init__(self, *, feature=None, threshold=None, left=None, right=None,
                 prediction=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
        self.proba = proba  # class probabilities at this leaf

    @property
    def is_leaf(self) -> bool:
        return self.prediction is not None


class _Tree:
    """A single CART tree with a random feature subset at each split."""

    def __init__(self, *, max_depth, min_samples_split, max_features, rng):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng
        self._root: _Node | None = None
        self._n_features = 0
        self.n_splits = 0  # used for feature importances

    def fit(self, x, y_idx, n_classes):
        self._n_features = x.shape[1]
        self._n_classes = n_classes
        self._root = self._grow(x, y_idx, depth=0)
        return self

    def predict_one(self, row):
        node = self._root
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.prediction

    def predict_proba_one(self, row):
        node = self._root
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.proba

    @staticmethod
    def _gini(counts):
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        return float(1.0 - np.sum(p ** 2))

    def _best_split(self, x, y_idx):
        n_samples, n_features = x.shape
        if self.max_features is None:
            features = np.arange(n_features)
        else:
            k = min(self.max_features, n_features)
            features = self.rng.choice(n_features, size=k, replace=False)
        parent_counts = np.bincount(y_idx, minlength=self._n_classes)
        parent_gini = self._gini(parent_counts)
        best = None
        for feature in features:
            order = np.argsort(x[:, feature], kind="stable")
            xs = x[order, feature]
            ys = y_idx[order]
            for i in range(1, n_samples):
                if xs[i] == xs[i - 1]:
                    continue
                left_counts = np.bincount(ys[:i], minlength=self._n_classes)
                right_counts = np.bincount(ys[i:], minlength=self._n_classes)
                weighted = (
                    i * self._gini(left_counts)
                    + (n_samples - i) * self._gini(right_counts)
                ) / n_samples
                gain = parent_gini - weighted
                if best is None or gain > best[2]:
                    best = (int(feature), float((xs[i] + xs[i - 1]) / 2.0), gain)
        if best is None or best[2] <= 0.0:
            return None
        return best

    def _grow(self, x, y_idx, depth):
        n_samples = x.shape[0]
        counts = np.bincount(y_idx, minlength=self._n_classes)
        node = _Node()
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
        self.n_splits += 1
        mask = x[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(x[mask], y_idx[mask], depth + 1)
        node.right = self._grow(x[~mask], y_idx[~mask], depth + 1)
        return node


class RandomForestClassifier:
    """Random-forest classifier: bagged CART trees with random splits.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest. Must be at least 1.
    max_depth : int or None
        Maximum depth of each tree. ``None`` grows until leaves are pure.
    min_samples_split : int
        Minimum samples a node needs before it may be split.
    max_features : {"sqrt", None} or int
        Number of features considered at each split: ``"sqrt"`` uses
        ``sqrt(n_features)``, ``None`` uses all features, an int uses
        exactly that many.
    random_state : int or None
        Seed for bootstrap sampling and feature subsets.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int | None = None,
                 min_samples_split: int = 2, max_features="sqrt",
                 random_state: int | None = None):
        if not isinstance(n_estimators, int) or isinstance(n_estimators, bool) \
                or n_estimators < 1:
            raise ValueError("n_estimators must be an integer of at least 1")
        if max_depth is not None and (
                not isinstance(max_depth, int) or isinstance(max_depth, bool)
                or max_depth < 1):
            raise ValueError("max_depth must be an integer of at least 1, or None")
        if (not isinstance(min_samples_split, int)
                or isinstance(min_samples_split, bool) or min_samples_split < 2):
            raise ValueError("min_samples_split must be an integer of at least 2")
        if max_features not in ("sqrt", None) and (
                not isinstance(max_features, int) or isinstance(max_features, bool)
                or max_features < 1):
            raise ValueError('max_features must be "sqrt", None, or a positive integer')
        if random_state is not None and (
                not isinstance(random_state, int) or isinstance(random_state, bool)):
            raise ValueError("random_state must be an integer or None")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self._trees: list[_Tree] = []
        self.classes_: np.ndarray | None = None
        self.feature_importances_: np.ndarray | None = None

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
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
        if isinstance(self.max_features, int) and self.max_features > x.shape[1]:
            raise ValueError(
                f"max_features={self.max_features} exceeds n_features={x.shape[1]}"
            )
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        n_samples, n_features = x.shape
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features is None:
            k = n_features
        else:
            k = self.max_features
        rng = np.random.default_rng(self.random_state)
        self._trees = []
        split_counts = np.zeros(n_features)
        for _ in range(self.n_estimators):
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            tree = _Tree(max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split,
                         max_features=k, rng=rng)
            tree.fit(x[boot_idx], y_idx[boot_idx], len(self.classes_))
            self._trees.append(tree)
            self._count_splits(tree._root, split_counts)
        total = split_counts.sum()
        self.feature_importances_ = (
            split_counts / total if total > 0 else np.zeros(n_features)
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = self._check_x(x)
        votes = np.zeros((x.shape[0], len(self.classes_)))
        for tree in self._trees:
            for i, row in enumerate(x):
                votes[i] += tree.predict_proba_one(row)
        return votes / len(self._trees)

    # -- internals ------------------------------------------------------
    def _check_fitted(self):
        if not self._trees or self.classes_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    def _check_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self._trees[0]._n_features:
            raise ValueError(
                "x has %d features, expected %d"
                % (x.shape[1], self._trees[0]._n_features)
            )
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        return x

    @staticmethod
    def _count_splits(node, counts):
        if node is None or node.is_leaf:
            return
        counts[node.feature] += 1
        RandomForestClassifier._count_splits(node.left, counts)
        RandomForestClassifier._count_splits(node.right, counts)
