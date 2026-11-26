"""Bootstrap-aggregated decision-tree classifier, from scratch with numpy only.

Each base learner is a small CART tree (axis-aligned splits chosen by Gini
impurity) trained on a bootstrap resample of the data; predictions are by
majority vote. Bagging reduces the variance of the single tree.
"""
from __future__ import annotations

import numpy as np


class _Node:
    """One node of a CART tree: either a split or a leaf prediction."""

    def __init__(self, *, feature=None, threshold=None, left=None, right=None,
                 prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

    @property
    def is_leaf(self) -> bool:
        return self.prediction is not None


class _DecisionTree:
    """Minimal CART tree used as the bagging base learner."""

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2,
                 max_features: int | None = None, random_state: int | None = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self._root: _Node | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_: int | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_DecisionTree":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_ = x.shape[1]
        y_idx = np.searchsorted(self.classes_, y)
        self._root = self._grow(x, y_idx, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.array([
            self.classes_[self._traverse(row, self._root)] for row in x
        ])

    @staticmethod
    def _gini(counts: np.ndarray) -> float:
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        return float(1.0 - np.sum(p ** 2))

    def _best_split(self, x, y_idx, features):
        n_samples = x.shape[0]
        n_classes = len(self.classes_)
        parent_counts = np.bincount(y_idx, minlength=n_classes)
        parent_gini = self._gini(parent_counts)
        best = None
        for feature in features:
            order = np.argsort(x[:, feature], kind="stable")
            xs = x[order, feature]
            ys = y_idx[order]
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
        counts = np.bincount(y_idx, minlength=len(self.classes_))
        node = _Node()
        prediction = int(np.argmax(counts))
        if (
            len(counts[counts > 0]) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
            or x.shape[0] < self.min_samples_split
        ):
            node.prediction = prediction
            return node
        rng = np.random.default_rng(self.random_state)
        n_features = x.shape[1]
        if self.max_features is not None and self.max_features < n_features:
            features = rng.choice(n_features, size=self.max_features, replace=False)
        else:
            features = np.arange(n_features)
        split = self._best_split(x, y_idx, features)
        if split is None:
            node.prediction = prediction
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


class BaggingClassifier:
    """Bootstrap-aggregated (bagged) decision-tree classifier.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the ensemble.
    max_depth : int or None
        Maximum depth of each base tree.
    min_samples_split : int
        Minimum samples a node needs before it may be split.
    max_features : int or None
        Features considered at each split (random subset when set);
        ``None`` considers all features.
    random_state : int or None
        Seed for the bootstrap resampling.
    """

    def __init__(self, n_estimators: int = 10, max_depth: int | None = None,
                 min_samples_split: int = 2, max_features: int | None = None,
                 random_state: int | None = None):
        if not isinstance(n_estimators, int) or isinstance(n_estimators, bool):
            raise ValueError("n_estimators must be an integer")
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if max_depth is not None:
            if not isinstance(max_depth, int) or isinstance(max_depth, bool):
                raise ValueError("max_depth must be an integer or None")
            if max_depth < 1:
                raise ValueError("max_depth must be at least 1")
        if not isinstance(min_samples_split, int) or isinstance(min_samples_split, bool):
            raise ValueError("min_samples_split must be an integer")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if max_features is not None:
            if not isinstance(max_features, int) or isinstance(max_features, bool):
                raise ValueError("max_features must be an integer or None")
            if max_features < 1:
                raise ValueError("max_features must be at least 1")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_: list[_DecisionTree] = []
        self.classes_: np.ndarray | None = None

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaggingClassifier":
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
        if len(np.unique(y)) < 2:
            raise ValueError("need at least 2 classes to train a classifier")
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        n_samples = x.shape[0]
        self.estimators_ = []
        for i in range(self.n_estimators):
            idx = rng.integers(0, n_samples, size=n_samples)
            tree = _DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=None if self.random_state is None else int(rng.integers(0, 2 ** 31 - 1)),
            )
            tree.fit(x[idx], y[idx])
            self.estimators_.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = self._check_x(x)
        votes = np.array([tree.predict(x) for tree in self.estimators_])
        # Majority vote per sample.
        preds = []
        for col in votes.T:
            counts = np.bincount(
                np.searchsorted(self.classes_, col), minlength=len(self.classes_)
            )
            preds.append(self.classes_[int(np.argmax(counts))])
        return np.array(preds)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Fraction of trees voting for each class."""
        self._check_fitted()
        x = self._check_x(x)
        votes = np.array([tree.predict(x) for tree in self.estimators_])
        proba = np.zeros((x.shape[0], len(self.classes_)))
        for j, cls in enumerate(self.classes_):
            proba[:, j] = np.mean(votes == cls, axis=0)
        return proba

    # -- internals ------------------------------------------------------
    def _check_fitted(self):
        if not self.estimators_ or self.classes_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    def _check_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self.estimators_[0].n_features_:
            raise ValueError(
                "x has %d features, expected %d"
                % (x.shape[1], self.estimators_[0].n_features_)
            )
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        return x
