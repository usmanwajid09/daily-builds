"""Gradient-boosting regressor, from scratch with numpy only.

Builds an additive model of shallow regression trees. It starts from the
mean of ``y`` and, at each stage, fits a small tree to the current
residuals (the negative gradient of the squared loss) and adds a shrunken
copy of it to the prediction. Shrinkage (``learning_rate``) plus shallow
trees is what keeps the model from memorising the training data.
"""
from __future__ import annotations

import numpy as np


class _Node:
    """One node of a regression tree: a split or a leaf value."""

    def __init__(self, *, feature=None, threshold=None, left=None, right=None,
                 value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class _RegressionTree:
    """A small CART regression tree fit by minimising squared error."""

    def __init__(self, *, max_depth, min_samples_leaf):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._root: _Node | None = None

    def fit(self, x, y):
        self._root = self._grow(x, y, depth=0)
        return self

    def predict(self, x):
        return np.array([self._predict_one(row) for row in x])

    def _predict_one(self, row):
        node = self._root
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def _best_split(self, x, y):
        n_samples, n_features = x.shape
        parent_sse = float(np.sum((y - y.mean()) ** 2))
        best = None
        for feature in range(n_features):
            order = np.argsort(x[:, feature], kind="stable")
            xs = x[order, feature]
            ys = y[order]
            cumsum = np.cumsum(ys)
            total = cumsum[-1]
            for i in range(1, n_samples):
                if xs[i] == xs[i - 1]:
                    continue
                if i < self.min_samples_leaf or n_samples - i < self.min_samples_leaf:
                    continue
                left_mean = cumsum[i - 1] / i
                right_mean = (total - cumsum[i - 1]) / (n_samples - i)
                sse = float(
                    np.sum((ys[:i] - left_mean) ** 2)
                    + np.sum((ys[i:] - right_mean) ** 2)
                )
                gain = parent_sse - sse
                if best is None or gain > best[2]:
                    best = (feature, float((xs[i] + xs[i - 1]) / 2.0), gain)
        if best is None or best[2] <= 0.0:
            return None
        return best

    def _grow(self, x, y, depth):
        node = _Node(value=float(y.mean()))
        if (
            depth >= self.max_depth
            or len(y) < 2 * self.min_samples_leaf
            or np.all(y == y[0])
        ):
            return node
        split = self._best_split(x, y)
        if split is None:
            return node
        feature, threshold, _gain = split
        mask = x[:, feature] <= threshold
        node.value = None
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(x[mask], y[mask], depth + 1)
        node.right = self._grow(x[~mask], y[~mask], depth + 1)
        return node


class GradientBoostingRegressor:
    """Gradient boosting for regression with squared loss.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages (trees). Must be at least 1.
    learning_rate : float
        Shrinkage applied to each tree's contribution. Must be positive.
    max_depth : int
        Maximum depth of each regression tree. Must be at least 1.
    min_samples_leaf : int
        Minimum samples in a leaf. Must be at least 1.
    random_state : int or None
        Accepted for API compatibility; the fit itself is deterministic.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_leaf: int = 1,
                 random_state: int | None = None):
        if not isinstance(n_estimators, int) or isinstance(n_estimators, bool) \
                or n_estimators < 1:
            raise ValueError("n_estimators must be an integer of at least 1")
        if (not isinstance(learning_rate, (int, float))
                or isinstance(learning_rate, bool) or learning_rate <= 0):
            raise ValueError("learning_rate must be a positive number")
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) \
                or max_depth < 1:
            raise ValueError("max_depth must be an integer of at least 1")
        if (not isinstance(min_samples_leaf, int)
                or isinstance(min_samples_leaf, bool) or min_samples_leaf < 1):
            raise ValueError("min_samples_leaf must be an integer of at least 1")
        self.n_estimators = n_estimators
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._trees: list[_RegressionTree] = []
        self._init: float | None = None
        self._n_features: int | None = None

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of targets")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        if not np.isfinite(y).all():
            raise ValueError("y must not contain NaN or infinite values")
        self._n_features = x.shape[1]
        self._init = float(y.mean())
        self._trees = []
        pred = np.full_like(y, self._init)
        for _ in range(self.n_estimators):
            residual = y - pred
            tree = _RegressionTree(max_depth=self.max_depth,
                                   min_samples_leaf=self.min_samples_leaf)
            tree.fit(x, residual)
            self._trees.append(tree)
            pred = pred + self.learning_rate * tree.predict(x)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = self._check_x(x)
        pred = np.full(x.shape[0], self._init)
        for tree in self._trees:
            pred = pred + self.learning_rate * tree.predict(x)
        return pred

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Coefficient of determination (R^2) on the given data."""
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of targets")
        pred = self.predict(x)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot

    # -- internals ------------------------------------------------------
    def _check_fitted(self):
        if not self._trees or self._init is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    def _check_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self._n_features:
            raise ValueError(
                "x has %d features, expected %d" % (x.shape[1], self._n_features)
            )
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        return x
