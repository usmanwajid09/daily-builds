"""Extremely randomized trees regressor, from scratch with numpy only.

Like a random forest, but each candidate split threshold is drawn
uniformly at random between the feature's min and max inside the node,
and the best of a random subset of such candidates is kept. Splits are
scored by variance reduction (the regression analogue of information
gain). Full trees are grown on the whole dataset (or bootstrap samples),
and predictions are the mean of the trees' predictions.
"""
from __future__ import annotations

import numpy as np


class _Node:
    """One node of a regression tree: a split or a leaf value."""

    def __init__(self, *, feature=None, threshold=None, left=None,
                 right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class _ExtraTree:
    """A single extremely randomized regression tree."""

    def __init__(self, *, max_depth, min_samples_split, min_samples_leaf,
                 max_features, rng):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng
        self._root: _Node | None = None
        self.importances_: np.ndarray | None = None

    def fit(self, x, y, n_features):
        self.importances_ = np.zeros(n_features, dtype=float)
        self._root = self._grow(x, y, depth=0)
        return self

    def predict(self, x):
        return np.array([self._predict_one(row) for row in x])

    def _predict_one(self, row):
        node = self._root
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def _grow(self, x, y, depth):
        n = len(y)
        if (n < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)
                or np.all(y == y[0])):
            return _Node(value=float(y.mean()))
        split = self._best_random_split(x, y)
        if split is None:
            return _Node(value=float(y.mean()))
        feature, threshold, gain = split
        left_mask = x[:, feature] <= threshold
        self.importances_[feature] += gain * n
        return _Node(
            feature=feature, threshold=threshold,
            left=self._grow(x[left_mask], y[left_mask], depth + 1),
            right=self._grow(x[~left_mask], y[~left_mask], depth + 1))

    def _best_random_split(self, x, y):
        n, n_features = x.shape
        n_candidates = max(1, self.max_features)
        features = self.rng.choice(n_features, size=min(n_candidates, n_features),
                                   replace=False)
        parent_var = float(np.var(y))
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
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            gain = parent_var - (
                n_left * float(np.var(y[left_mask]))
                + n_right * float(np.var(y[~left_mask]))) / n
            if gain <= 0:
                continue
            if best is None or gain > best[2]:
                best = (int(feature), threshold, gain)
        return best


def _validate_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {x.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array, got shape {y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have inconsistent lengths: {x.shape[0]} vs {y.shape[0]}.")
    if x.shape[0] == 0:
        raise ValueError("X and y must not be empty.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values.")
    return x, y


class ExtraTreesRegressor:
    """Extremely randomized trees regressor (Extra-Trees), from scratch.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees in the forest. Must be >= 1.
    max_depth : int or None, default None
        Maximum tree depth. ``None`` grows trees until pure / split rules stop.
    min_samples_split : int, default 2
        Minimum samples required to split a node. Must be >= 2.
    min_samples_leaf : int, default 1
        Minimum samples required in a leaf. Must be >= 1.
    max_features : int, float or None, default 1.0
        Number of features considered per split. An int is used as-is, a
        float in (0, 1] is a fraction of the features, and ``None`` means all.
    bootstrap : bool, default False
        If True, each tree is fit on a bootstrap sample; otherwise on the
        whole dataset.
    random_state : int or None, default None
        Seed for reproducibility.
    """

    def __init__(self, *, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=1.0, bootstrap=False,
                 random_state=None):
        if not isinstance(n_estimators, (int, np.integer)) or n_estimators < 1:
            raise ValueError(f"n_estimators must be an integer >= 1, got {n_estimators!r}.")
        if max_depth is not None and (not isinstance(max_depth, (int, np.integer)) or max_depth < 1):
            raise ValueError(f"max_depth must be a positive integer or None, got {max_depth!r}.")
        if not isinstance(min_samples_split, (int, np.integer)) or min_samples_split < 2:
            raise ValueError(f"min_samples_split must be an integer >= 2, got {min_samples_split!r}.")
        if not isinstance(min_samples_leaf, (int, np.integer)) or min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be an integer >= 1, got {min_samples_leaf!r}.")
        if isinstance(max_features, (int, np.integer)) and max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {max_features!r}.")
        if isinstance(max_features, float) and not 0 < max_features <= 1.0:
            raise ValueError(
                f"max_features as a fraction must be in (0, 1], got {max_features!r}.")
        if not (max_features is None or isinstance(max_features, (int, np.integer, float))):
            raise ValueError(f"max_features must be int, float or None, got {max_features!r}.")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self.trees_ = []
        self.feature_importances_ = None
        self.n_features_in_ = None

    def _resolve_max_features(self, n_features):
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, (int, np.integer)):
            if self.max_features < 1 or self.max_features > n_features:
                raise ValueError(
                    f"max_features must be between 1 and {n_features}, got {self.max_features!r}.")
            return int(self.max_features)
        if isinstance(self.max_features, float):
            if not 0 < self.max_features <= 1.0:
                raise ValueError(
                    f"max_features as a fraction must be in (0, 1], got {self.max_features!r}.")
            return max(1, int(round(self.max_features * n_features)))
        raise ValueError(f"max_features must be int, float or None, got {self.max_features!r}.")

    def fit(self, x, y):
        """Fit the forest. Returns self."""
        x, y = _validate_xy(x, y)
        n_features = x.shape[1]
        k = self._resolve_max_features(n_features)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        importances = np.zeros(n_features, dtype=float)
        for _ in range(self.n_estimators):
            tree_rng = np.random.default_rng(rng.integers(0, 2 ** 31 - 1))
            if self.bootstrap:
                idx = tree_rng.integers(0, len(y), size=len(y))
                x_t, y_t = x[idx], y[idx]
            else:
                x_t, y_t = x, y
            tree = _ExtraTree(max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split,
                              min_samples_leaf=self.min_samples_leaf,
                              max_features=k, rng=tree_rng)
            tree.fit(x_t, y_t, n_features)
            importances += tree.importances_
            self.trees_.append(tree)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        self.n_features_in_ = n_features
        return self

    def predict(self, x):
        """Predict regression values. Raises ValueError if not fitted."""
        if not self.trees_:
            raise ValueError("ExtraTreesRegressor is not fitted yet. Call fit() first.")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {x.shape}.")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {x.shape[1]} features, but the model was fit with {self.n_features_in_}.")
        if x.shape[0] == 0:
            raise ValueError("X must not be empty.")
        if not np.all(np.isfinite(x)):
            raise ValueError("X must contain only finite values.")
        preds = np.mean([tree.predict(x) for tree in self.trees_], axis=0)
        return preds

    def score(self, x, y):
        """Return the coefficient of determination R^2 of the prediction."""
        x, y = _validate_xy(x, y)
        y_pred = self.predict(x)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot
