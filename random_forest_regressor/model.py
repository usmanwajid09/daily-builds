"""Random-forest regressor, from scratch with numpy only.

Each tree is a small CART regression tree (variance reduction, greedy
splits) grown on a bootstrap resample of the data, and at every split
only a random subset of the features is considered. Predictions are the
mean of the trees' predictions. Samples left out of a tree's bootstrap
sample (out-of-bag) give an honest error estimate without a held-out
test set: the out-of-bag score is the R^2 of the OOB predictions.
"""
from __future__ import annotations

import numpy as np


class _Node:
    """One node of a tree: either a split or a leaf value."""

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


class _Tree:
    """A single CART regression tree with a random feature subset."""

    def __init__(self, *, max_depth, min_samples_split, max_features, rng):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng
        self._root: _Node | None = None
        self._n_features = 0
        self.gain_sums: np.ndarray | None = None  # variance cut per feature

    def fit(self, x, y):
        self._n_features = x.shape[1]
        self.gain_sums = np.zeros(self._n_features)
        self._root = self._grow(x, y, depth=0)
        return self

    def predict_one(self, row):
        node = self._root
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    @staticmethod
    def _variance(y):
        if y.size == 0:
            return 0.0
        return float(np.var(y))

    def _best_split(self, x, y):
        n_samples, n_features = x.shape
        if self.max_features is None:
            features = np.arange(n_features)
        else:
            k = min(self.max_features, n_features)
            features = self.rng.choice(n_features, size=k, replace=False)
        parent_var = self._variance(y)
        best = None
        for feature in features:
            order = np.argsort(x[:, feature], kind="stable")
            xs = x[order, feature]
            ys = y[order]
            for i in range(1, n_samples):
                if xs[i] == xs[i - 1]:
                    continue
                left_var = self._variance(ys[:i])
                right_var = self._variance(ys[i:])
                weighted = (i * left_var + (n_samples - i) * right_var) / n_samples
                gain = parent_var - weighted
                if best is None or gain > best[2]:
                    best = (int(feature), float((xs[i] + xs[i - 1]) / 2.0), gain)
        if best is None or best[2] <= 0.0:
            return None
        return best

    def _grow(self, x, y, depth):
        n_samples = x.shape[0]
        node = _Node()
        value = float(np.mean(y))
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or np.all(y == y[0])
        ):
            node.value = value
            return node
        split = self._best_split(x, y)
        if split is None:
            node.value = value
            return node
        feature, threshold, gain = split
        self.gain_sums[feature] += gain * n_samples
        mask = x[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(x[mask], y[mask], depth + 1)
        node.right = self._grow(x[~mask], y[~mask], depth + 1)
        return node


class RandomForestRegressor:
    """Random-forest regressor: bagged CART trees with random splits.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest. Must be at least 1.
    max_depth : int or None
        Maximum depth of each tree. ``None`` grows until leaves are pure
        or cannot be split further.
    min_samples_split : int
        Minimum samples a node needs before it may be split.
    max_features : {"sqrt", None} or int
        Number of features considered at each split: ``"sqrt"`` uses
        ``sqrt(n_features)``, ``None`` uses all features, an int uses
        exactly that many.
    random_state : int or None
        Seed for bootstrap sampling and feature subsets.

    Attributes
    -----------
    oob_score_ : float or None
        R^2 of the out-of-bag predictions, or None if every sample was
        in every bootstrap sample (only possible with tiny data).
    feature_importances_ : ndarray
        Share of total variance reduction contributed by each feature
        (sums to 1, or all zeros when no tree ever split).
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
        self.oob_score_: float | None = None
        self.feature_importances_: np.ndarray | None = None

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
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
        if isinstance(self.max_features, int) and self.max_features > x.shape[1]:
            raise ValueError(
                f"max_features={self.max_features} exceeds n_features={x.shape[1]}"
            )
        n_samples, n_features = x.shape
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features is None:
            k = n_features
        else:
            k = self.max_features
        rng = np.random.default_rng(self.random_state)
        self._trees = []
        gain_sums = np.zeros(n_features)
        oob_sum = np.zeros(n_samples)
        oob_count = np.zeros(n_samples, dtype=int)
        for _ in range(self.n_estimators):
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            in_boot = np.zeros(n_samples, dtype=bool)
            in_boot[boot_idx] = True
            tree = _Tree(max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split,
                         max_features=k, rng=rng)
            tree.fit(x[boot_idx], y[boot_idx])
            self._trees.append(tree)
            gain_sums += tree.gain_sums
            oob_idx = np.where(~in_boot)[0]
            for i in oob_idx:
                oob_sum[i] += tree.predict_one(x[i])
                oob_count[i] += 1
        total = gain_sums.sum()
        self.feature_importances_ = (
            gain_sums / total if total > 0 else np.zeros(n_features)
        )
        covered = oob_count > 0
        if covered.any():
            oob_pred = oob_sum[covered] / oob_count[covered]
            ss_res = float(np.sum((y[covered] - oob_pred) ** 2))
            ss_tot = float(np.sum((y[covered] - np.mean(y[covered])) ** 2))
            self.oob_score_ = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            self.oob_score_ = None
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = self._check_x(x)
        preds = np.zeros((x.shape[0], len(self._trees)))
        for j, tree in enumerate(self._trees):
            for i, row in enumerate(x):
                preds[i, j] = tree.predict_one(row)
        return preds.mean(axis=1)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """R^2 of predictions on (x, y)."""
        y = np.asarray(y, dtype=float)
        pred = self.predict(x)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot == 0:
            raise ValueError("cannot compute R^2: y has zero variance")
        return 1.0 - ss_res / ss_tot

    # -- internals ------------------------------------------------------
    def _check_fitted(self):
        if not self._trees:
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
