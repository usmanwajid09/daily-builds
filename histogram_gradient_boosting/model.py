"""Histogram-based gradient boosting regressor, from scratch with numpy.

Builds an additive model of shallow regression trees for squared loss.
Before training, each feature is binned into at most ``max_bins``
equal-frequency bins. Split finding then works on bin histograms
(sums of gradients and Hessians per bin) instead of sorting every
sample at every node, which is the core idea behind LightGBM-style
histogram boosting. For squared loss the gradient is ``pred - y`` and
the Hessian is 1, so a leaf's optimal value is
``-sum(grad) / (sum(hess) + lambda)`` and the split gain is the usual
second-order gain with ``gamma`` (min_split_gain) and ``lambda``
(L2 regularisation) penalties.
"""
from __future__ import annotations

import numpy as np


class _Node:
    """One node of a histogram regression tree: a split or a leaf value."""

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


class _HistTree:
    """A regression tree grown from binned features and gradients."""

    def __init__(self, *, binned, thresholds, max_depth, min_samples_leaf,
                 min_child_weight, reg_lambda, min_split_gain):
        self.binned = binned            # (n, p) int bin ids
        self.thresholds = thresholds    # per-feature bin edge values
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.min_split_gain = min_split_gain
        self._root: _Node | None = None

    def fit(self, idx, grad, hess):
        self._root = self._grow(idx, grad, hess, depth=0)
        return self

    def predict(self, x_binned):
        return np.array([self._predict_one(row) for row in x_binned])

    def _predict_one(self, row):
        node = self._root
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def _leaf_value(self, g, h):
        return -g / (h + self.reg_lambda)

    def _grow(self, idx, grad, hess, depth):
        g = float(grad[idx].sum())
        h = float(hess[idx].sum())
        if (len(idx) < 2 * self.min_samples_leaf
                or (self.max_depth is not None and depth >= self.max_depth)):
            return _Node(value=self._leaf_value(g, h))
        split = self._best_split(idx, grad, hess, g, h)
        if split is None:
            return _Node(value=self._leaf_value(g, h))
        feature, bin_edge, left_idx, right_idx = split
        return _Node(
            feature=feature, threshold=bin_edge,
            left=self._grow(left_idx, grad, hess, depth + 1),
            right=self._grow(right_idx, grad, hess, depth + 1))

    def _best_split(self, idx, grad, hess, g_total, h_total):
        best = None  # (gain, feature, bin_id, left_idx, right_idx)
        for feature in range(self.binned.shape[1]):
            bins = self.binned[idx, feature]
            order = np.argsort(bins, kind="stable")
            sorted_bins = bins[order]
            sorted_grad = grad[idx][order]
            sorted_hess = hess[idx][order]
            # Cumulative gradient / Hessian per bin boundary.
            uniq, starts = np.unique(sorted_bins, return_index=True)
            if len(uniq) < 2:
                continue
            cum_g = np.cumsum(sorted_grad)
            cum_h = np.cumsum(sorted_hess)
            # Split after each distinct bin except the last: left = bins <= b.
            for j in range(len(uniq) - 1):
                cut = starts[j + 1] - 1
                g_left, h_left = float(cum_g[cut]), float(cum_h[cut])
                g_right, h_right = g_total - g_left, h_total - h_left
                n_left = cut + 1
                n_right = len(idx) - n_left
                if (n_left < self.min_samples_leaf
                        or n_right < self.min_samples_leaf
                        or h_left < self.min_child_weight
                        or h_right < self.min_child_weight):
                    continue
                gain = (g_left ** 2 / (h_left + self.reg_lambda)
                        + g_right ** 2 / (h_right + self.reg_lambda)
                        - g_total ** 2 / (h_total + self.reg_lambda)) / 2.0
                if gain <= self.min_split_gain:
                    continue
                if best is None or gain > best[0]:
                    left_idx = idx[order[:cut + 1]]
                    right_idx = idx[order[cut + 1:]]
                    # Threshold is a bin id: predictions compare binned
                    # feature values, so the split is bins <= uniq[j].
                    best = (gain, feature, int(uniq[j]), left_idx, right_idx)
        if best is None:
            return None
        return best[1], best[2], best[3], best[4]


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


class HistogramGradientBoostingRegressor:
    """Gradient boosting regressor with histogram-based split finding.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of boosting stages (trees). Must be >= 1.
    learning_rate : float, default 0.1
        Shrinkage applied to each tree's contribution. Must be > 0.
    max_depth : int or None, default 3
        Maximum depth of each tree. ``None`` grows until the leaf rules stop.
    max_bins : int, default 255
        Maximum number of bins per feature. Must be >= 2.
    min_samples_leaf : int, default 20
        Minimum samples in a leaf. Must be >= 1.
    min_child_weight : float, default 1.0
        Minimum sum of Hessians in a child. Must be >= 0.
    reg_lambda : float, default 1.0
        L2 regularisation on leaf values. Must be >= 0.
    min_split_gain : float, default 0.0
        Minimum gain required to split (``gamma``). Must be >= 0.
    subsample : float, default 1.0
        Fraction of samples used per tree, in (0, 1].
    random_state : int or None, default None
        Seed for reproducibility (used when ``subsample`` < 1).
    """

    def __init__(self, *, n_estimators=100, learning_rate=0.1, max_depth=3,
                 max_bins=255, min_samples_leaf=20, min_child_weight=1.0,
                 reg_lambda=1.0, min_split_gain=0.0, subsample=1.0,
                 random_state=None):
        if not isinstance(n_estimators, (int, np.integer)) or n_estimators < 1:
            raise ValueError(f"n_estimators must be an integer >= 1, got {n_estimators!r}.")
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate!r}.")
        if max_depth is not None and (not isinstance(max_depth, (int, np.integer)) or max_depth < 1):
            raise ValueError(f"max_depth must be a positive integer or None, got {max_depth!r}.")
        if not isinstance(max_bins, (int, np.integer)) or max_bins < 2:
            raise ValueError(f"max_bins must be an integer >= 2, got {max_bins!r}.")
        if not isinstance(min_samples_leaf, (int, np.integer)) or min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be an integer >= 1, got {min_samples_leaf!r}.")
        if min_child_weight < 0:
            raise ValueError(f"min_child_weight must be >= 0, got {min_child_weight!r}.")
        if reg_lambda < 0:
            raise ValueError(f"reg_lambda must be >= 0, got {reg_lambda!r}.")
        if min_split_gain < 0:
            raise ValueError(f"min_split_gain must be >= 0, got {min_split_gain!r}.")
        if not 0 < subsample <= 1.0:
            raise ValueError(f"subsample must be in (0, 1], got {subsample!r}.")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.max_bins = int(max_bins)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_child_weight = float(min_child_weight)
        self.reg_lambda = float(reg_lambda)
        self.min_split_gain = float(min_split_gain)
        self.subsample = float(subsample)
        self.random_state = random_state
        self.init_ = None
        self.trees_ = []
        self.bin_thresholds_ = None
        self.n_features_in_ = None
        self.train_score_ = None

    # -- binning ---------------------------------------------------------
    def _fit_bins(self, x):
        """Equal-frequency bin edges per feature; returns (binned, thresholds)."""
        n, p = x.shape
        n_bins = min(self.max_bins, n)
        binned = np.empty((n, p), dtype=np.int64)
        thresholds = []
        for j in range(p):
            col = x[:, j]
            qs = np.linspace(0, 100, n_bins + 1)[1:-1]
            edges = np.unique(np.percentile(col, qs))
            thresholds.append(edges)
            binned[:, j] = np.searchsorted(edges, col, side="right")
        return binned, thresholds

    def _apply_bins(self, x):
        binned = np.empty(x.shape, dtype=np.int64)
        for j, edges in enumerate(self.bin_thresholds_):
            binned[:, j] = np.searchsorted(edges, x[:, j], side="right")
        return binned

    # -- fitting ---------------------------------------------------------
    def fit(self, x, y):
        """Fit the boosting model. Returns self."""
        x, y = _validate_xy(x, y)
        n, p = x.shape
        if n < 2 * self.min_samples_leaf and n > 1:
            # Still allow fitting; leaf rules just stop splitting early.
            pass
        rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = p
        binned, thresholds = self._fit_bins(x)
        self.bin_thresholds_ = thresholds
        self.init_ = float(y.mean())
        pred = np.full(n, self.init_)
        self.trees_ = []
        self.train_score_ = []
        for _ in range(self.n_estimators):
            grad = pred - y            # d/dpred of 0.5*(pred - y)^2
            hess = np.ones(n)
            if self.subsample < 1.0:
                idx = np.sort(rng.choice(n, size=max(1, int(self.subsample * n)),
                                         replace=False))
            else:
                idx = np.arange(n)
            tree = _HistTree(binned=binned, thresholds=thresholds,
                             max_depth=self.max_depth,
                             min_samples_leaf=self.min_samples_leaf,
                             min_child_weight=self.min_child_weight,
                             reg_lambda=self.reg_lambda,
                             min_split_gain=self.min_split_gain)
            tree.fit(idx, grad, hess)
            update = self.learning_rate * tree.predict(binned)
            pred = pred + update
            self.trees_.append(tree)
            self.train_score_.append(float(np.mean((y - pred) ** 2)))
        return self

    def predict(self, x):
        """Predict regression values. Raises ValueError if not fitted."""
        if not self.trees_:
            raise ValueError("HistogramGradientBoostingRegressor is not fitted yet. Call fit() first.")
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
        binned = self._apply_bins(x)
        pred = np.full(x.shape[0], self.init_)
        for tree in self.trees_:
            pred = pred + self.learning_rate * tree.predict(binned)
        return pred

    def score(self, x, y):
        """Return the coefficient of determination R^2 of the prediction."""
        x, y = _validate_xy(x, y)
        y_pred = self.predict(x)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot
