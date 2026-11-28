"""Histogram-based gradient boosting classifier from scratch, numpy only."""

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class _TreeNode:
    __slots__ = ("feature", "bin_threshold", "left", "right", "value")

    def __init__(self, value=None, feature=None, bin_threshold=None,
                 left=None, right=None):
        self.value = value  # leaf output (None for internal nodes)
        self.feature = feature
        self.bin_threshold = bin_threshold
        self.left = left
        self.right = right


class _RegressionTree:
    """Small regression tree fit to gradients/hessians, split on binned features."""

    def __init__(self, max_depth=3, min_samples_leaf=5, reg_lambda=1.0,
                 min_gain=1e-7):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.min_gain = min_gain

    def fit(self, X_binned, grad, hess, n_bins):
        self.n_bins_ = n_bins
        self.root_ = self._build(X_binned, grad, hess, depth=0)
        return self

    def _leaf_value(self, grad, hess):
        return -grad.sum() / (hess.sum() + self.reg_lambda)

    def _build(self, X_binned, grad, hess, depth):
        n = len(grad)
        value = self._leaf_value(grad, hess)
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return _TreeNode(value=value)
        total_g, total_h = grad.sum(), hess.sum()
        base_score = total_g ** 2 / (total_h + self.reg_lambda)
        best_gain, best_split = 0.0, None
        for feature in range(X_binned.shape[1]):
            bins = X_binned[:, feature]
            for threshold in range(self.n_bins_ - 1):
                left_mask = bins <= threshold
                n_left = int(left_mask.sum())
                n_right = n - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                g_l, h_l = grad[left_mask].sum(), hess[left_mask].sum()
                g_r, h_r = total_g - g_l, total_h - h_l
                gain = (g_l ** 2 / (h_l + self.reg_lambda)
                        + g_r ** 2 / (h_r + self.reg_lambda)
                        - base_score)
                if gain > best_gain:
                    best_gain, best_split = gain, (feature, threshold, left_mask)
        if best_split is None or best_gain <= self.min_gain:
            return _TreeNode(value=value)
        feature, threshold, left_mask = best_split
        left = self._build(X_binned[left_mask], grad[left_mask],
                           hess[left_mask], depth + 1)
        right = self._build(X_binned[~left_mask], grad[~left_mask],
                            hess[~left_mask], depth + 1)
        return _TreeNode(feature=feature, bin_threshold=threshold,
                         left=left, right=right)

    def _predict_one(self, node, x_binned):
        while node.value is None:
            if x_binned[node.feature] <= node.bin_threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X_binned):
        return np.array([self._predict_one(self.root_, row)
                         for row in X_binned])


class HistogramGBClassifier:
    """Gradient boosting classifier with histogram-binned splits.

    Features are first quantised into ``max_bins`` quantile bins; each
    boosting round then fits a small regression tree to the log-loss
    gradients using only bin-level gradient/hessian sums, which keeps
    the split search fast. Binary classification optimises logistic
    loss directly; multiclass problems use a one-vs-rest ensemble.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of boosting rounds (trees). Must be positive.
    learning_rate : float, default 0.1
        Shrinkage applied to each tree's output. Must be positive.
    max_depth : int, default 3
        Maximum depth of each tree. Must be >= 1.
    max_bins : int, default 16
        Number of quantile bins per feature. Must be >= 2.
    min_samples_leaf : int, default 5
        Minimum samples in a leaf. Must be positive.
    reg_lambda : float, default 1.0
        L2 regularisation on leaf values. Must be non-negative.
    random_state : int or None, default None
        Currently unused (fitting is deterministic); kept for API symmetry.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 max_bins=16, min_samples_leaf=5, reg_lambda=1.0,
                 random_state=None):
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {n_estimators!r}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate!r}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth!r}")
        if max_bins < 2:
            raise ValueError(f"max_bins must be >= 2, got {max_bins!r}")
        if min_samples_leaf <= 0:
            raise ValueError(
                f"min_samples_leaf must be positive, got {min_samples_leaf!r}")
        if reg_lambda < 0:
            raise ValueError(f"reg_lambda must be non-negative, got {reg_lambda!r}")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    # -- binning ---------------------------------------------------------
    def _fit_binner(self, X):
        n_features = X.shape[1]
        self.bin_edges_ = []
        for j in range(n_features):
            quantiles = np.linspace(0, 100, self.max_bins + 1)[1:-1]
            edges = np.unique(np.percentile(X[:, j], quantiles))
            # Guarantee at least one edge so every feature bins.
            if len(edges) == 0:
                edges = np.array([X[:, j].mean()])
            self.bin_edges_.append(edges)

    def _bin(self, X):
        X_binned = np.empty(X.shape, dtype=np.int64)
        for j, edges in enumerate(self.bin_edges_):
            X_binned[:, j] = np.searchsorted(edges, X[:, j], side="right")
        return X_binned

    # -- boosting --------------------------------------------------------
    def _fit_binary(self, X_binned, y01):
        n = len(y01)
        n_bins = self.max_bins
        # Start from the log-odds of the base rate.
        base_rate = float(np.clip(y01.mean(), 1e-6, 1 - 1e-6))
        init = np.log(base_rate / (1.0 - base_rate))
        raw = np.full(n, init)
        trees = []
        for _ in range(self.n_estimators):
            p = _sigmoid(raw)
            grad = p - y01
            hess = np.clip(p * (1.0 - p), 1e-6, None)
            tree = _RegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                reg_lambda=self.reg_lambda).fit(X_binned, grad, hess, n_bins)
            raw = raw + self.learning_rate * tree.predict(X_binned)
            trees.append(tree)
        return trees, init

    def fit(self, X, y):
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} != {y.shape[0]}")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or infinite values")
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError(
                f"Need at least 2 classes, got {len(self.classes_)}")
        self._fit_binner(X)
        X_binned = self._bin(X)
        self.boosters_ = []  # list of (trees, init) per class
        for cls in self.classes_:
            y01 = (y == cls).astype(float)
            self.boosters_.append(self._fit_binary(X_binned, y01))
        return self

    def _raw_scores(self, X):
        if not hasattr(self, "boosters_"):
            raise RuntimeError(
                "HistogramGBClassifier is not fitted yet; call fit() first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[1] != len(self.bin_edges_):
            raise ValueError(
                f"X has {X.shape[1]} features, expected {len(self.bin_edges_)}")
        X_binned = self._bin(X)
        scores = np.empty((X.shape[0], len(self.classes_)))
        for k, (trees, init) in enumerate(self.boosters_):
            raw = np.full(X.shape[0], init)
            for tree in trees:
                raw = raw + self.learning_rate * tree.predict(X_binned)
            scores[:, k] = raw
        return scores

    def predict_proba(self, X):
        """Class probabilities (sigmoid per class, normalised to sum to 1)."""
        scores = self._raw_scores(X)
        probs = _sigmoid(scores)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Predict class labels."""
        scores = self._raw_scores(X)  # raises RuntimeError if not fitted
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        """Mean accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
