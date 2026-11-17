"""CART regression tree from scratch (numpy only).

Splits are chosen to maximise variance reduction: for a candidate split
the weighted mean squared error of the two children is compared against
the parent node's MSE; the split with the largest decrease wins.
Candidate thresholds are the midpoints between consecutive sorted unique
values of each feature. Leaves predict the mean target of their samples.
"""

import numpy as np


class DecisionTreeRegressor:
    """CART regression tree using variance-reduction (MSE) splits.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree. ``None`` grows until the stopping
        rules (``min_samples_split`` / ``min_samples_leaf`` / pure node)
        stop it. Must be >= 1 when given.
    min_samples_split : int
        Minimum samples a node needs before it may be split. Must be >= 2.
    min_samples_leaf : int
        Minimum samples a leaf must contain. Splits that would create a
        smaller leaf are rejected. Must be >= 1.
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if max_depth is not None and (not isinstance(max_depth, (int, np.integer)) or max_depth < 1):
            raise ValueError("max_depth must be an integer >= 1 or None")
        if not isinstance(min_samples_split, (int, np.integer)) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if not isinstance(min_samples_leaf, (int, np.integer)) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1")
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)

    # -- public API -----------------------------------------------------
    def fit(self, X, y):
        """Fit the tree to ``X`` (2D) and ``y`` (1D). Returns self."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("X and y must be non-empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have mismatched lengths: {X.shape[0]} != {y.shape[0]}"
            )
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values (no NaN or inf)")
        if not np.isfinite(y).all():
            raise ValueError("y must contain only finite values (no NaN or inf)")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build(X, y, depth=0)
        self.max_depth_ = self._depth(self.tree_)
        self.n_leaves_ = self._count_leaves(self.tree_)
        return self

    def predict(self, X):
        """Predict target values for ``X`` (leaf means)."""
        if not hasattr(self, "tree_"):
            raise ValueError("DecisionTreeRegressor is not fitted yet; call fit first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values (no NaN or inf)")
        return np.array([self._predict_one(row, self.tree_) for row in X])

    def score(self, X, y):
        """Return the coefficient of determination R² of the prediction."""
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot

    # -- tree building --------------------------------------------------
    def _build(self, X, y, depth):
        n_samples = X.shape[0]
        node = {"value": float(np.mean(y)), "n_samples": int(n_samples)}
        if (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or np.all(y == y[0])
        ):
            return node
        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold, left_idx, right_idx = split
        node.update(
            {
                "feature": int(feature),
                "threshold": float(threshold),
                "left": self._build(X[left_idx], y[left_idx], depth + 1),
                "right": self._build(X[right_idx], y[right_idx], depth + 1),
            }
        )
        return node

    def _best_split(self, X, y):
        """Return (feature, threshold, left_mask, right_mask) or None."""
        n_samples = X.shape[0]
        parent_mse = float(np.mean((y - np.mean(y)) ** 2))
        best_gain = 0.0
        best = None
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            if values.size < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                n_left = int(left_idx.sum())
                n_right = int(right_idx.sum())
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                mse_left = float(np.mean((y[left_idx] - np.mean(y[left_idx])) ** 2))
                mse_right = float(np.mean((y[right_idx] - np.mean(y[right_idx])) ** 2))
                weighted_mse = (n_left * mse_left + n_right * mse_right) / n_samples
                gain = parent_mse - weighted_mse
                if gain > best_gain:
                    best_gain = gain
                    best = (feature, float(threshold), left_idx, right_idx)
        return best

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _predict_one(row, node):
        while "feature" in node:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        return node["value"]

    @staticmethod
    def _depth(node):
        if "feature" not in node:
            return 0
        return 1 + max(DecisionTreeRegressor._depth(node["left"]),
                       DecisionTreeRegressor._depth(node["right"]))

    @staticmethod
    def _count_leaves(node):
        if "feature" not in node:
            return 1
        return (DecisionTreeRegressor._count_leaves(node["left"])
                + DecisionTreeRegressor._count_leaves(node["right"]))
