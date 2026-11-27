"""Stacking regressor, from scratch with numpy only.

Three small base regressors — ridge regression, k-nearest neighbours and a
small CART decision tree — are combined by a ridge meta-learner. The
meta-learner is trained on *out-of-fold* predictions: each base learner is
trained on K-1 folds and predicts the held-out fold, so the meta-features
for every training sample come from a model that never saw that sample.
That avoids the leakage you would get by training the meta-learner on
in-sample predictions (where a flexible base learner such as a tree would
look perfect and hog all the weight). After the meta-learner is fitted,
each base learner is refit on the full training set for use at predict time.
"""
from __future__ import annotations

import numpy as np


def _validate_xy(X, y):
    """Validate X (2D numeric) and y (1D numeric), return float arrays."""
    try:
        X = np.asarray(X, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("X must be a numeric array") from None
    try:
        y = np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("y must be a numeric array") from None
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, got {X.ndim}D")
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array, got {y.ndim}D")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have mismatched lengths: X has {X.shape[0]} rows, "
            f"y has {y.shape[0]} values"
        )
    if X.shape[0] == 0:
        raise ValueError("X and y must be non-empty")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite values")
    return X, y


def _ridge_fit(X, y, alpha):
    """Closed-form ridge with an unpenalised intercept.

    Returns (intercept, coef) for y ~ intercept + X @ coef.
    """
    n_samples, n_features = X.shape
    x_mean = X.mean(axis=0)
    y_mean = float(np.mean(y))
    Xc = X - x_mean
    yc = y - y_mean
    A = Xc.T @ Xc + alpha * np.eye(n_features)
    coef = np.linalg.solve(A, Xc.T @ yc)
    intercept = y_mean - float(x_mean @ coef)
    return intercept, coef


class RidgeRegressor:
    """Linear regression with L2 regularisation, solved in closed form."""

    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        self.intercept_, self.coef_ = _ridge_fit(X, y, self.alpha)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("RidgeRegressor must be fitted before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        return self.intercept_ + X @ self.coef_


class KNNRegressor:
    """k-nearest neighbours regression, Euclidean distance, uniform weights."""

    def __init__(self, n_neighbors=5):
        if not isinstance(n_neighbors, (int, np.integer)) or isinstance(
            n_neighbors, bool
        ):
            raise ValueError(
                f"n_neighbors must be an int, got {type(n_neighbors).__name__}"
            )
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
        self.n_neighbors = int(n_neighbors)
        self.X_train_ = None
        self.y_train_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        self.X_train_ = X
        self.y_train_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if self.X_train_ is None:
            raise RuntimeError("KNNRegressor must be fitted before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        k = min(self.n_neighbors, self.X_train_.shape[0])
        # Squared Euclidean distances, chunked to bound memory.
        preds = np.empty(X.shape[0], dtype=float)
        for start in range(0, X.shape[0], 256):
            chunk = X[start : start + 256]
            dists = (
                np.sum(chunk**2, axis=1, keepdims=True)
                - 2.0 * chunk @ self.X_train_.T
                + np.sum(self.X_train_**2, axis=1)
            )
            nearest = np.argpartition(dists, k - 1, axis=1)[:, :k]
            preds[start : start + 256] = np.mean(
                self.y_train_[nearest], axis=1
            )
        return preds


class DecisionTreeRegressor:
    """Small CART regression tree with squared-error splits."""

    def __init__(self, max_depth=5, min_samples_split=2):
        if max_depth is not None and (
            not isinstance(max_depth, (int, np.integer))
            or isinstance(max_depth, bool)
            or max_depth < 1
        ):
            raise ValueError(f"max_depth must be a positive int, got {max_depth}")
        if (
            not isinstance(min_samples_split, (int, np.integer))
            or isinstance(min_samples_split, bool)
            or min_samples_split < 2
        ):
            raise ValueError(
                f"min_samples_split must be an int >= 2, got {min_samples_split}"
            )
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.tree_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        """Best (feature, threshold, sse) split, or None if no split helps."""
        n_samples, n_features = X.shape
        total_sse = float(np.sum((y - y.mean()) ** 2))
        best = None
        for j in range(n_features):
            order = np.argsort(X[:, j], kind="mergesort")
            xs = X[order, j]
            ys = y[order]
            # Candidate thresholds: midpoints between distinct values.
            distinct = xs[1:] != xs[:-1]
            if not np.any(distinct):
                continue
            cum = np.cumsum(ys)
            cum2 = np.cumsum(ys * ys)
            total = cum[-1]
            total2 = cum2[-1]
            counts = np.arange(1, n_samples)
            left_sse = cum2[:-1] - cum[:-1] ** 2 / counts
            right_sse = (total2 - cum2[:-1]) - (total - cum[:-1]) ** 2 / (
                n_samples - counts
            )
            sse = left_sse + right_sse
            sse = np.where(distinct, sse, np.inf)
            idx = int(np.argmin(sse))
            if sse[idx] < total_sse - 1e-12 and (
                best is None or sse[idx] < best[2]
            ):
                best = (j, float((xs[idx] + xs[idx + 1]) / 2.0), float(sse[idx]))
        return best

    def _build(self, X, y, depth):
        node = {"value": float(np.mean(y)), "n_samples": int(X.shape[0])}
        if (
            X.shape[0] >= self.min_samples_split
            and (self.max_depth is None or depth < self.max_depth)
            and np.ptp(y) > 0
        ):
            split = self._best_split(X, y)
            if split is not None:
                j, threshold, _ = split
                left = X[:, j] <= threshold
                right = ~left
                if left.sum() >= 1 and right.sum() >= 1:
                    node.update(
                        {
                            "feature": j,
                            "threshold": threshold,
                            "left": self._build(X[left], y[left], depth + 1),
                            "right": self._build(X[right], y[right], depth + 1),
                        }
                    )
        return node

    def _predict_one(self, node, x):
        while "feature" in node:
            node = (
                node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
            )
        return node["value"]

    def predict(self, X):
        if self.tree_ is None:
            raise RuntimeError(
                "DecisionTreeRegressor must be fitted before predict"
            )
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        return np.array([self._predict_one(self.tree_, row) for row in X])


def _make_base_learners():
    """The three base learners, in a fixed order."""
    return [
        RidgeRegressor(alpha=1.0),
        KNNRegressor(n_neighbors=5),
        DecisionTreeRegressor(max_depth=5, min_samples_split=2),
    ]


class StackingRegressor:
    """Stacked regression: ridge meta-learner over out-of-fold base predictions.

    Parameters
    ----------
    n_folds : int, default 5
        Number of folds for the out-of-fold meta-features.
    alpha : float, default 1.0
        Ridge penalty for the meta-learner.
    random_state : int or None, default 42
        Seed for shuffling the folds.
    """

    def __init__(self, n_folds=5, alpha=1.0, random_state=42):
        if isinstance(n_folds, bool) or not isinstance(
            n_folds, (int, np.integer)
        ):
            raise ValueError(
                f"n_folds must be an int, got {type(n_folds).__name__}"
            )
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self.n_folds = int(n_folds)
        self.alpha = float(alpha)
        self.random_state = random_state
        self.base_models_ = None
        self.meta_model_ = None
        self.meta_features_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        n_samples = X.shape[0]
        if n_samples < self.n_folds:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_folds ({self.n_folds})"
            )
        self.n_features_in_ = X.shape[1]

        rng = np.random.default_rng(self.random_state)
        indices = rng.permutation(n_samples)
        folds = np.array_split(indices, self.n_folds)

        # Out-of-fold predictions: one meta-feature column per base learner.
        meta_features = np.empty((n_samples, 3), dtype=float)
        for fold in folds:
            mask = np.ones(n_samples, dtype=bool)
            mask[fold] = False
            for k, learner in enumerate(_make_base_learners()):
                learner.fit(X[mask], y[mask])
                meta_features[fold, k] = learner.predict(X[fold])
        self.meta_features_ = meta_features

        # Ridge meta-learner on the out-of-fold predictions.
        intercept, coef = _ridge_fit(meta_features, y, self.alpha)
        self.meta_model_ = np.concatenate([[intercept], coef])

        # Refit each base learner on the full data for prediction.
        self.base_models_ = _make_base_learners()
        for learner in self.base_models_:
            learner.fit(X, y)
        return self

    def _meta_features(self, X):
        return np.column_stack(
            [learner.predict(X) for learner in self.base_models_]
        )

    def predict(self, X):
        if self.base_models_ is None or self.meta_model_ is None:
            raise RuntimeError("StackingRegressor must be fitted before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        meta = self._meta_features(X)
        return self.meta_model_[0] + meta @ self.meta_model_[1:]
