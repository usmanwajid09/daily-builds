"""Bagging regressor, from scratch with numpy only.

Bootstrap aggregation ("bagging") of decision-stump regressors: each stump
is a single-feature, single-threshold split that predicts the mean of its
leaf. A single stump has high variance — a small change in the data can
move its split — so averaging many stumps, each trained on a bootstrap
resample, reduces the variance of the prediction without raising the bias
much. If ``max_features`` is set, each stump also sees only a random
subset of the features, which decorrelates the stumps further (the
random-forest idea, with stumps instead of trees).
"""
from __future__ import annotations

import numpy as np


def _validate_xy(X, y):
    """Validate X (2D) and y (1D), return float arrays."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
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


class DecisionStumpRegressor:
    """A one-split regression tree: one feature, one threshold, leaf means.

    After fitting, ``feature_index_``, ``threshold_``, ``left_value_`` and
    ``right_value_`` describe the split. If no split improves on predicting
    the overall mean (e.g. all features constant), the stump predicts the
    mean for every input.
    """

    def __init__(self):
        self.feature_index_ = None
        self.threshold_ = None
        self.left_value_ = None
        self.right_value_ = None
        self._fitted = False

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        n_samples, n_features = X.shape
        overall_mean = float(np.mean(y))

        best_mse = float(np.sum((y - overall_mean) ** 2))
        best = None
        # Candidate thresholds: midpoints of sorted unique values, capped
        # at 100 per feature so large inputs stay fast.
        for feature in range(n_features):
            values = X[:, feature]
            unique = np.unique(values)
            if unique.size < 2:
                continue  # constant feature: no split possible
            midpoints = (unique[:-1] + unique[1:]) / 2.0
            if midpoints.size > 100:
                idx = np.linspace(0, midpoints.size - 1, 100).astype(int)
                midpoints = midpoints[idx]
            for threshold in midpoints:
                left = values <= threshold
                right = ~left
                if not left.any() or not right.any():
                    continue
                left_mean = float(np.mean(y[left]))
                right_mean = float(np.mean(y[right]))
                sse = float(np.sum((y[left] - left_mean) ** 2)
                            + np.sum((y[right] - right_mean) ** 2))
                if sse < best_mse:
                    best_mse = sse
                    best = (feature, float(threshold), left_mean, right_mean)

        if best is None:
            self.feature_index_ = None
            self.threshold_ = None
            self.left_value_ = overall_mean
            self.right_value_ = overall_mean
        else:
            (self.feature_index_, self.threshold_,
             self.left_value_, self.right_value_) = best
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise ValueError("DecisionStumpRegressor is not fitted yet; call fit first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if self.feature_index_ is None:
            return np.full(X.shape[0], self.left_value_)
        return np.where(X[:, self.feature_index_] <= self.threshold_,
                        self.left_value_, self.right_value_)


class BaggingRegressor:
    """Bootstrap-aggregated decision-stump regressors.

    Each of ``n_estimators`` stumps is trained on a bootstrap resample of
    the training data (sampling with replacement, same size as the training
    set). If ``max_features`` is set, each stump sees only a random subset
    of that many features. Predictions are the mean of the stumps' predictions.
    """

    def __init__(self, n_estimators=50, random_state=None, max_features=None):
        if not isinstance(n_estimators, (int, np.integer)) or isinstance(n_estimators, bool):
            raise ValueError(
                f"n_estimators must be a positive integer, got {n_estimators!r}"
            )
        if n_estimators <= 0:
            raise ValueError(
                f"n_estimators must be a positive integer, got {n_estimators}"
            )
        if max_features is not None:
            if not isinstance(max_features, (int, np.integer)) or isinstance(max_features, bool):
                raise ValueError(
                    f"max_features must be a positive integer or None, got {max_features!r}"
                )
            if max_features <= 0:
                raise ValueError(
                    f"max_features must be a positive integer or None, got {max_features}"
                )
        self.n_estimators = int(n_estimators)
        self.random_state = random_state
        self.max_features = None if max_features is None else int(max_features)
        self.estimators_ = []
        self._fitted = False

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        n_samples, n_features = X.shape
        if self.max_features is not None and self.max_features > n_features:
            raise ValueError(
                f"max_features={self.max_features} exceeds the number of "
                f"features ({n_features})"
            )
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n_samples, size=n_samples)  # bootstrap resample
            X_boot, y_boot = X[idx], y[idx]
            stump = DecisionStumpRegressor()
            if self.max_features is not None and self.max_features < n_features:
                features = rng.choice(n_features, size=self.max_features,
                                      replace=False)
                stump.fit(X_boot[:, features], y_boot)
                # Wrap so the stump's feature index refers to the full X.
                stump = _FeatureSubsetStump(stump, features)
            else:
                stump.fit(X_boot, y_boot)
            self.estimators_.append(stump)
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise ValueError("BaggingRegressor is not fitted yet; call fit first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        preds = np.column_stack([est.predict(X) for est in self.estimators_])
        return np.mean(preds, axis=1)


class _FeatureSubsetStump:
    """A fitted stump that only sees a subset of the full feature set."""

    def __init__(self, stump, features):
        self._stump = stump
        self._features = np.asarray(features)

    def predict(self, X):
        return self._stump.predict(np.asarray(X, dtype=float)[:, self._features])
