"""Voting regression with three small numpy-only base regressors.

Base learners (all implemented from scratch, no sklearn):
- ``RidgeRegressor``: closed-form ridge regression with an L2 penalty
  and an optional intercept.
- ``KNNRegressor``: k-nearest-neighbours regression with Euclidean
  distance and uniform weights.
- ``DecisionTreeRegressor``: a small CART regression tree that picks
  squared-error-optimal splits, with ``max_depth`` and
  ``min_samples_split`` controls.

``VotingRegressor`` fits all three base learners on the full training
data and combines their predictions per sample with one of three rules:

- ``"mean"``: the plain average of the three predictions.
- ``"median"``: the per-sample median of the three predictions. With
  three learners the median discards the highest and lowest prediction,
  so a single bad (e.g. badly overfit) base learner cannot drag the
  ensemble away from the other two.
- ``"weighted"``: a weighted average using ``weights`` (normalised to
  sum to 1). ``"mean"`` and ``"median"`` ignore ``weights``.
"""

import numpy as np

__all__ = [
    "RidgeRegressor",
    "KNNRegressor",
    "DecisionTreeRegressor",
    "VotingRegressor",
]


def _as_2d_float_array(X, name="X"):
    """Validate X as a 2D numeric array and return it as float."""
    try:
        X = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {type(X).__name__}") from exc
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {X.shape}")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} must contain only finite values")
    return X


def _as_1d_float_array(y, name="y"):
    """Validate y as a 1D numeric array and return it as float."""
    try:
        y = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {type(y).__name__}") from exc
    if y.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {y.shape}")
    if y.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{name} must contain only finite values")
    return y


def _check_fitted(obj, attrs):
    if not all(hasattr(obj, attr) for attr in attrs):
        raise ValueError(
            f"{type(obj).__name__} is not fitted yet; call fit(X, y) first"
        )


class RidgeRegressor:
    """Ridge regression via the closed-form normal equations.

    Solves ``min_w ||Xw + b - y||^2 + alpha * ||w||^2``. The intercept
    is not penalised. A tiny jitter is added if the system is singular.
    """

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)

    def fit(self, X, y):
        X = _as_2d_float_array(X)
        y = _as_1d_float_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}"
            )
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")

        self.n_features_in_ = X.shape[1]
        if self.fit_intercept:
            self.X_mean_ = X.mean(axis=0)
            self.y_mean_ = float(y.mean())
            Xc = X - self.X_mean_
            yc = y - self.y_mean_
        else:
            self.X_mean_ = np.zeros(X.shape[1])
            self.y_mean_ = 0.0
            Xc, yc = X, y

        n_features = X.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(n_features)
        try:
            self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.pinv(A) @ (Xc.T @ yc)
        self.intercept_ = float(self.y_mean_ - self.X_mean_ @ self.coef_)
        return self

    def predict(self, X):
        _check_fitted(self, ["coef_", "intercept_"])
        X = _as_2d_float_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        return X @ self.coef_ + self.intercept_


class KNNRegressor:
    """k-nearest-neighbours regression, Euclidean distance, uniform weights."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = int(n_neighbors)

    def fit(self, X, y):
        X = _as_2d_float_array(X)
        y = _as_1d_float_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}"
            )
        if self.n_neighbors < 1:
            raise ValueError(
                f"n_neighbors must be >= 1, got {self.n_neighbors}"
            )
        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) exceeds number of "
                f"samples ({X.shape[0]})"
            )
        self.X_train_ = X
        self.y_train_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        _check_fitted(self, ["X_train_", "y_train_"])
        X = _as_2d_float_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        # (n_test, n_train) squared Euclidean distances.
        diff = X[:, None, :] - self.X_train_[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        idx = np.argpartition(dist, self.n_neighbors - 1, axis=1)[
            :, : self.n_neighbors
        ]
        return self.y_train_[idx].mean(axis=1)


class _TreeNode:
    """One node of the CART regression tree."""

    __slots__ = ("value", "feature", "threshold", "left", "right")

    def __init__(self, value, feature=None, threshold=None, left=None, right=None):
        self.value = float(value)  # leaf prediction (mean of y in node)
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class DecisionTreeRegressor:
    """Small CART regression tree with squared-error splits.

    At each node the split (feature, threshold) minimising the total
    within-node sum of squared errors is chosen. Growth stops at
    ``max_depth`` or when a node has fewer than ``min_samples_split``
    samples. Ties in split quality are broken by the lowest feature
    index, then the lowest threshold, so results are deterministic.
    """

    def __init__(self, max_depth=5, min_samples_split=2, random_state=None):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.random_state = random_state

    def fit(self, X, y):
        X = _as_2d_float_array(X)
        y = _as_1d_float_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}"
            )
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")
        if self.min_samples_split < 2:
            raise ValueError(
                f"min_samples_split must be >= 2, got {self.min_samples_split}"
            )
        # random_state is accepted for API compatibility / reproducibility
        # of any tie-breaking; splits here are already deterministic.
        self._rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        """Return (feature, threshold, sse) of the best split, or None."""
        n, n_features = X.shape
        total_sse = float(np.sum((y - y.mean()) ** 2))
        best = None  # (sse, feature, threshold)
        for feature in range(n_features):
            order = np.argsort(X[:, feature], kind="stable")
            x_sorted = X[order, feature]
            y_sorted = y[order]
            # Candidate thresholds: midpoints between distinct values.
            distinct = x_sorted[1:] != x_sorted[:-1]
            if not np.any(distinct):
                continue
            cum_y = np.cumsum(y_sorted)
            cum_y2 = np.cumsum(y_sorted ** 2)
            counts = np.arange(1, n)
            left_sse = cum_y2[:-1] - cum_y[:-1] ** 2 / counts
            right_sse = (
                (cum_y2[-1] - cum_y2[:-1])
                - (cum_y[-1] - cum_y[:-1]) ** 2 / (n - counts)
            )
            sse = left_sse + right_sse
            sse[~distinct] = np.inf
            # Enforce min_samples_split on both sides.
            too_small = (counts < self.min_samples_split) | (
                (n - counts) < self.min_samples_split
            )
            sse[too_small] = np.inf
            j = int(np.argmin(sse))
            if np.isfinite(sse[j]) and sse[j] < total_sse:
                if best is None or sse[j] < best[0]:
                    threshold = float((x_sorted[j] + x_sorted[j + 1]) / 2)
                    best = (float(sse[j]), feature, threshold)
        return best

    def _build(self, X, y, depth):
        node = _TreeNode(value=float(y.mean()))
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return node
        split = self._best_split(X, y)
        if split is None:
            return node
        _, feature, threshold = split
        left_mask = X[:, feature] <= threshold
        if left_mask.all() or (~left_mask).all():
            return node
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _predict_one(self, node, x):
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        _check_fitted(self, ["tree_"])
        X = _as_2d_float_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        return np.array([self._predict_one(self.tree_, x) for x in X])


class VotingRegressor:
    """Voting ensemble of Ridge, k-NN and decision-tree regressors.

    Each base learner is fitted on the full training data and their
    predictions are combined per sample with ``voting``:

    - ``"mean"``: average of the three predictions.
    - ``"median"``: per-sample median of the three predictions (robust
      to a single bad base learner).
    - ``"weighted"``: weighted average with ``weights`` (normalised to
      sum to 1).

    ``weights`` is a list/array of 3 non-negative numbers, or ``None``
    for equal weights. ``random_state`` is accepted for reproducibility
    of any tie-breaking and defaults to 42.
    """

    _VALID_VOTING = ("mean", "median", "weighted")

    def __init__(self, weights=None, voting="mean", random_state=42):
        self.weights = weights
        self.voting = voting
        self.random_state = random_state

    def _validate_weights(self):
        if self.weights is None:
            return np.ones(3) / 3.0
        try:
            w = np.asarray(self.weights, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("weights must be a list/array of 3 numbers") from exc
        if w.ndim != 1 or w.shape[0] != 3:
            raise ValueError(
                f"weights must have length 3 (one per base learner), "
                f"got shape {w.shape}"
            )
        if np.any(w < 0):
            raise ValueError(f"weights must be non-negative, got {self.weights}")
        if not np.any(w > 0):
            raise ValueError("weights must not be all zero")
        return w / w.sum()

    def fit(self, X, y):
        X = _as_2d_float_array(X)
        y = _as_1d_float_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}"
            )
        if self.voting not in self._VALID_VOTING:
            raise ValueError(
                f"voting must be one of {list(self._VALID_VOTING)}, "
                f"got {self.voting!r}"
            )
        self.weights_ = self._validate_weights()
        self.n_features_in_ = X.shape[1]

        self.estimators_ = [
            RidgeRegressor(alpha=1.0),
            KNNRegressor(n_neighbors=5),
            DecisionTreeRegressor(max_depth=5, min_samples_split=2,
                                  random_state=self.random_state),
        ]
        for est in self.estimators_:
            est.fit(X, y)
        return self

    def _base_predictions(self, X):
        return np.column_stack([est.predict(X) for est in self.estimators_])

    def predict(self, X):
        _check_fitted(self, ["estimators_", "weights_"])
        X = _as_2d_float_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        preds = self._base_predictions(X)
        if self.voting == "mean":
            return preds.mean(axis=1)
        if self.voting == "median":
            return np.median(preds, axis=1)
        # "weighted"
        return preds @ self.weights_
