"""Isolation forest anomaly detection, from scratch with numpy only.

An isolation forest builds an ensemble of random isolation trees. Each
tree recursively splits the data by picking a random feature and a random
split value between that feature's min and max in the current node, until
every leaf holds one sample (or a depth limit is reached). Anomalies are
isolated in very few splits because they sit far from the rest of the
data, so their average path length across the forest is short; normal
points need many splits and have long path lengths. The anomaly score
is ``2 ** (-mean_path_length / c(n))``, where ``c(n)`` is the average path
length of an unsuccessful search in a binary search tree of ``n`` nodes.
"""
from __future__ import annotations

import math

import numpy as np


def _harmonic_number(n: int) -> float:
    """Harmonic number H(n) = 1 + 1/2 + ... + 1/n (H(0) = 0)."""
    if n <= 0:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, n + 1)))


def _c_factor(n: int) -> float:
    """Average path length of an unsuccessful BST search over n nodes.

    c(n) = 2 * H(n - 1) - 2 * (n - 1) / n, with c(1) = 0 and c(2) = 1.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * _harmonic_number(n - 1) - 2.0 * (n - 1) / n


class _IsolationTree:
    """A single random isolation tree.

    Nodes are tuples: ("leaf", size) or
    ("split", feature, split_value, left, right).
    """

    def __init__(self, rng: np.random.Generator, max_depth: int):
        self.rng = rng
        self.max_depth = max_depth
        self.root = None

    def fit(self, x: np.ndarray) -> "_IsolationTree":
        self.root = self._build(x, depth=0)
        return self

    def _build(self, x: np.ndarray, depth: int):
        n = x.shape[0]
        if n <= 1 or depth >= self.max_depth:
            return ("leaf", n)
        feature = int(self.rng.integers(x.shape[1]))
        col = x[:, feature]
        lo, hi = float(col.min()), float(col.max())
        if lo == hi:
            # All values identical on this feature: cannot split further.
            return ("leaf", n)
        split = float(self.rng.uniform(lo, hi))
        left_mask = col < split
        if not left_mask.any() or left_mask.all():
            # Degenerate split (should be rare): treat as a leaf.
            return ("leaf", n)
        return (
            "split",
            feature,
            split,
            self._build(x[left_mask], depth + 1),
            self._build(x[~left_mask], depth + 1),
        )

    def path_length(self, x: np.ndarray) -> np.ndarray:
        """Path length of each sample, with the c(size) adjustment for leaves."""
        out = np.empty(x.shape[0], dtype=float)
        for i, row in enumerate(x):
            out[i] = self._walk(self.root, row, depth=0)
        return out

    def _walk(self, node, row: np.ndarray, depth: int) -> float:
        if node[0] == "leaf":
            return depth + _c_factor(node[1])
        _, feature, split, left, right = node
        if row[feature] < split:
            return self._walk(left, row, depth + 1)
        return self._walk(right, row, depth + 1)


class IsolationForest:
    """Isolation forest anomaly detector.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees in the forest.
    max_samples : int or "auto"
        Number of samples drawn (without replacement) to train each tree.
        ``"auto"`` means ``min(256, n_samples)``.
    contamination : float
        Expected fraction of outliers in the training data, in (0, 0.5].
        Used to set the decision threshold: the lowest ``contamination``
        fraction of training scores are labelled outliers.
    random_state : int or None
        Seed for reproducible subsampling and splits.
    """

    def __init__(self, n_estimators: int = 100, max_samples=256,
                 contamination: float = 0.1, random_state: int | None = None):
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int):
            raise ValueError("n_estimators must be an integer")
        if n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if isinstance(max_samples, str):
            if max_samples != "auto":
                raise ValueError('max_samples must be a positive integer or "auto"')
        elif isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise ValueError('max_samples must be a positive integer or "auto"')
        elif max_samples < 1:
            raise ValueError("max_samples must be a positive integer")
        if not isinstance(contamination, (int, float)) or isinstance(contamination, bool):
            raise ValueError("contamination must be a number in (0, 0.5]")
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, int)
        ):
            raise ValueError("random_state must be an integer or None")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = float(contamination)
        self.random_state = random_state
        self.trees_: list[_IsolationTree] = []
        self.threshold_: float | None = None
        self.offset_: float | None = None
        self.n_features_in_: int | None = None
        self._n_train: int | None = None

    def fit(self, x: np.ndarray) -> "IsolationForest":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        n_samples, n_features = x.shape
        if self.max_samples == "auto":
            subsample_size = min(256, n_samples)
        else:
            subsample_size = min(self.max_samples, n_samples)
        max_depth = math.ceil(math.log2(subsample_size)) if subsample_size > 1 else 0
        self.trees_ = []
        for i in range(self.n_estimators):
            rng = np.random.default_rng(
                None if self.random_state is None else self.random_state + i
            )
            idx = rng.choice(n_samples, size=subsample_size, replace=False)
            tree = _IsolationTree(rng, max_depth)
            tree.fit(x[idx])
            self.trees_.append(tree)
        self.n_features_in_ = n_features
        self._n_train = n_samples
        # Compute training scores directly (decision_function refuses to
        # run until threshold_ is set, and that is what we are computing).
        c = _c_factor(n_samples)
        if c == 0.0:
            train_scores = np.full(n_samples, 0.5)
        else:
            train_scores = 2.0 ** (-self._mean_path_length(x) / c)
        # Lowest `contamination` fraction of training scores are outliers,
        # so the threshold is the (1 - contamination) quantile.
        self.threshold_ = float(np.quantile(train_scores, 1.0 - self.contamination))
        self.offset_ = -self.threshold_
        return self

    def _check_fitted(self):
        if not self.trees_ or self.threshold_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    def _validate_predict_input(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if x.shape[0] == 0:
            raise ValueError("cannot predict on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                "x has %d features, expected %d" % (x.shape[1], self.n_features_in_)
            )
        return x

    def _mean_path_length(self, x: np.ndarray) -> np.ndarray:
        total = np.zeros(x.shape[0], dtype=float)
        for tree in self.trees_:
            total += tree.path_length(x)
        return total / len(self.trees_)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Anomaly score per sample: higher means more anomalous.

        Scores lie in (0, 1]; a score near 1 means the point was isolated
        in very few splits, near 0.5 means an average path length.
        """
        self._check_fitted()
        x = self._validate_predict_input(x)
        mean_path = self._mean_path_length(x)
        c = _c_factor(self._n_train)
        if c == 0.0:
            # A single training sample: everything looks equally normal.
            return np.full(x.shape[0], 0.5)
        return 2.0 ** (-mean_path / c)

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        """Alias for :meth:`decision_function`."""
        return self.decision_function(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return 1 for inliers and -1 for outliers."""
        scores = self.decision_function(x)
        return np.where(scores >= self.threshold_, -1, 1).astype(int)
