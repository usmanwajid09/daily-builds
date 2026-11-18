"""Gradient boosting classifier, from scratch with numpy only.

Binary classification uses the logistic (log) loss; multi-class uses the
multinomial deviance (softmax) loss with one regression tree per class per
stage. Each stage fits small CART regression trees to the negative
gradient (residuals) of the loss and adds them to the ensemble, scaled by
the learning rate.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


class _TreeNode:
    """One node of a regression tree: a split or a leaf value."""

    def __init__(self, *, feature=None, threshold=None, left=None,
                 right=None, value=None, n_samples=0, gain=0.0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples
        self.gain = gain  # variance reduction contributed by this split

    @property
    def is_leaf(self):
        return self.value is not None


class _RegressionTree:
    """Tiny CART regression tree fit by minimising squared error."""

    def __init__(self, *, max_depth=3, min_samples_split=2):
        if not isinstance(max_depth, (int, np.integer)) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer, "
                             f"got {max_depth!r}")
        if not isinstance(min_samples_split, (int, np.integer)) \
                or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2, "
                             f"got {min_samples_split!r}")
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.root = None
        self.importances_ = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if x.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {x.shape}")
        if len(x) != len(y):
            raise ValueError("X and y have different lengths "
                             f"({len(x)} vs {len(y)})")
        self.n_features_in_ = x.shape[1]
        self.importances_ = np.zeros(self.n_features_in_)
        self.root = self._grow(x, y, depth=0)
        total = self.importances_.sum()
        if total > 0:
            self.importances_ /= total
        return self

    def _grow(self, x, y, depth):
        node = _TreeNode(n_samples=len(y))
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or np.all(y == y[0])):
            node.value = float(np.mean(y))
            return node
        feature, threshold, gain = self._best_split(x, y)
        if feature is None or gain <= 0:
            node.value = float(np.mean(y))
            return node
        node.feature = feature
        node.threshold = threshold
        node.gain = gain
        self.importances_[feature] += gain
        left_mask = x[:, feature] <= threshold
        node.left = self._grow(x[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(x[~left_mask], y[~left_mask], depth + 1)
        return node

    @staticmethod
    def _best_split(x, y):
        n, n_features = x.shape
        total_var = float(np.var(y)) * n
        best_feature, best_threshold, best_gain = None, None, 0.0
        for feature in range(n_features):
            order = np.argsort(x[:, feature], kind="stable")
            xs = x[order, feature]
            ys = y[order]
            # Candidate thresholds: midpoints between distinct values.
            distinct = xs[1:] != xs[:-1]
            if not np.any(distinct):
                continue
            idx = np.flatnonzero(distinct)
            cum = np.cumsum(ys)
            total = cum[-1]
            counts_left = idx + 1
            counts_right = n - counts_left
            sums_left = cum[idx]
            sums_right = total - sums_left
            # Weighted variance of the two children.
            var_left = (np.cumsum(ys * ys)[idx]
                        - sums_left ** 2 / counts_left)
            var_right = ((np.cumsum(ys * ys)[-1] - np.cumsum(ys * ys)[idx])
                         - sums_right ** 2 / counts_right)
            gains = total_var - (var_left + var_right)
            j = int(np.argmax(gains))
            if gains[j] > best_gain:
                best_gain = float(gains[j])
                best_feature = feature
                best_threshold = float((xs[idx[j]] + xs[idx[j] + 1]) / 2.0)
        return best_feature, best_threshold, best_gain

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {x.shape}")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {x.shape[1]} features, expected "
                             f"{self.n_features_in_}")
        out = np.empty(len(x))
        for i, row in enumerate(x):
            node = self.root
            while not node.is_leaf:
                node = (node.left if row[node.feature] <= node.threshold
                        else node.right)
            out[i] = node.value
        return out


class GradientBoostingClassifier:
    """Gradient boosting classifier with log-loss / multinomial deviance.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages.
    learning_rate : float
        Shrinkage applied to each stage's contribution.
    max_depth : int
        Maximum depth of each regression tree.
    min_samples_split : int
        Minimum samples a node needs before it may be split.
    subsample : float
        Fraction of samples used to fit each tree (1.0 = all samples).
        Values below 1.0 add stochasticity; requires ``random_state`` for
        reproducibility.
    random_state : int or None
        Seed for the subsampling RNG.
    """

    def __init__(self, *, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, subsample=1.0, random_state=None):
        if not isinstance(n_estimators, (int, np.integer)) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer, "
                             f"got {n_estimators!r}")
        if not (0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1], "
                             f"got {learning_rate!r}")
        if not isinstance(max_depth, (int, np.integer)) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer, "
                             f"got {max_depth!r}")
        if not (0 < subsample <= 1.0):
            raise ValueError(f"subsample must be in (0, 1], got {subsample!r}")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.subsample = float(subsample)
        self.random_state = random_state

    # -- fitting ---------------------------------------------------------
    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {x.shape}")
        if len(x) != len(y):
            raise ValueError("X and y have different lengths "
                             f"({len(x)} vs {len(y)})")
        if len(x) == 0:
            raise ValueError("Cannot fit on an empty dataset")
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("Need at least 2 classes to fit, "
                             f"got {len(classes)}")
        self.classes_ = classes
        self.n_classes_ = len(classes)
        self.n_features_in_ = x.shape[1]
        y_idx = np.searchsorted(classes, y)

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self.feature_importances_ = np.zeros(self.n_features_in_)

        if self.n_classes_ == 2:
            prior = float(np.mean(y_idx == 1))
            prior = np.clip(prior, 1e-6, 1 - 1e-6)
            raw = np.full(len(y), np.log(prior / (1 - prior)))
            for _ in range(self.n_estimators):
                p = _sigmoid(raw)
                residual = (y_idx == 1).astype(float) - p
                tree = self._fit_tree(x, residual, rng)
                raw = raw + self.learning_rate * tree.predict(x)
                self.trees_.append([tree])
            self.init_raw_ = np.log(prior / (1 - prior))
        else:
            priors = np.array([np.mean(y_idx == k)
                               for k in range(self.n_classes_)])
            priors = np.clip(priors, 1e-6, None)
            priors = priors / priors.sum()
            raw = np.tile(np.log(priors), (len(y), 1))
            one_hot = np.eye(self.n_classes_)[y_idx]
            for _ in range(self.n_estimators):
                p = _softmax(raw)
                stage = []
                for k in range(self.n_classes_):
                    residual = one_hot[:, k] - p[:, k]
                    tree = self._fit_tree(x, residual, rng)
                    raw[:, k] += self.learning_rate * tree.predict(x)
                    stage.append(tree)
                self.trees_.append(stage)
            self.init_raw_ = np.log(priors)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _fit_tree(self, x, residual, rng):
        if self.subsample < 1.0:
            n = int(self.subsample * len(x))
            idx = rng.choice(len(x), size=n, replace=False)
            x_sub, r_sub = x[idx], residual[idx]
        else:
            x_sub, r_sub = x, residual
        tree = _RegressionTree(max_depth=self.max_depth,
                               min_samples_split=self.min_samples_split)
        tree.fit(x_sub, r_sub)
        self.feature_importances_ += tree.importances_ * len(x_sub)
        return tree

    # -- prediction ------------------------------------------------------
    def _raw_scores(self, x, n_stages=None):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {x.shape}")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {x.shape[1]} features, expected "
                             f"{self.n_features_in_}")
        stages = self.trees_ if n_stages is None else self.trees_[:n_stages]
        if self.n_classes_ == 2:
            raw = np.full(len(x), float(self.init_raw_))
            for (tree,) in stages:
                raw = raw + self.learning_rate * tree.predict(x)
            return raw
        raw = np.tile(np.asarray(self.init_raw_, dtype=float), (len(x), 1))
        for stage in stages:
            for k, tree in enumerate(stage):
                raw[:, k] += self.learning_rate * tree.predict(x)
        return raw

    def predict_proba(self, x):
        self._check_fitted()
        raw = self._raw_scores(x)
        if self.n_classes_ == 2:
            p1 = _sigmoid(raw)
            return np.column_stack([1 - p1, p1])
        return _softmax(raw)

    def predict(self, x):
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]

    def staged_predict(self, x):
        """Yield class predictions after each boosting stage."""
        self._check_fitted()
        for n in range(1, len(self.trees_) + 1):
            raw = self._raw_scores(x, n_stages=n)
            if self.n_classes_ == 2:
                proba = np.column_stack([1 - _sigmoid(raw), _sigmoid(raw)])
            else:
                proba = _softmax(raw)
            yield self.classes_[np.argmax(proba, axis=1)]

    def score(self, x, y):
        """Return classification accuracy on (x, y)."""
        return float(np.mean(self.predict(x) == np.asarray(y)))

    def _check_fitted(self):
        if not hasattr(self, "trees_"):
            raise ValueError("Model is not fitted yet — call fit() first")
