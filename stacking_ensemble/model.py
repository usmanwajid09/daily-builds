"""Stacking classifier from scratch, with numpy only.

A stacking ensemble trains several diverse base classifiers, collects
their out-of-fold predictions on the training data (so the meta learner
never sees predictions made on data a base learner was trained on), and
trains a meta learner on those predictions. At predict time each base
learner predicts on the new sample and the meta learner combines them.

Two tiny built-in learners are provided so the ensemble is usable with
no other dependency: :class:`SimpleLogisticRegression` (gradient
descent) and :class:`DecisionStump` (a one-level decision tree).
"""

from __future__ import annotations

import numpy as np


def _check_xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("X and y must not be empty")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}"
        )
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    return X, y


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class SimpleLogisticRegression:
    """Binary logistic regression trained by gradient descent."""

    def __init__(self, lr: float = 0.1, n_iter: int = 500):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if n_iter <= 0:
            raise ValueError(f"n_iter must be positive, got {n_iter}")
        self.lr = lr
        self.n_iter = n_iter
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def fit(self, X, y):
        X, y = _check_xy(X, y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"SimpleLogisticRegression is binary, got {len(classes)} classes"
            )
        self.classes_ = classes
        y_bin = (y == classes[1]).astype(float)
        w = np.zeros(X.shape[1])
        b = 0.0
        for _ in range(self.n_iter):
            p = _sigmoid(X @ w + b)
            grad_w = X.T @ (p - y_bin) / len(y_bin)
            grad_b = float(np.mean(p - y_bin))
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.coef_ = w
        self.intercept_ = b
        return self

    def predict_proba(self, X):
        if self.coef_ is None:
            raise ValueError("SimpleLogisticRegression is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.coef_):
            raise ValueError(
                f"Expected X with {len(self.coef_)} features, got shape {X.shape}"
            )
        p1 = _sigmoid(X @ self.coef_ + self.intercept_)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class DecisionStump:
    """One-level decision tree: the single best feature/threshold split."""

    def __init__(self):
        self.feature_ = None
        self.threshold_ = None
        self.left_class_ = None
        self.right_class_ = None
        self.classes_ = None

    def fit(self, X, y):
        X, y = _check_xy(X, y)
        self.classes_ = np.unique(y)
        best_gain = -1.0
        best = None
        parent_majority = self.classes_[np.argmax(
            [np.sum(y == c) for c in self.classes_]
        )]
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = X[:, feature] <= t
                right = ~left
                if not left.any() or not right.any():
                    continue
                left_majority = self.classes_[np.argmax(
                    [np.sum(y[left] == c) for c in self.classes_]
                )]
                right_majority = self.classes_[np.argmax(
                    [np.sum(y[right] == c) for c in self.classes_]
                )]
                correct = int(np.sum(y[left] == left_majority)) + int(
                    np.sum(y[right] == right_majority)
                )
                if correct > best_gain:
                    best_gain = correct
                    best = (feature, t, left_majority, right_majority)
        if best is None:
            # No split possible (constant features): predict the majority.
            self.feature_, self.threshold_ = 0, np.inf
            self.left_class_ = self.right_class_ = parent_majority
        else:
            self.feature_, self.threshold_, self.left_class_, self.right_class_ = best
        return self

    def predict(self, X):
        if self.feature_ is None:
            raise ValueError("DecisionStump is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] <= self.feature_:
            raise ValueError(f"Expected X with > {self.feature_} features, got shape {X.shape}")
        return np.where(
            X[:, self.feature_] <= self.threshold_, self.left_class_, self.right_class_
        )

    def predict_proba(self, X):
        if self.feature_ is None:
            raise ValueError("DecisionStump is not fitted yet")
        preds = self.predict(X)
        proba = np.zeros((len(preds), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            proba[:, i] = (preds == c).astype(float)
        return proba


def _clone(estimator):
    """Shallow-copy an estimator so each fold gets a fresh instance."""
    import copy

    return copy.deepcopy(estimator)


class StackingClassifier:
    """Stacking ensemble classifier.

    Parameters
    ----------
    base_estimators : list
        Base classifiers, each with ``fit``/``predict`` and (optionally)
        ``predict_proba``. Each is cloned before fitting.
    meta_estimator : estimator
        Classifier trained on the base learners' out-of-fold predictions.
    n_folds : int
        Number of folds for the out-of-fold predictions. Default 5.
    use_proba : bool
        If True (default), the meta features are the base learners'
        predicted probabilities; otherwise their predicted class labels.
    """

    def __init__(self, base_estimators, meta_estimator, n_folds: int = 5,
                 use_proba: bool = True):
        if not base_estimators:
            raise ValueError("base_estimators must not be empty")
        if meta_estimator is None:
            raise ValueError("meta_estimator must not be None")
        if not isinstance(n_folds, int) or n_folds < 2:
            raise ValueError(f"n_folds must be an integer >= 2, got {n_folds}")
        self.base_estimators = list(base_estimators)
        self.meta_estimator = meta_estimator
        self.n_folds = n_folds
        self.use_proba = use_proba
        self.classes_ = None
        self.base_estimators_ = None
        self.meta_estimator_ = None

    def _meta_features(self, estimators, X):
        cols = []
        for est in estimators:
            if self.use_proba and hasattr(est, "predict_proba"):
                proba = est.predict_proba(X)
                if proba.shape[1] == len(self.classes_):
                    cols.append(proba)
                    continue
            cols.append(est.predict(X).reshape(-1, 1).astype(float))
        return np.hstack(cols)

    def fit(self, X, y):
        X, y = _check_xy(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("Need at least 2 classes to fit a classifier")
        if X.shape[0] < self.n_folds:
            raise ValueError(
                f"Need at least n_folds={self.n_folds} samples, got {X.shape[0]}"
            )
        n = X.shape[0]
        indices = np.arange(n)
        rng = np.random.default_rng(0)
        rng.shuffle(indices)
        folds = np.array_split(indices, self.n_folds)

        oof = np.zeros((n, 0))
        # Build out-of-fold meta features column by column.
        oof_parts = []
        for est in self.base_estimators:
            oof_est = np.zeros((n, len(self.classes_) if self.use_proba else 1))
            for fold in folds:
                mask = np.ones(n, dtype=bool)
                mask[fold] = False
                clone = _clone(est).fit(X[mask], y[mask])
                if self.use_proba and hasattr(clone, "predict_proba"):
                    proba = clone.predict_proba(X[fold])
                    if proba.shape[1] == len(self.classes_):
                        oof_est[fold] = proba
                        continue
                oof_est[fold] = clone.predict(X[fold]).reshape(-1, 1)
            oof_parts.append(oof_est)
        oof = np.hstack(oof_parts)

        # Refit every base learner on the full training set.
        self.base_estimators_ = [
            _clone(est).fit(X, y) for est in self.base_estimators
        ]
        self.meta_estimator_ = _clone(self.meta_estimator).fit(oof, y)
        return self

    def predict(self, X):
        if self.meta_estimator_ is None:
            raise ValueError("StackingClassifier is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        meta_X = self._meta_features(self.base_estimators_, X)
        return self.meta_estimator_.predict(meta_X)

    def predict_proba(self, X):
        if self.meta_estimator_ is None:
            raise ValueError("StackingClassifier is not fitted yet")
        if not hasattr(self.meta_estimator_, "predict_proba"):
            raise ValueError("Meta estimator does not support predict_proba")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        meta_X = self._meta_features(self.base_estimators_, X)
        return self.meta_estimator_.predict_proba(meta_X)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
