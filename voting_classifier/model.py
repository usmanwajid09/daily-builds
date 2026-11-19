"""Hard/soft voting ensemble classifier, from scratch with numpy only.

A voting ensemble combines several base classifiers:

- **Hard voting**: each base learner casts one vote (optionally
  weighted); the class with the most votes wins.
- **Soft voting**: the predicted class probabilities of the base
  learners are averaged (optionally weighted); the class with the
  highest average probability wins. Soft voting needs every base
  learner to support ``predict_proba``.

Two tiny built-in learners are provided so the ensemble is usable with
no other dependency: :class:`SimpleLogisticRegression` (gradient
descent) and :class:`SimpleKNNClassifier`.
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
            w -= self.lr * (X.T @ (p - y_bin) / len(y_bin))
            b -= self.lr * float(np.mean(p - y_bin))
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


class SimpleKNNClassifier:
    """k-nearest neighbours classifier (Euclidean distance, majority vote)."""

    def __init__(self, k: int = 5):
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k}")
        self.k = k
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None

    def fit(self, X, y):
        X, y = _check_xy(X, y)
        if self.k > X.shape[0]:
            raise ValueError(
                f"k={self.k} is larger than the number of samples {X.shape[0]}"
            )
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        if self.X_train_ is None:
            raise ValueError("SimpleKNNClassifier is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"Expected X with {self.X_train_.shape[1]} features, got shape {X.shape}"
            )
        dists = np.sqrt(((X[:, None, :] - self.X_train_[None, :, :]) ** 2).sum(axis=2))
        nearest = np.argsort(dists, axis=1)[:, : self.k]
        proba = np.zeros((len(X), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            proba[:, i] = np.mean(self.y_train_[nearest] == c, axis=1)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class VotingClassifier:
    """Voting ensemble classifier.

    Parameters
    ----------
    estimators : list
        Base classifiers, each with ``fit``/``predict``. Soft voting
        additionally requires ``predict_proba`` on every estimator.
    voting : {"hard", "soft"}
        Voting strategy. Default "soft".
    weights : list of float or None
        Weight per estimator. If None, all estimators count equally.
    """

    def __init__(self, estimators, voting: str = "soft", weights=None):
        if not estimators:
            raise ValueError("estimators must not be empty")
        if voting not in ("hard", "soft"):
            raise ValueError(f"voting must be 'hard' or 'soft', got {voting!r}")
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if len(weights) != len(estimators):
                raise ValueError(
                    f"weights has length {len(weights)} but there are "
                    f"{len(estimators)} estimators"
                )
            if np.any(weights < 0):
                raise ValueError("weights must not be negative")
            if weights.sum() == 0:
                raise ValueError("weights must not all be zero")
        self.estimators = list(estimators)
        self.voting = voting
        self.weights = weights
        self.classes_ = None
        self.estimators_ = None

    def fit(self, X, y):
        X, y = _check_xy(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("Need at least 2 classes to fit a classifier")
        if self.voting == "soft":
            for est in self.estimators:
                if not hasattr(est, "predict_proba"):
                    raise ValueError(
                        "Soft voting requires every estimator to support "
                        f"predict_proba, but {type(est).__name__} does not"
                    )
        import copy

        self.estimators_ = [copy.deepcopy(est).fit(X, y) for est in self.estimators]
        return self

    def _weights(self):
        if self.weights is None:
            return np.ones(len(self.estimators_))
        return self.weights

    def predict_proba(self, X):
        if self.voting != "soft":
            raise ValueError("predict_proba is only available with soft voting")
        if self.estimators_ is None:
            raise ValueError("VotingClassifier is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        w = self._weights()
        avg = np.zeros((X.shape[0], len(self.classes_)))
        for est, wi in zip(self.estimators_, w):
            avg += wi * est.predict_proba(X)
        return avg / w.sum()

    def predict(self, X):
        if self.estimators_ is None:
            raise ValueError("VotingClassifier is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if self.voting == "soft":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        # Hard voting: weighted vote counts per class.
        w = self._weights()
        votes = np.zeros((X.shape[0], len(self.classes_)))
        for est, wi in zip(self.estimators_, w):
            preds = est.predict(X)
            for i, c in enumerate(self.classes_):
                votes[:, i] += wi * (preds == c)
        return self.classes_[np.argmax(votes, axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
