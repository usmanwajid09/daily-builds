"""Adaline classifier from scratch (gradient descent, standardisation), numpy only."""

import numpy as np


class AdalineClassifier:
    """Adaptive Linear Neuron classifier.

    Trains a linear model by minimising the mean squared error between
    the raw (linear) net input and the target labels with batch gradient
    descent. Features are standardised by default, which is important
    for Adaline because the gradient step size depends on feature scale.

    Binary classification uses labels -1 / +1 internally. Multiclass
    problems are handled with a one-vs-rest ensemble of binary Adalines.

    Parameters
    ----------
    learning_rate : float, default 0.01
        Step size for gradient descent. Must be positive.
    n_iter : int, default 1000
        Maximum number of gradient-descent epochs. Must be positive.
    standardise : bool, default True
        Whether to standardise features (zero mean, unit variance)
        using the training data statistics.
    tol : float, default 1e-6
        Stop early when the change in cost between epochs is below this.
    random_state : int or None, default None
        Seed used only to shuffle nothing here; kept for API symmetry
        with other estimators (no randomness is used).
    """

    def __init__(self, learning_rate=0.01, n_iter=1000, standardise=True,
                 tol=1e-6, random_state=None):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive, "
                             f"got {learning_rate!r}")
        if n_iter <= 0:
            raise ValueError(f"n_iter must be positive, got {n_iter!r}")
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol!r}")
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.standardise = standardise
        self.tol = tol
        self.random_state = random_state

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _check_xy(X, y):
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
        return X, y

    def _scale(self, X, fit):
        if not self.standardise:
            return X
        if fit:
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.scale_[self.scale_ == 0.0] = 1.0
        return (X - self.mean_) / self.scale_

    def _fit_binary(self, X, y_signed):
        """Fit one binary Adaline; y_signed holds -1 / +1."""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        costs = []
        for _ in range(self.n_iter):
            net = X @ w + b
            errors = y_signed - net
            grad_w = -(2.0 / n_samples) * (X.T @ errors)
            grad_b = -(2.0 / n_samples) * errors.sum()
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
            cost = float(np.mean(errors ** 2))
            costs.append(cost)
            if len(costs) > 1 and abs(costs[-2] - costs[-1]) < self.tol:
                break
        return w, b, costs

    # -- API -------------------------------------------------------------
    def fit(self, X, y):
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels (any hashable type; at least 2 classes needed).
        """
        X, y = self._check_xy(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError(
                f"Need at least 2 classes, got {len(self.classes_)}")
        X = self._scale(X, fit=True)
        self.coef_ = []
        self.intercept_ = []
        self.costs_ = []
        for cls in self.classes_:
            y_signed = np.where(y == cls, 1.0, -1.0)
            w, b, costs = self._fit_binary(X, y_signed)
            self.coef_.append(w)
            self.intercept_.append(b)
            self.costs_.append(costs)
        self.coef_ = np.asarray(self.coef_)
        self.intercept_ = np.asarray(self.intercept_)
        self.n_iter_ = max(len(c) for c in self.costs_)
        return self

    def decision_function(self, X):
        """Raw net input per class, shape (n_samples, n_classes)."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("AdalineClassifier is not fitted yet; call fit() first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[1] != self.coef_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.coef_.shape[1]}")
        X = self._scale(X, fit=False)
        return X @ self.coef_.T + self.intercept_

    def predict(self, X):
        """Predict class labels."""
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Class probabilities via a sigmoid over the net inputs."""
        scores = self.decision_function(X)
        # Sigmoid, clipped for numerical safety.
        probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / row_sums

    def score(self, X, y):
        """Mean accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
