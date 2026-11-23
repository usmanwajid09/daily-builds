"""Ridge classifier, from scratch with numpy only.

Fits a ridge regression (closed-form least squares with an L2 penalty)
against one-hot class targets, then classifies by the largest decision
value. For two classes this is equivalent to taking the sign of the
single decision function; for more classes it is one-vs-rest.
"""
from __future__ import annotations

import numpy as np


class RidgeClassifier:
    """Ridge regression adapted as a classifier.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength. Must be positive; larger values
        shrink the coefficients harder toward zero. The intercept is
        not penalised.
    fit_intercept : bool
        Whether to fit an intercept term.
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"alpha must be a positive number, got {alpha!r}")
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    # -- validation -----------------------------------------------------
    @staticmethod
    def _check_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must have at least one sample and one feature")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X

    def _check_fitted(self):
        if self.coef_ is None:
            raise RuntimeError("RidgeClassifier is not fitted yet; call fit() first")

    # -- fitting --------------------------------------------------------
    def fit(self, X, y):
        """Fit the ridge classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels (any hashable type; stored as seen).
        """
        X = self._check_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if len(y) != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {len(y)} labels"
            )
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError(
                f"need at least 2 classes to fit, got {len(self.classes_)}"
            )

        n_samples, n_features = X.shape
        # One-hot targets, coded as -1/+1 so a binary problem uses sign().
        Y = np.full((n_samples, len(self.classes_)), -1.0)
        Y[np.arange(n_samples), y_idx] = 1.0

        X_design = X
        if self.fit_intercept:
            X_design = np.hstack([np.ones((n_samples, 1)), X])

        penalty = self.alpha * np.eye(X_design.shape[1])
        if self.fit_intercept:
            penalty[0, 0] = 0.0  # never penalise the intercept
        # Closed form: W = (X'X + alpha*I)^-1 X'Y
        try:
            W = np.linalg.solve(X_design.T @ X_design + penalty, X_design.T @ Y)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"could not solve ridge system: {exc}") from exc

        if self.fit_intercept:
            self.intercept_ = W[0]
            self.coef_ = W[1:]
        else:
            self.intercept_ = np.zeros(len(self.classes_))
            self.coef_ = W
        return self

    # -- prediction -----------------------------------------------------
    def decision_function(self, X):
        """Raw decision values, shape (n_samples, n_classes)."""
        self._check_fitted()
        X = self._check_X(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"expected {self.coef_.shape[0]} features, got {X.shape[1]}"
            )
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """Predicted class labels."""
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        """Mean accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
