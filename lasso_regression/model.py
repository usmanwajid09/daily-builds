"""Lasso regression, from scratch with numpy only.

Minimises ``0.5 * ||y - Xb||^2 + alpha * ||b||_1`` by coordinate descent:
each coefficient is updated in turn with the soft-thresholding operator,
which is what drives unhelpful coefficients to exactly zero (feature
selection). Features are standardised before fitting and the target is
centred, so the penalty treats every feature on the same scale; the
returned coefficients are converted back to the original feature scale.
"""
from __future__ import annotations

import numpy as np


class LassoRegression:
    """Lasso (L1-penalised) linear regression via coordinate descent.

    Parameters
    ----------
    alpha : float
        Strength of the L1 penalty. Must be non-negative; ``alpha=0``
        reduces to ordinary least squares. Larger values zero out more
        coefficients.
    max_iter : int
        Maximum number of full passes over the coefficients.
    tol : float
        Convergence tolerance: stop when the largest coefficient change in
        a full pass is smaller than this. Must be positive.
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000,
                 tol: float = 1e-4):
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) \
                or alpha < 0:
            raise ValueError("alpha must be a non-negative number")
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) \
                or max_iter < 1:
            raise ValueError("max_iter must be an integer of at least 1")
        if not isinstance(tol, (int, float)) or isinstance(tol, bool) or tol <= 0:
            raise ValueError("tol must be a positive number")
        self.alpha = float(alpha)
        self.max_iter = max_iter
        self.tol = float(tol)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_iter_: int | None = None

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray) -> "LassoRegression":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of targets")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        if not np.isfinite(y).all():
            raise ValueError("y must not contain NaN or infinite values")
        n_samples, n_features = x.shape
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x_std = np.where(x_std == 0.0, 1.0, x_std)  # constant columns stay 0
        y_mean = float(y.mean())
        xs = (x - x_mean) / x_std
        yc = y - y_mean
        beta = np.zeros(n_features)
        denom = np.sum(xs ** 2, axis=0)  # = n_samples for varying columns
        self.n_iter_ = 0
        for it in range(1, self.max_iter + 1):
            max_change = 0.0
            for j in range(n_features):
                if denom[j] == 0.0:
                    beta[j] = 0.0
                    continue
                # Partial residual: yc - xs @ beta + beta_j * xs[:, j]
                rho = xs[:, j] @ (yc - xs @ beta + beta[j] * xs[:, j])
                new_beta = self._soft_threshold(rho, self.alpha) / denom[j]
                max_change = max(max_change, abs(new_beta - beta[j]))
                beta[j] = new_beta
            self.n_iter_ = it
            if max_change < self.tol:
                break
        self.coef_ = beta / x_std
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = self._check_x(x)
        return x @ self.coef_ + self.intercept_

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Coefficient of determination (R^2) on the given data."""
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of targets")
        pred = self.predict(x)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot

    # -- internals ------------------------------------------------------
    @staticmethod
    def _soft_threshold(z: float, alpha: float) -> float:
        if z > alpha:
            return z - alpha
        if z < -alpha:
            return z + alpha
        return 0.0

    def _check_fitted(self):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    def _check_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                "x has %d features, expected %d" % (x.shape[1], self.coef_.shape[0])
            )
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        return x
