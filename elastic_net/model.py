"""Elastic Net regression, from scratch with numpy only.

Minimises ``0.5 * ||y - Xb||^2 + alpha * (l1_ratio * ||b||_1
+ 0.5 * (1 - l1_ratio) * ||b||_2^2)`` by coordinate descent: each
coefficient is updated in turn with the soft-thresholding operator,
with the L2 part shrinking the update through the denominator. The L1
term selects features (driving some coefficients to exactly zero) while
the L2 term keeps correlated features together instead of picking one
arbitrarily, which is where Elastic Net beats pure Lasso. Features are
standardised before fitting and the target is centred, so the penalty
treats every feature on the same scale; the returned coefficients are
converted back to the original feature scale.
"""
from __future__ import annotations

import numpy as np


class ElasticNetRegression:
    """Elastic Net (L1 + L2 penalised) linear regression via coordinate descent.

    Parameters
    ----------
    alpha : float
        Overall penalty strength. Must be non-negative; ``alpha=0``
        reduces to ordinary least squares.
    l1_ratio : float
        Mix between L1 and L2: 1.0 is pure Lasso, 0.0 is pure ridge.
        Must lie in [0, 1].
    max_iter : int
        Maximum number of full passes over the coefficients.
    tol : float
        Convergence tolerance: stop when the largest coefficient change
        in a full pass is smaller than this. Must be positive.
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5,
                 max_iter: int = 1000, tol: float = 1e-4):
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) \
                or alpha < 0:
            raise ValueError("alpha must be a non-negative number")
        if not isinstance(l1_ratio, (int, float)) or isinstance(l1_ratio, bool) \
                or not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be a number in [0, 1]")
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) \
                or max_iter < 1:
            raise ValueError("max_iter must be an integer of at least 1")
        if not isinstance(tol, (int, float)) or isinstance(tol, bool) or tol <= 0:
            raise ValueError("tol must be a positive number")
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = max_iter
        self.tol = float(tol)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_iter_: int | None = None

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray) -> "ElasticNetRegression":
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
            raise ValueError("x must contain only finite values")
        if not np.isfinite(y).all():
            raise ValueError("y must contain only finite values")

        n, p = x.shape
        self._x_mean = x.mean(axis=0)
        self._x_std = x.std(axis=0)
        self._x_std[self._x_std == 0.0] = 1.0  # constant feature: no scaling
        xs = (x - self._x_mean) / self._x_std
        y_mean = y.mean()
        yc = y - y_mean

        beta = np.zeros(p)
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1.0 - self.l1_ratio)
        # Column norms of the standardised design matrix.
        col_sq = np.sum(xs ** 2, axis=0)
        for it in range(1, self.max_iter + 1):
            max_change = 0.0
            for j in range(p):
                # Partial residual without feature j.
                r = yc - xs @ beta + xs[:, j] * beta[j]
                rho = xs[:, j] @ r
                denom = col_sq[j] + l2
                new_beta = np.sign(rho) * max(abs(rho) - l1, 0.0) / denom
                max_change = max(max_change, abs(new_beta - beta[j]))
                beta[j] = new_beta
            if max_change < self.tol:
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.max_iter

        # Convert back to the original feature scale.
        self.coef_ = beta / self._x_std
        self.intercept_ = float(y_mean - self._x_mean @ self.coef_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("model is not fitted yet; call fit() first")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if x.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"x has {x.shape[1]} features but the model was fitted on "
                f"{self.coef_.shape[0]}"
            )
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        return x @ self.coef_ + self.intercept_

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2."""
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(x)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot
