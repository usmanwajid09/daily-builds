"""Ridge regression (L2-regularised linear regression) from scratch.

The weights are found with the closed-form solution

    w = (X^T X + alpha * I)^-1 X^T y

where the intercept is fitted by centring the data and is *not*
penalised. Larger ``alpha`` shrinks the coefficients towards zero, which
helps when features are correlated or the sample is small. With
``alpha=0`` this reduces to ordinary least squares (solved via
``np.linalg.lstsq`` when the design matrix is singular).
"""

from __future__ import annotations

import numpy as np


class RidgeRegression:
    """Linear regression with an L2 penalty on the coefficients.

    Args:
        alpha: regularisation strength, must be >= 0. 0 means no
            regularisation (ordinary least squares).
    """

    def __init__(self, alpha: float = 1.0):
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating)):
            raise ValueError(f"alpha must be a number >= 0, got {alpha!r}")
        if not np.isfinite(alpha) or float(alpha) < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """Fit the model to training data.

        Args:
            x: 2D array of shape (n_samples, n_features).
            y: 1D array of shape (n_samples,).

        Returns:
            self, with ``coef_`` and ``intercept_`` set.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of shape (n_samples,)")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("x and y must contain only finite values")

        x_mean = x.mean(axis=0)
        y_mean = y.mean()
        x_centred = x - x_mean
        y_centred = y - y_mean
        n_features = x.shape[1]
        gram = x_centred.T @ x_centred
        if self.alpha > 0.0:
            gram = gram + self.alpha * np.eye(n_features)
            try:
                coef = np.linalg.solve(gram, x_centred.T @ y_centred)
            except np.linalg.LinAlgError:
                coef = np.linalg.pinv(gram) @ (x_centred.T @ y_centred)
        else:
            coef, *_ = np.linalg.lstsq(x_centred, y_centred, rcond=None)
        self.coef_ = coef
        self.intercept_ = float(y_mean - x_mean @ coef)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict target values for ``x``."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                "x has %d features, expected %d"
                % (x.shape[1], self.coef_.shape[0])
            )
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        return x @ self.coef_ + self.intercept_

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the predictions.

        R^2 is 1 for a perfect fit and 0 for a model that only predicts
        the mean. It can be negative for a model worse than the mean.
        """
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of shape (n_samples,)")
        y_pred = self.predict(x)
        if y.shape[0] != y_pred.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot
