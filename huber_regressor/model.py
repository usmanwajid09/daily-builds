"""Robust linear regression with the Huber loss, fitted by IRLS.

Model
-----
    y = X @ w + b + eps

Ordinary least squares (OLS) minimises the sum of squared residuals, so a
single gross outlier can drag the fitted line far off. The Huber loss
keeps the squared loss for small residuals but switches to a *linear*
loss for large ones, capping the influence of any one point::

    L(r) = 0.5 * r^2                     if |r| <= c
    L(r) = c * |r| - 0.5 * c^2           if |r| >  c

where ``c = epsilon * scale`` and ``scale`` is a robust estimate of the
residual spread (1.4826 * MAD, the median absolute deviation, which makes
it consistent with the standard deviation for Gaussian noise). The
default ``epsilon = 1.35`` gives 95% efficiency at the Gaussian model
while still down-weighting outliers.

Fitting is done by iteratively reweighted least squares (IRLS): start
from the OLS fit, then repeat

    1. residuals r = y - y_pred
    2. scale = 1.4826 * median(|r - median(r)|)   (fallback below if 0)
    3. weights w = 1                  if |r| <= epsilon * scale
               w = epsilon * scale / |r|   otherwise
    4. solve the weighted least-squares problem with those weights

until the coefficient vector stops moving (max change < ``tol``) or
``max_iter`` iterations are reached. Each weighted least-squares step is
solved with ``numpy.linalg.lstsq`` on the sqrt-weight-scaled design
matrix, which also handles rank-deficient designs gracefully.

The intercept is fitted by augmenting the design matrix with a column
of ones when ``fit_intercept=True``; with ``fit_intercept=False`` the
line is forced through the origin and ``intercept_`` is 0.0.

Only NumPy and the standard library are used.
"""

import numpy as np


class HuberRegressor:
    """Robust linear regression with the Huber loss, fitted by IRLS.

    Parameters
    ----------
    epsilon : float, default=1.35
        Outlier threshold in units of the residual scale (1.4826 * MAD).
        Residuals larger than ``epsilon * scale`` are down-weighted.
        Must be > 0.
    max_iter : int, default=100
        Maximum number of IRLS iterations. Must be >= 1.
    tol : float, default=1e-6
        Convergence tolerance: stop when the largest absolute change in
        any fitted parameter (slopes and intercept) falls below this.
        Must be > 0.
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    """

    def __init__(self, epsilon=1.35, max_iter=100, tol=1e-6, fit_intercept=True):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.fit_intercept = bool(fit_intercept)
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _validate_X_y(X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("X and y must be non-empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have mismatched lengths: {X.shape[0]} vs {y.shape[0]}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")
        return X, y

    def _design(self, X):
        """Design matrix: X with an optional leading column of ones."""
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    @staticmethod
    def _robust_scale(residuals):
        """1.4826 * MAD of the residuals, with fallbacks if MAD is 0."""
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        scale = 1.4826 * mad
        if scale <= 0:
            # All residuals (nearly) identical, e.g. a perfect fit: fall
            # back to the mean absolute deviation from the median.
            scale = float(np.mean(np.abs(residuals - med)))
        if scale <= 0:
            # Perfect fit (all residuals 0): any positive scale works,
            # since every weight will be 1 anyway.
            scale = 1.0
        return scale

    # -- main API --------------------------------------------------------
    def fit(self, X, y):
        """Fit the Huber regression model by IRLS.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = self._validate_X_y(X, y)
        X_aug = self._design(X)

        # Start from plain OLS (all weights equal to 1).
        params, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

        self.n_iter_ = 0
        for it in range(1, self.max_iter + 1):
            residuals = y - X_aug @ params
            scale = self._robust_scale(residuals)
            cutoff = self.epsilon * scale
            abs_r = np.abs(residuals)
            weights = np.where(abs_r <= cutoff, 1.0, cutoff / np.maximum(abs_r, 1e-300))
            # Weighted least squares via sqrt-weight scaling.
            sw = np.sqrt(weights)
            new_params, *_ = np.linalg.lstsq(X_aug * sw[:, None], y * sw, rcond=None)
            change = float(np.max(np.abs(new_params - params)))
            params = new_params
            self.n_iter_ = it
            if change < self.tol:
                break

        if self.fit_intercept:
            self.intercept_ = float(params[0])
            self.coef_ = np.asarray(params[1:], dtype=float)
        else:
            self.intercept_ = 0.0
            self.coef_ = np.asarray(params, dtype=float)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Predict target values for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
        """
        if self.coef_ is None:
            raise ValueError("HuberRegressor is not fitted yet; call fit() first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        R^2 = 1 - SS_res / SS_tot. It is 1.0 for a perfect fit and can be
        negative for a model worse than predicting the mean.
        """
        X, y = self._validate_X_y(X, y)
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot
