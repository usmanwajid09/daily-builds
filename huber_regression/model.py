"""Robust linear regression with the Huber loss, fitted by IRLS.

The Huber loss is quadratic near zero (like least squares) and linear far
away, so large residuals — outliers — pull the fit far less than OLS does.

    huber(x) = 0.5 * x**2          if |x| <= delta
             = delta * (|x| - 0.5 * delta)  otherwise

The model is fitted by iteratively reweighted least squares (IRLS):
weights w = 1 for inliers and w = delta / |r| for outliers, with residuals
scaled by a robust sigma estimate (1.4826 * MAD) so `delta` is expressed in
units of robust sigma and behaves sensibly regardless of the noise scale.
"""

import numpy as np


def huber_loss(r, delta=1.35):
    """Huber loss of residual(s) ``r`` with threshold ``delta``.

    Quadratic for |r| <= delta, linear beyond it. Accepts scalars or arrays.
    """
    r = np.asarray(r, dtype=float)
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    abs_r = np.abs(r)
    quadratic = 0.5 * r ** 2
    linear = delta * (abs_r - 0.5 * delta)
    return np.where(abs_r <= delta, quadratic, linear)


def ols_fit(X, y):
    """Plain ordinary least squares fit.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)

    Returns
    -------
    (coef, intercept) : tuple of ndarray and float
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have mismatched lengths: {X.shape[0]} vs {y.shape[0]}"
        )
    if X.shape[0] == 0:
        raise ValueError("X and y must be non-empty")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("X and y must contain only finite values")
    X_aug = np.column_stack([X, np.ones(X.shape[0])])
    theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return theta[:-1], float(theta[-1])


class HuberRegression:
    """Robust linear regression with Huber loss, fitted by IRLS.

    Parameters
    ----------
    delta : float, default 1.35
        Huber threshold in units of robust sigma (1.4826 * MAD of the
        residuals). 1.35 gives ~95% efficiency at the Gaussian.
    n_iter : int, default 100
        Maximum number of IRLS iterations.
    tol : float, default 1e-6
        Convergence tolerance on the change in the coefficient vector.

    Attributes
    ----------
    coef_ : ndarray, shape (n_features,)
    intercept_ : float
    n_iter_run_ : int
        Number of IRLS iterations actually run.
    weights_ : ndarray, shape (n_samples,)
        Final IRLS weights (1 for inliers, < 1 for outliers).
    """

    def __init__(self, delta=1.35, n_iter=100, tol=1e-6):
        if not isinstance(delta, (int, float)) or isinstance(delta, bool):
            raise ValueError(f"delta must be a positive number, got {delta!r}")
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        if not isinstance(n_iter, (int, np.integer)) or isinstance(n_iter, bool):
            raise ValueError(f"n_iter must be a positive int, got {n_iter!r}")
        if n_iter <= 0:
            raise ValueError(f"n_iter must be a positive int, got {n_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")
        self.delta = float(delta)
        self.n_iter = int(n_iter)
        self.tol = float(tol)

    def _validate_xy(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have mismatched lengths: {X.shape[0]} vs {y.shape[0]}"
            )
        if X.shape[0] == 0:
            raise ValueError("X and y must be non-empty")
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError("X and y must contain only finite values")
        return X, y

    def fit(self, X, y):
        """Fit the model by iteratively reweighted least squares."""
        X, y = self._validate_xy(X, y)
        n = X.shape[0]
        X_aug = np.column_stack([X, np.ones(n)])

        # Start from the OLS solution.
        theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

        weights = np.ones(n)
        self.n_iter_run_ = 0
        for it in range(1, self.n_iter + 1):
            residuals = y - X_aug @ theta
            # Robust scale: 1.4826 * MAD, floored so delta stays meaningful
            # when residuals are (nearly) all zero.
            mad = np.median(np.abs(residuals - np.median(residuals)))
            sigma = max(1.4826 * mad, 1e-8)
            scaled = residuals / sigma
            abs_scaled = np.abs(scaled)
            weights = np.where(
                abs_scaled <= self.delta, 1.0, self.delta / abs_scaled
            )
            # Weighted least squares via sqrt(w)-scaled design.
            sqrt_w = np.sqrt(weights)
            X_w = X_aug * sqrt_w[:, None]
            y_w = y * sqrt_w
            new_theta, *_ = np.linalg.lstsq(X_w, y_w, rcond=None)
            self.n_iter_run_ = it
            if np.max(np.abs(new_theta - theta)) < self.tol:
                theta = new_theta
                break
            theta = new_theta

        self.coef_ = theta[:-1]
        self.intercept_ = float(theta[-1])
        self.weights_ = weights
        return self

    def predict(self, X):
        """Predict target values for X."""
        if not hasattr(self, "coef_"):
            raise ValueError("HuberRegression is not fitted yet; call fit() first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features but model expects "
                f"{self.coef_.shape[0]}"
            )
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        return X @ self.coef_ + self.intercept_
