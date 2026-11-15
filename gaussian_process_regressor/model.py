"""Gaussian process regression, from scratch with numpy only.

A Gaussian process places a prior over functions: any finite set of
function values is jointly Gaussian with covariance given by a kernel.
Conditioning that prior on noisy observations gives a closed-form
posterior at new points, with both a predictive mean and a predictive
variance — so the model knows how uncertain it is, growing less sure
far from the training data. This module uses the RBF (squared
exponential) kernel, ``k(x, z) = sigma_f^2 * exp(-||x - z||^2 /
(2 * length_scale^2))``, plus observation noise ``sigma_n^2``. The
training covariance is factorised once with a Cholesky decomposition
(numerically safer than inverting the matrix), and predictions reuse
that factorisation. ``predict`` returns both the mean and the standard
deviation; ``log_marginal_likelihood`` scores the fit of the chosen
hyperparameters, which is what makes kernel tuning principled.
"""
from __future__ import annotations

import numpy as np


class GaussianProcessRegressor:
    """Gaussian process regression with an RBF kernel.

    Parameters
    ----------
    length_scale : float
        RBF kernel length scale: how far apart two points must be
        before their function values become nearly uncorrelated.
        Must be positive.
    sigma_f : float
        Signal standard deviation: the typical scale of the function's
        variation. Must be positive.
    sigma_n : float
        Observation-noise standard deviation. Must be positive.
    jitter : float
        Tiny value added to the kernel diagonal for numerical
        stability of the Cholesky factorisation. Must be positive.
    """

    def __init__(self, length_scale: float = 1.0, sigma_f: float = 1.0,
                 sigma_n: float = 0.1, jitter: float = 1e-8):
        for name, value in (("length_scale", length_scale),
                            ("sigma_f", sigma_f),
                            ("sigma_n", sigma_n),
                            ("jitter", jitter)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        self.length_scale = float(length_scale)
        self.sigma_f = float(sigma_f)
        self.sigma_n = float(sigma_n)
        self.jitter = float(jitter)
        self.x_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianProcessRegressor":
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

        self.x_train_ = x
        self.y_train_ = y
        kernel = self._kernel(x, x)
        kernel[np.diag_indices_from(kernel)] += self.sigma_n ** 2 + self.jitter
        self._chol = np.linalg.cholesky(kernel)
        # Solve K alpha = y via the Cholesky factors (no explicit inverse).
        self._alpha = np.linalg.solve(
            self._chol.T, np.linalg.solve(self._chol, y))
        return self

    def predict(self, x: np.ndarray,
                return_std: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict the mean at new points; with ``return_std=True`` also
        return the predictive standard deviation."""
        if self.x_train_ is None:
            raise ValueError("model is not fitted yet; call fit() first")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if x.shape[1] != self.x_train_.shape[1]:
            raise ValueError(
                f"x has {x.shape[1]} features but the model was fitted on "
                f"{self.x_train_.shape[1]}"
            )
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")

        k_star = self._kernel(self.x_train_, x)  # (n_train, n_new)
        mean = k_star.T @ self._alpha
        if not return_std:
            return mean
        # Predictive variance: k(x, x) - k_star^T K^-1 k_star.
        v = np.linalg.solve(self._chol, k_star)
        var = self.sigma_f ** 2 - np.sum(v ** 2, axis=0)
        std = np.sqrt(np.maximum(var, 0.0))
        return mean, std

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2."""
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(x)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot

    def log_marginal_likelihood(self) -> float:
        """Return log p(y | X) under the current hyperparameters."""
        if self.x_train_ is None:
            raise ValueError("model is not fitted yet; call fit() first")
        n = len(self.y_train_)
        quad = float(self.y_train_ @ self._alpha)
        log_det = 2.0 * float(np.sum(np.log(np.diag(self._chol))))
        return -0.5 * quad - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)

    # -- internals ------------------------------------------------------
    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sq_a = np.sum(a ** 2, axis=1, keepdims=True)
        sq_b = np.sum(b ** 2, axis=1)
        dist_sq = np.maximum(sq_a + sq_b - 2.0 * (a @ b.T), 0.0)
        return self.sigma_f ** 2 * np.exp(-dist_sq / (2.0 * self.length_scale ** 2))
