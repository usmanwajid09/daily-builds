"""Kernel ridge regression, from scratch with numpy only.

Kernel ridge regression fits a function in the reproducing-kernel
Hilbert space of a kernel: the prediction at a new point is a weighted
sum of kernel similarities to the training points,
``f(x) = sum_i alpha_i * k(x, x_i)``. The dual weights have a
closed-form solution, ``alpha = (K + lambda * I)^-1 y``, where ``K``
is the kernel matrix on the training data and ``lambda`` is the ridge
regularisation strength. This module uses the RBF (Gaussian) kernel,
``k(x, z) = exp(-gamma * ||x - z||^2)``, which lets a linear method
fit genuinely non-linear relationships such as a sine wave. The
kernel matrix is solved with ``np.linalg.solve`` (falling back to a
least-squares solve if it is singular), and predictions need only the
stored training points and dual weights.
"""
from __future__ import annotations

import numpy as np


class KernelRidgeRegressor:
    """Kernel ridge regression with an RBF kernel.

    Parameters
    ----------
    alpha : float
        Ridge regularisation strength (``lambda`` in the dual solution).
        Must be positive; larger values shrink the fit toward zero and
        smooth it.
    gamma : float
        RBF kernel bandwidth parameter: ``k(x, z) =
        exp(-gamma * ||x - z||^2)``. Must be positive; larger values
        make the kernel narrower and the fit more local. If omitted,
        defaults to ``1 / n_features``.
    """

    def __init__(self, alpha: float = 1.0, gamma: float | None = None):
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) \
                or alpha <= 0:
            raise ValueError("alpha must be a positive number")
        if gamma is not None and (
                not isinstance(gamma, (int, float)) or isinstance(gamma, bool)
                or gamma <= 0):
            raise ValueError("gamma must be a positive number or None")
        self.alpha = float(alpha)
        self.gamma = None if gamma is None else float(gamma)
        self.dual_coef_: np.ndarray | None = None
        self.x_train_: np.ndarray | None = None
        self.gamma_: float | None = None

    # -- public API -----------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegressor":
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

        self.gamma_ = self.gamma if self.gamma is not None else 1.0 / x.shape[1]
        kernel = self._rbf_kernel(x, x)
        kernel[np.diag_indices_from(kernel)] += self.alpha
        try:
            dual = np.linalg.solve(kernel, y)
        except np.linalg.LinAlgError:
            dual, *_ = np.linalg.lstsq(kernel, y, rcond=None)
        self.dual_coef_ = dual
        self.x_train_ = x
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.dual_coef_ is None:
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
        return self._rbf_kernel(x, self.x_train_) @ self.dual_coef_

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2."""
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(x)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot

    # -- internals ------------------------------------------------------
    def _rbf_kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sq_a = np.sum(a ** 2, axis=1, keepdims=True)
        sq_b = np.sum(b ** 2, axis=1)
        dist_sq = np.maximum(sq_a + sq_b - 2.0 * (a @ b.T), 0.0)
        return np.exp(-self.gamma_ * dist_sq)
