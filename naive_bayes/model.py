"""Gaussian Naive Bayes classifier, from scratch with numpy only.

Each feature is modelled per class as an independent Gaussian, so the
log-posterior of a class is its log-prior plus the sum of the per-feature
log-likelihoods. A small variance floor (``var_smoothing``) keeps the
Gaussian well-defined when a feature is constant inside a class.
"""
from __future__ import annotations

import numpy as np


class GaussianNB:
    """Gaussian Naive Bayes classifier.

    Parameters
    ----------
    var_smoothing : float
        Fraction of the largest feature variance added to every class
        variance as a floor, so no variance is ever zero.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        if var_smoothing < 0:
            raise ValueError("var_smoothing must be non-negative")
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self._mean: np.ndarray | None = None   # (n_classes, n_features)
        self._var: np.ndarray | None = None    # (n_classes, n_features)
        self._log_prior: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianNB":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on empty data")
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = x.shape[1]
        self._mean = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))
        self._log_prior = np.zeros(n_classes)
        epsilon = self.var_smoothing * np.var(x, axis=0).max()
        for i, cls in enumerate(self.classes_):
            rows = x[y == cls]
            self._mean[i] = rows.mean(axis=0)
            self._var[i] = rows.var(axis=0) + epsilon
            self._log_prior[i] = np.log(len(rows) / len(y))
        return self

    def _log_likelihood(self, x: np.ndarray) -> np.ndarray:
        # (n_samples, n_classes, n_features) -> sum over features.
        diff = x[:, None, :] - self._mean[None, :, :]
        log_pdf = -0.5 * (
            np.log(2.0 * np.pi * self._var[None, :, :])
            + diff ** 2 / self._var[None, :, :]
        )
        return log_pdf.sum(axis=2) + self._log_prior[None, :]

    def _check_fitted(self):
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    def _check_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self._mean.shape[1]:
            raise ValueError(
                "x has %d features, expected %d" % (x.shape[1], self._mean.shape[1])
            )
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = self._check_x(x)
        return self.classes_[np.argmax(self._log_likelihood(x), axis=1)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities via a numerically stable softmax of log-likelihoods."""
        self._check_fitted()
        x = self._check_x(x)
        log_post = self._log_likelihood(x)
        shifted = log_post - log_post.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        return probs / probs.sum(axis=1, keepdims=True)
