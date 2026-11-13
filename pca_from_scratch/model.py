"""Principal component analysis, from scratch with numpy only.

The data are centred, the covariance matrix is formed, and its
eigenvectors (sorted by decreasing eigenvalue) become the principal
components. ``transform`` projects onto the first ``n_components``
components; ``inverse_transform`` reconstructs the data from them.
"""
from __future__ import annotations

import numpy as np


class PCA:
    """PCA via eigendecomposition of the covariance matrix.

    Parameters
    ----------
    n_components : int or None
        Number of components to keep. ``None`` keeps all of them.
    """

    def __init__(self, n_components: int | None = None):
        if n_components is not None:
            if not isinstance(n_components, int) or isinstance(n_components, bool):
                raise ValueError("n_components must be an integer or None")
            if n_components < 1:
                raise ValueError("n_components must be at least 1")
        self.n_components = n_components
        self.components_: np.ndarray | None = None   # (n_components, n_features)
        self.mean_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "PCA":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if x.shape[0] < 2:
            raise ValueError("need at least 2 samples to fit PCA")
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        n_samples, n_features = x.shape
        n_components = n_features if self.n_components is None else self.n_components
        if n_components > n_features:
            raise ValueError(
                "n_components (%d) cannot exceed the number of features (%d)"
                % (n_components, n_features)
            )
        self.mean_ = x.mean(axis=0)
        x_centered = x - self.mean_
        cov = (x_centered.T @ x_centered) / (n_samples - 1)
        # Covariance is symmetric: eigh returns real, sorted eigenvalues.
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        total_variance = eigenvalues.sum()
        if total_variance == 0:
            raise ValueError("cannot fit PCA on data with zero total variance")
        self.components_ = eigenvectors[:, :n_components].T
        self.explained_variance_ = eigenvalues[:n_components]
        self.explained_variance_ratio_ = eigenvalues[:n_components] / total_variance
        return self

    def _check_fitted(self):
        if self.components_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first")

    def _check_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("x must be a 1D or 2D array")
        if x.shape[1] != self.components_.shape[1]:
            raise ValueError(
                "x has %d features, expected %d"
                % (x.shape[1], self.components_.shape[1])
            )
        if not np.isfinite(x).all():
            raise ValueError("x must not contain NaN or infinite values")
        return x

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project data onto the principal components."""
        self._check_fitted()
        x = self._check_x(x)
        return (x - self.mean_) @ self.components_.T

    def inverse_transform(self, x_transformed: np.ndarray) -> np.ndarray:
        """Reconstruct data from its principal-component representation."""
        self._check_fitted()
        x_transformed = np.asarray(x_transformed, dtype=float)
        if x_transformed.ndim == 1:
            x_transformed = x_transformed.reshape(1, -1)
        if x_transformed.ndim != 2:
            raise ValueError("x_transformed must be a 1D or 2D array")
        if x_transformed.shape[1] != self.components_.shape[0]:
            raise ValueError(
                "x_transformed has %d components, expected %d"
                % (x_transformed.shape[1], self.components_.shape[0])
            )
        if not np.isfinite(x_transformed).all():
            raise ValueError("x_transformed must not contain NaN or infinite values")
        return x_transformed @ self.components_ + self.mean_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)
