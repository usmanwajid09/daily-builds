"""Kernel PCA, from scratch with numpy only.

Non-linear dimensionality reduction: the data is implicitly mapped to a
high-dimensional feature space via a kernel (RBF by default), the centred
kernel matrix is eigendecomposed, and the data is projected onto the
leading eigenvectors. The RBF kernel makes concentric rings linearly
separable in the projected space, where plain PCA fails.
"""
from __future__ import annotations

import numpy as np


class KernelPCA:
    """Kernel principal component analysis.

    Parameters
    ----------
    n_components : int
        Number of components to keep.
    kernel : str
        Kernel to use: ``"rbf"`` (Gaussian) or ``"linear"``.
    gamma : float or None
        RBF kernel coefficient. ``None`` uses ``1 / n_features``.
    """

    def __init__(self, *, n_components=2, kernel="rbf", gamma=None):
        if not isinstance(n_components, (int, np.integer)) or n_components < 1:
            raise ValueError("n_components must be a positive integer, "
                             f"got {n_components!r}")
        if kernel not in ("rbf", "linear"):
            raise ValueError(f'kernel must be "rbf" or "linear", got {kernel!r}')
        if gamma is not None and (not np.isfinite(gamma) or gamma <= 0):
            raise ValueError(f"gamma must be positive, got {gamma!r}")
        self.n_components = int(n_components)
        self.kernel = kernel
        self.gamma = gamma

    # -- kernels ---------------------------------------------------------
    def _rbf(self, x, z):
        gamma = self.gamma if self.gamma is not None \
            else 1.0 / x.shape[1]
        sq = (np.sum(x ** 2, axis=1)[:, None]
              + np.sum(z ** 2, axis=1)[None, :]
              - 2.0 * x @ z.T)
        return np.exp(-gamma * np.maximum(sq, 0.0))

    def _kernel_matrix(self, x, z):
        if self.kernel == "rbf":
            return self._rbf(x, z)
        return x @ z.T

    # -- fitting ---------------------------------------------------------
    def fit(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {x.shape}")
        if len(x) == 0:
            raise ValueError("Cannot fit on an empty dataset")
        if self.n_components > len(x):
            raise ValueError("n_components cannot exceed the number of "
                             f"samples ({len(x)}), got {self.n_components}")
        self.X_fit_ = x
        self.n_features_in_ = x.shape[1]
        self.n_samples_fit_ = len(x)

        k = self._kernel_matrix(x, x)
        n = len(x)
        self._col_mean_ = k.mean(axis=0)          # mean over rows, per column
        self._total_mean_ = k.mean()
        kc = k - self._col_mean_[None, :] - self._col_mean_[:, None] \
            + self._total_mean_

        # Eigendecomposition of the symmetric centred kernel matrix.
        eigvals, eigvecs = np.linalg.eigh(kc)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        # Numerical noise can give tiny negative eigenvalues; clip them.
        eigvals = np.clip(eigvals, 0.0, None)
        if eigvals[self.n_components - 1] <= 0:
            raise ValueError("Not enough positive eigenvalues to extract "
                             f"{self.n_components} components")
        self.lambdas_ = eigvals[:self.n_components]
        self.alphas_ = eigvecs[:, :self.n_components] \
            / np.sqrt(self.lambdas_)[None, :]
        positive = eigvals[eigvals > 0]
        self.explained_variance_ratio_ = \
            self.lambdas_ / positive.sum()
        return self

    def transform(self, x):
        """Project X onto the fitted kernel principal components."""
        self._check_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {x.shape}")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {x.shape[1]} features, expected "
                             f"{self.n_features_in_}")
        k = self._kernel_matrix(x, self.X_fit_)
        kc = k - self._col_mean_[None, :] \
            - k.mean(axis=1)[:, None] + self._total_mean_
        return kc @ self.alphas_

    def fit_transform(self, x):
        """Fit the model and return the projected training data."""
        return self.fit(x).transform(x)

    def _check_fitted(self):
        if not hasattr(self, "alphas_"):
            raise ValueError("Model is not fitted yet — call fit() first")
