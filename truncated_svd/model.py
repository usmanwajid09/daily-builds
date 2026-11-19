"""Truncated singular value decomposition, from scratch with numpy only.

Truncated SVD keeps only the top ``n_components`` singular values and
vectors, which is the standard way to do low-rank approximation and
dimensionality reduction on data that should not be centred (unlike
PCA, e.g. sparse term-document matrices).

It is computed via eigendecomposition of the smaller Gram matrix:
``X.T @ X`` when there are fewer features than samples, otherwise
``X @ X.T``. The eigenvectors give the right singular vectors, the
eigenvalues give the squared singular values, and the left singular
vectors follow from ``U = X @ V / sigma``.
"""

from __future__ import annotations

import numpy as np


class TruncatedSVD:
    """Truncated SVD via eigendecomposition of the Gram matrix.

    Parameters
    ----------
    n_components : int
        Number of singular values/vectors to keep. Must satisfy
        ``1 <= n_components <= min(n_samples, n_features)``.
    """

    def __init__(self, n_components: int = 2):
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError(
                f"n_components must be a positive integer, got {n_components}"
            )
        self.n_components = n_components
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_features_in_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must not be empty")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        n_samples, n_features = X.shape
        if self.n_components > min(n_samples, n_features):
            raise ValueError(
                f"n_components={self.n_components} must be <= "
                f"min(n_samples, n_features)={min(n_samples, n_features)}"
            )
        self.n_features_in_ = n_features

        # Eigendecompose the smaller Gram matrix.
        if n_features <= n_samples:
            gram = X.T @ X
            eigvals, eigvecs = np.linalg.eigh(gram)
            order = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(eigvals[order], 0.0)
            eigvecs = eigvecs[:, order]
            V = eigvecs[:, : self.n_components]
            singular_values = np.sqrt(eigvals[: self.n_components])
        else:
            gram = X @ X.T
            eigvals, eigvecs = np.linalg.eigh(gram)
            order = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(eigvals[order], 0.0)
            eigvecs = eigvecs[:, order]
            singular_values = np.sqrt(eigvals[: self.n_components])
            U = eigvecs[:, : self.n_components]
            # V = X.T @ U / sigma (guard against zero singular values).
            with np.errstate(divide="ignore", invalid="ignore"):
                V = np.where(
                    singular_values > 0,
                    (X.T @ U) / np.where(singular_values > 0, singular_values, 1.0),
                    0.0,
                )

        self.components_ = V.T
        self.singular_values_ = singular_values
        # Explained variance: eigenvalues / (n_samples - 1), matching PCA.
        if n_samples > 1:
            total_var = float(np.sum(eigvals) / (n_samples - 1))
            self.explained_variance_ = eigvals[: self.n_components] / (n_samples - 1)
            self.explained_variance_ratio_ = (
                self.explained_variance_ / total_var if total_var > 0 else np.zeros(self.n_components)
            )
        else:
            self.explained_variance_ = eigvals[: self.n_components]
            self.explained_variance_ratio_ = np.zeros(self.n_components)
        return self

    def transform(self, X):
        if self.components_ is None:
            raise ValueError("TruncatedSVD is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected X with {self.n_features_in_} features, got shape {X.shape}"
            )
        return X @ self.components_.T

    def inverse_transform(self, X_transformed):
        if self.components_ is None:
            raise ValueError("TruncatedSVD is not fitted yet")
        X_transformed = np.asarray(X_transformed, dtype=float)
        if X_transformed.ndim != 2 or X_transformed.shape[1] != self.n_components:
            raise ValueError(
                f"Expected input with {self.n_components} components, "
                f"got shape {X_transformed.shape}"
            )
        return X_transformed @ self.components_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def reconstruction_error(self, X):
        """Mean squared reconstruction error using the kept components."""
        X = np.asarray(X, dtype=float)
        recon = self.inverse_transform(self.transform(X))
        return float(np.mean((X - recon) ** 2))
