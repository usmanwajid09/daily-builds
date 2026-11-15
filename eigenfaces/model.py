"""Eigenfaces face recognition from scratch (PCA + nearest neighbour).

Faces are flattened to vectors, the mean face is subtracted, and PCA
(eigendecomposition of the covariance matrix) gives the eigenfaces.
A new face is recognised by projecting it into eigenface space and
finding the nearest training face there. Reconstruction from a small
number of eigenfaces shows how much information the components keep.
Only numpy is used.
"""

import numpy as np


class Eigenfaces:
    """PCA-based face recogniser.

    Parameters
    ----------
    n_components : int
        Number of eigenfaces to keep. Must be >= 1 and at most
        ``min(n_samples - 1, n_pixels)``.
    """

    def __init__(self, n_components=10):
        if not isinstance(n_components, (int, np.integer)) or isinstance(n_components, bool):
            raise ValueError("n_components must be a positive integer")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        self.n_components = int(n_components)
        self.mean_face_ = None
        self.components_ = None  # shape (n_components, n_pixels)
        self.explained_variance_ratio_ = None
        self._train_projections = None
        self._train_labels = None
        self._image_shape = None

    def _as_matrix(self, images):
        arr = np.asarray(images, dtype=float)
        if arr.ndim == 2:  # single image
            arr = arr[np.newaxis, :, :]
        if arr.ndim != 3:
            raise ValueError("images must be a 2D array (one image) or 3D array (n, height, width)")
        if arr.shape[0] < 1:
            raise ValueError("at least one image is required")
        if arr.shape[1] < 2 or arr.shape[2] < 2:
            raise ValueError("images must be at least 2x2 pixels")
        if not np.isfinite(arr).all():
            raise ValueError("images must contain only finite values")
        self._check_shape(arr)
        return arr.reshape(arr.shape[0], -1)

    def _check_shape(self, arr):
        if self._image_shape is not None and arr.shape[1:] != self._image_shape:
            raise ValueError(
                f"all images must have shape {self._image_shape}, got {arr.shape[1:]}"
            )

    def fit(self, images, labels=None):
        """Fit eigenfaces to training images.

        Parameters
        ----------
        images : array-like, shape (n_samples, height, width)
        labels : array-like of length n_samples, optional
            Identity label per image; defaults to 0..n_samples-1.
        """
        X = self._as_matrix(images)
        n_samples, n_pixels = X.shape
        if n_samples < 2:
            raise ValueError("at least 2 training images are required")
        max_components = min(n_samples - 1, n_pixels)
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} is too large for "
                f"{n_samples} images of {n_pixels} pixels (max {max_components})"
            )
        if labels is None:
            labels = np.arange(n_samples)
        labels = np.asarray(labels)
        if labels.shape[0] != n_samples:
            raise ValueError("labels must have one entry per image")

        self._image_shape = np.asarray(images).shape[1:]
        self.mean_face_ = X.mean(axis=0)
        Xc = X - self.mean_face_
        # Eigendecomposition of the small (n x n) Gram matrix, then lift
        # the eigenvectors back to pixel space (the standard trick).
        gram = Xc @ Xc.T / n_samples
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        components = eigvecs.T @ Xc  # (n_samples, n_pixels)
        norms = np.linalg.norm(components, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        components = components / norms
        self.components_ = components[: self.n_components]
        total = eigvals.sum()
        self.explained_variance_ratio_ = (
            eigvals[: self.n_components] / total if total > 0 else np.zeros(self.n_components)
        )
        self._train_projections = self.transform(images)
        self._train_labels = labels
        return self

    def _require_fit(self):
        if self.components_ is None:
            raise ValueError("Eigenfaces is not fitted yet; call fit() first")

    def transform(self, images):
        """Project images into eigenface space. Returns (n, n_components)."""
        self._require_fit()
        X = self._as_matrix(images)
        return (X - self.mean_face_) @ self.components_.T

    def reconstruct(self, images):
        """Reconstruct images from their eigenface projections.

        Returns an array of the same shape as the input images.
        """
        self._require_fit()
        X = self._as_matrix(images)
        projections = (X - self.mean_face_) @ self.components_.T
        rebuilt = projections @ self.components_ + self.mean_face_
        return rebuilt.reshape((-1,) + self._image_shape)

    def reconstruction_error(self, images):
        """Mean squared reconstruction error per image."""
        self._require_fit()
        X = self._as_matrix(images)
        rebuilt = self.reconstruct(images).reshape(X.shape)
        return np.mean((X - rebuilt) ** 2, axis=1)

    def predict(self, images):
        """Recognise images: label of the nearest training face in eigenface space."""
        self._require_fit()
        projections = self.transform(images)
        out = []
        for p in projections:
            dists = np.linalg.norm(self._train_projections - p, axis=1)
            out.append(self._train_labels[int(np.argmin(dists))])
        return np.asarray(out)
