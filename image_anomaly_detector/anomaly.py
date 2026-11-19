"""Image anomaly detection via PCA reconstruction error (numpy only).

Normal image patches lie close to a low-dimensional linear subspace:
a smooth gradient patch is well described by a few principal
components. A patch containing something the training set never saw —
a bright block, a scratch, a foreign object — reconstructs badly, so
its mean squared reconstruction error is far larger than a normal
patch's. :class:`PCAAnomalyDetector` learns that subspace from normal
patches with an SVD, and flags patches whose reconstruction error
exceeds a threshold (the 95th percentile of the training errors by
default).
"""

import numpy as np

__all__ = [
    "PCAAnomalyDetector",
    "detect_anomalies",
    "extract_patches",
    "reconstruction_error",
    "train_detector",
]


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive int, got {value}")
    return value


def _validate_image(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(
            f"image must be a 2D grayscale array, got shape {image.shape}"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    return image


def extract_patches(image, patch_size=8, stride=8):
    """Split a grayscale ``image`` into flattened patches.

    Returns an array of shape ``(n_patches, patch_size * patch_size)``.
    Patches are taken on a grid starting at the top-left corner; when
    the image size is not divisible by the stride, the leftover rows
    and columns at the bottom/right edges are dropped.
    """
    patch_size = _positive_int("patch_size", patch_size)
    stride = _positive_int("stride", stride)
    image = _validate_image(image)
    height, width = image.shape
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"image of shape {image.shape} is smaller than "
            f"patch_size {patch_size}"
        )
    n_rows = (height - patch_size) // stride + 1
    n_cols = (width - patch_size) // stride + 1
    patches = np.empty((n_rows * n_cols, patch_size * patch_size))
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            top, left = r * stride, c * stride
            patches[idx] = image[
                top:top + patch_size, left:left + patch_size
            ].ravel()
            idx += 1
    return patches


class PCAAnomalyDetector:
    """PCA model of normal image patches, scored by reconstruction error.

    ``n_components`` is either an int (number of principal components
    to keep, 1..n_features) or a float in (0, 1] (the fraction of total
    variance the kept components must explain). After :meth:`fit`,
    ``threshold_`` holds the 95th percentile of the training
    reconstruction errors; :meth:`predict` flags patches above it.
    """

    def __init__(self, n_components=0.95):
        if isinstance(n_components, bool):
            raise ValueError(
                f"n_components must be an int >= 1 or a float in (0, 1], "
                f"got {n_components!r}"
            )
        if isinstance(n_components, (int, np.integer)):
            if int(n_components) < 1:
                raise ValueError(
                    f"n_components must be >= 1, got {n_components}"
                )
        elif isinstance(n_components, (float, np.floating)):
            if not 0.0 < float(n_components) <= 1.0:
                raise ValueError(
                    "n_components as a variance ratio must lie in (0, 1], "
                    f"got {n_components}"
                )
        else:
            raise ValueError(
                "n_components must be an int >= 1 or a float in (0, 1], "
                f"got {n_components!r}"
            )
        self.n_components = n_components

    def _check_fitted(self):
        if not hasattr(self, "components_"):
            raise ValueError(
                "detector has not been fitted yet; call fit() first"
            )

    def _check_patches(self, patches):
        patches = np.asarray(patches, dtype=np.float64)
        if patches.ndim != 2:
            raise ValueError(
                f"patches must be a 2D array, got shape {patches.shape}"
            )
        if patches.shape[0] == 0:
            raise ValueError("patches must not be empty")
        if patches.shape[1] == 0:
            raise ValueError("patches must have at least one feature")
        if not np.isfinite(patches).all():
            raise ValueError("patches must contain only finite values")
        return patches

    def fit(self, patches):
        """Learn the normal-patch subspace from ``patches``.

        Sets ``mean_``, ``components_``, ``explained_variance_ratio_``,
        ``n_components_`` and ``threshold_`` (95th percentile of the
        training reconstruction errors). Returns ``self``.
        """
        patches = self._check_patches(patches)
        n_samples, n_features = patches.shape
        if n_samples < 2:
            raise ValueError(
                "need at least 2 patches to fit PCA, "
                f"got {n_samples}"
            )
        if isinstance(self.n_components, (int, np.integer)):
            k = int(self.n_components)
            if k > n_features:
                raise ValueError(
                    f"n_components {k} exceeds n_features {n_features}"
                )
        self.mean_ = patches.mean(axis=0)
        centred = patches - self.mean_
        _, singular, vt = np.linalg.svd(centred, full_matrices=False)
        variances = singular ** 2 / (n_samples - 1)
        total_variance = variances.sum()
        if total_variance == 0.0:
            # All patches identical: any single component reconstructs
            # them exactly; there is no variance to explain.
            k = 1
            explained = np.zeros(1)
        else:
            ratios = variances / total_variance
            if isinstance(self.n_components, (float, np.floating)):
                ratio = float(self.n_components)
                k = int(np.searchsorted(np.cumsum(ratios), ratio)) + 1
                k = min(k, n_features)
            explained = ratios[:k]
        self.n_components_ = k
        self.components_ = vt[:k]
        self.explained_variance_ratio_ = explained
        self._train_errors_ = self.reconstruction_error(patches)
        self.threshold_ = float(np.percentile(self._train_errors_, 95))
        return self

    def transform(self, patches):
        """Project ``patches`` onto the kept principal components."""
        self._check_fitted()
        patches = self._check_patches(patches)
        return (patches - self.mean_) @ self.components_.T

    def reconstruct(self, patches):
        """Reconstruct ``patches`` from their PCA projections."""
        self._check_fitted()
        return self.transform(patches) @ self.components_ + self.mean_

    def reconstruction_error(self, patches):
        """Per-patch mean squared reconstruction error."""
        self._check_fitted()
        patches = self._check_patches(patches)
        return np.mean((patches - self.reconstruct(patches)) ** 2, axis=1)

    def score_samples(self, patches):
        """Anomaly score per patch: higher means more normal.

        Defined as the negative reconstruction error, following the
        scikit-learn convention.
        """
        return -self.reconstruction_error(patches)

    def set_threshold(self, percentile=95.0, value=None):
        """Change ``threshold_`` after fitting.

        Either pass ``value`` to set the threshold directly, or pass a
        ``percentile`` (0..100) of the training reconstruction errors.
        Returns ``self``.
        """
        self._check_fitted()
        if value is not None:
            value = float(value)
            if value < 0:
                raise ValueError(
                    f"threshold value must be >= 0, got {value}"
                )
            self.threshold_ = value
        else:
            percentile = float(percentile)
            if not 0.0 <= percentile <= 100.0:
                raise ValueError(
                    f"percentile must lie in [0, 100], got {percentile}"
                )
            self.threshold_ = float(
                np.percentile(self._train_errors_, percentile)
            )
        return self

    def predict(self, patches):
        """Return a boolean array: ``True`` where a patch is anomalous.

        A patch is anomalous when its reconstruction error exceeds
        ``threshold_``.
        """
        return self.reconstruction_error(patches) > self.threshold_


def reconstruction_error(patches, detector):
    """Per-patch mean squared reconstruction error under ``detector``.

    Convenience wrapper around
    :meth:`PCAAnomalyDetector.reconstruction_error`; ``detector`` must
    already be fitted.
    """
    if not isinstance(detector, PCAAnomalyDetector):
        raise ValueError(
            f"detector must be a PCAAnomalyDetector, got {type(detector)}"
        )
    return detector.reconstruction_error(patches)


def train_detector(images, patch_size=8, stride=8, n_components=0.95):
    """Fit a :class:`PCAAnomalyDetector` on patches from normal images.

    ``images`` is a list of 2D grayscale images; patches are extracted
    from each with :func:`extract_patches` and pooled for fitting.
    """
    if not images:
        raise ValueError("images must not be empty")
    patch_size = _positive_int("patch_size", patch_size)
    stride = _positive_int("stride", stride)
    all_patches = [extract_patches(img, patch_size, stride) for img in images]
    detector = PCAAnomalyDetector(n_components=n_components)
    return detector.fit(np.concatenate(all_patches, axis=0))


def detect_anomalies(image, detector, patch_size=8, stride=8):
    """Flag anomalous patches in ``image`` under a fitted ``detector``.

    Returns ``(mask, errors)``: ``mask`` is a 2D boolean array over the
    patch grid (``True`` = anomalous patch) and ``errors`` is the 2D
    array of per-patch reconstruction errors on the same grid.
    """
    if not isinstance(detector, PCAAnomalyDetector):
        raise ValueError(
            f"detector must be a PCAAnomalyDetector, got {type(detector)}"
        )
    patch_size = _positive_int("patch_size", patch_size)
    stride = _positive_int("stride", stride)
    image = _validate_image(image)
    height, width = image.shape
    n_rows = (height - patch_size) // stride + 1
    n_cols = (width - patch_size) // stride + 1
    if n_rows < 1 or n_cols < 1:
        raise ValueError(
            f"image of shape {image.shape} is smaller than "
            f"patch_size {patch_size}"
        )
    patches = extract_patches(image, patch_size, stride)
    errors = detector.reconstruction_error(patches).reshape(n_rows, n_cols)
    return errors > detector.threshold_, errors
