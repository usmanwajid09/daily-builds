"""LBP texture classification from scratch (numpy only).

Textures are described by Local Binary Pattern (LBP) histograms: each
pixel is encoded by comparing its 8 neighbours to the centre pixel, and
the 256-bin histogram of those codes is the texture's fingerprint.
A small kNN classifier on top of the histograms separates synthetic
textures (stripes, checkerboard, noise, dots) with high accuracy.
"""

import numpy as np

CLASSES = ("stripes", "checkerboard", "noise", "dots")


def _validate_image(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D grayscale, got shape {image.shape}")
    if image.shape[0] < 3 or image.shape[1] < 3:
        raise ValueError(
            f"image must be at least 3x3 for LBP, got {image.shape}"
        )
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    return image


def _add_gaussian_noise(image, sigma, rng):
    if sigma <= 0:
        return image
    noisy = image + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0.0, 255.0)


def make_stripes(size=64, period=8, orientation="horizontal", rng=None,
                 noise_sigma=5.0):
    """Return a striped texture (values 0/255) with optional noise.

    Raises:
        ValueError: if ``orientation`` is not "horizontal"/"vertical"
            or ``period`` is smaller than 2.
    """
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(
            f"orientation must be 'horizontal' or 'vertical', got {orientation!r}"
        )
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    rng = np.random.default_rng(rng)
    coords = np.arange(size)
    bands = (coords // period) % 2
    if orientation == "horizontal":
        image = np.tile((bands * 255.0)[:, None], (1, size))
    else:
        image = np.tile((bands * 255.0)[None, :], (size, 1))
    return _add_gaussian_noise(image, noise_sigma, rng)


def make_checkerboard(size=64, block=8, rng=None, noise_sigma=5.0):
    """Return a checkerboard texture (values 0/255) with optional noise."""
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    rng = np.random.default_rng(rng)
    rows, cols = np.indices((size, size))
    image = (((rows // block) + (cols // block)) % 2) * 255.0
    return _add_gaussian_noise(image, noise_sigma, rng)


def make_noise(size=64, rng=None):
    """Return a uniform random-noise texture (values 0-255)."""
    rng = np.random.default_rng(rng)
    return rng.random((size, size)) * 255.0


def make_dots(size=64, spacing=12, radius=3, rng=None, noise_sigma=5.0):
    """Return a dotted texture: bright discs on a dark grid, with noise."""
    if spacing < 2 * radius + 1:
        raise ValueError(
            f"spacing {spacing} too small for radius {radius}"
        )
    rng = np.random.default_rng(rng)
    image = np.zeros((size, size), dtype=np.float64)
    rows, cols = np.indices((size, size))
    for cy in range(spacing // 2, size, spacing):
        for cx in range(spacing // 2, size, spacing):
            jitter_y = int(rng.integers(-2, 3))
            jitter_x = int(rng.integers(-2, 3))
            dist2 = (rows - (cy + jitter_y)) ** 2 + (cols - (cx + jitter_x)) ** 2
            image[dist2 <= radius * radius] = 255.0
    return _add_gaussian_noise(image, noise_sigma, rng)


def make_texture(name, size=64, rng=None):
    """Generate one texture of class ``name`` (see :data:`CLASSES`)."""
    if name == "stripes":
        orientation = "horizontal" if np.random.default_rng(rng).random() < 0.5 else "vertical"
        return make_stripes(size=size, orientation=orientation, rng=rng)
    if name == "checkerboard":
        return make_checkerboard(size=size, rng=rng)
    if name == "noise":
        return make_noise(size=size, rng=rng)
    if name == "dots":
        return make_dots(size=size, rng=rng)
    raise ValueError(f"unknown texture class {name!r}, expected one of {CLASSES}")


def lbp_histogram(image, n_bins=256):
    """Return the normalised LBP histogram of a grayscale ``image``.

    Each interior pixel is coded by comparing its 8 neighbours to the
    centre pixel (neighbour >= centre sets the bit). The returned
    histogram has ``n_bins`` entries and sums to 1.
    """
    image = _validate_image(image)
    centre = image[1:-1, 1:-1]
    code = np.zeros(centre.shape, dtype=np.int32)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dr, dc) in enumerate(offsets):
        neighbour = image[1 + dr:image.shape[0] - 1 + dr,
                          1 + dc:image.shape[1] - 1 + dc]
        code += ((neighbour >= centre).astype(np.int32) << bit)
    hist = np.bincount(code.ravel(), minlength=n_bins).astype(np.float64)
    hist /= hist.sum()
    return hist


def extract_features(images):
    """Return an ``(n, 256)`` array of LBP histograms for ``images``."""
    images = list(images)
    if not images:
        raise ValueError("need at least one image to extract features from")
    return np.stack([lbp_histogram(img) for img in images])


class TextureKNN:
    """k-Nearest Neighbour classifier over LBP histograms."""

    def __init__(self, k=3):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._train_X = None
        self._train_y = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"got {len(X)} samples but {len(y)} labels")
        if len(X) < self.k:
            raise ValueError(
                f"need at least k={self.k} training samples, got {len(X)}"
            )
        self._train_X = X
        self._train_y = y
        return self

    def predict(self, X):
        if self._train_X is None:
            raise ValueError("call fit() before predict()")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self._train_X.shape[1]:
            raise ValueError(
                f"expected {self._train_X.shape[1]} features, "
                f"got {X.shape[1]}"
            )
        preds = []
        for x in X:
            dists = np.linalg.norm(self._train_X - x, axis=1)
            nearest = np.argsort(dists)[: self.k]
            labels, counts = np.unique(
                self._train_y[nearest], return_counts=True
            )
            preds.append(labels[np.argmax(counts)])
        return np.array(preds)


def accuracy(classifier, images, labels):
    """Fraction of ``images`` the fitted ``classifier`` gets right."""
    preds = classifier.predict(extract_features(images))
    labels = np.asarray(labels)
    if len(preds) != len(labels):
        raise ValueError("images/labels length mismatch")
    return float(np.mean(preds == labels))
