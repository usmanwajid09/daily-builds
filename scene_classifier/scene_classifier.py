"""Simple scene classifier from scratch.

Colour-histogram features + k-nearest neighbours, numpy only.
Synthetic scenes (beach / forest / city) are generated in code so the
whole thing runs with no data files and no external libraries.
"""

import numpy as np

__all__ = [
    "extract_features",
    "generate_scene",
    "SceneClassifier",
    "train_classifier",
    "predict",
    "SCENE_KINDS",
]

SCENE_KINDS = ("beach", "forest", "city")


def _validate_image(image):
    """Return the image as a float array in [0, 1], or raise ValueError."""
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"image must be a numpy array, got {type(image).__name__}"
        )
    if image.ndim != 3:
        raise ValueError(
            f"image must be 3D (height, width, 3), got shape {image.shape}"
        )
    if image.shape[2] != 3:
        raise ValueError(
            f"image must have 3 colour channels (RGB), got shape {image.shape}"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.dtype == np.uint8:
        return image.astype(np.float64) / 255.0
    arr = np.asarray(image, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("image values must be finite (no NaN or inf)")
    if arr.min() < 0.0 or arr.max() > 1.0:
        raise ValueError(
            f"float image values must be in [0, 1], got range "
            f"[{arr.min():.3f}, {arr.max():.3f}]"
        )
    return arr


def extract_features(image, bins=8):
    """Return a normalised RGB histogram feature vector.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3)
        RGB image, float in [0, 1] or uint8 in [0, 255].
    bins : int
        Histogram bins per colour channel (must be >= 1).

    Returns
    -------
    np.ndarray, shape (3 * bins,)
        Concatenated per-channel histograms, normalised to sum to 1.
    """
    if not isinstance(bins, (int, np.integer)) or isinstance(bins, bool):
        raise ValueError(f"bins must be an integer >= 1, got {bins!r}")
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    img = _validate_image(image)
    feats = []
    for channel in range(3):
        hist, _ = np.histogram(img[:, :, channel], bins=bins, range=(0.0, 1.0))
        feats.append(hist.astype(np.float64))
    vec = np.concatenate(feats)
    total = vec.sum()
    if total == 0:
        raise ValueError("image has no pixels to histogram")
    return vec / total


def generate_scene(kind, seed=0, size=64):
    """Build a synthetic scene image, shape (size, size, 3), float in [0, 1].

    Parameters
    ----------
    kind : str
        One of "beach", "forest", "city".
    seed : int
        Random seed, so the same scene is reproducible.
    size : int
        Image height and width in pixels (must be >= 8).
    """
    if kind not in SCENE_KINDS:
        raise ValueError(
            f"unknown scene kind {kind!r}; expected one of {list(SCENE_KINDS)}"
        )
    if not isinstance(size, (int, np.integer)) or isinstance(size, bool):
        raise ValueError(f"size must be an integer >= 8, got {size!r}")
    if size < 8:
        raise ValueError(f"size must be >= 8, got {size}")
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.float64)
    horizon = size // 2

    if kind == "beach":
        # Blue sky on top, warm sand below, a darker sea band at the horizon.
        img[:horizon] = [0.45, 0.75, 0.95]          # sky
        img[horizon:horizon + size // 8] = [0.10, 0.45, 0.75]  # sea
        img[horizon + size // 8:] = [0.93, 0.82, 0.60]        # sand
    elif kind == "forest":
        # Green canopy above, brown trunk in the middle, dark ground below.
        img[: 3 * size // 4] = [0.15, 0.45, 0.15]   # canopy
        trunk_w = max(2, size // 8)
        c0 = size // 2 - trunk_w // 2
        img[size // 4:, c0:c0 + trunk_w] = [0.45, 0.28, 0.12]  # trunk
        img[3 * size // 4:] = [0.25, 0.20, 0.12]    # ground
    else:  # city
        # Pale sky, grey buildings of varying heights, dark road at the bottom.
        img[:] = [0.75, 0.80, 0.85]                 # sky
        x = 0
        while x < size:
            width = int(rng.integers(size // 8, size // 4))
            height = int(rng.integers(size // 4, 3 * size // 4))
            shade = float(rng.uniform(0.30, 0.60))
            img[size - height:size - size // 8, x:min(x + width, size)] = [
                shade, shade, shade + 0.05,
            ]
            x += width
        img[size - size // 8:] = [0.15, 0.15, 0.15]  # road

    # Light per-pixel noise so no two seeds are identical, clipped to [0, 1].
    img = np.clip(img + rng.normal(0.0, 0.03, img.shape), 0.0, 1.0)
    return img


class SceneClassifier:
    """k-nearest-neighbour scene classifier on RGB histogram features.

    Parameters
    ----------
    k : int
        Number of neighbours to vote (must be >= 1 and <= number of
        training samples at fit time).
    bins : int
        Histogram bins per channel used when fitting raw images.
    """

    def __init__(self, k=3, bins=8):
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
            raise ValueError(f"k must be an integer >= 1, got {k!r}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not isinstance(bins, (int, np.integer)) or isinstance(bins, bool):
            raise ValueError(f"bins must be an integer >= 1, got {bins!r}")
        if bins < 1:
            raise ValueError(f"bins must be >= 1, got {bins}")
        self.k = int(k)
        self.bins = int(bins)
        self._X = None
        self._y = None
        self.classes_ = None

    def fit(self, X, y):
        """Fit on feature vectors (or raw images) with labels.

        X may be a 2D array of feature vectors, shape (n, 3*bins), or a
        4D array of raw images, shape (n, H, W, 3) — raw images are
        converted with :func:`extract_features` first.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 4:
            X = np.array([extract_features(img, bins=self.bins) for img in X])
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D feature vectors or 4D images, got shape {X.shape}"
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one training sample")
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]} labels"
            )
        if self.k > X.shape[0]:
            raise ValueError(
                f"k={self.k} is larger than the number of training samples "
                f"({X.shape[0]}); k must be <= number of training samples"
            )
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        self._X = X.astype(np.float64)
        self._y = y
        self.classes_ = np.unique(y)
        return self

    def _check_fitted(self):
        if self._X is None:
            raise ValueError(
                "classifier is not fitted yet; call fit() before predict()"
            )

    def _features(self, X):
        X = np.asarray(X)
        if X.ndim == 3:  # single raw image
            return extract_features(X, bins=self.bins)[None, :]
        if X.ndim == 4:  # batch of raw images
            return np.array(
                [extract_features(img, bins=self.bins) for img in X]
            )
        if X.ndim != 2:
            raise ValueError(
                f"X must be a single image (3D), feature vectors (2D) or "
                f"a batch of images (4D), got shape {X.shape}"
            )
        return X.astype(np.float64)

    def _nearest(self, X):
        self._check_fitted()
        feats = self._features(X)
        # Euclidean distance to every training vector.
        diff = feats[:, None, :] - self._X[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        return np.argsort(dist, axis=1)[:, : self.k]

    def predict(self, X):
        """Return the predicted label for each sample in X."""
        idx = self._nearest(X)
        preds = []
        for row in idx:
            labels, counts = np.unique(self._y[row], return_counts=True)
            preds.append(labels[np.argmax(counts)])
        return np.array(preds)

    def predict_proba(self, X):
        """Return vote fractions per class, shape (n_samples, n_classes).

        Columns follow ``self.classes_``; each row sums to 1.
        """
        idx = self._nearest(X)
        proba = np.zeros((idx.shape[0], len(self.classes_)))
        for i, row in enumerate(idx):
            labels, counts = np.unique(self._y[row], return_counts=True)
            for label, count in zip(labels, counts):
                j = int(np.where(self.classes_ == label)[0][0])
                proba[i, j] = count / self.k
        return proba

    def score(self, X, y):
        """Return the fraction of correct predictions."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


def train_classifier(X, y, k=3, bins=8):
    """Train and return a :class:`SceneClassifier` on X, y."""
    return SceneClassifier(k=k, bins=bins).fit(X, y)


def predict(classifier, X):
    """Predict labels for X with a fitted classifier."""
    if not isinstance(classifier, SceneClassifier):
        raise ValueError(
            f"classifier must be a SceneClassifier, got "
            f"{type(classifier).__name__}"
        )
    return classifier.predict(X)
