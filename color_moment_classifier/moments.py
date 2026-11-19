"""Colour-moment image classifier from scratch.

Per-channel colour moments (mean / std / skewness per RGB channel)
give a 9-dim feature vector, classified with k-nearest neighbours,
numpy only. Synthetic colour scenes (sunset / forest / ocean) are
generated in code so the whole thing runs with no data files and no
external libraries.
"""

import numpy as np

__all__ = [
    "SCENE_KINDS",
    "ColorMomentClassifier",
    "color_moments",
    "extract_features",
    "generate_scene",
    "predict",
    "train_classifier",
]

SCENE_KINDS = ("sunset", "forest", "ocean")


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
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
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


def color_moments(image):
    """Return the 9-dim per-channel colour-moment vector of an image.

    For each RGB channel, in order: mean, standard deviation,
    skewness, where skewness = E[(x - mean)^3] / std^3 and is 0.0
    when the channel is constant (std == 0).

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3)
        RGB image, float in [0, 1] or uint8 in [0, 255].

    Returns
    -------
    np.ndarray, shape (9,)
        [R mean, R std, R skew, G mean, G std, G skew,
         B mean, B std, B skew]
    """
    img = _validate_image(image)
    feats = []
    for channel in range(3):
        pixels = img[:, :, channel].ravel()
        mean = float(pixels.mean())
        std = float(pixels.std())
        if std <= 1e-12:
            # Constant (or numerically constant) channel: no spread,
            # so skewness is defined as 0.0. The tolerance matters:
            # floating-point noise can leave std ~1e-16 instead of
            # exactly 0, which would otherwise make the skew ratio
            # blow up to +/-1.
            skew = 0.0
        else:
            skew = float(np.mean((pixels - mean) ** 3) / std ** 3)
        feats.extend([mean, std, skew])
    return np.array(feats, dtype=np.float64)


def extract_features(image):
    """Return the 9-dim colour-moment feature vector of an image.

    Alias/wrapper around :func:`color_moments`.
    """
    return color_moments(image)


def generate_scene(kind, size=32, seed=None):
    """Build a synthetic colour scene, shape (size, size, 3), float in [0, 1].

    Parameters
    ----------
    kind : str
        One of "sunset", "forest", "ocean".
    size : int
        Image height and width in pixels (must be >= 1).
    seed : int or None
        Random seed, so the same scene is reproducible (None = random).
    """
    if kind not in SCENE_KINDS:
        raise ValueError(
            f"unknown scene kind {kind!r}; expected one of {list(SCENE_KINDS)}"
        )
    if not isinstance(size, (int, np.integer)) or isinstance(size, bool):
        raise ValueError(f"size must be an integer >= 1, got {size!r}")
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.float64)

    if kind == "sunset":
        # Warm red/orange sky, with a bright horizontal band (the sun)
        # across the middle rows.
        img[:] = [0.90, 0.42, 0.12]
        band_h = max(1, size // 6)
        band_start = size // 2 - band_h // 2
        img[band_start:band_start + band_h] = [1.00, 0.78, 0.35]
    elif kind == "forest":
        # Green-dominant canopy, darker green ground at the bottom.
        img[:] = [0.15, 0.52, 0.16]
        img[3 * size // 4:] = [0.10, 0.35, 0.10]
    else:  # ocean
        # Blue-dominant water, slightly lighter blue sky at the top.
        img[:] = [0.08, 0.42, 0.85]
        img[: size // 4] = [0.30, 0.60, 0.95]

    # Light per-pixel noise so no two seeds are identical, clipped to [0, 1].
    img = np.clip(img + rng.normal(0.0, 0.03, img.shape), 0.0, 1.0)
    return img


class ColorMomentClassifier:
    """k-nearest-neighbour classifier on 9-dim colour-moment features.

    Training features are standardised (per-feature mean / std, with
    std == 0 guarded to 1.0) and neighbours are found by Euclidean
    distance in that standardised space.

    Parameters
    ----------
    k : int
        Number of neighbours to vote (must be >= 1 and <= number of
        training samples at fit time).
    """

    def __init__(self, k=3):
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
            raise ValueError(f"k must be an integer >= 1, got {k!r}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)
        self._X_std = None
        self._y = None
        self.classes_ = None
        self.feature_mean_ = None
        self.feature_std_ = None

    def fit(self, X, y):
        """Fit on colour-moment feature vectors with labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, 9)
            Colour-moment feature vectors.
        y : array-like, shape (n_samples,)
            Labels.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D feature vectors of shape (n_samples, 9), "
                f"got shape {X.shape}"
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one training sample")
        if X.shape[1] != 9:
            raise ValueError(
                f"X must have 9 features per sample, got {X.shape[1]}"
            )
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]} labels"
            )
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        if self.k > X.shape[0]:
            raise ValueError(
                f"k={self.k} is larger than the number of training samples "
                f"({X.shape[0]}); k must be <= number of training samples"
            )
        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0)
        # Guard constant features: dividing by 0 would give NaN/inf.
        self.feature_std_ = np.where(self.feature_std_ == 0.0, 1.0, self.feature_std_)
        self._X_std = (X - self.feature_mean_) / self.feature_std_
        self._y = y
        self.classes_ = np.unique(y)
        return self

    def _check_fitted(self):
        if self._X_std is None:
            raise ValueError(
                "classifier is not fitted yet; call fit() before predict()"
            )

    def _features(self, X):
        """Return X as a 2D array of standardised feature vectors."""
        arr = np.asarray(X)
        if arr.ndim == 3:  # single raw image
            return self._standardise(extract_features(arr)[None, :])
        if arr.ndim == 4:  # batch of raw images
            feats = np.array([extract_features(img) for img in arr])
            return self._standardise(feats)
        if arr.ndim == 1:  # single feature vector
            if arr.shape[0] != 9:
                raise ValueError(
                    f"feature vector must have 9 values, got {arr.shape[0]}"
                )
            return self._standardise(arr[None, :])
        if arr.ndim != 2:
            raise ValueError(
                f"X must be a single image (3D), feature vectors (2D) or "
                f"a batch of images (4D), got shape {arr.shape}"
            )
        if arr.shape[1] != 9:
            raise ValueError(
                f"X must have 9 features per sample, got {arr.shape[1]}"
            )
        return self._standardise(arr.astype(np.float64))

    def _standardise(self, feats):
        return (feats - self.feature_mean_) / self.feature_std_

    def predict(self, X):
        """Return the predicted label for each sample in X.

        Majority vote among the k nearest training samples; ties are
        broken deterministically by the smallest label order.
        """
        self._check_fitted()
        feats = self._features(X)
        diff = feats[:, None, :] - self._X_std[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        idx = np.argsort(dist, axis=1)[:, : self.k]
        preds = []
        for row in idx:
            # classes_ is sorted, so the first max-count class wins ties.
            best_label, best_count = None, -1
            for label in self.classes_:
                count = int((self._y[row] == label).sum())
                if count > best_count:
                    best_label, best_count = label, count
            preds.append(best_label)
        return np.array(preds)

    def score(self, X, y):
        """Return the fraction of correct predictions, as a float in [0, 1]."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


def train_classifier(images, labels, k=3):
    """Train and return a :class:`ColorMomentClassifier`.

    Parameters
    ----------
    images : list of np.ndarray, or array of shape (n_samples, 9)
        Raw RGB images (converted with :func:`extract_features`) or
        ready-made 9-dim feature vectors.
    labels : array-like, shape (n_samples,)
        Labels.
    k : int
        Number of neighbours to vote.
    """
    arr = np.asarray(images)
    if arr.ndim == 2 and arr.shape[1] == 9:
        X = arr.astype(np.float64)
    else:
        X = np.array([extract_features(img) for img in images])
    return ColorMomentClassifier(k=k).fit(X, np.asarray(labels))


def predict(images, classifier):
    """Predict labels for images (or feature vectors) with a fitted classifier.

    Parameters
    ----------
    images : list of np.ndarray, or array of shape (n_samples, 9)
        Raw RGB images or ready-made 9-dim feature vectors.
    classifier : ColorMomentClassifier
        A fitted classifier.
    """
    if not isinstance(classifier, ColorMomentClassifier):
        raise ValueError(
            f"classifier must be a ColorMomentClassifier, got "
            f"{type(classifier).__name__}"
        )
    return classifier.predict(images)
