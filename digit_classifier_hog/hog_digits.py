"""HOG features + kNN classifier for synthetic 8x8 digit images.

Each digit 0-9 is rendered from the hand-defined 8x8 binary templates
in :mod:`digit_recognizer.digits`, then corrupted with random pixel
flips. This module extracts a small Histogram of Oriented Gradients
(HOG) descriptor from each image -- cell orientation histograms with
L2-normalised blocks -- and classifies it with a k-nearest-neighbours
classifier (k=3 by default, Euclidean distance).

A raw-pixel nearest-template baseline
(:func:`template_match_accuracy`) is provided so the HOG + kNN
recogniser can be compared against plain template matching on the
same noisy images.

The HOG extractor here is written for 8x8 images from scratch; it
does not import anything from the ``hog_features`` folder.
"""

import numpy as np

from digit_recognizer.digits import TEMPLATE_SHAPE, add_noise, recognize, template

IMAGE_SHAPE = TEMPLATE_SHAPE
DEFAULT_CELL_SIZE = 4
DEFAULT_N_BINS = 9
DEFAULT_K = 3


def _validate_image(image):
    """Validate an 8x8 image and return it as a float64 array."""
    try:
        image = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"image must be an 8x8 numeric array, got {image!r}") from exc
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got {image.ndim}D with shape {image.shape}")
    if image.shape != IMAGE_SHAPE:
        raise ValueError(f"image must have shape {IMAGE_SHAPE}, got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("image values must lie in [0, 1]")
    return image


def _validate_cell_size(cell_size):
    if not isinstance(cell_size, (int, np.integer)) or isinstance(cell_size, bool):
        raise ValueError(f"cell_size must be an integer, got {cell_size!r}")
    cell_size = int(cell_size)
    if cell_size < 1 or cell_size > IMAGE_SHAPE[0]:
        raise ValueError(f"cell_size must be in [1, {IMAGE_SHAPE[0]}], got {cell_size}")
    if IMAGE_SHAPE[0] % cell_size != 0 or IMAGE_SHAPE[1] % cell_size != 0:
        raise ValueError(
            f"cell_size must divide the image shape {IMAGE_SHAPE}, got {cell_size}"
        )
    return cell_size


def _validate_n_bins(n_bins):
    if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool):
        raise ValueError(f"n_bins must be an integer, got {n_bins!r}")
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    return n_bins


def hog_features(image, cell_size=DEFAULT_CELL_SIZE, n_bins=DEFAULT_N_BINS):
    """Return the HOG descriptor of an 8x8 image as a 1D float array.

    Gradients are computed with central differences (``numpy.gradient``),
    orientations are unsigned (0-180 degrees) and binned into
    ``n_bins`` bins, each pixel voting with its gradient magnitude.
    Cells of ``cell_size`` x ``cell_size`` pixels form orientation
    histograms; overlapping 2x2-cell blocks are L2-normalised
    (with a small epsilon) and concatenated.

    With the defaults (``cell_size=4``, ``n_bins=9``) the image has
    2x2 cells in a single 2x2-cell block, so the descriptor has
    4 * 9 = 36 entries. With ``cell_size=2`` there are 4x4 cells and
    3x3 overlapping blocks, giving 9 * 4 * 9 = 324 entries.

    Raises:
        ValueError: if the image is not a valid 8x8 array in [0, 1],
            or ``cell_size`` / ``n_bins`` are invalid.
    """
    image = _validate_image(image)
    cell_size = _validate_cell_size(cell_size)
    n_bins = _validate_n_bins(n_bins)

    # Central-difference gradients; edges use one-sided differences.
    gy, gx = np.gradient(image)
    magnitude = np.hypot(gx, gy)
    # Unsigned orientation: 0-180 degrees.
    orientation = np.degrees(np.arctan2(gy, gx)) % 180.0

    n_cells_y = IMAGE_SHAPE[0] // cell_size
    n_cells_x = IMAGE_SHAPE[1] // cell_size
    bin_width = 180.0 / n_bins
    bin_index = np.minimum((orientation // bin_width).astype(int), n_bins - 1)

    # Per-cell orientation histograms, weighted by gradient magnitude.
    cell_hist = np.zeros((n_cells_y, n_cells_x, n_bins), dtype=np.float64)
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            ys = slice(cy * cell_size, (cy + 1) * cell_size)
            xs = slice(cx * cell_size, (cx + 1) * cell_size)
            np.add.at(
                cell_hist[cy, cx],
                bin_index[ys, xs].ravel(),
                magnitude[ys, xs].ravel(),
            )

    # Overlapping 2x2-cell blocks, each L2-normalised.
    block_cells = 2
    blocks = []
    for by in range(n_cells_y - block_cells + 1):
        for bx in range(n_cells_x - block_cells + 1):
            block = cell_hist[
                by : by + block_cells, bx : bx + block_cells, :
            ].ravel()
            norm = np.linalg.norm(block)
            blocks.append(block / np.sqrt(norm * norm + 1e-6))
    return np.concatenate(blocks)


def hog_vector_length(cell_size=DEFAULT_CELL_SIZE, n_bins=DEFAULT_N_BINS):
    """Return the length of :func:`hog_features` output for given settings."""
    cell_size = _validate_cell_size(cell_size)
    n_bins = _validate_n_bins(n_bins)
    n_cells = IMAGE_SHAPE[0] // cell_size
    n_blocks = (n_cells - 2 + 1) ** 2
    return n_blocks * 4 * n_bins


def _validate_features_labels(features, labels):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2:
        raise ValueError(
            f"features must be a 2D array (n_samples, n_features), got shape {features.shape}"
        )
    if features.shape[0] == 0:
        raise ValueError("training set must not be empty")
    if not np.isfinite(features).all():
        raise ValueError("features must contain only finite values")
    if labels.shape != (features.shape[0],):
        raise ValueError(
            f"got {features.shape[0]} feature rows but {len(labels)} labels"
        )
    if labels.size == 0:
        raise ValueError("labels must not be empty")
    try:
        labels = labels.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"labels must be integers, got {labels!r}") from exc
    if np.any(labels < 0) or np.any(labels > 9):
        raise ValueError("labels must be digits in 0-9")
    return features, labels


def _validate_k(k, n_train=None):
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise ValueError(f"k must be an integer, got {k!r}")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n_train is not None and k > n_train:
        raise ValueError(
            f"k={k} is larger than the training set size ({n_train})"
        )
    return k


class HOGKNNClassifier:
    """k-nearest-neighbours classifier over HOG feature vectors.

    Majority vote among the ``k`` nearest training vectors (Euclidean
    distance); ties are broken by the smallest digit, so predictions
    are deterministic.
    """

    def __init__(self, k=DEFAULT_K):
        self.k = _validate_k(k)
        self._features = None
        self._labels = None

    def fit(self, features, labels):
        """Store ``features`` (n_samples, n_features) and ``labels``.

        Raises:
            ValueError: if the training set is empty, the labels do
                not match the rows, or ``k`` exceeds the training size.
        """
        features, labels = _validate_features_labels(features, labels)
        _validate_k(self.k, n_train=features.shape[0])
        self._features = features
        self._labels = labels
        return self

    @property
    def is_fitted(self):
        return self._features is not None

    def _check_fitted(self, features):
        if not self.is_fitted:
            raise ValueError("classifier has not been fitted yet; call fit() first")
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2:
            raise ValueError(
                f"features must be a 1D vector or 2D array, got shape {features.shape}"
            )
        if features.shape[0] == 0:
            raise ValueError("need at least one feature vector to predict")
        if features.shape[1] != self._features.shape[1]:
            raise ValueError(
                f"feature length must be {self._features.shape[1]}, "
                f"got {features.shape[1]}"
            )
        if not np.isfinite(features).all():
            raise ValueError("features must contain only finite values")
        return features

    def predict(self, features):
        """Predict a digit (0-9) for each feature vector in ``features``.

        Accepts a single 1D vector (returns a 1-element array) or a
        2D array (returns one prediction per row).
        """
        features = self._check_fitted(features)
        # Euclidean distances to every training vector.
        diff = features[:, None, :] - self._features[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        nearest = np.argpartition(dists, self.k - 1, axis=1)[:, : self.k]
        predictions = np.empty(features.shape[0], dtype=int)
        for i, idx in enumerate(nearest):
            votes = np.bincount(self._labels[idx], minlength=10)
            predictions[i] = int(np.argmax(votes))
        return predictions

    def predict_one(self, feature_vector):
        """Predict a single digit for one 1D HOG vector."""
        return int(self.predict(feature_vector)[0])

    def score(self, features, labels):
        """Return the fraction of ``features`` correctly predicted."""
        features = self._check_fitted(features)
        labels = np.asarray(labels)
        if labels.shape != (features.shape[0],):
            raise ValueError(
                f"got {features.shape[0]} feature rows but {len(labels)} labels"
            )
        return float(np.mean(self.predict(features) == labels))


def train(features, labels, k=DEFAULT_K):
    """Fit and return a :class:`HOGKNNClassifier` with ``k`` neighbours."""
    return HOGKNNClassifier(k=k).fit(features, labels)


def predict(classifier, features):
    """Predict digits for ``features`` using a fitted ``classifier``."""
    if not isinstance(classifier, HOGKNNClassifier):
        raise ValueError(
            f"classifier must be a HOGKNNClassifier, got {type(classifier).__name__}"
        )
    return classifier.predict(features)


def accuracy(features, labels, classifier):
    """Fraction of ``features`` the fitted ``classifier`` gets right."""
    if not isinstance(classifier, HOGKNNClassifier):
        raise ValueError(
            f"classifier must be a HOGKNNClassifier, got {type(classifier).__name__}"
        )
    return classifier.score(features, labels)


def make_images(n_variants=20, flip_prob=0.05, seed=42):
    """Generate noisy digit images: ``n_variants`` per digit 0-9.

    Returns:
        ``(images, labels)`` -- a list of 8x8 arrays and a matching
        list of digit labels, digits in order 0-9.

    Raises:
        ValueError: if ``n_variants`` < 1 or ``flip_prob`` not in [0, 1].
    """
    if not isinstance(n_variants, (int, np.integer)) or isinstance(n_variants, bool):
        raise ValueError(f"n_variants must be an integer, got {n_variants!r}")
    n_variants = int(n_variants)
    if n_variants < 1:
        raise ValueError(f"n_variants must be >= 1, got {n_variants}")
    if not 0.0 <= float(flip_prob) <= 1.0:
        raise ValueError(f"flip_prob must be in [0, 1], got {flip_prob!r}")
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for digit in range(10):
        clean = template(digit)
        for _ in range(n_variants):
            images.append(add_noise(clean, flip_prob, rng))
            labels.append(digit)
    return images, labels


def _image_features(image, cell_size, n_bins, append_pixels=False):
    """HOG vector for ``image``, optionally with raw pixels appended."""
    hog = hog_features(image, cell_size=cell_size, n_bins=n_bins)
    if append_pixels:
        return np.concatenate([hog, _validate_image(image).ravel()])
    return hog


def make_dataset(n_variants=20, flip_prob=0.05, seed=42,
                 cell_size=2, n_bins=DEFAULT_N_BINS, append_pixels=False):
    """Generate a HOG feature dataset of noisy digit images.

    ``cell_size`` defaults to 2 (324-dim descriptors), which beats
    ``cell_size=4`` (36-dim) on these tiny images because the finer
    2x2-pixel cells keep more edge detail. With ``append_pixels=True``
    the flattened raw image is appended to the HOG vector, which
    closes most of the gap to template matching (see the demo).

    Returns:
        ``(features, labels)`` -- ``features`` is a float array of
        shape (10 * n_variants, hog length), ``labels`` an int array
        of matching digits.
    """
    images, labels = make_images(n_variants, flip_prob, seed)
    features = np.array(
        [
            _image_features(img, cell_size, n_bins, append_pixels)
            for img in images
        ]
    )
    return features, np.array(labels, dtype=int)


def template_match_accuracy(images, labels):
    """Nearest-template (raw-pixel Euclidean) accuracy on ``images``.

    The baseline :func:`digit_recognizer.digits.recognize` is run on
    each image; returns the fraction whose predicted digit matches.
    """
    images = list(images)
    labels = list(labels)
    if len(images) != len(labels):
        raise ValueError(
            f"got {len(images)} images but {len(labels)} labels"
        )
    if not images:
        raise ValueError("need at least one image to score")
    correct = sum(recognize(img)[0] == label for img, label in zip(images, labels))
    return correct / len(images)
