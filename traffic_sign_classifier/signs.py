"""Synthetic traffic-sign generation + colour/shape features + kNN.

Signs are drawn programmatically as small RGB images (numpy only, no
PIL/OpenCV) so no external image files are needed. Each class has a
distinct colour + shape, as in real road signs:

- ``"stop"``        -- red octagon, white border ring, white centre bar
- ``"yield"``       -- red-bordered inverted triangle, white inside
- ``"speed_limit"`` -- white circle, red border ring, dark centre bar
- ``"no_entry"``    -- red circle, white horizontal bar
- ``"go"``          -- blue circle, white up-arrow
- ``"warning"``     -- yellow upright triangle, black exclamation mark

Variants add scale/translation jitter, background variation and a
little Gaussian noise, so a classifier must generalise rather than
memorise one exact picture. Features combine colour proportions,
shape statistics and a coarse downsampled colour histogram, and the
classifier is k-nearest neighbours (k=3 by default) on standardised
features.
"""

import numpy as np

CLASSES = ("stop", "yield", "speed_limit", "no_entry", "go", "warning")

# Canonical colours (uint8 RGB).
_RED = np.array([205, 35, 35], dtype=np.float64)
_WHITE = np.array([242, 242, 242], dtype=np.float64)
_BLUE = np.array([35, 85, 205], dtype=np.float64)
_YELLOW = np.array([238, 205, 30], dtype=np.float64)
_BLACK = np.array([30, 30, 30], dtype=np.float64)

_FEATURE_LENGTH = 62  # 6 colour + 8 shape + 48 (4x4x3) histogram


def _resolve_rng(rng):
    """Accept an int seed, a Generator, or None; return a Generator."""
    if rng is None or isinstance(rng, (int, np.integer)):
        return np.random.default_rng(rng)
    if isinstance(rng, np.random.Generator):
        return rng
    raise ValueError(f"rng must be an int seed, a Generator, or None, got {rng!r}")


def _grids(size):
    yy, xx = np.mgrid[0:size, 0:size]
    return xx.astype(np.float64), yy.astype(np.float64)


def _circle(cx, cy, r, xx, yy):
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _octagon(cx, cy, r, xx, yy):
    dx = np.abs(xx - cx)
    dy = np.abs(yy - cy)
    return (dx <= 0.924 * r) & (dy <= 0.924 * r) & (dx + dy <= 1.707 * r)


def _triangle(p1, p2, p3, xx, yy):
    """Filled triangle from vertices (x, y); barycentric sign test."""

    def _sign(ax, ay, bx, by, px, py):
        return (px - bx) * (ay - by) - (ax - bx) * (py - by)

    d1 = _sign(p1[0], p1[1], p2[0], p2[1], xx, yy)
    d2 = _sign(p2[0], p2[1], p3[0], p3[1], xx, yy)
    d3 = _sign(p3[0], p3[1], p1[0], p1[1], xx, yy)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def _rect(img, cx, cy, half_w, half_h, colour):
    h, w = img.shape[:2]
    x0, x1 = max(0, int(round(cx - half_w))), min(w, int(round(cx + half_w)) + 1)
    y0, y1 = max(0, int(round(cy - half_h))), min(h, int(round(cy + half_h)) + 1)
    img[y0:y1, x0:x1] = colour


def _paint(img, mask, colour):
    img[mask] = colour


def make_sign(sign_type, rng=None, size=32):
    """Generate one synthetic traffic-sign image.

    Args:
        sign_type: one of :data:`CLASSES`.
        rng: int seed, ``numpy.random.Generator``, or None.
        size: image side in pixels (square ``size`` x ``size``).

    Returns:
        ``numpy.ndarray`` of shape ``(size, size, 3)``, dtype uint8.

    Raises:
        ValueError: if ``sign_type`` is unknown or ``size`` is invalid.
    """
    if sign_type not in CLASSES:
        raise ValueError(
            f"sign_type must be one of {list(CLASSES)}, got {sign_type!r}"
        )
    if not isinstance(size, (int, np.integer)) or int(size) < 16:
        raise ValueError(f"size must be an integer >= 16, got {size!r}")
    size = int(size)
    rng = _resolve_rng(rng)

    # Background: light grey, varied per image.
    base = float(rng.integers(195, 226))
    bg = np.array([base, base, base - float(rng.integers(0, 8))])
    img = np.tile(bg, (size, size, 1)).astype(np.float64)

    xx, yy = _grids(size)
    scale = float(rng.uniform(0.85, 1.12))
    cx = size / 2.0 + float(rng.uniform(-2, 2))
    cy = size / 2.0 + float(rng.uniform(-2, 2))
    r = 11.0 * scale

    if sign_type == "stop":
        _paint(img, _octagon(cx, cy, r, xx, yy), _RED)
        _paint(img, _octagon(cx, cy, r * 0.84, xx, yy), _WHITE)
        _paint(img, _octagon(cx, cy, r * 0.68, xx, yy), _RED)
        _rect(img, cx, cy, 5.5 * scale, 1.8 * scale, _WHITE)
    elif sign_type == "yield":
        tri = _triangle(
            (cx, cy + r), (cx - r, cy - 0.8 * r), (cx + r, cy - 0.8 * r), xx, yy
        )
        _paint(img, tri, _RED)
        tri_in = _triangle(
            (cx, cy + 0.72 * r),
            (cx - 0.72 * r, cy - 0.58 * r),
            (cx + 0.72 * r, cy - 0.58 * r),
            xx,
            yy,
        )
        _paint(img, tri_in, _WHITE)
    elif sign_type == "speed_limit":
        _paint(img, _circle(cx, cy, r, xx, yy), _WHITE)
        ring = _circle(cx, cy, r, xx, yy) & ~_circle(cx, cy, r * 0.78, xx, yy)
        _paint(img, ring, _RED)
        _rect(img, cx, cy, 5.0 * scale, 1.8 * scale, _BLACK)
    elif sign_type == "no_entry":
        _paint(img, _circle(cx, cy, r, xx, yy), _RED)
        _rect(img, cx, cy, 6.5 * scale, 2.2 * scale, _WHITE)
    elif sign_type == "go":
        _paint(img, _circle(cx, cy, r, xx, yy), _BLUE)
        _rect(img, cx, cy + 2.5 * scale, 2.2 * scale, 4.5 * scale, _WHITE)
        head = _triangle(
            (cx, cy - 8.5 * scale),
            (cx - 4.5 * scale, cy - 0.5 * scale),
            (cx + 4.5 * scale, cy - 0.5 * scale),
            xx,
            yy,
        )
        _paint(img, head, _WHITE)
    elif sign_type == "warning":
        tri = _triangle(
            (cx, cy - r), (cx - r, cy + 0.8 * r), (cx + r, cy + 0.8 * r), xx, yy
        )
        _paint(img, tri, _BLACK)
        tri_in = _triangle(
            (cx, cy - 0.72 * r),
            (cx - 0.72 * r, cy + 0.58 * r),
            (cx + 0.72 * r, cy + 0.58 * r),
            xx,
            yy,
        )
        _paint(img, tri_in, _YELLOW)
        _rect(img, cx, cy - 1.0 * scale, 1.4 * scale, 3.5 * scale, _BLACK)
        _rect(img, cx, cy + 5.0 * scale, 1.4 * scale, 1.4 * scale, _BLACK)

    # Small Gaussian noise so no two variants are pixel-identical.
    img += rng.normal(0.0, 6.0, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _validate_image(image):
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"image must be a 3D RGB array with shape (H, W, 3), got shape {arr.shape}"
        )
    if arr.shape[0] < 8 or arr.shape[1] < 8:
        raise ValueError(
            f"image must be at least 8x8 pixels, got {arr.shape[:2]}"
        )
    if not np.isfinite(arr.astype(np.float64)).all():
        raise ValueError("image must contain only finite values")
    img = arr.astype(np.float64)
    if img.min() < 0 or img.max() > 255:
        raise ValueError("image values must lie in [0, 255]")
    # Accept float images in [0, 1] by scaling them up.
    if arr.dtype.kind == "f" and img.max() <= 1.0:
        img = img * 255.0
    return img


def extract_features(image):
    """Extract a 62-length feature vector from an RGB sign image.

    Features: 6 colour proportions (red/blue/yellow/white/black/grey),
    8 shape statistics of the foreground mask, and a coarse 4x4
    downsampled colour histogram (48 values, each channel / 255).

    Raises:
        ValueError: if ``image`` is not a 3D RGB array.
    """
    img = _validate_image(image)
    h, w, _ = img.shape
    r_ch, g_ch, b_ch = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    total = float(h * w)

    red = (r_ch > 150) & (g_ch < 100) & (b_ch < 100)
    blue = (b_ch > 120) & (b_ch > r_ch + 50) & (r_ch < 120)
    yellow = (r_ch > 180) & (g_ch > 150) & (b_ch < 100)
    white = (r_ch > 200) & (g_ch > 200) & (b_ch > 200)
    black = (r_ch < 80) & (g_ch < 80) & (b_ch < 80)
    grey = (~red) & (~blue) & (~yellow) & (~white) & (~black)
    colour = np.array(
        [m.mean() for m in (red, blue, yellow, white, black, grey)],
        dtype=np.float64,
    )

    # Foreground = pixels far from the corner (background) colour.
    corner = np.median(
        np.concatenate(
            [
                img[:4, :4].reshape(-1, 3),
                img[:4, -4:].reshape(-1, 3),
                img[-4:, :4].reshape(-1, 3),
                img[-4:, -4:].reshape(-1, 3),
            ]
        ),
        axis=0,
    )
    mask = np.max(np.abs(img - corner), axis=2) > 40
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        shape = np.zeros(8, dtype=np.float64)
    else:
        r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        bbox = mask[r0 : r1 + 1, c0 : c1 + 1]
        fill = float(mask.mean() * h * w / (bh * bw)) if bh * bw else 0.0
        # Fraction of the bbox corner blocks that are foreground:
        # octagons/circles fill corners less than triangles do.
        ch, cw = max(1, bh // 5), max(1, bw // 5)
        corner_blocks = np.concatenate(
            [
                bbox[:ch, :cw].ravel(),
                bbox[:ch, -cw:].ravel(),
                bbox[-ch:, :cw].ravel(),
                bbox[-ch:, -cw:].ravel(),
            ]
        )
        corner_frac = float(corner_blocks.mean())
        lr_sym = float(np.mean(mask == np.fliplr(mask)))
        tb_sym = float(np.mean(mask == np.flipud(mask)))
        cy_off = float((rows.mean() - h / 2.0) / h)
        cx_off = float((cols.mean() - w / 2.0) / w)
        shape = np.array(
            [
                fill,
                bw / bh,
                float(mask.mean()),
                corner_frac,
                lr_sym,
                tb_sym,
                cy_off,
                cx_off,
            ],
            dtype=np.float64,
        )

    # Coarse 4x4 downsampled colour histogram (spatial layout matters:
    # e.g. yield points down, warning points up).
    hist = []
    edges_r = np.linspace(0, h, 5).astype(int)
    edges_c = np.linspace(0, w, 5).astype(int)
    for i in range(4):
        for j in range(4):
            block = img[edges_r[i] : edges_r[i + 1], edges_c[j] : edges_c[j + 1]]
            hist.extend(block.mean(axis=(0, 1)) / 255.0)
    features = np.concatenate([colour, shape, np.array(hist, dtype=np.float64)])
    assert len(features) == _FEATURE_LENGTH
    return features


def make_dataset(n_per_class, seed=0):
    """Generate a shuffled synthetic dataset.

    Args:
        n_per_class: number of variants to generate per class.
        seed: random seed (deterministic for a given seed).

    Returns:
        ``(images, labels)`` -- a list of uint8 RGB arrays and a list
        of class-name strings, shuffled with the same seed.

    Raises:
        ValueError: if ``n_per_class`` is not a positive integer.
    """
    if not isinstance(n_per_class, (int, np.integer)) or int(n_per_class) < 1:
        raise ValueError(
            f"n_per_class must be a positive integer, got {n_per_class!r}"
        )
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for sign_type in CLASSES:
        for _ in range(int(n_per_class)):
            images.append(make_sign(sign_type, rng=rng))
            labels.append(sign_type)
    order = rng.permutation(len(images))
    return [images[i] for i in order], [labels[i] for i in order]


class TrafficSignClassifier:
    """k-nearest-neighbour traffic-sign classifier (k=3 by default).

    Features are standardised with the training mean/std before the
    Euclidean distance is computed.
    """

    def __init__(self, k=3):
        if not isinstance(k, (int, np.integer)) or int(k) < 1:
            raise ValueError(f"k must be an integer >= 1, got {k!r}")
        self.k = int(k)
        self._mean = None
        self._std = None
        self._train_X = None
        self._train_y = None

    def fit(self, images, labels):
        """Fit on a list of RGB images and matching class labels.

        Raises:
            ValueError: if the training set is empty, lengths mismatch,
                or ``k`` exceeds the number of training examples.
        """
        images = list(images)
        labels = list(labels)
        if not images:
            raise ValueError("need at least one training image to fit")
        if len(images) != len(labels):
            raise ValueError(
                f"got {len(images)} images but {len(labels)} labels"
            )
        if self.k > len(images):
            raise ValueError(
                f"k={self.k} exceeds the number of training examples "
                f"({len(images)})"
            )
        for label in labels:
            if label not in CLASSES:
                raise ValueError(
                    f"labels must be one of {list(CLASSES)}, got {label!r}"
                )
        feats = np.array([extract_features(img) for img in images])
        self._mean = feats.mean(axis=0)
        self._std = feats.std(axis=0)
        self._std[self._std == 0] = 1.0
        self._train_X = (feats - self._mean) / self._std
        self._train_y = labels
        return self

    def _standardise(self, images):
        feats = np.array([extract_features(img) for img in images])
        return (feats - self._mean) / self._std

    def predict(self, images):
        """Predict a class label for each image in ``images``.

        Returns a list of class-name strings.

        Raises:
            ValueError: if the classifier has not been fitted yet.
        """
        if self._train_X is None:
            raise ValueError("classifier has not been fitted yet; call fit() first")
        images = list(images)
        if not images:
            raise ValueError("need at least one image to predict")
        feats = self._standardise(images)
        preds = []
        for vec in feats:
            dists = np.linalg.norm(self._train_X - vec, axis=1)
            nearest = np.argsort(dists, kind="stable")[: self.k]
            votes = {}
            for idx in nearest:
                votes[self._train_y[idx]] = votes.get(self._train_y[idx], 0) + 1
            # Most votes wins; ties broken alphabetically (deterministic).
            preds.append(max(sorted(votes), key=lambda lab: votes[lab]))
        return preds

    def predict_one(self, image):
        """Predict the class label for a single image."""
        return self.predict([image])[0]

    def score(self, images, labels):
        """Fraction of ``images`` correctly classified against ``labels``."""
        preds = self.predict(images)
        return float(np.mean([p == t for p, t in zip(preds, labels)]))


def accuracy(images, labels, classifier=None):
    """Fraction correct.

    Two uses: ``accuracy(images, labels, classifier)`` scores a fitted
    :class:`TrafficSignClassifier` on images, and
    ``accuracy(y_true, y_pred)`` compares two label lists directly.

    Raises:
        ValueError: if the inputs are empty or lengths mismatch.
    """
    if classifier is not None:
        return classifier.score(images, labels)
    y_true, y_pred = list(images), list(labels)
    if not y_true:
        raise ValueError("need at least one label to score")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"got {len(y_true)} true labels but {len(y_pred)} predictions"
        )
    return float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
