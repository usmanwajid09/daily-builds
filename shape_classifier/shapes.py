"""Hu-moment shape classification from scratch (numpy only).

Shapes (circle, square, triangle, star) are drawn as binary masks on a
canvas. Each shape is described by its 7 Hu moment invariants — features
that are (to first order) invariant to translation, scale, and rotation —
and a small kNN classifier on the log-transformed Hu features separates
the four classes.
"""

import numpy as np

CLASSES = ("circle", "square", "triangle", "star")


def _rotate(points, angle_deg, centre):
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return (points - centre) @ rot.T + centre


def _fill_polygon(canvas, vertices):
    """Fill a polygon given as ``(x, y)`` vertices into ``canvas`` (in place)."""
    height, width = canvas.shape
    verts = np.asarray(vertices, dtype=np.float64)
    n = len(verts)
    for y in range(height):
        yc = y + 0.5
        xs = []
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            if (y1 <= yc < y2) or (y2 <= yc < y1):
                xs.append(x1 + (yc - y1) * (x2 - x1) / (y2 - y1))
        xs.sort()
        for i in range(0, len(xs) - 1, 2):
            x_start = max(0, int(np.ceil(xs[i] - 0.5)))
            x_end = min(width, int(np.floor(xs[i + 1] - 0.5)) + 1)
            if x_end > x_start:
                canvas[y, x_start:x_end] = 1.0
    return canvas


def _regular_polygon_vertices(n_sides, radius, centre, angle_deg=0.0):
    angles = np.deg2rad(np.arange(n_sides) * (360.0 / n_sides) - 90.0)
    points = np.stack(
        [centre[0] + radius * np.cos(angles),
         centre[1] + radius * np.sin(angles)],
        axis=1,
    )
    return _rotate(points, angle_deg, np.asarray(centre, dtype=np.float64))


def make_circle(size=64, radius=20, centre=None):
    """Return a binary image with a filled circle."""
    centre = (size / 2.0, size / 2.0) if centre is None else centre
    rows, cols = np.indices((size, size))
    dist2 = (rows - centre[1]) ** 2 + (cols - centre[0]) ** 2
    return (dist2 <= radius * radius).astype(np.float64)


def make_square(size=64, side=36, centre=None, angle_deg=0.0):
    """Return a binary image with a filled (optionally rotated) square."""
    centre = np.array((size / 2.0, size / 2.0)) if centre is None else np.asarray(centre, dtype=np.float64)
    half = side / 2.0
    corners = np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half]]
    ) + centre
    canvas = np.zeros((size, size), dtype=np.float64)
    return _fill_polygon(canvas, _rotate(corners, angle_deg, centre))


def make_triangle(size=64, side=40, centre=None, angle_deg=0.0):
    """Return a binary image with a filled (optionally rotated) triangle."""
    centre = np.array((size / 2.0, size / 2.0)) if centre is None else np.asarray(centre, dtype=np.float64)
    radius = side / np.sqrt(3.0)
    canvas = np.zeros((size, size), dtype=np.float64)
    return _fill_polygon(
        canvas, _regular_polygon_vertices(3, radius, centre, angle_deg)
    )


def make_star(size=64, outer=22, inner=9, centre=None, angle_deg=0.0):
    """Return a binary image with a filled 5-pointed star."""
    centre = np.array((size / 2.0, size / 2.0)) if centre is None else np.asarray(centre, dtype=np.float64)
    angles = np.deg2rad(np.arange(10) * 36.0 - 90.0)
    radii = np.where(np.arange(10) % 2 == 0, outer, inner)
    points = np.stack(
        [centre[0] + radii * np.cos(angles),
         centre[1] + radii * np.sin(angles)],
        axis=1,
    )
    canvas = np.zeros((size, size), dtype=np.float64)
    return _fill_polygon(canvas, _rotate(points, angle_deg, centre))


def make_shape(name, size=64, rng=None):
    """Generate one shape of class ``name`` with random scale/position/rotation."""
    rng = np.random.default_rng(rng)
    centre = np.array(
        [size / 2.0 + rng.uniform(-6, 6), size / 2.0 + rng.uniform(-6, 6)]
    )
    angle = float(rng.uniform(0, 360))
    if name == "circle":
        return make_circle(size=size, radius=float(rng.uniform(14, 24)),
                           centre=centre)
    if name == "square":
        return make_square(size=size, side=float(rng.uniform(26, 40)),
                           centre=centre, angle_deg=angle)
    if name == "triangle":
        return make_triangle(size=size, side=float(rng.uniform(32, 46)),
                             centre=centre, angle_deg=angle)
    if name == "star":
        return make_star(size=size, outer=float(rng.uniform(18, 25)),
                         inner=float(rng.uniform(7, 11)),
                         centre=centre, angle_deg=angle)
    raise ValueError(f"unknown shape class {name!r}, expected one of {CLASSES}")


def _validate_binary(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    if image.sum() <= 0:
        raise ValueError("image is empty: no foreground pixels found")
    return (image > 0).astype(np.float64)


def hu_moments(image):
    """Return the 7 Hu moment invariants of a binary ``image``.

    Moments are computed from the foreground pixels, normalised for
    scale, then log-transformed (``-sign(h) * log10(|h|)``) so the
    features live on a comparable scale for kNN.
    """
    binary = _validate_binary(image)
    ys, xs = np.nonzero(binary)
    m00 = float(len(xs))
    x_bar, y_bar = xs.mean(), ys.mean()
    dx, dy = xs - x_bar, ys - y_bar
    mu20 = float((dx ** 2).sum())
    mu02 = float((dy ** 2).sum())
    mu11 = float((dx * dy).sum())
    mu30 = float((dx ** 3).sum())
    mu03 = float((dy ** 3).sum())
    mu12 = float((dx * dy ** 2).sum())
    mu21 = float((dx ** 2 * dy).sum())
    eta20 = mu20 / m00 ** 2
    eta02 = mu02 / m00 ** 2
    eta11 = mu11 / m00 ** 2
    eta30 = mu30 / m00 ** 2.5
    eta03 = mu03 / m00 ** 2.5
    eta12 = mu12 / m00 ** 2.5
    eta21 = mu21 / m00 ** 2.5
    h = np.array([
        eta20 + eta02,
        (eta20 - eta02) ** 2 + 4 * eta11 ** 2,
        (eta30 - 3 * eta12) ** 2 + (3 * eta21 - eta03) ** 2,
        (eta30 + eta12) ** 2 + (eta21 + eta03) ** 2,
        (eta30 - 3 * eta12) * (eta30 + eta12)
        * ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2)
        + (3 * eta21 - eta03) * (eta21 + eta03)
        * (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2),
        (eta20 - eta02) * ((eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
        + 4 * eta11 * (eta30 + eta12) * (eta21 + eta03),
        (3 * eta21 - eta03) * (eta30 + eta12)
        * ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2)
        - (eta30 - 3 * eta12) * (eta21 + eta03)
        * (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2),
    ], dtype=np.float64)
    abs_h = np.abs(h)
    log_h = np.zeros_like(h)
    nonzero = abs_h > 0
    log_h[nonzero] = -np.sign(h[nonzero]) * np.log10(abs_h[nonzero])
    return log_h


def extract_features(images):
    """Return an ``(n, 4)`` array of log-Hu features for ``images``.

    Only the first four invariants are used: h5-h7 vanish for
    mirror-symmetric shapes, so their log transform is dominated by
    rasterisation noise (random sign flips of a near-zero value) and
    would swamp the distance metric.
    """
    images = list(images)
    if not images:
        raise ValueError("need at least one image to extract features from")
    return np.stack([hu_moments(img)[:4] for img in images])


class ShapeKNN:
    """k-Nearest Neighbour classifier over log-Hu moment features.

    Features are z-scored on the training set before the Euclidean
    distance is computed, so no single invariant dominates the metric.
    """

    def __init__(self, k=3):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._train_X = None
        self._train_y = None
        self._mean = None
        self._scale = None

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
        self._mean = X.mean(axis=0)
        self._scale = X.std(axis=0)
        self._scale[self._scale == 0] = 1.0
        self._train_X = (X - self._mean) / self._scale
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
        X = (X - self._mean) / self._scale
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
