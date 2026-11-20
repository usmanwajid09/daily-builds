"""Hand-gesture classification on synthetic binary shapes (numpy only).

Four gesture classes — circle, square, triangle, star — are drawn as
filled binary masks with random scale, position and rotation. Each
gesture is described by three rotation/scale/translation invariant
features — the log-transformed first Hu moment invariant, the radial
spread of boundary pixels, and the solidity (area / convex-hull
area) — and a k-Nearest Neighbour classifier with vote-share
probabilities separates the classes.
"""

import numpy as np

GESTURES = ("circle", "square", "triangle", "star")


def _rotate(points, angle_deg, centre):
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return (points - centre) @ rot.T + centre


def _fill_polygon(canvas, vertices):
    """Fill a polygon given as ``(x, y)`` vertices into ``canvas``."""
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
    centre = (np.array((size / 2.0, size / 2.0)) if centre is None
              else np.asarray(centre, dtype=np.float64))
    half = side / 2.0
    corners = np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half]]
    ) + centre
    canvas = np.zeros((size, size), dtype=np.float64)
    return _fill_polygon(canvas, _rotate(corners, angle_deg, centre))


def make_triangle(size=64, side=40, centre=None, angle_deg=0.0):
    """Return a binary image with a filled (optionally rotated) triangle."""
    centre = (np.array((size / 2.0, size / 2.0)) if centre is None
              else np.asarray(centre, dtype=np.float64))
    radius = side / np.sqrt(3.0)
    canvas = np.zeros((size, size), dtype=np.float64)
    return _fill_polygon(
        canvas, _regular_polygon_vertices(3, radius, centre, angle_deg)
    )


def make_star(size=64, outer=22, inner=9, centre=None, angle_deg=0.0):
    """Return a binary image with a filled 5-pointed star."""
    centre = (np.array((size / 2.0, size / 2.0)) if centre is None
              else np.asarray(centre, dtype=np.float64))
    angles = np.deg2rad(np.arange(10) * 36.0 - 90.0)
    radii = np.where(np.arange(10) % 2 == 0, outer, inner)
    points = np.stack(
        [centre[0] + radii * np.cos(angles),
         centre[1] + radii * np.sin(angles)],
        axis=1,
    )
    canvas = np.zeros((size, size), dtype=np.float64)
    return _fill_polygon(canvas, _rotate(points, angle_deg, centre))


def make_gesture(name, size=64, rng=None):
    """Generate one gesture of class ``name`` with random geometry."""
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
    raise ValueError(f"unknown gesture {name!r}, expected one of {GESTURES}")


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
    """Return the 7 log-transformed Hu moment invariants of ``image``."""
    binary = _validate_binary(image).astype(bool)
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


def radial_spread(image):
    """Spread of boundary-pixel distances from the centroid.

    For each boundary pixel (foreground with a background 4-neighbour)
    take its distance from the shape centroid and return std/mean.
    Rotation, scale and translation invariant: ~0.01 for a circle,
    ~0.11 for a square, ~0.22 for a triangle, ~0.24 for a star. It
    separates the near-circular classes that the log-Hu features
    alone leave close together (a perfect disc has near-zero higher
    Hu invariants, so their log transform is rasterisation noise).
    """
    binary = _validate_binary(image).astype(bool)
    ys, xs = np.nonzero(binary)
    cy, cx = ys.mean(), xs.mean()
    padded = np.pad(binary, 1)
    inner = padded[1:-1, 1:-1]
    neighbours = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                  & padded[1:-1, :-2] & padded[1:-1, 2:])
    boundary = inner & ~neighbours
    by, bx = np.nonzero(boundary)
    if len(bx) < 3:
        raise ValueError("image has too few boundary pixels")
    dists = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)
    return float(dists.std() / dists.mean())


def _convex_hull_area(points):
    """Area of the convex hull of ``points`` (Andrew's monotone chain).

    ``points`` is an ``(n, 2)`` array of ``(x, y)`` pixel centres;
    the area comes from the shoelace formula over the hull vertices.
    """
    pts = sorted(set(map(tuple, np.asarray(points).tolist())))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    x = np.array([p[0] for p in hull], dtype=np.float64)
    y = np.array([p[1] for p in hull], dtype=np.float64)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1))
                           - np.dot(y, np.roll(x, -1))))


def solidity(image):
    """Foreground area / convex-hull area of a binary ``image``.

    ~1.0 for convex shapes (circle, square, triangle) and ~0.6 for the
    concave 5-pointed star. The ratio is clipped at 1.0: pixel-centre
    hulls slightly underestimate the true hull of curved shapes.
    """
    binary = _validate_binary(image)
    ys, xs = np.nonzero(binary)
    hull_area = _convex_hull_area(np.stack([xs, ys], axis=1))
    if hull_area <= 0:
        raise ValueError("image has too few pixels for a convex hull")
    return float(min(1.0, len(xs) / hull_area))


def extract_features(images):
    """Return an ``(n, 3)`` array of features for ``images``.

    The columns are: the log-transformed first Hu invariant, the
    radial spread from :func:`radial_spread`, and the solidity from
    :func:`solidity`. Only h1 of the Hu invariants is used as a
    feature: h2-h7 are (numerically) zero for these symmetric shapes,
    so their log transform is pure rasterisation noise and would
    swamp the distance metric — the full 7 invariants are still
    available from :func:`hu_moments`.
    """
    images = list(images)
    if not images:
        raise ValueError("need at least one image to extract features from")
    feats = np.array([[hu_moments(img)[0], radial_spread(img),
                       solidity(img)] for img in images])
    return feats


class GestureClassifier:
    """k-Nearest Neighbour gesture classifier over log-Hu features.

    Features are z-scored on the training set. ``predict_proba``
    returns the vote share of the k nearest neighbours per class, so
    the confidence of a prediction is visible.
    """

    def __init__(self, k=3):
        if int(k) < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)
        self._train_X = None
        self._train_y = None
        self._mean = None
        self._scale = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"got {len(X)} samples but {len(y)} labels")
        if len(X) < self.k:
            raise ValueError(
                f"need at least k={self.k} training samples, got {len(X)}")
        unknown = set(np.unique(y)) - set(GESTURES)
        if unknown:
            raise ValueError(f"unknown gesture label(s): {sorted(unknown)}")
        self.classes_ = np.array(sorted(set(GESTURES) & set(np.unique(y))))
        self._mean = X.mean(axis=0)
        self._scale = X.std(axis=0)
        self._scale[self._scale == 0] = 1.0
        self._train_X = (X - self._mean) / self._scale
        self._train_y = y
        return self

    def _check_fitted(self):
        if self._train_X is None:
            raise ValueError("call fit() before predicting")

    def _nearest(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self._train_X.shape[1]:
            raise ValueError(
                f"expected {self._train_X.shape[1]} features, "
                f"got {X.shape[1]}")
        X = (X - self._mean) / self._scale
        out = []
        for x in X:
            dists = np.linalg.norm(self._train_X - x, axis=1)
            out.append(np.argsort(dists)[: self.k])
        return out

    def predict(self, X):
        """Predicted gesture label for each row of ``X``."""
        preds = []
        for nearest in self._nearest(X):
            labels, counts = np.unique(
                self._train_y[nearest], return_counts=True)
            preds.append(labels[np.argmax(counts)])
        return np.array(preds)

    def predict_proba(self, X):
        """Vote-share probabilities, shape ``(n, len(classes_))``."""
        probas = []
        for nearest in self._nearest(X):
            row = np.zeros(len(self.classes_))
            labels, counts = np.unique(
                self._train_y[nearest], return_counts=True)
            for label, count in zip(labels, counts):
                row[list(self.classes_).index(label)] = count / self.k
            probas.append(row)
        return np.array(probas)

    def classify(self, image):
        """Classify a single binary gesture ``image`` -> label string."""
        return str(self.predict(extract_features([image]))[0])


def accuracy(classifier, images, labels):
    """Fraction of ``images`` the fitted ``classifier`` gets right."""
    preds = classifier.predict(extract_features(images))
    labels = np.asarray(labels)
    if len(preds) != len(labels):
        raise ValueError("images/labels length mismatch")
    return float(np.mean(preds == labels))
