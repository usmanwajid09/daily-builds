"""Sliding-window object detector from scratch (numpy only).

Pipeline::

    image -> sliding windows (multi-position, optional multi-scale)
          -> HOG-like gradient features per window
          -> linear (logistic) classifier score
          -> threshold -> non-maximum suppression -> detections

The classifier is trained on synthetic windows: filled circles are the
positive class, blank / noisy / square windows are negatives. No OpenCV,
no torch — gradients, histograms and the classifier are all hand-rolled.
"""

import numpy as np

__all__ = [
    "hog_features",
    "LogisticClassifier",
    "iou",
    "non_max_suppression",
    "SlidingWindowDetector",
    "make_circle_window",
    "make_negative_window",
    "make_scene",
]


# ---------------------------------------------------------------------------
# HOG-like features
# ---------------------------------------------------------------------------
def hog_features(window, cell_size=4, n_bins=9):
    """Gradient-histogram descriptor for one grayscale window.

    Returns a 1-D float array of length
    ``(H // cell) * (W // cell) * n_bins``, L2-normalised per 2x2-cell
    block (with zero blocks left as zeros).
    """
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2:
        raise ValueError(f"window must be 2-D, got shape {window.shape}.")
    if window.shape[0] == 0 or window.shape[1] == 0:
        raise ValueError(f"window must not be empty, got shape {window.shape}.")
    if not np.all(np.isfinite(window)):
        raise ValueError("window must contain only finite values.")
    if not isinstance(cell_size, int) or cell_size <= 0:
        raise ValueError(f"cell_size must be a positive int, got {cell_size!r}.")
    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError(f"n_bins must be a positive int, got {n_bins!r}.")
    h, w = window.shape
    if h < cell_size or w < cell_size:
        raise ValueError(
            f"window {window.shape} is smaller than cell size {cell_size}."
        )
    gy, gx = np.gradient(window)
    magnitude = np.hypot(gx, gy)
    orientation = np.degrees(np.arctan2(gy, gx)) % 180.0
    bin_width = 180.0 / n_bins
    bin_index = np.minimum((orientation // bin_width).astype(int), n_bins - 1)

    n_cells_y, n_cells_x = h // cell_size, w // cell_size
    cells = np.zeros((n_cells_y, n_cells_x, n_bins))
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            ys = slice(cy * cell_size, (cy + 1) * cell_size)
            xs = slice(cx * cell_size, (cx + 1) * cell_size)
            for b in range(n_bins):
                mask = bin_index[ys, xs] == b
                cells[cy, cx, b] = magnitude[ys, xs][mask].sum()

    # Block normalisation over 2x2 cells (stride 1 cell).
    blocks = []
    for by in range(max(n_cells_y - 1, 1)):
        for bx in range(max(n_cells_x - 1, 1)):
            block = cells[by:by + 2, bx:bx + 2, :].reshape(-1)
            norm = np.linalg.norm(block)
            blocks.append(block / norm if norm > 1e-8 else block)
    if not blocks:  # single-cell image: fall back to the raw cell histogram
        flat = cells.reshape(-1)
        norm = np.linalg.norm(flat)
        return flat / norm if norm > 1e-8 else flat
    return np.concatenate(blocks)


def feature_length(window_shape, cell_size=4, n_bins=9):
    """Length of the descriptor hog_features returns for this window shape."""
    h, w = window_shape
    n_cells_y, n_cells_x = h // cell_size, w // cell_size
    n_blocks_y = max(n_cells_y - 1, 1)
    n_blocks_x = max(n_cells_x - 1, 1)
    return n_blocks_y * n_blocks_x * 4 * n_bins


# ---------------------------------------------------------------------------
# Linear classifier
# ---------------------------------------------------------------------------
class LogisticClassifier:
    """Binary logistic regression trained by batch gradient descent."""

    def __init__(self, n_features, lr=0.1, epochs=300, seed=0):
        if not isinstance(n_features, int) or n_features <= 0:
            raise ValueError(f"n_features must be a positive int, got {n_features!r}.")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}.")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(f"epochs must be a positive int, got {epochs!r}.")
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.01, size=n_features)
        self.bias = 0.0
        self.lr = lr
        self.epochs = epochs
        self._fitted = False

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, features), got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if X.shape[1] != len(self.weights):
            raise ValueError(
                f"expected {len(self.weights)} features, got {X.shape[1]}."
            )
        if y.shape != (X.shape[0],):
            raise ValueError(f"y must have shape ({X.shape[0]},), got {y.shape}.")
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError("y must contain only 0/1 labels.")
        for _ in range(self.epochs):
            probs = self._sigmoid(X @ self.weights + self.bias)
            error = probs - y
            grad_w = X.T @ error / len(y)
            grad_b = float(error.mean())
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
        self._fitted = True
        return self

    def decision_scores(self, X):
        if not self._fitted:
            raise ValueError("LogisticClassifier must be fitted before scoring.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise ValueError(
                f"X must have shape (N, {len(self.weights)}), got {X.shape}."
            )
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        return (self.decision_scores(X) >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def iou(box_a, box_b):
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    if min(aw, ah, bw, bh) <= 0:
        raise ValueError(f"boxes must have positive width/height: {box_a}, {box_b}.")
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(detections, iou_threshold=0.3):
    """Keep the highest-scoring box, drop any overlapping it too much.

    ``detections`` is a list of dicts with keys x, y, w, h, score.
    """
    if not isinstance(iou_threshold, (int, float)) or not 0 <= iou_threshold <= 1:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold!r}.")
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    for det in ordered:
        box = (det["x"], det["y"], det["w"], det["h"])
        if all(iou(box, (k["x"], k["y"], k["w"], k["h"])) <= iou_threshold
               for k in kept):
            kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class SlidingWindowDetector:
    """Detect filled circles in a grayscale image.

    Parameters
    ----------
    window_size : side of the square sliding window (pixels).
    stride : step between neighbouring window positions.
    cell_size, n_bins : HOG descriptor settings.
    threshold : minimum classifier score for a detection.
    iou_threshold : NMS overlap threshold.
    """

    def __init__(self, window_size=16, stride=4, cell_size=4, n_bins=9,
                 threshold=0.5, iou_threshold=0.3):
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"window_size must be a positive int, got {window_size!r}.")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError(f"stride must be a positive int, got {stride!r}.")
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {threshold!r}.")
        self.window_size = window_size
        self.stride = stride
        self.cell_size = cell_size
        self.n_bins = n_bins
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.classifier = LogisticClassifier(
            feature_length((window_size, window_size), cell_size, n_bins))
        self._fitted = False

    def _features(self, window):
        return hog_features(window, self.cell_size, self.n_bins)

    def fit(self, positive_windows, negative_windows, **kwargs):
        """Train the classifier on lists of 16x16 (or window-sized) images."""
        if len(positive_windows) == 0:
            raise ValueError("need at least one positive window.")
        if len(negative_windows) == 0:
            raise ValueError("need at least one negative window.")
        X, y = [], []
        for win in positive_windows:
            win = np.asarray(win, dtype=np.float64)
            if win.shape != (self.window_size, self.window_size):
                raise ValueError(
                    f"windows must be {self.window_size}x{self.window_size}, "
                    f"got {win.shape}."
                )
            X.append(self._features(win))
            y.append(1)
        for win in negative_windows:
            win = np.asarray(win, dtype=np.float64)
            if win.shape != (self.window_size, self.window_size):
                raise ValueError(
                    f"windows must be {self.window_size}x{self.window_size}, "
                    f"got {win.shape}."
                )
            X.append(self._features(win))
            y.append(0)
        self.classifier.fit(np.stack(X), np.array(y))
        self._fitted = True
        return self

    def detect(self, image):
        """Return a list of detections: dicts with x, y, w, h, score."""
        if not self._fitted:
            raise ValueError("detector must be fitted before detect() is called.")
        image = np.asarray(image, dtype=np.float64)
        if image.ndim != 2:
            raise ValueError(f"image must be 2-D grayscale, got shape {image.shape}.")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image must not be empty.")
        if not np.all(np.isfinite(image)):
            raise ValueError("image must contain only finite values.")
        h, w = image.shape
        if h < self.window_size or w < self.window_size:
            raise ValueError(
                f"image {image.shape} is smaller than window size {self.window_size}."
            )
        raw = []
        for top in range(0, h - self.window_size + 1, self.stride):
            for left in range(0, w - self.window_size + 1, self.stride):
                window = image[top:top + self.window_size,
                               left:left + self.window_size]
                score = float(self.classifier.decision_scores(self._features(window))[0])
                if score >= self.threshold:
                    raw.append({"x": int(left), "y": int(top),
                                "w": self.window_size, "h": self.window_size,
                                "score": score})
        return non_max_suppression(raw, self.iou_threshold)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def make_circle_window(size=16, centre=None, radius=None, rng=None):
    """16x16 window with a filled circle (positive example)."""
    rng = np.random.default_rng(rng)
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    centre = centre or (size // 2, size // 2)
    radius = radius if radius is not None else size // 4
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}.")
    yy, xx = np.mgrid[0:size, 0:size]
    return ((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius ** 2).astype(np.float64)


def make_negative_window(size=16, kind="blank", rng=None):
    """16x16 negative window: 'blank', 'noise', or 'square'."""
    rng = np.random.default_rng(rng)
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    if kind == "blank":
        return np.zeros((size, size))
    if kind == "noise":
        return rng.random((size, size))
    if kind == "square":
        img = np.zeros((size, size))
        img[size // 4:3 * size // 4, size // 4:3 * size // 4] = 1.0
        return img
    raise ValueError(f"unknown negative kind {kind!r}; expected blank/noise/square.")


def make_scene(size=48, circles=((16, 16), (34, 32)), radius=5, rng=None):
    """A scene image with filled circles at the given (row, col) centres."""
    rng = np.random.default_rng(rng)
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}.")
    img = np.zeros((size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    for (cy, cx) in circles:
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        img[mask] = 1.0
    return img
