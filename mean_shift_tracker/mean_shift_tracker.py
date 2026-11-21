"""Mean-shift colour-histogram object tracking from scratch.

The target is modelled by a normalised RGB colour histogram taken
from the initial bounding box. In each new frame the tracker
back-projects that model into a weight image — every pixel gets a
weight of ``sqrt(q[u] / p[u])`` where ``q`` is the target histogram,
``p`` the histogram of the current window and ``u`` the pixel's
colour bin — and then shifts the window centre to the weighted mean
of the pixel positions inside the window (the mean-shift step). The
step repeats until the centre barely moves or the iteration budget
runs out. Pixels whose colour never occurred in the target get weight
zero, so the window is pulled towards the target and away from
differently coloured background.

Frames are RGB arrays with values in 0..255 (uint8 or float);
bounding boxes are ``(y, x, height, width)`` with ``(y, x)`` the
top-left corner.
"""

import numpy as np

__all__ = [
    "MeanShiftTracker",
    "color_histogram",
    "moving_square_frame",
]


def _check_frame(frame, name="frame"):
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"{name} must be an RGB image of shape (height, width, 3), "
            f"got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _check_bins(bins):
    if not isinstance(bins, (int, np.integer)) or int(bins) < 2:
        raise ValueError(f"bins must be an integer >= 2, got {bins!r}")
    return int(bins)


def _bin_indices(frame, bins):
    """Per-pixel colour-bin index in ``0 .. bins**3 - 1``."""
    arr = np.asarray(frame, dtype=float)
    arr = np.clip(arr, 0, 255)
    scaled = (arr * bins / 256.0).astype(int)
    scaled = np.clip(scaled, 0, bins - 1)
    return (scaled[..., 0] * bins + scaled[..., 1]) * bins + scaled[..., 2]


def color_histogram(frame, bins=8):
    """Normalised RGB colour histogram of ``frame`` (length ``bins**3``)."""
    frame = _check_frame(frame)
    bins = _check_bins(bins)
    indices = _bin_indices(frame, bins)
    hist = np.bincount(indices.ravel(), minlength=bins ** 3).astype(float)
    total = hist.sum()
    if total == 0:
        raise ValueError("cannot build a histogram from an empty image")
    return hist / total


def _check_bbox(bbox, height, width, name="bbox"):
    bbox = tuple(bbox)
    if len(bbox) != 4:
        raise ValueError(f"{name} must be (y, x, height, width), got {bbox!r}")
    y, x, h, w = (float(v) for v in bbox)
    if h <= 0 or w <= 0:
        raise ValueError(f"{name} height and width must be positive, got {bbox!r}")
    if y < 0 or x < 0 or y + h > height or x + w > width:
        raise ValueError(
            f"{name} {bbox!r} does not fit inside a {height}x{width} frame")
    return y, x, h, w


def _window_histogram(frame, center_y, center_x, height, width, bins):
    """Histogram of the window centred at ``(center_y, center_x)``.

    The window is clipped to the frame; returns ``(hist, ys, xs)``
    with the pixel coordinates of the (possibly clipped) window.
    """
    h2, w2 = height // 2, width // 2
    # Round-half-up (not banker's rounding): a centre of 42.5 must put
    # the window at 43, or the window stalls half a pixel short of the
    # target and the centroid can never advance onto it.
    cy = int(np.floor(center_y + 0.5))
    cx = int(np.floor(center_x + 0.5))
    y0 = max(0, cy - h2)
    y1 = min(frame.shape[0], cy - h2 + height)
    x0 = max(0, cx - w2)
    x1 = min(frame.shape[1], cx - w2 + width)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("tracking window lies outside the frame")
    indices = _bin_indices(frame[y0:y1, x0:x1], bins)
    hist = np.bincount(indices.ravel(), minlength=bins ** 3).astype(float)
    hist /= hist.sum()
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return hist, ys.astype(float), xs.astype(float), indices


def mean_shift_step(frame, center, size, target_hist, bins=8):
    """One mean-shift step; returns the new ``(y, x)`` centre."""
    frame = _check_frame(frame)
    bins = _check_bins(bins)
    center_y, center_x = float(center[0]), float(center[1])
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"size must be positive (height, width), got {size!r}")
    cand_hist, ys, xs, indices = _window_histogram(
        frame, center_y, center_x, height, width, bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cand_hist > 0, target_hist / np.maximum(cand_hist, 1e-12), 0.0)
    weights = np.sqrt(np.maximum(ratio[indices], 0.0))
    total = weights.sum()
    if total <= 0:
        return center_y, center_x  # nothing in the window matches the target
    new_y = float(np.sum(weights * ys) / total)
    new_x = float(np.sum(weights * xs) / total)
    return new_y, new_x


class MeanShiftTracker:
    """Stateful mean-shift tracker for one target.

    Parameters
    ----------
    bins : int
        Colour bins per channel (histogram has ``bins**3`` entries).
    max_iterations : int
        Mean-shift steps per frame.
    epsilon : float
        Stop when the centre moves less than this many pixels.
    """

    def __init__(self, bins=8, max_iterations=20, epsilon=0.5):
        self.bins = _check_bins(bins)
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations!r}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon!r}")
        self.max_iterations = int(max_iterations)
        self.epsilon = float(epsilon)
        self.target_hist_ = None
        self.center_ = None
        self.size_ = None

    def init(self, frame, bbox):
        """Learn the target histogram from ``bbox`` in ``frame``.

        Returns the initial ``(y, x, height, width)`` box (floats).
        """
        frame = _check_frame(frame)
        y, x, h, w = _check_bbox(bbox, frame.shape[0], frame.shape[1])
        self.target_hist_ = color_histogram(
            frame[int(y):int(y + h), int(x):int(x + w)], bins=self.bins)
        self.center_ = (y + h / 2.0, x + w / 2.0)
        self.size_ = (h, w)
        return self.bbox

    @property
    def bbox(self):
        """Current ``(y, x, height, width)`` box (floats)."""
        if self.center_ is None:
            raise RuntimeError("tracker is not initialised; call init() first")
        cy, cx = self.center_
        h, w = self.size_
        return (cy - h / 2.0, cx - w / 2.0, h, w)

    def update(self, frame):
        """Track the target in ``frame``; returns the new bbox."""
        if self.target_hist_ is None:
            raise RuntimeError("tracker is not initialised; call init() first")
        frame = _check_frame(frame)
        cy, cx = self.center_
        for _ in range(self.max_iterations):
            new_y, new_x = mean_shift_step(
                frame, (cy, cx), self.size_, self.target_hist_, bins=self.bins)
            shift = float(np.hypot(new_y - cy, new_x - cx))
            cy, cx = new_y, new_x
            if shift < self.epsilon:
                break
        self.center_ = (cy, cx)
        return self.bbox


def moving_square_frame(shape=(60, 80), center=(30, 40), size=12, seed=0,
                        background=(40, 40, 40), color=(220, 30, 30)):
    """Synthetic frame: a solid coloured square on a noisy background.

    The background is dark grey with a little per-pixel noise; the
    square is bright red, so its colour histogram is distinct. Moving
    ``center`` between frames gives a known ground-truth motion.
    """
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError(f"shape must be (height, width) of positive ints, got {shape!r}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size!r}")
    rng = np.random.default_rng(seed)
    frame = np.zeros((shape[0], shape[1], 3), dtype=float)
    for c in range(3):
        frame[..., c] = background[c] + rng.normal(0.0, 4.0, size=shape)
    half = size // 2
    y0 = int(round(center[0])) - half
    x0 = int(round(center[1])) - half
    y1, x1 = y0 + size, x0 + size
    if y0 < 0 or x0 < 0 or y1 > shape[0] or x1 > shape[1]:
        raise ValueError(f"square centred at {center!r} does not fit in {shape!r}")
    frame[y0:y1, x0:x1] = color
    return np.clip(frame, 0, 255)
