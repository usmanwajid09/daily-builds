"""CamShift: continuously adaptive mean-shift colour tracking.

Implemented from scratch with numpy only — no OpenCV. A colour
histogram is built from a target region, each new frame is turned into
a backprojection (probability) map, and mean-shift is iterated on that
map while the search window is resized from the zeroth moment, so the
tracker follows a target that moves and changes size.
"""

import numpy as np

__all__ = [
    "quantise_colour",
    "build_histogram",
    "backproject",
    "mean_shift",
    "camshift_track",
    "track_sequence",
]


def _validate_frame(frame):
    if frame is None:
        raise ValueError("frame must not be None")
    try:
        arr = np.asarray(frame, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("frame must be a numeric array") from None
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"frame must have shape (H, W, 3), got {arr.shape}")
    if arr.size == 0:
        raise ValueError("frame must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("frame must contain only finite values")
    if np.any(arr < 0) or np.any(arr > 255):
        raise ValueError("frame values must lie in [0, 255]")
    return arr


def _validate_bins(bins):
    if not isinstance(bins, (int, np.integer)) or isinstance(bins, bool):
        raise ValueError(f"bins must be an integer, got {bins!r}")
    if bins < 2 or bins > 64:
        raise ValueError(f"bins must be between 2 and 64, got {bins}")
    return int(bins)


def _validate_window(window, height, width):
    if len(window) != 4:
        raise ValueError(f"window must be (x, y, w, h), got {window!r}")
    x, y, w, h = (int(v) for v in window)
    if w <= 0 or h <= 0:
        raise ValueError(f"window width and height must be positive, got w={w}, h={h}")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"window {window} lies outside the {width}x{height} frame")
    return (x, y, w, h)


def quantise_colour(frame, bins):
    """Quantise each RGB channel into ``bins`` levels; returns int array.

    Bin index = floor(value / 256 * bins), clipped to [0, bins - 1].
    """
    arr = _validate_frame(frame)
    bins = _validate_bins(bins)
    return np.clip((arr / 256.0 * bins).astype(np.int64), 0, bins - 1)


def build_histogram(frame, window, bins=16):
    """Return a normalised 3D colour histogram of the window region.

    The histogram has shape (bins, bins, bins) and sums to 1. Raises
    ValueError for a window outside the frame or an empty region.
    """
    arr = _validate_frame(frame)
    bins = _validate_bins(bins)
    x, y, w, h = _validate_window(window, arr.shape[0], arr.shape[1])
    quant = quantise_colour(arr, bins)
    region = quant[y:y + h, x:x + w]
    hist = np.zeros((bins, bins, bins), dtype=np.float64)
    flat = region.reshape(-1, 3)
    np.add.at(hist, (flat[:, 0], flat[:, 1], flat[:, 2]), 1.0)
    total = hist.sum()
    if total == 0:
        raise ValueError("histogram region is empty")
    return hist / total


def backproject(frame, histogram):
    """Return a probability map: P(colour) for every pixel in the frame.

    Values lie in [0, 1]; pixels whose colour was never seen in the
    target histogram get 0.
    """
    arr = _validate_frame(frame)
    hist = np.asarray(histogram, dtype=np.float64)
    if hist.ndim != 3 or hist.shape[0] != hist.shape[1] or hist.shape[1] != hist.shape[2]:
        raise ValueError(f"histogram must be a cubic 3D array, got shape {hist.shape}")
    if hist.size == 0 or np.any(hist < 0) or not np.all(np.isfinite(hist)):
        raise ValueError("histogram must be non-empty, finite and non-negative")
    bins = hist.shape[0]
    quant = quantise_colour(arr, bins)
    return hist[quant[:, :, 0], quant[:, :, 1], quant[:, :, 2]]


def mean_shift(prob_map, window, max_iter=10, epsilon=1.0):
    """Iterate mean-shift on a probability map; return the new window.

    The window centre moves to the centroid of the probability mass
    inside it. Stops when the centre moves less than ``epsilon`` pixels
    or ``max_iter`` iterations are reached. If the window contains no
    probability mass, it is returned unchanged.
    """
    if prob_map is None:
        raise ValueError("prob_map must not be None")
    prob = np.asarray(prob_map, dtype=np.float64)
    if prob.ndim != 2 or prob.size == 0:
        raise ValueError("prob_map must be a non-empty 2D array")
    if not np.all(np.isfinite(prob)) or np.any(prob < 0):
        raise ValueError("prob_map must be finite and non-negative")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    height, width = prob.shape
    x, y, w, h = _validate_window(window, height, width)
    for _ in range(int(max_iter)):
        region = prob[y:y + h, x:x + w]
        m00 = float(region.sum())
        if m00 <= 0:
            break
        rows, cols = np.mgrid[0:h, 0:w].astype(np.float64)
        m10 = float(np.sum(cols * region))
        m01 = float(np.sum(rows * region))
        cx = x + m10 / m00
        cy = y + m01 / m00
        new_x = int(round(cx - w / 2.0))
        new_y = int(round(cy - h / 2.0))
        new_x = max(0, min(new_x, width - w))
        new_y = max(0, min(new_y, height - h))
        shift = abs(new_x - x) + abs(new_y - y)
        x, y = new_x, new_y
        if shift < epsilon:
            break
    return (x, y, w, h)


def camshift_track(frame, window, histogram, max_iter=10, epsilon=1.0):
    """One CamShift step: backproject, mean-shift, then resize the window.

    The new window size scales with sqrt(m00) of the probability mass
    inside the shifted window (clamped to [8, frame size]), so the
    window grows/shrinks with the target. Returns ``(window, mass)``
    where mass is the total probability inside the final window.
    """
    arr = _validate_frame(frame)
    height, width = arr.shape[0], arr.shape[1]
    x, y, w, h = _validate_window(window, height, width)
    prob = backproject(arr, histogram)
    x, y, w, h = mean_shift(prob, (x, y, w, h), max_iter=max_iter, epsilon=epsilon)
    region = prob[y:y + h, x:x + w]
    m00 = float(region.sum())
    if m00 > 0:
        side = int(round(2.0 * np.sqrt(m00)))
        side = max(8, min(side, width, height))
        cx, cy = x + w // 2, y + h // 2
        new_w = new_h = side
        new_x = max(0, min(cx - new_w // 2, width - new_w))
        new_y = max(0, min(cy - new_h // 2, height - new_h))
        x, y, w, h = new_x, new_y, new_w, new_h
    return (x, y, w, h), m00


def track_sequence(frames, window, histogram, max_iter=10, epsilon=1.0):
    """Track through a list of frames; return the window after each frame."""
    if not frames:
        raise ValueError("frames must be a non-empty list")
    current = tuple(int(v) for v in window)
    windows = []
    for frame in frames:
        current, _mass = camshift_track(frame, current, histogram,
                                        max_iter=max_iter, epsilon=epsilon)
        windows.append(current)
    return windows
