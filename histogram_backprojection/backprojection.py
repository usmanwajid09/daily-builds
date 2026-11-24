"""Histogram backprojection for colour-based object localisation.

Implemented from scratch with numpy only — no OpenCV. A colour
histogram is built from a region of interest, then every pixel of a
(new) image is scored by how likely its colour is under that histogram.
The result is a probability map whose peak / centroid localises the
object that matches the model's colour.
"""

import numpy as np

__all__ = [
    "build_histogram",
    "backproject",
    "locate_peak",
    "locate_centroid",
    "localise",
]


def _validate_image(image):
    if image is None:
        raise ValueError("image must not be None")
    try:
        arr = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("image must be a numeric array") from None
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("image must contain only finite values")
    if np.any(arr < 0) or np.any(arr > 255):
        raise ValueError("image values must lie in [0, 255]")
    return arr


def _validate_bins(bins):
    if not isinstance(bins, (int, np.integer)) or isinstance(bins, bool):
        raise ValueError(f"bins must be an integer, got {bins!r}")
    if bins < 2 or bins > 64:
        raise ValueError(f"bins must be between 2 and 64, got {bins}")
    return int(bins)


def _validate_roi(roi, height, width):
    if len(roi) != 4:
        raise ValueError(f"roi must be (x, y, w, h), got {roi!r}")
    x, y, w, h = (int(v) for v in roi)
    if w <= 0 or h <= 0:
        raise ValueError(f"roi width and height must be positive, got w={w}, h={h}")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"roi {roi} lies outside the {width}x{height} image")
    return (x, y, w, h)


def _quantise(arr, bins):
    return np.clip((arr / 256.0 * bins).astype(np.int64), 0, bins - 1)


def build_histogram(image, roi, bins=16):
    """Return a normalised 3D RGB histogram of the ROI.

    Shape (bins, bins, bins), sums to 1.
    """
    arr = _validate_image(image)
    bins = _validate_bins(bins)
    x, y, w, h = _validate_roi(roi, arr.shape[0], arr.shape[1])
    quant = _quantise(arr, bins)
    region = quant[y:y + h, x:x + w].reshape(-1, 3)
    hist = np.zeros((bins, bins, bins), dtype=np.float64)
    np.add.at(hist, (region[:, 0], region[:, 1], region[:, 2]), 1.0)
    total = hist.sum()
    if total == 0:
        raise ValueError("histogram region is empty")
    return hist / total


def backproject(image, histogram):
    """Return a probability map in [0, 1] for every pixel's colour.

    The map is scaled so its maximum is 1 (unless the histogram is
    all zeros, which raises ValueError).
    """
    arr = _validate_image(image)
    hist = np.asarray(histogram, dtype=np.float64)
    if hist.ndim != 3 or hist.shape[0] != hist.shape[1] or hist.shape[1] != hist.shape[2]:
        raise ValueError(f"histogram must be a cubic 3D array, got shape {hist.shape}")
    if hist.size == 0 or np.any(hist < 0) or not np.all(np.isfinite(hist)):
        raise ValueError("histogram must be non-empty, finite and non-negative")
    if hist.max() <= 0:
        raise ValueError("histogram is all zeros: nothing to backproject")
    bins = hist.shape[0]
    quant = _quantise(arr, bins)
    prob = hist[quant[:, :, 0], quant[:, :, 1], quant[:, :, 2]]
    return prob / prob.max() if prob.max() > 0 else prob


def locate_peak(prob_map):
    """Return ``(x, y, value)`` of the maximum of a probability map."""
    prob = np.asarray(prob_map, dtype=np.float64)
    if prob.ndim != 2 or prob.size == 0:
        raise ValueError("prob_map must be a non-empty 2D array")
    if not np.all(np.isfinite(prob)):
        raise ValueError("prob_map must contain only finite values")
    y, x = np.unravel_index(int(np.argmax(prob)), prob.shape)
    return (int(x), int(y), float(prob[y, x]))


def locate_centroid(prob_map, threshold=0.5):
    """Return the centroid (x, y) of pixels above ``threshold`` * max.

    Raises ValueError if no pixel clears the threshold.
    """
    prob = np.asarray(prob_map, dtype=np.float64)
    if prob.ndim != 2 or prob.size == 0:
        raise ValueError("prob_map must be a non-empty 2D array")
    if not np.all(np.isfinite(prob)):
        raise ValueError("prob_map must contain only finite values")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}")
    mask = prob >= threshold * prob.max()
    if not np.any(mask):
        raise ValueError("no pixel clears the threshold")
    ys, xs = np.nonzero(mask)
    weights = prob[mask]
    return (float(np.sum(xs * weights) / weights.sum()),
            float(np.sum(ys * weights) / weights.sum()))


def localise(image, histogram, threshold=0.5):
    """Backproject and return a dict with peak, centroid and peak value."""
    prob = backproject(image, histogram)
    x, y, value = locate_peak(prob)
    cx, cy = locate_centroid(prob, threshold=threshold)
    return {
        "probability_map": prob,
        "peak": (x, y),
        "peak_value": value,
        "centroid": (cx, cy),
    }
