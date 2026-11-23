"""Prewitt edge detection, implemented from scratch (numpy only)."""

import numpy as np

PREWITT_X = np.array([[-1.0, 0.0, 1.0],
                      [-1.0, 0.0, 1.0],
                      [-1.0, 0.0, 1.0]], dtype=np.float64)
PREWITT_Y = np.array([[-1.0, -1.0, -1.0],
                      [0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0]], dtype=np.float64)


def _validate_image(image):
    """Return ``image`` as a float64 2D array or raise ValueError."""
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"image must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"image must be a 2D grayscale array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    return arr


def _convolve_3x3(image, kernel):
    """Convolve a 2D image with a 3x3 kernel using replicate padding."""
    padded = np.pad(image, 1, mode="edge")
    out = np.zeros_like(image)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * padded[i:i + image.shape[0], j:j + image.shape[1]]
    return out


def prewitt_gradients(image):
    """Return the (gx, gy) Prewitt gradient images of a grayscale image.

    Prewitt uses uniform weights (-1, -1, -1): it averages the three
    neighbouring rows/columns, so it smooths a little while differentiating.
    Output shape matches the input; borders use replicate padding.
    """
    image = _validate_image(image)
    return _convolve_3x3(image, PREWITT_X), _convolve_3x3(image, PREWITT_Y)


def prewitt_magnitude(image):
    """Return the Prewitt gradient magnitude image, same shape as input."""
    gx, gy = prewitt_gradients(image)
    return np.hypot(gx, gy)


def prewitt_direction(image):
    """Return the gradient direction in radians, in [-pi, pi]."""
    gx, gy = prewitt_gradients(image)
    return np.arctan2(gy, gx)


def prewitt_edges(image):
    """Return ``(magnitude, direction)`` for a grayscale image."""
    gx, gy = prewitt_gradients(image)
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def threshold_edges(magnitude, threshold):
    """Return a boolean edge mask: True where magnitude >= threshold."""
    mag = np.asarray(magnitude, dtype=np.float64)
    if mag.ndim != 2:
        raise ValueError(f"magnitude must be a 2D array, got shape {mag.shape}")
    if mag.size == 0:
        raise ValueError("magnitude must not be empty")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.floating)):
        raise ValueError(f"threshold must be a number, got {threshold!r}")
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be a non-negative finite number, got {threshold}")
    if (mag < 0).any():
        raise ValueError("magnitude must not contain negative values")
    return mag >= threshold


def edge_pixel_count(magnitude, threshold):
    """Count edge pixels: magnitude values >= ``threshold``."""
    return int(threshold_edges(magnitude, threshold).sum())
