"""Roberts cross edge detection, implemented from scratch (numpy only).

The Roberts cross operator is the smallest gradient operator: two 2x2
kernels that difference diagonally adjacent pixels. Because a 2x2 kernel
has no centre pixel, the output is one row and one column smaller than
the input — element (i, j) of the result describes the 2x2 block whose
top-left corner is (i, j).
"""

import numpy as np

ROBERTS_X = np.array([[1.0, 0.0],
                      [0.0, -1.0]], dtype=np.float64)
ROBERTS_Y = np.array([[0.0, 1.0],
                      [-1.0, 0.0]], dtype=np.float64)


def _validate_image(image):
    """Return ``image`` as a float64 2D array or raise ValueError."""
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"image must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"image must be a 2D grayscale array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(
            f"image must be at least 2x2 for the Roberts operator, got shape {arr.shape}"
        )
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    return arr


def roberts_gradients(image):
    """Return the (gx, gy) Roberts gradient images, shape (H-1, W-1)."""
    image = _validate_image(image)
    gx = image[:-1, :-1] - image[1:, 1:]
    gy = image[:-1, 1:] - image[1:, :-1]
    return gx, gy


def roberts_magnitude(image):
    """Return the Roberts gradient magnitude, shape (H-1, W-1)."""
    gx, gy = roberts_gradients(image)
    return np.hypot(gx, gy)


def roberts_direction(image):
    """Return the gradient direction in radians, in [-pi, pi]."""
    gx, gy = roberts_gradients(image)
    return np.arctan2(gy, gx)


def roberts_edges(image):
    """Return ``(magnitude, direction)``, each of shape (H-1, W-1)."""
    gx, gy = roberts_gradients(image)
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
