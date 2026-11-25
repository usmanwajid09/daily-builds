"""Shi-Tomasi corner detection, implemented from scratch (numpy only)."""

import numpy as np

_SOBEL_X = np.array(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float64
)
_SOBEL_Y = np.array(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float64
)


def _validate_image(image):
    """Return the image as a float64 2D array, raising ValueError on bad input."""
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")
    if image.ndim != 2:
        raise ValueError(f"image must be 2D (grayscale), got {image.ndim}D")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("image must have a numeric dtype")
    arr = image.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("image must contain only finite values")
    return arr


def _convolve3x3(image, kernel):
    """Convolve a 2D array with a 3x3 kernel using replicate padding."""
    padded = np.pad(image, 1, mode="edge")
    out = np.zeros_like(image, dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            weight = kernel[di, dj]
            if weight != 0.0:
                out += weight * padded[di:di + image.shape[0], dj:dj + image.shape[1]]
    return out


def _box_mean(image, radius):
    """Mean over a (2*radius+1) square window, via an integral image."""
    if radius == 0:
        return image.astype(np.float64)
    padded = np.pad(image, radius, mode="edge")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1))
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    r = radius
    total = (
        integral[2 * r + 1:, 2 * r + 1:]
        - integral[: -2 * r - 1, 2 * r + 1:]
        - integral[2 * r + 1:, : -2 * r - 1]
        + integral[: -2 * r - 1, : -2 * r - 1]
    )
    return total / float((2 * r + 1) ** 2)


def _validate_window(window_size):
    if not isinstance(window_size, int) or isinstance(window_size, bool):
        raise ValueError("window_size must be an integer")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")


def corner_response(image, window_size=3):
    """Shi-Tomasi corner response: the smaller eigenvalue of the
    structure tensor at each pixel.

    The structure tensor sums the outer product of the Sobel gradient
    over a ``window_size`` x ``window_size`` neighbourhood. A pixel is
    a corner when both eigenvalues are large, i.e. when the smaller
    one is large — flat regions score ~0, edges score ~0 (one
    eigenvalue is small), corners score high.

    Returns
    -------
    np.ndarray
        Non-negative float64 response, same shape as the input.
    """
    _validate_window(window_size)
    arr = _validate_image(image)
    gx = _convolve3x3(arr, _SOBEL_X)
    gy = _convolve3x3(arr, _SOBEL_Y)
    radius = window_size // 2
    sxx = _box_mean(gx * gx, radius)
    syy = _box_mean(gy * gy, radius)
    sxy = _box_mean(gx * gy, radius)
    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    discriminant = np.maximum(trace * trace / 4.0 - det, 0.0)
    response = trace / 2.0 - np.sqrt(discriminant)
    return np.maximum(response, 0.0)


def detect_corners(image, max_corners=100, quality_level=0.01,
                   min_distance=5, window_size=3):
    """Detect Shi-Tomasi corners in a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image (uint8 or float).
    max_corners : int
        Maximum number of corners to return (strongest first).
    quality_level : float
        Fraction of the strongest response a pixel must reach to be a
        candidate, in (0, 1].
    min_distance : int
        Minimum pixel distance between returned corners (non-maximum
        suppression radius).
    window_size : int
        Odd side length of the structure-tensor window.

    Returns
    -------
    list of (row, col, response)
        Corners sorted by descending response strength.
    """
    _validate_window(window_size)
    if not isinstance(max_corners, int) or isinstance(max_corners, bool):
        raise ValueError("max_corners must be an integer")
    if max_corners < 1:
        raise ValueError("max_corners must be at least 1")
    if not (0.0 < quality_level <= 1.0):
        raise ValueError("quality_level must be in (0, 1]")
    if not isinstance(min_distance, int) or isinstance(min_distance, bool):
        raise ValueError("min_distance must be an integer")
    if min_distance < 1:
        raise ValueError("min_distance must be at least 1")

    response = corner_response(image, window_size=window_size)
    strongest = response.max()
    if strongest <= 0.0:
        return []
    threshold = quality_level * strongest
    rows, cols = np.nonzero(response >= threshold)
    order = np.argsort(-response[rows, cols])
    corners = []
    for idx in order:
        row, col = int(rows[idx]), int(cols[idx])
        if all(
            (row - r) ** 2 + (col - c) ** 2 >= min_distance ** 2
            for r, c, _ in corners
        ):
            corners.append((row, col, float(response[row, col])))
            if len(corners) >= max_corners:
                break
    return corners
