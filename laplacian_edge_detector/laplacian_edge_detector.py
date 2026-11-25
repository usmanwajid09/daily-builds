"""Laplacian edge detection, implemented from scratch (numpy only)."""

import numpy as np

_KERNELS = {
    "4-neighbour": np.array(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64
    ),
    "8-neighbour": np.array(
        [[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64
    ),
}


def _validate_image(image):
    """Return the image as a float64 array, raising ValueError on bad input."""
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")
    if image.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2D (grayscale) or 3D (RGB), got {image.ndim}D"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(
            f"RGB image must have 3 channels, got {image.shape[2]}"
        )
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


def laplacian(image, kernel="4-neighbour"):
    """Apply the 3x3 Laplacian operator to an image.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale or 3D RGB image (uint8 or float).
    kernel : str
        "4-neighbour" (cross kernel, centre weight -4) or "8-neighbour"
        (all neighbours, centre weight -8).

    Returns
    -------
    np.ndarray
        Laplacian response, same shape as the input, float64. For RGB
        images the operator is applied independently per channel.
    """
    if kernel not in _KERNELS:
        raise ValueError(
            f"unknown kernel {kernel!r}; expected one of {sorted(_KERNELS)}"
        )
    arr = _validate_image(image)
    kern = _KERNELS[kernel]
    if arr.ndim == 2:
        return _convolve3x3(arr, kern)
    out = np.empty_like(arr)
    for channel in range(3):
        out[:, :, channel] = _convolve3x3(arr[:, :, channel], kern)
    return out


def laplacian_magnitude(image, kernel="4-neighbour"):
    """Absolute value of the Laplacian response (edge strength)."""
    return np.abs(laplacian(image, kernel=kernel))


def zero_crossings(response, threshold=0.0):
    """Find zero crossings in a Laplacian response.

    A pixel is a zero crossing when the response changes sign between
    it and one of its 4-neighbours and at least one of the two pixels
    has magnitude >= ``threshold`` (which suppresses weak, noisy
    crossings).

    Parameters
    ----------
    response : np.ndarray
        2D Laplacian response (float).
    threshold : float
        Minimum magnitude for a crossing to count (non-negative).

    Returns
    -------
    np.ndarray
        Boolean mask, True where a zero crossing occurs.
    """
    if not isinstance(response, np.ndarray):
        raise ValueError("response must be a numpy array")
    if response.ndim != 2:
        raise ValueError(f"response must be 2D, got {response.ndim}D")
    if response.size == 0:
        raise ValueError("response must not be empty")
    if not np.issubdtype(response.dtype, np.number):
        raise ValueError("response must have a numeric dtype")
    resp = response.astype(np.float64)
    if not np.all(np.isfinite(resp)):
        raise ValueError("response must contain only finite values")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    padded = np.pad(resp, 1, mode="edge")
    centre = padded[1:-1, 1:-1]
    crossings = np.zeros(resp.shape, dtype=bool)
    neighbours = (
        padded[:-2, 1:-1],  # up
        padded[2:, 1:-1],  # down
        padded[1:-1, :-2],  # left
        padded[1:-1, 2:],  # right
    )
    for neighbour in neighbours:
        opposite_sign = centre * neighbour < 0
        strong_enough = (np.abs(centre) >= threshold) | (
            np.abs(neighbour) >= threshold
        )
        crossings |= opposite_sign & strong_enough
    return crossings


def edge_mask(image, threshold, kernel="4-neighbour"):
    """Binary edge mask: True where |Laplacian| >= ``threshold``.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale or 3D RGB image.
    threshold : float
        Non-negative magnitude threshold. For RGB images a pixel
        counts as an edge if any channel exceeds the threshold.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    magnitude = laplacian_magnitude(image, kernel=kernel)
    if magnitude.ndim == 3:
        magnitude = magnitude.max(axis=2)
    return magnitude >= threshold


def edge_pixel_count(image, threshold, kernel="4-neighbour"):
    """Number of pixels whose |Laplacian| is >= ``threshold``."""
    return int(edge_mask(image, threshold, kernel=kernel).sum())
