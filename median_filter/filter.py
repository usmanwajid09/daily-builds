"""Median filtering for salt-and-pepper denoising, from scratch (numpy only).

Each pixel is replaced by the median of its neighbourhood window. A
single bright or dark spike (salt-and-pepper noise) is outvoted by its
neighbours, so it vanishes — while a real step edge survives, because
the median of a window straddling an edge still sits on one side of
it. That edge-preserving behaviour is why the median filter, not a
mean, is the standard cure for impulse noise.
"""

import numpy as np


def _validate_image(image):
    """Return ``image`` as a float64 array (2D or (H, W, 3)) or raise."""
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"image must be numeric, got dtype {arr.dtype}")
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] != 3):
        raise ValueError(
            f"image must be 2D grayscale or (H, W, 3) RGB, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    return arr


def _validate_kernel_size(kernel_size):
    if isinstance(kernel_size, bool) or not isinstance(
            kernel_size, (int, np.integer)):
        raise ValueError(
            f"kernel_size must be an odd positive integer, got {kernel_size!r}")
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be an odd positive integer, got {kernel_size}")
    return kernel_size


def _median_2d(channel, kernel_size):
    """Median-filter one 2D channel with replicate padding."""
    radius = kernel_size // 2
    height, width = channel.shape
    padded = np.pad(channel, radius, mode="edge")
    out = np.empty_like(channel)
    for row in range(height):
        for col in range(width):
            window = padded[row:row + kernel_size, col:col + kernel_size]
            out[row, col] = np.median(window)
    return out


def median_filter(image, kernel_size=3):
    """Median-filter a grayscale or RGB image.

    Each pixel is replaced by the median of its ``kernel_size`` x
    ``kernel_size`` neighbourhood (replicate padding at the border, so
    a flat border stays flat). ``kernel_size`` must be an odd positive
    integer; 1 returns the input unchanged.

    Returns:
        A float64 array with the same shape as the input.

    Raises:
        ValueError: on a bad image or an invalid ``kernel_size``.
    """
    image = _validate_image(image)
    kernel_size = _validate_kernel_size(kernel_size)
    if kernel_size == 1:
        return image.copy()
    if image.ndim == 2:
        return _median_2d(image, kernel_size)
    return np.stack(
        [_median_2d(image[:, :, c], kernel_size) for c in range(3)], axis=2)


def salt_and_pepper_pixel_count(image, low=0.0, high=255.0):
    """Count pixels sitting exactly at the salt (``high``) or pepper (``low``) value."""
    arr = _validate_image(image)
    return int(np.count_nonzero((arr == low) | (arr == high)))
