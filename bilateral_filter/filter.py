"""Bilateral (edge-preserving) smoothing from scratch (numpy only).

A bilateral filter smooths flat regions like a Gaussian blur but keeps
sharp edges: each neighbour's weight is the product of a spatial
Gaussian (how close the pixel is) and a range Gaussian (how similar
its intensity is).  Works on grayscale images and RGB images.
"""

import numpy as np

__all__ = ["bilateral_filter", "gaussian_kernel_2d"]


def gaussian_kernel_2d(diameter, sigma):
    """Return a normalised 2D Gaussian kernel.

    Parameters
    ----------
    diameter : int
        Kernel size; must be a positive odd integer.
    sigma : float
        Gaussian standard deviation; must be positive.
    """
    if not isinstance(diameter, (int, np.integer)) or diameter <= 0 \
            or diameter % 2 == 0:
        raise ValueError("diameter must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = diameter // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def bilateral_filter(image, diameter=5, sigma_color=25.0, sigma_space=5.0):
    """Apply bilateral smoothing to a grayscale or RGB image.

    Parameters
    ----------
    image : 2D array (grayscale) or 3D array with 3 channels (RGB)
        Pixel values in any range; the output is a weighted average of
        the input, so it stays in the same range.
    diameter : int
        Neighbourhood size; must be a positive odd integer.
    sigma_color : float
        Range sigma: how different two intensities may be and still
        mix.  Must be positive.  Small values preserve edges more.
    sigma_space : float
        Spatial sigma: how far neighbours reach.  Must be positive.

    Returns
    -------
    float64 array of the same shape as ``image``.
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2D grayscale or 3D RGB, got shape {arr.shape}")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError(
            f"RGB image must have 3 channels, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    if not isinstance(diameter, (int, np.integer)) or diameter <= 0 \
            or diameter % 2 == 0:
        raise ValueError("diameter must be a positive odd integer")
    if sigma_color <= 0:
        raise ValueError("sigma_color must be positive")
    if sigma_space <= 0:
        raise ValueError("sigma_space must be positive")

    gray = arr.ndim == 2
    radius = diameter // 2
    spatial = gaussian_kernel_2d(diameter, sigma_space)
    if gray:
        padded = np.pad(arr, radius, mode="reflect")
    else:
        padded = np.pad(arr, ((radius, radius), (radius, radius), (0, 0)),
                        mode="reflect")
    height, width = arr.shape[:2]
    out = np.empty_like(arr)
    for y in range(height):
        for x in range(width):
            if gray:
                window = padded[y:y + diameter, x:x + diameter]
                centre = arr[y, x]
                range_w = np.exp(-((window - centre) ** 2) /
                                 (2.0 * sigma_color ** 2))
                weights = spatial * range_w
                out[y, x] = np.sum(weights * window) / np.sum(weights)
            else:
                window = padded[y:y + diameter, x:x + diameter, :]
                centre = arr[y, x, :]
                diff2 = np.sum((window - centre) ** 2, axis=2)
                range_w = np.exp(-diff2 / (2.0 * sigma_color ** 2))
                weights = spatial * range_w
                out[y, x, :] = (np.sum(weights[..., None] * window,
                                       axis=(0, 1)) / np.sum(weights))
    return out
