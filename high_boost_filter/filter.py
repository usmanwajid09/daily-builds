"""High-boost image sharpening, from scratch (numpy only).

High-boost filtering sharpens an image by adding a scaled copy of its
high-frequency detail back onto itself::

    sharpened = original + k * (original - blurred)

The term in brackets is the unsharp mask — everything the blur removed,
i.e. edges and fine detail. ``k = 1`` is classic unsharp masking;
larger ``k`` boosts edges harder. Output is clipped to the 8-bit range
0-255.
"""

import numpy as np


def gaussian_kernel_1d(sigma, kernel_size=None):
    """Return a normalised 1D Gaussian kernel.

    Args:
        sigma: standard deviation; must be a positive finite number.
        kernel_size: odd positive integer; defaults to
            ``2 * ceil(3 * sigma) + 1``.

    Raises:
        ValueError: if ``sigma`` is not a positive finite number or
            ``kernel_size`` is not an odd positive integer.
    """
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float, np.floating)):
        raise ValueError(f"sigma must be a positive number, got {sigma!r}")
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be a positive number, got {sigma}")
    if kernel_size is None:
        kernel_size = 2 * int(np.ceil(3 * sigma)) + 1
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, (int, np.integer)):
        raise ValueError(f"kernel_size must be an odd positive integer, got {kernel_size!r}")
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be an odd positive integer, got {kernel_size}")
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


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


def _validate_number(value, name, minimum=None, exclusive_minimum=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if minimum is not None:
        if exclusive_minimum and value <= minimum:
            raise ValueError(f"{name} must be > {minimum}, got {value}")
        if not exclusive_minimum and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _blur_2d(channel, kernel):
    """Blur one 2D channel with a separable kernel (replicate padding)."""
    radius = len(kernel) // 2
    height, width = channel.shape
    padded = np.pad(channel, radius, mode="edge")
    tmp = np.zeros_like(channel)
    for i in range(len(kernel)):
        tmp += kernel[i] * padded[radius:radius + height, i:i + width]
    padded = np.pad(tmp, radius, mode="edge")
    out = np.zeros_like(channel)
    for i in range(len(kernel)):
        out += kernel[i] * padded[i:i + height, radius:radius + width]
    return out


def _blur(image, sigma):
    kernel = gaussian_kernel_1d(sigma)
    if image.ndim == 2:
        return _blur_2d(image, kernel)
    return np.stack([_blur_2d(image[:, :, c], kernel) for c in range(3)], axis=2)


def sharpening_mask(image, sigma=1.0):
    """Return ``image - blur(image, sigma)``: the detail the blur removed."""
    image = _validate_image(image)
    sigma = _validate_number(sigma, "sigma", minimum=0.0, exclusive_minimum=True)
    return image - _blur(image, sigma)


def high_boost_filter(image, k=2.0, sigma=1.0):
    """Sharpen ``image`` with high-boost filtering.

    Returns ``clip(image + k * (image - blur(image, sigma)), 0, 255)``
    as a float64 array with the same shape as the input. ``k = 0``
    returns the input unchanged; ``k = 1`` is classic unsharp masking.

    Raises:
        ValueError: on a bad image, ``k < 0``, or ``sigma <= 0``.
    """
    image = _validate_image(image)
    k = _validate_number(k, "k", minimum=0.0)
    sigma = _validate_number(sigma, "sigma", minimum=0.0, exclusive_minimum=True)
    mask = image - _blur(image, sigma)
    return np.clip(image + k * mask, 0.0, 255.0)
