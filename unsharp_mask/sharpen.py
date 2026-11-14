"""Unsharp masking / image sharpening, implemented from scratch (numpy only).

Unsharp masking sharpens an image by blurring it, subtracting the blur
from the original to isolate the fine detail, and adding that detail back,
amplified:

    sharpened = image + amount * (image - blurred)

A ``threshold`` can suppress the boost where the detail is tiny, so flat
regions (and their noise) are left untouched.
"""

import numpy as np


def _validate_image(image):
    """Return ``image`` as a float64 array (2D or (H, W, 3)) or raise."""
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"image must be numeric, got dtype {arr.dtype}")
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] != 3):
        raise ValueError(
            f"image must be 2D grayscale or (H, W, 3) RGB, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("image must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    return arr


def _validate_positive_finite(value, name):
    """Return ``value`` as a float, or raise ValueError if not > 0 and finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value}")
    return value


def _validate_nonnegative_finite(value, name):
    """Return ``value`` as a float, or raise ValueError if not >= 0 and finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative number, got {value}")
    return value


def _gaussian_kernel_1d(sigma):
    """Return a normalised 1D Gaussian kernel covering +/-3 sigma."""
    kernel_size = 2 * int(np.ceil(3 * sigma)) + 1
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _blur_2d(channel, kernel):
    """Blur one 2D channel with a separable kernel (replicate padding)."""
    radius = len(kernel) // 2
    # Convolve along rows, then along columns: separable means two 1D
    # passes instead of one 2D pass (2k ops per pixel instead of k^2).
    # Each pass pads only the axis it convolves along.
    padded = np.pad(channel, ((0, 0), (radius, radius)), mode="edge")
    tmp = np.zeros_like(channel, dtype=np.float64)
    for i, weight in enumerate(kernel):
        tmp += weight * padded[:, i:i + channel.shape[1]]
    padded = np.pad(tmp, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(channel, dtype=np.float64)
    for j, weight in enumerate(kernel):
        out += weight * padded[j:j + channel.shape[0], :]
    return out


def gaussian_blur(image, sigma=1.0):
    """Return ``image`` blurred with a Gaussian kernel.

    Args:
        image: 2D grayscale array, or (H, W, 3) RGB array.
        sigma: standard deviation of the Gaussian blur; must be > 0.
            The kernel size is ``2 * ceil(3 * sigma) + 1`` and the kernel
            is normalised to sum to 1. Replicate (edge) padding is used
            at the borders, so a constant image is returned unchanged.

    Returns:
        A float64 array with the same shape as the input.

    Raises:
        ValueError: on a bad image (see :func:`_validate_image`) or a
            non-positive / non-finite ``sigma``.
    """
    arr = _validate_image(image)
    sigma = _validate_positive_finite(sigma, "sigma")
    kernel = _gaussian_kernel_1d(sigma)
    if arr.ndim == 2:
        return _blur_2d(arr, kernel)
    return np.stack([_blur_2d(arr[:, :, c], kernel) for c in range(3)], axis=2)


def unsharp_mask(image, amount=1.0, radius=1.0, threshold=0.0):
    """Return ``image`` sharpened by unsharp masking.

    The blurred image is subtracted from the original to get the detail
    (``image - blurred``), which is amplified by ``amount`` and added
    back. The boost is only applied where ``|image - blurred|`` exceeds
    ``threshold``; elsewhere the original pixel is kept, so flat regions
    stay untouched when ``threshold > 0``.

    Args:
        image: 2D grayscale array, or (H, W, 3) RGB array.
        amount: strength of the sharpening; must be >= 0. ``0`` returns
            the image unchanged.
        radius: sigma of the Gaussian blur; must be > 0. Larger values
            sharpen broader (lower-frequency) detail.
        threshold: minimum detail magnitude (in intensity units) that
            gets boosted; must be >= 0. Detail at or below the threshold
            is left as the original pixel.

    Returns:
        A float64 array with the same shape as the input, clipped to
        ``[0, 255]``.

    Raises:
        ValueError: on a bad image (see :func:`_validate_image`) or a
            bad ``amount`` / ``radius`` / ``threshold``.
    """
    arr = _validate_image(image)
    amount = _validate_nonnegative_finite(amount, "amount")
    radius = _validate_positive_finite(radius, "radius")
    threshold = _validate_nonnegative_finite(threshold, "threshold")
    if amount == 0:
        return np.clip(arr, 0, 255)
    blurred = gaussian_blur(arr, sigma=radius)
    detail = arr - blurred
    boost = np.where(np.abs(detail) > threshold, amount * detail, 0.0)
    return np.clip(arr + boost, 0, 255)
