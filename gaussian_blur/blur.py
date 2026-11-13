"""Gaussian blur, implemented from scratch (numpy only)."""

import numpy as np


def gaussian_kernel_1d(sigma, kernel_size=None):
    """Return a normalised 1D Gaussian kernel.

    Args:
        sigma: standard deviation of the Gaussian; must be > 0.
        kernel_size: odd positive integer, the kernel length. Defaults to
            ``2 * ceil(3 * sigma) + 1`` so the kernel covers +/-3 sigma.

    Returns:
        A 1D float64 array of length ``kernel_size`` that sums to 1.

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
            f"image must be 2D grayscale or (H, W, 3) RGB, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("image must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    return arr


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


def gaussian_blur(image, sigma=1.0, kernel_size=None):
    """Return ``image`` blurred with a Gaussian kernel.

    Args:
        image: 2D grayscale array, or (H, W, 3) RGB array.
        sigma: standard deviation of the Gaussian blur; must be > 0.
        kernel_size: odd positive integer kernel length; defaults to
            ``2 * ceil(3 * sigma) + 1``.

    Returns:
        A float64 array with the same shape as the input.

    Raises:
        ValueError: on a bad image (see :func:`_validate_image`) or bad
            ``sigma`` / ``kernel_size`` (see :func:`gaussian_kernel_1d`).
    """
    arr = _validate_image(image)
    kernel = gaussian_kernel_1d(sigma, kernel_size)
    if arr.ndim == 2:
        return _blur_2d(arr, kernel)
    return np.stack([_blur_2d(arr[:, :, c], kernel) for c in range(3)], axis=2)
