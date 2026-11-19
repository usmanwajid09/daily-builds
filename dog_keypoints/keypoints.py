"""Difference-of-Gaussians (DoG) keypoint detection, from scratch (numpy only).

A DoG image is the difference of the same image blurred at two scales
(sigma1 < sigma2): it responds strongly to blobs and corners whose size
matches the scale pair, and barely at all to flat areas. Keypoints are
the local maxima of |DoG| that clear a contrast threshold — the same
idea behind the first stage of SIFT.
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


def _validate_grayscale(image):
    """Return ``image`` as a 2D float64 array or raise ValueError."""
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"image must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"image must be 2D grayscale, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    return arr


def _validate_sigma(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value}")
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


def gaussian_blur(image, sigma):
    """Blur a grayscale image with a Gaussian of standard deviation ``sigma``."""
    image = _validate_grayscale(image)
    sigma = _validate_sigma(sigma, "sigma")
    return _blur_2d(image, gaussian_kernel_1d(sigma))


def difference_of_gaussians(image, sigma1=1.0, sigma2=1.6):
    """Return ``blur(image, sigma1) - blur(image, sigma2)``.

    Raises:
        ValueError: if either sigma is invalid or ``sigma2 <= sigma1``.
    """
    image = _validate_grayscale(image)
    sigma1 = _validate_sigma(sigma1, "sigma1")
    sigma2 = _validate_sigma(sigma2, "sigma2")
    if sigma2 <= sigma1:
        raise ValueError(f"sigma2 ({sigma2}) must be larger than sigma1 ({sigma1})")
    return gaussian_blur(image, sigma1) - gaussian_blur(image, sigma2)


def detect_keypoints(image, sigma1=1.0, sigma2=1.6, contrast_threshold=5.0,
                     max_keypoints=None):
    """Detect DoG keypoints in a grayscale image.

    A pixel is a keypoint when ``|DoG|`` there is at least
    ``contrast_threshold`` and is the strict local maximum of its 3x3
    neighbourhood (ties are resolved by keeping the first maximum, so
    flat plateaus yield no keypoints). Border pixels are never
    keypoints, since their neighbourhood is incomplete.

    Args:
        image: 2D grayscale array.
        sigma1, sigma2: blur scales, with ``sigma2 > sigma1``.
        contrast_threshold: minimum |DoG| response; must be >= 0.
        max_keypoints: optional cap; the strongest keypoints are kept.

    Returns:
        A list of ``(row, col, response)`` tuples, sorted by response
        (strongest first). ``response`` is the |DoG| value at the point.

    Raises:
        ValueError: on bad input, bad sigmas, a negative threshold, or
            a non-positive ``max_keypoints``.
    """
    dog = difference_of_gaussians(image, sigma1, sigma2)
    if isinstance(contrast_threshold, bool) or not isinstance(
            contrast_threshold, (int, float, np.floating)):
        raise ValueError(
            f"contrast_threshold must be a number >= 0, got {contrast_threshold!r}")
    contrast_threshold = float(contrast_threshold)
    if not np.isfinite(contrast_threshold) or contrast_threshold < 0:
        raise ValueError(
            f"contrast_threshold must be a number >= 0, got {contrast_threshold}")
    if max_keypoints is not None:
        if isinstance(max_keypoints, bool) or not isinstance(
                max_keypoints, (int, np.integer)) or int(max_keypoints) < 1:
            raise ValueError(
                f"max_keypoints must be a positive integer, got {max_keypoints!r}")
        max_keypoints = int(max_keypoints)
    response = np.abs(dog)
    height, width = response.shape
    keypoints = []
    if height < 3 or width < 3:
        return keypoints
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            value = response[row, col]
            if value < contrast_threshold or value == 0.0:
                continue
            window = response[row - 1:row + 2, col - 1:col + 2]
            # Keep only the first maximum of the window, so a flat
            # plateau does not report all nine of its pixels.
            if np.argmax(window) == 4 and value == window.max():
                keypoints.append((row, col, float(value)))
    keypoints.sort(key=lambda item: item[2], reverse=True)
    if max_keypoints is not None:
        keypoints = keypoints[:max_keypoints]
    return keypoints


def keypoint_count(keypoints):
    """Return how many keypoints a list from :func:`detect_keypoints` holds."""
    return len(list(keypoints))
