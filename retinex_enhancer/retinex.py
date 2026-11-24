"""Retinex illumination enhancement: single- and multi-scale Retinex.

Implemented from scratch with numpy only — no OpenCV. Retinex models
human colour constancy: an image is the product of illumination and
reflectance, so reflectance (the detail we want) is recovered as
log(image) - log(image blurred with a Gaussian). Multi-scale Retinex
averages several blur scales; MSRCR adds colour restoration so colours
do not wash out.
"""

import numpy as np

__all__ = [
    "gaussian_kernel_1d",
    "gaussian_blur",
    "single_scale_retinex",
    "multi_scale_retinex",
    "multi_scale_retinex_cr",
    "enhance",
]


def _validate_image(image):
    if image is None:
        raise ValueError("image must not be None")
    try:
        arr = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("image must be a numeric array") from None
    if arr.ndim not in (2, 3):
        raise ValueError(f"image must be 2D grayscale or 3D RGB, got {arr.ndim}D")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError(f"RGB image must have 3 channels, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("image must contain only finite values")
    if np.any(arr < 0) or np.any(arr > 255):
        raise ValueError("image values must lie in [0, 255]")
    return arr


def _validate_sigma(sigma, name="sigma"):
    if not isinstance(sigma, (int, float, np.floating)) or isinstance(sigma, bool):
        raise ValueError(f"{name} must be a number, got {sigma!r}")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"{name} must be positive, got {sigma}")
    return float(sigma)


def _validate_sigmas(sigmas):
    """Return sigmas as a validated non-empty list of positive floats."""
    if sigmas is None:
        raise ValueError("sigmas must not be None")
    try:
        sigmas = [float(s) for s in sigmas]
    except (TypeError, ValueError):
        raise ValueError("sigmas must be a sequence of numbers") from None
    if len(sigmas) == 0:
        raise ValueError("sigmas must be non-empty")
    for s in sigmas:
        _validate_sigma(s, name="sigma in sigmas")
    return sigmas


def _stretch_to_255(values):
    """Min-max stretch a float array to [0, 255]; constant input -> 128."""
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.full(values.shape, 128.0, dtype=np.float64)
    return (values - lo) / (hi - lo) * 255.0


def gaussian_kernel_1d(sigma):
    """Return a 1D Gaussian kernel covering +/-3 sigma; sums to 1."""
    sigma = _validate_sigma(sigma)
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def _convolve_1d(image, kernel, axis):
    """Separable 1D convolution with edge (replicate) padding."""
    pad = len(kernel) // 2
    padded = np.pad(image, [(pad, pad) if i == axis else (0, 0)
                            for i in range(image.ndim)], mode="edge")
    out = np.zeros_like(image, dtype=np.float64)
    for k, weight in enumerate(kernel):
        sl = [slice(None)] * image.ndim
        sl[axis] = slice(k, k + image.shape[axis])
        out += weight * padded[tuple(sl)]
    return out


def gaussian_blur(image, sigma):
    """Gaussian blur of a grayscale or RGB image (separable, edge padding)."""
    arr = _validate_image(image)
    sigma = _validate_sigma(sigma)
    kernel = gaussian_kernel_1d(sigma)
    blurred = _convolve_1d(arr, kernel, axis=0)
    blurred = _convolve_1d(blurred, kernel, axis=1)
    return blurred


def single_scale_retinex(image, sigma=80.0):
    """Single-scale Retinex; returns a float image in [0, 255].

    SSR = log(I + 1) - log(blur(I, sigma) + 1), min-max stretched to
    [0, 255]. A flat image maps to a constant mid-grey (128).
    """
    arr = _validate_image(image)
    sigma = _validate_sigma(sigma)
    log_img = np.log1p(arr)
    log_blur = np.log1p(gaussian_blur(arr, sigma))
    return _stretch_to_255(log_img - log_blur)


def multi_scale_retinex(image, sigmas=(15.0, 80.0, 250.0)):
    """Multi-scale Retinex: mean of single-scale Retinex over ``sigmas``.

    Returns a float image in [0, 255]. ``sigmas`` must be a non-empty
    sequence of positive numbers.
    """
    arr = _validate_image(image)
    if sigmas is None:
        raise ValueError("sigmas must not be None")
    try:
        sigmas = [float(s) for s in sigmas]
    except (TypeError, ValueError):
        raise ValueError("sigmas must be a sequence of numbers") from None
    if len(sigmas) == 0:
        raise ValueError("sigmas must be non-empty")
    for s in sigmas:
        _validate_sigma(s, name="sigma in sigmas")
    log_img = np.log1p(arr)
    total = np.zeros_like(arr)
    for sigma in sigmas:
        total += log_img - np.log1p(gaussian_blur(arr, sigma))
    return _stretch_to_255(total / len(sigmas))


def multi_scale_retinex_cr(image, sigmas=(15.0, 80.0, 250.0),
                           alpha=125.0, beta=46.0, gain=255.0, offset=0.0):
    """MSR with colour restoration (MSRCR); returns a float image in [0, 255].

    The log-domain MSR is multiplied per channel by a colour-restoration
    factor beta * log(1 + alpha * I / sum_c I_c), which boosts channels
    that dominate the pixel's colour, then min-max stretched to [0, 255].
    ``gain``/``offset`` apply a final linear tweak (defaults leave the
    stretched result unchanged). Grayscale images pass through with the
    MSR result unchanged. ``alpha`` and ``beta`` must be positive;
    ``gain`` must be positive.
    """
    arr = _validate_image(image)
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"alpha and beta must be positive, got alpha={alpha}, beta={beta}")
    if gain <= 0:
        raise ValueError(f"gain must be positive, got {gain}")
    log_img = np.log1p(arr)
    msr_log = np.zeros_like(arr)
    for sigma in _validate_sigmas(sigmas):
        msr_log += log_img - np.log1p(gaussian_blur(arr, sigma))
    msr_log /= len(_validate_sigmas(sigmas))
    if arr.ndim == 2:
        combined = msr_log
    else:
        channel_sum = np.sum(arr, axis=2, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            colour_factor = beta * np.log1p(alpha * arr / np.maximum(channel_sum, 1e-6))
        colour_factor = np.where(np.isfinite(colour_factor), colour_factor, 0.0)
        combined = msr_log * colour_factor
    stretched = _stretch_to_255(combined)
    return np.clip(stretched * (gain / 255.0) + offset, 0.0, 255.0)


def enhance(image, method="msr", **kwargs):
    """Enhance an image; returns a uint8 array in [0, 255].

    ``method`` is ``"ssr"``, ``"msr"`` or ``"msrcr"``. Extra keyword
    arguments are forwarded to the chosen Retinex function.
    """
    arr = _validate_image(image)
    if method == "ssr":
        result = single_scale_retinex(arr, **kwargs)
    elif method == "msr":
        result = multi_scale_retinex(arr, **kwargs)
    elif method == "msrcr":
        result = multi_scale_retinex_cr(arr, **kwargs)
    else:
        raise ValueError(f"unknown method {method!r}: choose 'ssr', 'msr' or 'msrcr'")
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)
