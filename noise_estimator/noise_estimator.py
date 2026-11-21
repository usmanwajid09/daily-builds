"""Image noise estimation and a small denoising guide (numpy only).

Two Laplacian-based estimators of the Gaussian noise standard
deviation ``sigma`` of a grayscale image:

* ``immerkaer`` (Immerkaer, 1996): the image is convolved with the
  3x3 mask ``[[1, -2, 1], [-2, 4, -2], [1, -2, 1]]``. Its rows and
  columns each sum to zero, so it annihilates smooth and linear
  image content, while white noise of std ``sigma`` gives a
  response of std ``6 * sigma``; the mean absolute response
  therefore estimates ``sigma`` as
  ``sqrt(pi / 2) * mean(|response|) / 6``.
* ``mad``: a robust variant using the median absolute deviation of
  the same Laplacian response, ``median(|response|) / 0.6745 / 6``,
  which is less sensitive to the strong responses that real edges
  produce.

Both are single-image estimators — no clean reference is needed.
``recommend_denoising`` then turns the estimate into a concrete,
actionable suggestion (method + parameters), and
``gaussian_denoise`` applies the light Gaussian blur it recommends
for low-to-moderate noise.
"""

import numpy as np

__all__ = [
    "estimate_sigma",
    "gaussian_denoise",
    "noise_level",
    "psnr",
    "recommend_denoising",
]

_LAPLACIAN = np.array([[1.0, -2.0, 1.0],
                       [-2.0, 4.0, -2.0],
                       [1.0, -2.0, 1.0]])


def _check_image(image, name="image"):
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D grayscale image, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        raise ValueError(
            f"{name} must be at least 3x3 for the Laplacian mask, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _laplacian_response(image):
    """Convolution with the 3x3 Immerkaer mask (valid region only)."""
    padded = np.pad(image, 1, mode="edge")
    response = np.zeros_like(image)
    for dy in range(3):
        for dx in range(3):
            response += _LAPLACIAN[dy, dx] * padded[dy:dy + image.shape[0],
                                                    dx:dx + image.shape[1]]
    return response[1:-1, 1:-1]


def estimate_sigma(image, method="immerkaer"):
    """Estimate the Gaussian noise std ``sigma`` of a grayscale image.

    ``method`` is ``"immerkaer"`` (mean absolute Laplacian response)
    or ``"mad"`` (median absolute deviation of the same response).
    Returns a non-negative float.
    """
    image = _check_image(image)
    if method not in ("immerkaer", "mad"):
        raise ValueError(
            f"method must be 'immerkaer' or 'mad', got {method!r}")
    response = _laplacian_response(image)
    if method == "immerkaer":
        sigma = float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(response)) / 6.0)
    else:
        sigma = float(np.median(np.abs(response)) / 0.6745 / 6.0)
    return max(sigma, 0.0)


def noise_level(sigma):
    """Plain-language noise band for an estimated ``sigma`` (0..255 scale)."""
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma!r}")
    if sigma < 2:
        return "very low"
    if sigma < 5:
        return "low"
    if sigma < 10:
        return "moderate"
    if sigma < 20:
        return "high"
    return "very high"


def recommend_denoising(sigma):
    """Denoising suggestion for an estimated ``sigma`` (0..255 scale).

    Returns a dict with the noise ``level``, the recommended
    ``method``, its ``parameters`` and a short ``note``. The advice
    assumes 8-bit (0..255) images; for a 0..1 image, scale sigma by
    255 first.
    """
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma!r}")
    level = noise_level(sigma)
    if sigma < 2:
        return {"level": level, "method": "none",
                "parameters": {},
                "note": "Noise is negligible; denoising would only blur detail."}
    if sigma < 5:
        return {"level": level, "method": "gaussian_blur",
                "parameters": {"sigma": 0.8},
                "note": "A light Gaussian blur (sigma 0.8) removes most of it."}
    if sigma < 10:
        return {"level": level, "method": "gaussian_blur",
                "parameters": {"sigma": 1.2},
                "note": "A Gaussian blur (sigma 1.2) is a good trade-off here."}
    if sigma < 20:
        return {"level": level, "method": "bilateral_filter",
                "parameters": {"diameter": 9, "sigma_color": 2 * sigma,
                               "sigma_space": 2.0},
                "note": "Noise is strong; prefer an edge-preserving "
                        "bilateral filter over a plain blur."}
    return {"level": level, "method": "non_local_means",
            "parameters": {"patch_size": 7, "search_window": 21,
                           "filter_strength": sigma},
            "note": "Noise is very strong; use non-local means and expect "
                    "some loss of fine detail."}


def _gaussian_kernel_1d(sigma):
    radius = max(1, int(np.ceil(3 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_denoise(image, sigma=1.0):
    """Separable Gaussian blur (a light denoise for low/moderate noise)."""
    image = _check_image(image)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    kernel = _gaussian_kernel_1d(float(sigma))
    radius = len(kernel) // 2
    padded = np.pad(image, radius, mode="edge")
    blurred = np.zeros_like(image)
    for i, weight in enumerate(kernel):
        blurred += weight * padded[i:i + image.shape[0], radius:-radius or None]
    padded = np.pad(blurred, radius, mode="edge")
    out = np.zeros_like(image)
    for i, weight in enumerate(kernel):
        out += weight * padded[radius:-radius or None, i:i + image.shape[1]]
    return out


def psnr(original, restored, data_range=None):
    """Peak signal-to-noise ratio in dB between two images."""
    original = _check_image(original, name="original")
    restored = np.asarray(restored, dtype=float)
    if restored.shape != original.shape:
        raise ValueError(
            f"restored must have the same shape as original, got {restored.shape} "
            f"vs {original.shape}")
    if data_range is None:
        data_range = float(original.max() - original.min()) or 1.0
    mse = float(np.mean((original - restored) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))
