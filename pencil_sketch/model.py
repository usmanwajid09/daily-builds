"""Pencil-sketch effect from a grayscale or RGB image.

Classic recipe (numpy only, no OpenCV):

1. Convert to grayscale (luminance weights).
2. Invert the grayscale image.
3. Gaussian-blur the inverted image.
4. Colour-dodge blend the grayscale image with the blurred inverted
   image — flat bright areas stay white, edges turn dark.

Public API: :func:`pencil_sketch`, :func:`gaussian_blur`,
:func:`rgb_to_grayscale`, :func:`color_dodge_blend`.
"""

import math

import numpy as np

__all__ = [
    "gaussian_blur",
    "rgb_to_grayscale",
    "color_dodge_blend",
    "pencil_sketch",
]

_LUMINANCE = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def _validate_sigma(sigma):
    """Return ``sigma`` as a float, or raise for a bad value."""
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float, np.floating, np.integer)):
        raise TypeError(f"sigma must be a number, got {type(sigma).__name__}")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be a positive finite number, got {sigma}")
    return sigma


def _gaussian_kernel_1d(sigma):
    """1D Gaussian kernel covering +/-3 sigma, normalised to sum 1."""
    radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_blur(image, sigma=1.0):
    """Separable Gaussian blur of a 2D float array.

    Uses replicate (edge) padding so the output keeps the input shape
    and flat regions stay flat right up to the border. The kernel
    covers +/-3 sigma in each direction.
    """
    sigma = _validate_sigma(sigma)
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.ndim != 2:
        raise ValueError(f"gaussian_blur expects a 2D array, got shape {image.shape}")
    if image.size == 0:
        raise ValueError("image must be non-empty")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"image must be numeric, got dtype {image.dtype}")

    arr = image.astype(np.float64, copy=False)
    kernel = _gaussian_kernel_1d(sigma)
    radius = len(kernel) // 2

    # Horizontal pass, then vertical pass, both with replicate padding.
    padded = np.pad(arr, ((0, 0), (radius, radius)), mode="edge")
    blurred = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), axis=1, arr=padded
    )
    padded = np.pad(blurred, ((radius, radius), (0, 0)), mode="edge")
    blurred = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="valid"), axis=0, arr=padded
    )
    return blurred


def rgb_to_grayscale(image):
    """Convert an RGB image to 2D grayscale with luminance weights.

    Uses weights (0.299, 0.587, 0.114). A 2D input is passed through
    unchanged (as float64); a 3D input must have exactly 3 channels.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.size == 0:
        raise ValueError("image must be non-empty")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"image must be numeric, got dtype {image.dtype}")
    if image.ndim == 2:
        return image.astype(np.float64, copy=False)
    if image.ndim == 3 and image.shape[2] == 3:
        return np.tensordot(
            image.astype(np.float64, copy=False), _LUMINANCE, axes=([2], [0])
        )
    raise ValueError(
        f"image must be 2D grayscale or 3D RGB with 3 channels, got shape {image.shape}"
    )


def color_dodge_blend(base, blend):
    """Colour-dodge blend of two float arrays in [0, 255].

    ``result = base / (255 - blend) * 255``, clipped to [0, 255].
    Where ``blend == 255`` the denominator is zero and the result is
    defined as 255 (full white).
    """
    if not isinstance(base, np.ndarray) or not isinstance(blend, np.ndarray):
        raise TypeError("base and blend must be numpy arrays")
    if base.shape != blend.shape:
        raise ValueError(f"base and blend must have the same shape, got {base.shape} and {blend.shape}")
    base_f = base.astype(np.float64, copy=False)
    blend_f = blend.astype(np.float64, copy=False)
    denom = 255.0 - blend_f
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom == 0, 255.0, base_f / denom * 255.0)
    return np.clip(result, 0.0, 255.0)


def pencil_sketch(image, sigma=1.0):
    """Return a uint8 grayscale pencil-sketch of ``image``.

    Accepts a 2D grayscale or 3D RGB numpy array (any numeric dtype).
    Pipeline: grayscale -> invert -> Gaussian blur the inverted image
    -> colour-dodge blend of the grayscale image with the blurred
    inverted image. The result is a uint8 array in [0, 255] with the
    same height and width as the input.

    Notes on flat regions: a flat white image stays white, and a flat
    black image stays black — colour-dodge of base 0 is 0 no matter
    how bright the blend layer is.
    """
    sigma = _validate_sigma(sigma)
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.size == 0:
        raise ValueError("image must be non-empty")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"image must be numeric, got dtype {image.dtype}")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError(
            f"image must be 2D grayscale or 3D RGB with 3 channels, got shape {image.shape}"
        )

    gray = rgb_to_grayscale(image)
    inverted = 255.0 - gray
    blurred_inverted = gaussian_blur(inverted, sigma=sigma)
    sketch = color_dodge_blend(gray, blurred_inverted)
    return np.clip(np.rint(sketch), 0, 255).astype(np.uint8)
