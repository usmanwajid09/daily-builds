"""Guided filter edge-preserving smoothing, from scratch (numpy only).

Implements the guided filter of He, Sun and Tang (2013): in each
local window the output is a linear transform of the guide image,
``q = a * I + b``, where ``a`` and ``b`` come from a ridge regression
of the input on the guide inside the window. Averaging the per-window
models over all windows that contain a pixel gives the final output.
Flat guide regions smooth strongly (``a`` ~ 0); across a strong
guide edge ``a`` ~ 1, so the edge passes through unsmoothed.
"""

import numpy as np


def _validate_image(image, name):
    """Return the image as a float64 2D array, raising ValueError on bad input."""
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if image.ndim != 2:
        raise ValueError(f"{name} must be 2D (grayscale), got {image.ndim}D")
    if image.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"{name} must have a numeric dtype")
    arr = image.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _box_mean(image, radius):
    """Mean over a (2*radius+1) square window, via an integral image
    (replicate padding at the borders)."""
    if radius == 0:
        return image.astype(np.float64)
    padded = np.pad(image, radius, mode="edge")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1))
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    r = radius
    total = (
        integral[2 * r + 1:, 2 * r + 1:]
        - integral[: -2 * r - 1, 2 * r + 1:]
        - integral[2 * r + 1:, : -2 * r - 1]
        + integral[: -2 * r - 1, : -2 * r - 1]
    )
    return total / float((2 * r + 1) ** 2)


def guided_filter(guide, src, radius=2, eps=0.01):
    """Filter ``src`` using ``guide`` as the edge reference.

    Parameters
    ----------
    guide : np.ndarray
        2D grayscale guide image; edges in the guide are preserved.
    src : np.ndarray
        2D grayscale image to filter, same shape as ``guide``.
    radius : int
        Window radius (window side is ``2*radius+1``), non-negative.
    eps : float
        Regularisation strength; larger values smooth more and blur
        weaker edges. Must be positive.

    Returns
    -------
    np.ndarray
        Filtered image, float64, same shape as the input.
    """
    guide_arr = _validate_image(guide, "guide")
    src_arr = _validate_image(src, "src")
    if guide_arr.shape != src_arr.shape:
        raise ValueError(
            f"guide and src must have the same shape, got "
            f"{guide_arr.shape} vs {src_arr.shape}"
        )
    if not isinstance(radius, int) or isinstance(radius, bool):
        raise ValueError("radius must be an integer")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if isinstance(eps, bool) or not isinstance(eps, (int, float, np.integer, np.floating)):
        raise ValueError("eps must be a number")
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive")

    mean_guide = _box_mean(guide_arr, radius)
    mean_src = _box_mean(src_arr, radius)
    corr_guide = _box_mean(guide_arr * guide_arr, radius)
    corr_guide_src = _box_mean(guide_arr * src_arr, radius)

    var_guide = corr_guide - mean_guide * mean_guide
    cov = corr_guide_src - mean_guide * mean_src

    a = cov / (var_guide + eps)
    b = mean_src - a * mean_guide
    return _box_mean(a, radius) * guide_arr + _box_mean(b, radius)


def edge_preserving_smooth(image, radius=2, eps=0.01):
    """Smooth an image while preserving its edges: the guided filter
    with the image itself as the guide.

    Noise in flat areas is averaged away; strong edges survive
    because the local linear model follows the guide across them.
    """
    arr = _validate_image(image, "image")
    return guided_filter(arr, arr, radius=radius, eps=eps)
