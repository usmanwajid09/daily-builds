"""Otsu's thresholding, implemented from scratch (numpy only).

Otsu's method picks the threshold that best separates an image into
two classes (foreground and background) by maximising the
between-class variance — equivalently, minimising the within-class
variance. For every candidate threshold ``t`` it computes the class
weights and means from the 256-bin histogram and scores
``w0 * w1 * (mean0 - mean1) ** 2``; the best ``t`` wins. Ties are
broken by taking the smallest threshold with the best score.
"""

import numpy as np


def _to_uint8(image):
    """Validate ``image`` and return it as a 2D uint8 array.

    Values are clipped to [0, 255] and rounded to the nearest integer.
    Raises ``ValueError`` for non-numeric, non-2D, empty, or non-finite
    input.
    """
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"image must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"image must be a 2D grayscale array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def _validate_threshold(threshold):
    if isinstance(threshold, bool) or not isinstance(threshold, (int, np.integer)):
        raise ValueError(f"threshold must be an integer in [0, 255], got {threshold!r}")
    if not 0 <= int(threshold) <= 255:
        raise ValueError(f"threshold must be in [0, 255], got {threshold}")


def otsu_threshold(image):
    """Return the Otsu threshold (an int in [0, 255]) for ``image``.

    Pixels with values ``<= threshold`` are background, pixels with
    values ``> threshold`` are foreground. A constant image has no
    split to find, so its (single) value is returned.

    Args:
        image: 2D grayscale array with values in [0, 255] (values
            outside are clipped; floats are rounded).

    Returns:
        The threshold as a Python ``int``.
    """
    gray = _to_uint8(image)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    bin_values = np.arange(256, dtype=np.float64)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    mean_bg = np.cumsum(hist * bin_values)
    mean_fg = mean_bg[-1] - mean_bg
    # Between-class variance, ignoring splits that leave a class empty.
    valid = (weight_bg > 0) & (weight_fg > 0)
    variance = np.zeros(256, dtype=np.float64)
    variance[valid] = (
        weight_bg[valid] * weight_fg[valid]
        * (mean_bg[valid] / weight_bg[valid]
           - mean_fg[valid] / weight_fg[valid]) ** 2
    )
    if not valid.any():
        return int(gray.flat[0])  # constant image
    return int(np.argmax(variance))


def apply_threshold(image, threshold):
    """Return a boolean mask of pixels with value > ``threshold``.

    Args:
        image: 2D grayscale array with values in [0, 255].
        threshold: integer in [0, 255].

    Returns:
        A boolean ``numpy`` array with the same (H, W) shape.

    Raises:
        ValueError: on a bad image or a threshold outside [0, 255].
    """
    _validate_threshold(threshold)
    gray = _to_uint8(image)
    return gray > int(threshold)


def binarize(image):
    """Return ``(mask, threshold)``: Otsu thresholding in one call.

    ``mask`` is a boolean array (True = foreground) and ``threshold``
    is the Otsu threshold used to produce it.
    """
    threshold = otsu_threshold(image)
    return apply_threshold(image, threshold), threshold
