"""Histogram equalization and contrast stretching, from scratch.

Both functions work only on numpy. Equalization remaps intensities so
the output histogram is (approximately) flat, using the cumulative
distribution function of the input histogram. Contrast stretching is a
simpler linear remap: the low and high percentiles of the input are
stretched out to 0 and 255.
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


def equalize_histogram(image):
    """Return ``image`` with its histogram equalized.

    Every intensity ``v`` is remapped to ``255 * CDF(v)``, where ``CDF``
    is the cumulative distribution function of the input histogram
    (normalised so the smallest present intensity maps to 0). A
    constant image has no contrast to spread, so it is returned
    unchanged.

    Args:
        image: 2D grayscale array with values in [0, 255] (values
            outside are clipped; floats are rounded).

    Returns:
        A 2D uint8 array with the same shape as the input.
    """
    gray = _to_uint8(image)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    # First non-zero CDF value: the smallest intensity actually present.
    nonzero = cdf[cdf > 0]
    if len(nonzero) == 0 or nonzero[0] == cdf[-1]:
        return gray.copy()  # constant (or empty) image: nothing to spread
    cdf_min = nonzero[0]
    total = cdf[-1]
    lut = np.rint(255.0 * (cdf - cdf_min) / (total - cdf_min)).astype(np.uint8)
    return lut[gray]


def contrast_stretch(image, low_percentile=2.0, high_percentile=98.0):
    """Linearly stretch ``image`` so its percentiles span [0, 255].

    Pixels at or below the ``low_percentile`` become 0, pixels at or
    above the ``high_percentile`` become 255, and everything between is
    scaled linearly. Unlike equalization this keeps the relative spacing
    of intensities — it just widens the range. A constant image (or one
    where the two percentiles coincide) is returned unchanged.

    Args:
        image: 2D grayscale array with values in [0, 255].
        low_percentile: lower percentile in [0, 100).
        high_percentile: upper percentile in (0, 100]; must be above
            ``low_percentile``.

    Returns:
        A 2D uint8 array with the same shape as the input.

    Raises:
        ValueError: on a bad image (see :func:`_to_uint8`) or bad
            percentile arguments.
    """
    for name, value in (("low_percentile", low_percentile),
                        ("high_percentile", high_percentile)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise ValueError(f"{name} must be a number, got {value!r}")
        if not np.isfinite(value) or not 0.0 <= float(value) <= 100.0:
            raise ValueError(f"{name} must be in [0, 100], got {value}")
    if not float(low_percentile) < float(high_percentile):
        raise ValueError(
            "low_percentile must be below high_percentile, got "
            f"{low_percentile} and {high_percentile}"
        )
    gray = _to_uint8(image)
    lo = np.percentile(gray, float(low_percentile))
    hi = np.percentile(gray, float(high_percentile))
    if hi <= lo:
        return gray.copy()
    stretched = (gray.astype(np.float64) - lo) * 255.0 / (hi - lo)
    return np.clip(np.rint(stretched), 0, 255).astype(np.uint8)
