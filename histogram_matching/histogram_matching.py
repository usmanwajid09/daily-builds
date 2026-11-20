"""Histogram specification (matching): reshape one image's histogram.

Given a source image and a reference image, remap the source's pixel
values so its histogram matches the reference's histogram.  Works on
grayscale images and, per channel, on RGB images.  Only numpy is used.
"""

import numpy as np


def _validate_image(image, name):
    """Return ``image`` as an array, raising ValueError on bad input."""
    image = np.asarray(image)
    if image.ndim == 2:
        pass
    elif image.ndim == 3 and image.shape[2] == 3:
        pass
    else:
        raise ValueError(
            f"{name} must be a 2D grayscale or 3-channel RGB image, "
            f"got shape {image.shape}"
        )
    if image.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"{name} dtype {image.dtype} is not numeric")
    return image


def _to_255(image):
    """Convert an image to float64 in [0, 255]; return (array, kind).

    ``kind`` is "uint8", "float01" or "float255" so the result can be
    converted back to the source's range afterwards.
    """
    if image.dtype == np.uint8:
        return image.astype(np.float64), "uint8"
    arr = image.astype(np.float64)
    if arr.min() < 0.0 or arr.max() > 255.0:
        raise ValueError(
            "float images must lie in [0, 1] or [0, 255], "
            f"got range [{arr.min()}, {arr.max()}]"
        )
    if arr.max() <= 1.0:
        return arr * 255.0, "float01"
    return arr, "float255"


def _from_255(arr, kind):
    """Convert a float64 [0, 255] image back to the source's range."""
    arr = np.clip(arr, 0.0, 255.0)
    if kind == "uint8":
        return np.round(arr).astype(np.uint8)
    if kind == "float01":
        return arr / 255.0
    return arr


def compute_histogram(image, bins=256):
    """Histogram of a grayscale image over ``bins`` levels in [0, 255].

    Returns an integer array of length ``bins``.
    """
    image = _validate_image(image, "image")
    if image.ndim != 2:
        raise ValueError(
            f"compute_histogram expects a grayscale image, got shape "
            f"{image.shape}"
        )
    arr, _ = _to_255(image)
    levels = np.clip(np.round(arr), 0, bins - 1).astype(np.int64).ravel()
    return np.bincount(levels, minlength=bins).astype(np.int64)


def compute_cdf(image, bins=256):
    """Normalised cumulative histogram of a grayscale image."""
    hist = compute_histogram(image, bins=bins)
    total = hist.sum()
    if total == 0:
        raise ValueError("image must not be empty")
    return np.cumsum(hist, dtype=np.float64) / float(total)


def _match_channel(source, reference, bins=256):
    """Map one grayscale channel of ``source`` onto ``reference``."""
    src_cdf = compute_cdf(source, bins=bins)
    ref_cdf = compute_cdf(reference, bins=bins)
    # For each source level, pick the smallest reference level whose
    # CDF is at least the source CDF at that level.
    lut = np.searchsorted(ref_cdf, src_cdf, side="left")
    lut = np.clip(lut, 0, bins - 1)
    src_arr, _ = _to_255(source)
    levels = np.clip(np.round(src_arr), 0, bins - 1).astype(np.int64)
    return lut[levels].astype(np.float64)


def match_histogram(source, reference, bins=256):
    """Match ``source``'s histogram to ``reference``'s histogram.

    Parameters
    ----------
    source, reference : array
        Grayscale (H, W) or RGB (H, W, 3) images.  Both must have the
        same shape kind (grayscale with grayscale, RGB with RGB).
        uint8 or float in [0, 1] / [0, 255] are accepted.
    bins : int
        Number of histogram bins; must be >= 2.

    Returns
    -------
    numpy.ndarray
        Image with the source's content but the reference's histogram,
        in the source's dtype and range.
    """
    source = _validate_image(source, "source")
    reference = _validate_image(reference, "reference")
    if source.ndim != reference.ndim:
        raise ValueError(
            f"source and reference must both be grayscale or both RGB, "
            f"got shapes {source.shape} and {reference.shape}"
        )
    if not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError(f"bins must be an integer >= 2, got {bins}")

    _, kind = _to_255(source)
    if source.ndim == 2:
        matched = _match_channel(source, reference, bins=bins)
    else:
        channels = [
            _match_channel(source[:, :, c], reference[:, :, c], bins=bins)
            for c in range(3)
        ]
        matched = np.stack(channels, axis=-1)
    return _from_255(matched, kind)
