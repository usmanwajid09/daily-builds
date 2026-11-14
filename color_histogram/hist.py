"""Colour histogram utilities, implemented from scratch (numpy only).

A colour histogram counts how often each intensity occurs in a channel,
so it summarises *what colours* an image contains while throwing away
*where* they are. That makes histograms a cheap way to compare images
(similar scenes have similar colour distributions) and to find pixels
that look like a target colour (backprojection).
"""

import numpy as np

__all__ = ["channel_histograms", "compare_histograms", "backproject"]


def _validate_bins(bins):
    """Return ``bins`` as an int, or raise ValueError."""
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)):
        raise ValueError(f"bins must be a positive integer, got {bins!r}")
    bins = int(bins)
    if bins < 1:
        raise ValueError(f"bins must be a positive integer, got {bins}")
    return bins


def _validate_value_range(value_range):
    """Return ``value_range`` as a (low, high) float pair, or raise."""
    try:
        low, high = value_range
    except (TypeError, ValueError):
        raise ValueError(
            f"value_range must be a (low, high) pair, got {value_range!r}"
        ) from None
    if isinstance(low, bool) or isinstance(high, bool):
        raise ValueError(
            f"value_range must be a pair of finite numbers, got {value_range!r}"
        )
    try:
        low, high = float(low), float(high)
    except (TypeError, ValueError):
        raise ValueError(
            f"value_range must be a pair of finite numbers, got {value_range!r}"
        ) from None
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(
            f"value_range must be finite, got {(low, high)}"
        )
    if not low < high:
        raise ValueError(
            f"value_range must satisfy low < high, got {(low, high)}"
        )
    return low, high


def _validate_rgb_image(image):
    """Return ``image`` as a float64 (H, W, 3) array, or raise ValueError."""
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"image must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"image must be (H, W, 3) RGB, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("image must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    return arr


def _validate_histogram(histogram, name="histogram"):
    """Return ``histogram`` as a float64 1D array, or raise ValueError."""
    arr = np.asarray(histogram, dtype=np.float64)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    if (arr < 0).any():
        raise ValueError(f"{name} must not contain negative values")
    return arr


def _bin_index(values, bins, low, high):
    """Map values to bin indices in [0, bins), clipping out-of-range ones."""
    scaled = (values - low) / (high - low) * bins
    return np.clip(scaled.astype(np.int64), 0, bins - 1)


def channel_histograms(image, bins=256, value_range=(0, 255)):
    """Return the normalised per-channel histograms of an RGB image.

    Args:
        image: (H, W, 3) RGB array.
        bins: number of histogram bins; a positive integer.
        value_range: (low, high) intensity range covered by the bins.

    Returns:
        A list of 3 float64 arrays, one per channel (R, G, B), each of
        length ``bins`` and summing to 1. A channel with no values inside
        ``value_range`` yields an all-zero histogram.

    Raises:
        ValueError: on a bad image, ``bins`` or ``value_range``.
    """
    arr = _validate_rgb_image(image)
    bins = _validate_bins(bins)
    low, high = _validate_value_range(value_range)
    histograms = []
    for channel in range(3):
        counts, _ = np.histogram(
            arr[:, :, channel], bins=bins, range=(low, high)
        )
        counts = counts.astype(np.float64)
        total = counts.sum()
        histograms.append(counts / total if total > 0 else counts)
    return histograms


def compare_histograms(h1, h2, method="correlation"):
    """Compare two histograms and return a similarity/distance score.

    Args:
        h1, h2: 1D non-negative arrays of the same length.
        method: one of "correlation", "chi_square", "intersection".

    Returns:
        A float. For normalised histograms, comparing a histogram with
        itself gives correlation 1.0, chi-square 0.0, intersection 1.0.
        Correlation and intersection are similarities (higher = more
        alike, 1.0 = identical); chi-square is a distance (lower = more
        alike, 0.0 = identical).

    Raises:
        ValueError: on bad histograms, mismatched lengths, or an unknown
            ``method``.
    """
    a = _validate_histogram(h1, "h1")
    b = _validate_histogram(h2, "h2")
    if a.shape != b.shape:
        raise ValueError(
            f"histograms must have the same length, got {a.shape[0]} and {b.shape[0]}"
        )
    methods = ("correlation", "chi_square", "intersection")
    if method not in methods:
        raise ValueError(
            f"unknown method {method!r}; expected one of {methods}"
        )
    if method == "correlation":
        a_mean = a.mean()
        b_mean = b.mean()
        a_centred = a - a_mean
        b_centred = b - b_mean
        denom = np.sqrt((a_centred ** 2).sum() * (b_centred ** 2).sum())
        if denom == 0.0:
            # Both histograms are constant: identical means perfectly
            # correlated, different means uncorrelated.
            return 1.0 if np.array_equal(a, b) else 0.0
        return float((a_centred * b_centred).sum() / denom)
    if method == "chi_square":
        denom = a + b
        terms = np.zeros_like(a)
        nonzero = denom > 0
        terms[nonzero] = (a[nonzero] - b[nonzero]) ** 2 / denom[nonzero]
        return float(terms.sum())
    # intersection
    return float(np.minimum(a, b).sum())


def backproject(image, histogram, bins=256, value_range=(0, 255), channel=0):
    """Return a 2D probability map for ``histogram`` over ``image``.

    Each pixel is scored by looking up its intensity in ``histogram``
    on the chosen channel, so pixels whose colour on that channel the
    histogram rates highly score near the histogram's peak. This is
    the standard histogram-backprojection idea, applied to one channel
    (a single 1D histogram can only model one channel; pass the
    histogram of the channel you are tracking, e.g. the red channel
    of a red object, and use ``channel`` to pick it).

    Args:
        image: (H, W, 3) RGB array.
        histogram: 1D non-negative array of length ``bins``.
        bins: number of histogram bins; a positive integer.
        value_range: (low, high) intensity range covered by the bins.
        channel: which RGB channel (0 = R, 1 = G, 2 = B) to look up.

    Returns:
        A float64 array of shape (H, W) with values between 0 and the
        histogram's maximum.

    Raises:
        ValueError: on a bad image, histogram, ``bins``,
            ``value_range`` or ``channel``.
    """
    arr = _validate_rgb_image(image)
    hist = _validate_histogram(histogram)
    bins = _validate_bins(bins)
    low, high = _validate_value_range(value_range)
    if hist.shape[0] != bins:
        raise ValueError(
            f"histogram length ({hist.shape[0]}) must equal bins ({bins})"
        )
    if isinstance(channel, bool) or not isinstance(channel, (int, np.integer)):
        raise ValueError(f"channel must be 0, 1 or 2, got {channel!r}")
    channel = int(channel)
    if channel not in (0, 1, 2):
        raise ValueError(f"channel must be 0, 1 or 2, got {channel}")
    idx = _bin_index(arr[:, :, channel], bins, low, high)
    return hist[idx].astype(np.float64)
