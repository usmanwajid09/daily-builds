"""Gamma correction and brightness/contrast adjustment, from scratch (numpy only).

Pointwise intensity transforms for grayscale (2D) or RGB (H, W, 3)
images: gamma correction, log transform, percentile contrast stretch,
and a brightness/contrast adjustment. All functions take an image as
a numpy array (uint8, or float in [0, 1] or [0, 255]) and return a
float64 array of the same shape with values in [0, 255].
"""

import numpy as np


def _validate_image(image):
    """Return ``image`` as a float64 array in [0, 255] or raise.

    Accepts 2D grayscale or (H, W, 3) RGB arrays. Float input whose
    values all lie in [0, 1] is treated as a [0, 1] image and scaled
    to [0, 255]; float input with any value above 1 is taken as
    already being in [0, 255]. uint8 input is used as-is.
    """
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


def _to_255(image):
    """Validate ``image`` and return it as float64 in [0, 255]."""
    arr = np.asarray(image)
    was_float = arr.dtype.kind == "f"
    arr = _validate_image(image)
    if was_float and arr.size and arr.max() <= 1.0 and arr.min() >= 0.0:
        # Float image in [0, 1]: scale to [0, 255].
        arr = arr * 255.0
    return arr


def _validate_positive_number(value, name):
    """Return ``value`` as a float, requiring a positive finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value}")
    return value


def _validate_finite_number(value, name):
    """Return ``value`` as a float, requiring a finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value}")
    return value


def gamma_correct(image, gamma):
    """Return ``image`` gamma-corrected: ``255 * (image / 255) ** gamma``.

    Args:
        image: 2D grayscale array, or (H, W, 3) RGB array.
        gamma: positive finite number. ``gamma < 1`` brightens (lifts
            the mid-tones), ``gamma > 1`` darkens, ``gamma == 1``
            returns the image unchanged.

    Returns:
        A float64 array with the same shape as the input, values in
        [0, 255].

    Raises:
        ValueError: on a bad image (see :func:`_validate_image`) or a
            non-positive / non-finite ``gamma``.
    """
    arr = _to_255(image)
    gamma = _validate_positive_number(gamma, "gamma")
    if gamma == 1.0:
        return arr.copy()
    return 255.0 * np.power(arr / 255.0, gamma)


def log_transform(image, c=1.0):
    """Return ``image`` after a normalised log transform.

    The transform is monotonic, maps 0 to 0 and 255 to 255, and
    expands the dark tones while compressing the bright ones — the
    classic way to reveal detail hiding in the shadows.

    Formula (``c == 1`` is the standard form)::

        out = 255 * log(1 + image) / log(1 + 255)

    For a general ``c`` the same shape is used with the input scaled
    by ``c``::

        out = 255 * log(1 + c * image / 255) / log(1 + c)

    so that 0 still maps to 0 and 255 still maps to 255 for any ``c``.

    Args:
        image: 2D grayscale array, or (H, W, 3) RGB array.
        c: positive finite number controlling the strength of the
            compression; larger ``c`` compresses the bright end more.

    Returns:
        A float64 array with the same shape as the input, values in
        [0, 255].

    Raises:
        ValueError: on a bad image or a non-positive / non-finite ``c``.
    """
    arr = _to_255(image)
    c = _validate_positive_number(c, "c")
    if c == 1.0:
        return 255.0 * np.log1p(arr) / np.log1p(255.0)
    return 255.0 * np.log1p(c * arr / 255.0) / np.log1p(c)


def contrast_stretch(image, low=2, high=98):
    """Stretch ``image`` contrast by mapping percentiles to [0, 255].

    The ``low`` and ``high`` percentiles of the image are computed
    per channel for RGB (globally for grayscale) and the range
    [p_low, p_high] is mapped linearly onto [0, 255], clipping
    anything outside. A flat image (p_low == p_high) is returned
    unchanged.

    Args:
        image: 2D grayscale array, or (H, W, 3) RGB array.
        low: lower percentile; must satisfy ``0 <= low < high``.
        high: upper percentile; must satisfy ``low < high <= 100``.

    Returns:
        A float64 array with the same shape as the input, values in
        [0, 255].

    Raises:
        ValueError: on a bad image or a bad percentile range.
    """
    arr = _to_255(image)
    for name, value in (("low", low), ("high", high)):
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.floating)
        ):
            raise ValueError(f"{name} must be a number, got {value!r}")
    low = float(low)
    high = float(high)
    if not (0.0 <= low < high <= 100.0):
        raise ValueError(
            f"percentiles must satisfy 0 <= low < high <= 100, got low={low}, high={high}"
        )
    if arr.ndim == 2:
        p_low, p_high = np.percentile(arr, [low, high])
        if p_low == p_high:
            return arr.copy()
        return np.clip((arr - p_low) * 255.0 / (p_high - p_low), 0.0, 255.0)
    out = np.empty_like(arr)
    for channel in range(3):
        p_low, p_high = np.percentile(arr[:, :, channel], [low, high])
        if p_low == p_high:
            out[:, :, channel] = arr[:, :, channel]
        else:
            out[:, :, channel] = np.clip(
                (arr[:, :, channel] - p_low) * 255.0 / (p_high - p_low),
                0.0,
                255.0,
            )
    return out


def adjust_brightness_contrast(image, brightness=0.0, contrast=1.0):
    """Shift brightness and scale contrast around the midpoint 127.5.

    ``out = (image - 127.5) * contrast + 127.5 + brightness``,
    clipped to [0, 255]. Positive ``brightness`` lightens the whole
    image; ``contrast > 1`` pushes darks darker and brights brighter
    around the midpoint, ``contrast < 1`` flattens towards grey.

    Args:
        image: 2D grayscale array, or (H, W, 3) RGB array.
        brightness: finite number added to every pixel (in [0, 255]
            units); 0 leaves brightness unchanged.
        contrast: positive finite multiplier; 1 leaves contrast
            unchanged.

    Returns:
        A float64 array with the same shape as the input, values in
        [0, 255].

    Raises:
        ValueError: on a bad image, a non-finite ``brightness``, or a
            non-positive / non-finite ``contrast``.
    """
    arr = _to_255(image)
    brightness = _validate_finite_number(brightness, "brightness")
    contrast = _validate_positive_number(contrast, "contrast")
    out = (arr - 127.5) * contrast + 127.5 + brightness
    return np.clip(out, 0.0, 255.0)
