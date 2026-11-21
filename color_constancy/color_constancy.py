"""Colour constancy / white balance from scratch (numpy only).

Estimates the colour of the scene illuminant from an RGB image and
removes the cast by scaling each channel. Three classic estimators:

- ``gray_world`` — the average colour of the scene is grey, so scale
  each channel so all three channel means are equal.
- ``white_patch`` — the brightest pixels are white, so scale each
  channel so its high percentile maps to 255.
- ``shades_of_gray`` — a Minkowski (p-norm) generalisation of the
  gray-world idea; p = 1 is gray-world, p -> infinity is white-patch.

Images may be uint8 or float in [0, 1] / [0, 255]; the corrected image
comes back in the same dtype and range.
"""

import numpy as np


def _validate_image(image, name="image"):
    """Return ``image`` as a float64 array in [0, 255], plus its kind."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"{name} must be an RGB image with shape (H, W, 3), "
            f"got shape {image.shape}"
        )
    if image.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"{name} dtype {image.dtype} is not numeric")
    if image.dtype == np.uint8:
        return image.astype(np.float64), "uint8"
    arr = image.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if arr.min() < 0.0 or arr.max() > 255.0:
        raise ValueError(
            f"float {name} must lie in [0, 1] or [0, 255], "
            f"got range [{arr.min()}, {arr.max()}]"
        )
    if arr.max() <= 1.0:
        return arr * 255.0, "float01"
    return arr, "float255"


def _from_255(arr, kind):
    arr = np.clip(arr, 0.0, 255.0)
    if kind == "uint8":
        return np.round(arr).astype(np.uint8)
    if kind == "float01":
        return arr / 255.0
    return arr


def _gains_from_illuminant(illuminant):
    """Per-channel gains that make ``illuminant`` grey (mean 1)."""
    illuminant = np.asarray(illuminant, dtype=np.float64)
    if illuminant.shape != (3,):
        raise ValueError(
            f"illuminant must have shape (3,), got {illuminant.shape}"
        )
    if np.any(illuminant <= 0):
        raise ValueError(
            f"illuminant channels must all be positive, got {illuminant}"
        )
    mean = illuminant.mean()
    return mean / illuminant


def gray_world_gains(image):
    """Gains that equalise the three channel means (gray-world).

    Returns an array of shape (3,) with all entries positive;
    multiplying the image by these gains makes every channel mean
    equal to the overall mean.
    """
    arr, _ = _validate_image(image)
    means = arr.reshape(-1, 3).mean(axis=0)
    if np.any(means <= 0):
        raise ValueError(
            "gray-world needs a positive mean in every channel, "
            f"got means {means}"
        )
    return _gains_from_illuminant(means)


def white_patch_gains(image, percentile=99.0):
    """Gains that map each channel's ``percentile`` to 255.

    ``percentile`` must lie in (0, 100]. The default (99) is robust to
    a few clipped or specular pixels; 100 uses the true channel max.
    """
    if not isinstance(percentile, (int, float, np.integer, np.floating)):
        raise ValueError(f"percentile must be a number, got {percentile!r}")
    if not 0.0 < float(percentile) <= 100.0:
        raise ValueError(
            f"percentile must lie in (0, 100], got {percentile}"
        )
    arr, _ = _validate_image(image)
    peaks = np.percentile(arr.reshape(-1, 3), percentile, axis=0)
    if np.any(peaks <= 0):
        raise ValueError(
            "white-patch needs a positive peak in every channel, "
            f"got peaks {peaks}"
        )
    return 255.0 / peaks


def shades_of_gray_gains(image, p=6.0):
    """Gains from the p-norm (Minkowski) of each channel.

    ``p`` must be positive. p = 1 is the gray-world estimate; larger
    p weights bright pixels more, approaching white-patch as p grows.
    """
    if not isinstance(p, (int, float, np.integer, np.floating)):
        raise ValueError(f"p must be a number, got {p!r}")
    if float(p) <= 0:
        raise ValueError(f"p must be positive, got {p}")
    arr, _ = _validate_image(image)
    flat = arr.reshape(-1, 3)
    norms = (np.mean(flat ** float(p), axis=0)) ** (1.0 / float(p))
    if np.any(norms <= 0):
        raise ValueError(
            "shades-of-gray needs a positive p-norm in every channel, "
            f"got norms {norms}"
        )
    return _gains_from_illuminant(norms)


def estimate_illuminant(image, method="gray_world", **kwargs):
    """Estimate the scene illuminant as a normalised RGB triplet.

    The returned array sums to 1, so (1/3, 1/3, 1/3) means a neutral
    illuminant. ``method`` is one of "gray_world", "white_patch" or
    "shades_of_gray"; extra keyword arguments are passed through to
    the estimator (``percentile`` / ``p``).
    """
    methods = {
        "gray_world": gray_world_gains,
        "white_patch": white_patch_gains,
        "shades_of_gray": shades_of_gray_gains,
    }
    if method not in methods:
        raise ValueError(
            f"method must be one of {', '.join(methods)}, got {method!r}"
        )
    gains = methods[method](image, **kwargs)
    illuminant = 1.0 / gains
    return illuminant / illuminant.sum()


def apply_gains(image, gains):
    """Multiply each channel of ``image`` by ``gains`` (shape (3,))."""
    arr, kind = _validate_image(image)
    gains = np.asarray(gains, dtype=np.float64)
    if gains.shape != (3,):
        raise ValueError(f"gains must have shape (3,), got {gains.shape}")
    if not np.all(np.isfinite(gains)) or np.any(gains <= 0):
        raise ValueError(f"gains must be finite and positive, got {gains}")
    return _from_255(arr * gains, kind)


def white_balance(image, method="gray_world", **kwargs):
    """Correct the colour cast of ``image`` using ``method``.

    ``method`` is one of "gray_world", "white_patch" or
    "shades_of_gray"; extra keyword arguments are passed through to
    the estimator (``percentile`` / ``p``). Returns the corrected
    image in the input's dtype and range.
    """
    arr, kind = _validate_image(image)
    methods = {
        "gray_world": gray_world_gains,
        "white_patch": white_patch_gains,
        "shades_of_gray": shades_of_gray_gains,
    }
    if method not in methods:
        raise ValueError(
            f"method must be one of {', '.join(methods)}, got {method!r}"
        )
    gains = methods[method](image, **kwargs)
    return _from_255(arr * gains, kind)
