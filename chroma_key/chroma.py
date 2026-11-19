"""Green-screen (chroma key) compositing, from scratch (numpy only).

A pixel belongs to the green screen when its colour is close to the key
colour in RGB space. Foreground pixels are kept, screen pixels are
replaced by the background image, and an optional feather band blends
the two so the cut-out edge is not a hard cliff.
"""

import numpy as np


def _validate_rgb(image, name="image"):
    """Return ``image`` as a float64 (H, W, 3) array or raise ValueError."""
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{name} must be an (H, W, 3) RGB image, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_key_color(key_color):
    arr = np.asarray(key_color, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(
            f"key_color must be 3 RGB values, got {np.asarray(key_color)!r}")
    if not np.isfinite(arr).all() or (arr < 0).any() or (arr > 255).any():
        raise ValueError(f"key_color values must be in 0-255, got {arr}")
    return arr


def _validate_non_negative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be a number >= 0, got {value!r}")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a number >= 0, got {value}")
    return value


def color_distance(image, key_color=(0.0, 255.0, 0.0)):
    """Return the per-pixel Euclidean RGB distance to ``key_color``.

    Result is a 2D float64 array the same height/width as ``image``;
    0 means the pixel is exactly the key colour.
    """
    image = _validate_rgb(image)
    key = _validate_key_color(key_color)
    diff = image - key
    return np.sqrt((diff ** 2).sum(axis=2))


def green_mask(image, key_color=(0.0, 255.0, 0.0), tolerance=60.0, feather=0.0):
    """Return a foreground alpha map: 1 keeps the pixel, 0 replaces it.

    Pixels closer than ``tolerance`` to ``key_color`` are screen
    (alpha 0); pixels further away are foreground (alpha 1). With
    ``feather > 0`` the alpha ramps linearly from 0 to 1 over the band
    ``[tolerance, tolerance + feather]`` for a soft edge.
    """
    image = _validate_rgb(image)
    _validate_key_color(key_color)
    tolerance = _validate_non_negative(tolerance, "tolerance")
    feather = _validate_non_negative(feather, "feather")
    distance = color_distance(image, key_color)
    if feather == 0.0:
        return (distance > tolerance).astype(np.float64)
    return np.clip((distance - tolerance) / feather, 0.0, 1.0)


def composite(foreground, background, key_color=(0.0, 255.0, 0.0),
              tolerance=60.0, feather=0.0):
    """Composite ``foreground`` over ``background`` by chroma keying.

    Screen pixels (close to ``key_color``) are replaced by the matching
    background pixel; foreground pixels are kept; feathered pixels are
    blended. Both images must be (H, W, 3) with the same shape.

    Returns:
        A float64 (H, W, 3) image.

    Raises:
        ValueError: on bad images, mismatched shapes, a bad key colour,
            or a negative ``tolerance``/``feather``.
    """
    foreground = _validate_rgb(foreground, "foreground")
    background = _validate_rgb(background, "background")
    if foreground.shape != background.shape:
        raise ValueError(
            f"foreground shape {foreground.shape} != background shape "
            f"{background.shape}")
    alpha = green_mask(foreground, key_color, tolerance, feather)[..., None]
    return alpha * foreground + (1.0 - alpha) * background


def replaced_pixel_count(foreground, key_color=(0.0, 255.0, 0.0), tolerance=60.0):
    """Count the screen pixels that :func:`composite` would replace."""
    mask = green_mask(foreground, key_color, tolerance)
    return int(np.count_nonzero(mask == 0.0))
