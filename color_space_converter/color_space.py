"""Colour-space conversions from scratch (numpy only, no OpenCV).

Conventions:
- RGB channels are float64 in [0, 255] (uint8 input is accepted too).
- HSV: H in [0, 360) degrees, S and V in [0, 1].
- YCbCr: BT.601 full-range, Y/Cb/Cr all in [0, 255].
- Grayscale: BT.601 luma, in [0, 255].
"""

import numpy as np


def _validate_rgb(image):
    """Return ``image`` as a float64 (H, W, 3) array in [0, 255]."""
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"image must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    if arr.min() < 0 or arr.max() > 255:
        raise ValueError(
            f"RGB values must be in [0, 255], got range [{arr.min()}, {arr.max()}]"
        )
    return arr


def _validate_hsv(image):
    """Return ``image`` as a float64 (H, W, 3) HSV array, validated."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    h, s, v = arr[..., 0], arr[..., 1], arr[..., 2]
    if h.min() < 0 or h.max() >= 360:
        raise ValueError(f"hue must be in [0, 360), got range [{h.min()}, {h.max()}]")
    if s.min() < 0 or s.max() > 1 or v.min() < 0 or v.max() > 1:
        raise ValueError("saturation and value must be in [0, 1]")
    return arr


def rgb_to_hsv(image):
    """Convert an (H, W, 3) RGB image in [0, 255] to HSV.

    Returns float64 (H, W, 3): H in [0, 360), S and V in [0, 1].
    Achromatic pixels (max == min) get hue 0.
    """
    rgb = _validate_rgb(image) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = rgb.max(axis=2)
    cmin = rgb.min(axis=2)
    delta = cmax - cmin
    hue = np.zeros_like(cmax)
    mask = delta > 0
    # Red is the max channel.
    sel = mask & (cmax == r)
    hue[sel] = (60.0 * ((g[sel] - b[sel]) / delta[sel]) + 360.0) % 360.0
    # Green is the max channel.
    sel = mask & (cmax == g)
    hue[sel] = 60.0 * ((b[sel] - r[sel]) / delta[sel]) + 120.0
    # Blue is the max channel.
    sel = mask & (cmax == b)
    hue[sel] = 60.0 * ((r[sel] - g[sel]) / delta[sel]) + 240.0
    saturation = np.where(cmax == 0, 0.0, delta / np.where(cmax == 0, 1.0, cmax))
    return np.stack([hue, saturation, cmax], axis=2)


def hsv_to_rgb(image):
    """Convert an (H, W, 3) HSV image to RGB in [0, 255].

    H in [0, 360), S and V in [0, 1]. Returns float64 (H, W, 3).
    """
    hsv = _validate_hsv(image)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    c = v * s
    x = c * (1.0 - np.abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    zero = np.zeros_like(h)
    sectors = [
        (h < 60, c, x, zero),
        ((h >= 60) & (h < 120), x, c, zero),
        ((h >= 120) & (h < 180), zero, c, x),
        ((h >= 180) & (h < 240), zero, x, c),
        ((h >= 240) & (h < 300), x, zero, c),
        (h >= 300, c, zero, x),
    ]
    for sel, rr, gg, bb in sectors:
        r[sel], g[sel], b[sel] = rr[sel], gg[sel], bb[sel]
    return np.stack([r + m, g + m, b + m], axis=2) * 255.0


def rgb_to_ycbcr(image):
    """Convert RGB in [0, 255] to BT.601 full-range YCbCr in [0, 255]."""
    rgb = _validate_rgb(image)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=2)


def ycbcr_to_rgb(image):
    """Convert BT.601 full-range YCbCr in [0, 255] back to RGB in [0, 255]."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    y, cb, cr = arr[..., 0], arr[..., 1], arr[..., 2]
    # For valid RGB input, Y is in [0, 255] and Cb/Cr in [0.5, 255.5]
    # (pure red has Cr = 255.5), so allow Cb/Cr up to 255.5.
    if y.min() < 0 or y.max() > 255:
        raise ValueError(f"Y values must be in [0, 255], got range [{y.min()}, {y.max()}]")
    if cb.min() < 0 or cb.max() > 255.5 or cr.min() < 0 or cr.max() > 255.5:
        raise ValueError(
            "Cb/Cr values must be in [0, 255.5], got ranges "
            f"Cb [{cb.min()}, {cb.max()}] Cr [{cr.min()}, {cr.max()}]"
        )
    y, cb, cr = arr[..., 0], arr[..., 1] - 128.0, arr[..., 2] - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.clip(np.stack([r, g, b], axis=2), 0.0, 255.0)


def rgb_to_grayscale(image):
    """Convert RGB in [0, 255] to BT.601 luma grayscale in [0, 255].

    Returns a 2D float64 array.
    """
    rgb = _validate_rgb(image)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def grayscale_to_rgb(image):
    """Replicate a 2D grayscale image in [0, 255] across three channels."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"image must be a 2D grayscale array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    if arr.min() < 0 or arr.max() > 255:
        raise ValueError(
            f"grayscale values must be in [0, 255], got range [{arr.min()}, {arr.max()}]"
        )
    return np.stack([arr, arr, arr], axis=2)
