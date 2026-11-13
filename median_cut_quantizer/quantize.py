"""Median-cut colour quantization from scratch.

Reduces an RGB image to at most ``n_colors`` colours using the classic
median-cut algorithm (Heckbert, 1982):

1. Put all pixels in one box.
2. Repeat until there are ``n_colors`` boxes (or no box can split):
   pick the box with the largest colour range, split it along the
   channel with the largest range, at the median pixel of that channel,
   so each half holds roughly the same number of pixels.
3. The palette is the mean colour of each box; every pixel is replaced
   by its box's palette colour.

The result is a palette that spends more colours where the image
actually has pixels, unlike a uniform grid. Input is an (H, W, 3)
uint8 or float array; output is a uint8 image of the same shape plus
the palette as an (N, 3) uint8 array.
"""

from __future__ import annotations

import numpy as np


def _validate_image(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"image must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"image must be an (H, W, 3) RGB array, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("image must not be empty")
    out = arr.astype(np.float64)
    if not np.isfinite(out).all():
        raise ValueError("image must contain only finite values")
    if out.min() < 0.0 or out.max() > 255.0:
        raise ValueError("pixel values must be in the range [0, 255]")
    return out


def _validate_n_colors(n_colors):
    if isinstance(n_colors, bool) or not isinstance(n_colors, (int, np.integer)):
        raise ValueError(f"n_colors must be a positive integer, got {n_colors!r}")
    if int(n_colors) < 1:
        raise ValueError(f"n_colors must be at least 1, got {n_colors}")
    return int(n_colors)


def _split_box(pixels: np.ndarray):
    """Split one box of pixels along its widest channel, at the median.

    Returns ``(left, right)`` pixel arrays, or ``None`` if the box
    cannot split (all pixels identical).
    """
    ranges = pixels.max(axis=0) - pixels.min(axis=0)
    channel = int(np.argmax(ranges))
    if ranges[channel] == 0:
        return None
    order = np.argsort(pixels[:, channel], kind="stable")
    sorted_pixels = pixels[order]
    mid = len(sorted_pixels) // 2
    if mid == 0 or mid == len(sorted_pixels):
        return None
    return sorted_pixels[:mid], sorted_pixels[mid:]


def quantize(image, n_colors: int = 16):
    """Quantize an RGB image to at most ``n_colors`` colours.

    Args:
        image: (H, W, 3) array with values in [0, 255].
        n_colors: maximum number of colours in the palette.

    Returns:
        ``(quantized, palette)`` where ``quantized`` is a uint8 array
        with the same shape as ``image`` and ``palette`` is a uint8
        array of shape (N, 3) with N <= n_colors.
    """
    arr = _validate_image(image)
    n_colors = _validate_n_colors(n_colors)
    pixels = arr.reshape(-1, 3)
    boxes = [pixels]
    while len(boxes) < n_colors:
        # Split the box with the largest colour range that can split.
        best_idx, best_range = None, -1.0
        for i, box in enumerate(boxes):
            box_range = float((box.max(axis=0) - box.min(axis=0)).max())
            if box_range > best_range:
                best_idx, best_range = i, box_range
        if best_range == 0.0:
            break  # every box is a single colour already
        split = _split_box(boxes[best_idx])
        if split is None:
            break
        boxes[best_idx:best_idx + 1] = list(split)

    palette = np.array(
        [box.mean(axis=0) for box in boxes], dtype=np.float64
    )
    palette = np.clip(np.round(palette), 0, 255).astype(np.uint8)

    # Assign each pixel to its nearest palette colour. (Boxes were built
    # from reordered copies of the pixels, so the box a pixel came from
    # is not its position in the image.)
    flat = pixels.astype(np.float64)
    dists = np.sum(
        (flat[:, None, :] - palette[None, :, :].astype(np.float64)) ** 2, axis=2
    )
    nearest = np.argmin(dists, axis=1)
    quantized = palette[nearest].reshape(arr.shape)
    return quantized, palette


def unique_color_count(image) -> int:
    """Return the number of distinct RGB colours in ``image``."""
    arr = _validate_image(image)
    pixels = arr.reshape(-1, 3)
    return int(np.unique(pixels, axis=0).shape[0])
