"""Image resizing from scratch, with numpy only.

Two interpolation methods, for grayscale (H, W) and RGB (H, W, 3)
images, up- or downscaling:

- ``nearest``: nearest-neighbour — each output pixel copies the closest
  input pixel. Fast and exact for integer scale factors, but blocky.
- ``bilinear``: bilinear interpolation — each output pixel is a weighted
  average of its four surrounding input pixels. Smoother, and exact for
  a linear gradient.

Coordinates use the "align corners" convention: the centres of the
corner pixels of the output map onto the centres of the corner pixels
of the input, so resizing to the same size returns the input unchanged
and a 1-pixel-wide output is the centre column of the input.
"""

from __future__ import annotations

import numpy as np


def _validate_image(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"image must be numeric, got dtype {arr.dtype}")
    if arr.ndim == 2:
        pass
    elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        pass
    else:
        raise ValueError(
            f"image must be 2D grayscale or (H, W, C) with C in (1, 3, 4), "
            f"got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("image must not be empty")
    out = arr.astype(np.float64)
    if not np.isfinite(out).all():
        raise ValueError("image must contain only finite values")
    return out


def _validate_size(new_height, new_width):
    for name, value in (("new_height", new_height), ("new_width", new_width)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if int(value) < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    return int(new_height), int(new_width)


def _source_coords(out_len: int, in_len: int) -> np.ndarray:
    """Map output pixel indices to (float) source coordinates.

    Align-corners: output index 0 maps to source 0, output index
    out_len - 1 maps to source in_len - 1. A single output pixel maps
    to the centre of the input.
    """
    if out_len == 1:
        return np.array([(in_len - 1) / 2.0])
    return np.linspace(0, in_len - 1, out_len)


def resize_nearest(image, new_height, new_width):
    """Resize ``image`` with nearest-neighbour interpolation.

    Args:
        image: 2D grayscale or (H, W, C) array.
        new_height, new_width: target size, positive integers.

    Returns:
        A float64 array of shape (new_height, new_width) or
        (new_height, new_width, C), same channel count as the input.
    """
    arr = _validate_image(image)
    new_height, new_width = _validate_size(new_height, new_width)
    h, w = arr.shape[0], arr.shape[1]
    row_idx = np.clip(np.round(_source_coords(new_height, h)).astype(int), 0, h - 1)
    col_idx = np.clip(np.round(_source_coords(new_width, w)).astype(int), 0, w - 1)
    return arr[row_idx[:, None], col_idx]


def resize_bilinear(image, new_height, new_width):
    """Resize ``image`` with bilinear interpolation.

    Args:
        image: 2D grayscale or (H, W, C) array.
        new_height, new_width: target size, positive integers.

    Returns:
        A float64 array of shape (new_height, new_width) or
        (new_height, new_width, C), same channel count as the input.
    """
    arr = _validate_image(image)
    new_height, new_width = _validate_size(new_height, new_width)
    h, w = arr.shape[0], arr.shape[1]

    row_f = _source_coords(new_height, h)
    col_f = _source_coords(new_width, w)
    r0 = np.floor(row_f).astype(int)
    c0 = np.floor(col_f).astype(int)
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    # Weights broadcast over the channel axis when present.
    if arr.ndim == 3:
        wr = (row_f - r0)[:, None, None]
        wc = (col_f - c0)[None, :, None]
    else:
        wr = (row_f - r0)[:, None]
        wc = (col_f - c0)[None, :]

    top_left = arr[r0[:, None], c0]
    top_right = arr[r0[:, None], c1]
    bottom_left = arr[r1[:, None], c0]
    bottom_right = arr[r1[:, None], c1]
    top = top_left * (1 - wc) + top_right * wc
    bottom = bottom_left * (1 - wc) + bottom_right * wc
    return top * (1 - wr) + bottom * wr


def resize(image, new_height, new_width, method="bilinear"):
    """Resize ``image`` to (new_height, new_width).

    Args:
        image: 2D grayscale or (H, W, C) array.
        new_height, new_width: target size, positive integers.
        method: "nearest" or "bilinear" (default).
    """
    if method == "nearest":
        return resize_nearest(image, new_height, new_width)
    if method == "bilinear":
        return resize_bilinear(image, new_height, new_width)
    raise ValueError(f"unknown method {method!r}, expected 'nearest' or 'bilinear'")
