"""Flood-fill segmentation (numpy only).

Stack-based flood fill from a seed pixel on a 2D grayscale image: a
pixel joins the region when its value is within ``tolerance`` of the
seed pixel's value. 4- or 8-connectivity is supported. :func:`fill_holes`
fills background holes fully enclosed by foreground pixels in a binary
mask, using a flood fill from the image border on the inverted mask.
"""

import math

import numpy as np

__all__ = ["flood_fill", "region_grow", "fill_holes"]


def _validate_image(image):
    """Check ``image`` is a non-empty 2D numeric numpy array."""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"image must be a numpy array, got {type(image).__name__}"
        )
    if image.ndim != 2:
        raise ValueError(
            f"image must be 2D (grayscale), got {image.ndim}D "
            f"with shape {image.shape}"
        )
    if image.size == 0:
        raise ValueError("image must be non-empty")
    if image.dtype == bool or image.dtype.kind not in "iuf":
        raise TypeError(
            f"image must have a numeric dtype (uint8/int/float), "
            f"got {image.dtype}"
        )
    return image


def _validate_seed(seed, shape):
    """Check ``seed`` is a pair of ints inside an image of ``shape``."""
    if isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed must be a pair of ints (row, col)")
    try:
        row, col = seed
    except (TypeError, ValueError):
        raise TypeError(
            f"seed must be a pair of ints (row, col), got {seed!r}"
        ) from None
    for value, name in ((row, "row"), (col, "col")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(
                f"seed {name} must be an int, got {value!r} "
                f"({type(value).__name__})"
            )
    row, col = int(row), int(col)
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        raise ValueError(
            f"seed ({row}, {col}) is outside the image of shape {shape}"
        )
    return row, col


def _validate_tolerance(tolerance):
    """Check ``tolerance`` is a non-negative finite number."""
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(
        tolerance, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"tolerance must be a number, got {tolerance!r} "
            f"({type(tolerance).__name__})"
        )
    tolerance = float(tolerance)
    if not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be finite, got {tolerance}")
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    return tolerance


def _validate_connectivity(connectivity):
    """Check ``connectivity`` is 4 or 8."""
    if isinstance(connectivity, (bool, np.bool_)) or not isinstance(
        connectivity, (int, np.integer)
    ):
        raise TypeError(
            f"connectivity must be 4 or 8, got {connectivity!r} "
            f"({type(connectivity).__name__})"
        )
    if int(connectivity) not in (4, 8):
        raise ValueError(
            f"connectivity must be 4 or 8, got {connectivity}"
        )
    return int(connectivity)


def _neighbour_offsets(connectivity):
    """Return (dr, dc) offsets for the requested connectivity."""
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return offsets


def flood_fill(image, seed, tolerance=0.0, connectivity=4):
    """Flood-fill from ``seed`` and return a boolean mask of the region.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale image with a numeric dtype.
    seed : tuple of (row, col)
        Starting pixel; must lie inside the image.
    tolerance : float, optional
        A pixel joins the region when ``abs(value - seed_value)``
        is ``<= tolerance``. Default 0.0 (exact-value fill).
    connectivity : int, optional
        4 or 8. Default 4.

    Returns
    -------
    numpy.ndarray
        Boolean mask, True for pixels in the filled region.
    """
    image = _validate_image(image)
    row, col = _validate_seed(seed, image.shape)
    tolerance = _validate_tolerance(tolerance)
    connectivity = _validate_connectivity(connectivity)

    # Compare in float64 so uint8/int images cannot overflow on subtract.
    values = image.astype(np.float64)
    seed_value = float(values[row, col])
    height, width = image.shape
    offsets = _neighbour_offsets(connectivity)

    mask = np.zeros((height, width), dtype=bool)
    stack = [(row, col)]
    mask[row, col] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                if abs(values[nr, nc] - seed_value) <= tolerance:
                    mask[nr, nc] = True
                    stack.append((nr, nc))
    return mask


def region_grow(image, seed, tolerance=0.0, connectivity=4):
    """Region growing from ``seed``; returns the grown-region mask.

    Same behaviour as :func:`flood_fill`: a pixel joins the region
    when ``abs(value - seed_value) <= tolerance``. Provided as a
    separate named entry point with its own validation.
    """
    return flood_fill(image, seed, tolerance=tolerance,
                      connectivity=connectivity)


def fill_holes(mask):
    """Fill background holes fully enclosed by True pixels.

    Parameters
    ----------
    mask : numpy.ndarray
        2D boolean mask.

    Returns
    -------
    numpy.ndarray
        Boolean mask with enclosed False regions set to True.
        Background connected to the image border is left alone.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(
            f"mask must be a numpy array, got {type(mask).__name__}"
        )
    if mask.ndim != 2:
        raise ValueError(
            f"mask must be 2D, got {mask.ndim}D with shape {mask.shape}"
        )
    if mask.size == 0:
        raise ValueError("mask must be non-empty")
    if mask.dtype != bool:
        raise TypeError(
            f"mask must have boolean dtype, got {mask.dtype}"
        )

    height, width = mask.shape
    background = ~mask
    # Flood the border-connected background (4-connected) on the
    # inverted mask; whatever background is left is an enclosed hole.
    reached = np.zeros((height, width), dtype=bool)
    stack = []
    for r in range(height):
        for c in (0, width - 1):
            if background[r, c] and not reached[r, c]:
                reached[r, c] = True
                stack.append((r, c))
    for c in range(width):
        for r in (0, height - 1):
            if background[r, c] and not reached[r, c]:
                reached[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < height and 0 <= nc < width
                    and background[nr, nc] and not reached[nr, nc]):
                reached[nr, nc] = True
                stack.append((nr, nc))
    holes = background & ~reached
    return mask | holes
