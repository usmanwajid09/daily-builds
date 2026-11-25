"""Region growing segmentation, implemented from scratch (numpy only)."""

from collections import deque

import numpy as np


def _validate_image(image):
    """Return the image as a float64 2D array, raising ValueError on bad input."""
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")
    if image.ndim != 2:
        raise ValueError(f"image must be 2D (grayscale), got {image.ndim}D")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("image must have a numeric dtype")
    arr = image.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("image must contain only finite values")
    return arr


def _validate_seeds(seeds, shape):
    if seeds is None:
        raise ValueError("seeds must not be empty")
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("seeds must not be empty")
    height, width = shape
    cleaned = []
    for seed in seed_list:
        try:
            row, col = seed
        except (TypeError, ValueError):
            raise ValueError(
                f"each seed must be a (row, col) pair, got {seed!r}"
            ) from None
        if isinstance(row, bool) or isinstance(col, bool):
            raise ValueError(f"seed coordinates must be integers, got {seed!r}")
        if not isinstance(row, (int, np.integer)) or not isinstance(col, (int, np.integer)):
            raise ValueError(f"seed coordinates must be integers, got {seed!r}")
        row, col = int(row), int(col)
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(
                f"seed {(row, col)} is outside the {height}x{width} image"
            )
        cleaned.append((row, col))
    return cleaned


def _validate_tolerance(tolerance):
    if isinstance(tolerance, bool) or not isinstance(
        tolerance, (int, float, np.integer, np.floating)
    ):
        raise ValueError("tolerance must be a number")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    return tolerance


def _neighbour_offsets(connectivity):
    if connectivity == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        return (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        )
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")


def grow_from_seed(image, seed, tolerance, connectivity=4):
    """Grow one region from a single seed pixel.

    A neighbour joins the region when its intensity differs from the
    seed pixel's intensity by at most ``tolerance``. Growth is a
    breadth-first flood fill over ``connectivity``-neighbours.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image (uint8 or float).
    seed : (row, col)
        Starting pixel, inside the image.
    tolerance : float
        Maximum allowed |pixel - seed value|, non-negative.
    connectivity : int
        4 or 8 neighbour connectivity.

    Returns
    -------
    np.ndarray
        Boolean mask of the grown region.
    """
    arr = _validate_image(image)
    (seed_row, seed_col), = _validate_seeds([seed], arr.shape)
    tolerance = _validate_tolerance(tolerance)
    offsets = _neighbour_offsets(connectivity)

    height, width = arr.shape
    reference = arr[seed_row, seed_col]
    mask = np.zeros(arr.shape, dtype=bool)
    mask[seed_row, seed_col] = True
    queue = deque([(seed_row, seed_col)])
    while queue:
        row, col = queue.popleft()
        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                if abs(arr[nr, nc] - reference) <= tolerance:
                    mask[nr, nc] = True
                    queue.append((nr, nc))
    return mask


def region_grow(image, seeds, tolerance, connectivity=4):
    """Grow regions from several seeds and return their union.

    Each seed grows its own region against its own seed intensity
    (see :func:`grow_from_seed`); the result is the union of all
    regions. Use this to segment several similar objects at once, or
    to seed one object in several places.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image (uint8 or float).
    seeds : iterable of (row, col)
        At least one seed pixel, each inside the image.
    tolerance : float
        Maximum allowed |pixel - seed value|, non-negative.
    connectivity : int
        4 or 8 neighbour connectivity.

    Returns
    -------
    np.ndarray
        Boolean mask of the union of the grown regions.
    """
    arr = _validate_image(image)
    seed_list = _validate_seeds(seeds, arr.shape)
    tolerance = _validate_tolerance(tolerance)
    offsets = _neighbour_offsets(connectivity)
    # Validate connectivity up front even when grow_from_seed would too.
    mask = np.zeros(arr.shape, dtype=bool)
    for seed in seed_list:
        mask |= grow_from_seed(arr, seed, tolerance, connectivity=connectivity)
    return mask
