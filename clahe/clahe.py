"""Contrast-limited adaptive histogram equalization (CLAHE), from scratch.

The image is split into a grid of tiles. Each tile gets its own histogram,
which is clipped at a limit (excess pixels are redistributed evenly across
all bins) so noise in flat regions is not amplified. Each clipped histogram
is turned into a lookup table via its cumulative distribution function, and
neighbouring tile tables are blended with bilinear interpolation so tile
boundaries stay invisible.
"""

import numpy as np

__all__ = ["apply_clahe", "clahe"]


def _validate(image):
    """Check the input and return it as a float array; raise ValueError otherwise."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(
            f"CLAHE expects a 2D grayscale image, got {arr.ndim}D array "
            f"with shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("CLAHE expects a non-empty image, got an empty array")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(
            f"CLAHE expects numeric pixel values, got dtype {arr.dtype}"
        )
    arr = arr.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("CLAHE expects finite pixel values, got NaN or inf")
    return arr


def _tile_lut(tile, clip_limit):
    """Lookup table (256 entries) for one tile from its clipped histogram."""
    if tile.min() == tile.max():
        # A flat tile has no contrast to enhance; leave it exactly as-is.
        return np.arange(256, dtype=np.float64)
    hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
    limit = clip_limit * hist.mean()
    excess = np.maximum(hist - limit, 0.0).sum()
    hist = np.minimum(hist, limit)
    hist += excess / 256.0  # redistribute clipped pixels evenly
    cdf = np.cumsum(hist)
    cdf_min = cdf[np.nonzero(hist)[0][0]]
    lut = (cdf - cdf_min) / (cdf[-1] - cdf_min) * 255.0
    return np.clip(lut, 0, 255)


def _tile_boundaries(length, count):
    """Split an axis of `length` pixels into `count` tiles; return boundaries."""
    edges = np.linspace(0, length, count + 1)
    bounds = np.unique(np.round(edges).astype(int))
    if len(bounds) < 2:  # count larger than the axis: one tile per pixel
        bounds = np.arange(length + 1)
    return bounds


def _blend_weights(length, bounds):
    """For every pixel position, the two bracketing tile centres and weights.

    Returns (lower_index, upper_index, weight) arrays of shape (length,).
    Outside the first/last tile centre the nearest tile is used (weight 0/1),
    which is how edge tiles are handled.
    """
    centres = (bounds[:-1] + bounds[1:] - 1) / 2.0
    n = len(centres)
    if n == 1:
        return (
            np.zeros(length, dtype=int),
            np.zeros(length, dtype=int),
            np.zeros(length, dtype=float),
        )
    pos = np.arange(length, dtype=float)
    lower = np.searchsorted(centres, pos, side="right") - 1
    lower = np.clip(lower, 0, n - 2)
    upper = lower + 1
    span = centres[upper] - centres[lower]
    weight = np.clip((pos - centres[lower]) / span, 0.0, 1.0)
    return lower, upper, weight


def apply_clahe(image, tile_grid=(8, 8), clip_limit=2.0):
    """Apply CLAHE to a 2D grayscale image and return a uint8 image.

    Parameters
    ----------
    image : array_like, shape (H, W)
        Grayscale image, uint8 or float (values clipped to 0-255).
    tile_grid : (int, int), default (8, 8)
        Number of tiles as (rows, cols).
    clip_limit : float, default 2.0
        Histogram clip limit as a multiple of the mean bin count. Higher
        values allow more contrast (and more noise amplification).

    Returns
    -------
    np.ndarray, dtype uint8, same shape as `image`.
    """
    arr = _validate(image)
    if (
        len(tile_grid) != 2
        or any(
            isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t < 1
            for t in tile_grid
        )
    ):
        raise ValueError(
            f"tile_grid must be two positive integers, got {tile_grid!r}"
        )
    if not np.isfinite(clip_limit) or clip_limit <= 0:
        raise ValueError(f"clip_limit must be a positive number, got {clip_limit!r}")

    img = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    height, width = img.shape
    tile_rows, tile_cols = int(tile_grid[0]), int(tile_grid[1])

    row_bounds = _tile_boundaries(height, tile_rows)
    col_bounds = _tile_boundaries(width, tile_cols)
    n_rows, n_cols = len(row_bounds) - 1, len(col_bounds) - 1

    # One lookup table per tile.
    luts = np.empty((n_rows, n_cols, 256), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            tile = img[row_bounds[i] : row_bounds[i + 1], col_bounds[j] : col_bounds[j + 1]]
            luts[i, j] = _tile_lut(tile, clip_limit)

    # Bilinear blend of the four neighbouring tile tables at every pixel.
    r_lo, r_hi, w_r = _blend_weights(height, row_bounds)
    c_lo, c_hi, w_c = _blend_weights(width, col_bounds)
    w_r = w_r[:, None]
    w_c = w_c[None, :]

    out = (
        (1 - w_r) * (1 - w_c) * luts[r_lo[:, None], c_lo[None, :], img]
        + (1 - w_r) * w_c * luts[r_lo[:, None], c_hi[None, :], img]
        + w_r * (1 - w_c) * luts[r_hi[:, None], c_lo[None, :], img]
        + w_r * w_c * luts[r_hi[:, None], c_hi[None, :], img]
    )
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def clahe(image, tile_grid=(8, 8), clip_limit=2.0):
    """Alias for :func:`apply_clahe`."""
    return apply_clahe(image, tile_grid=tile_grid, clip_limit=clip_limit)
