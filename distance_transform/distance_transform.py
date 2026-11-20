"""Two-pass chamfer distance transform.

For every foreground pixel, the distance to the nearest background
pixel.  Exact Euclidean distance is expensive; the chamfer transform
approximates it with two raster passes (forward, then backward) using
small integer masks: 3-4 (orthogonal 3, diagonal 4) or 5-7-11
(orthogonal 5, diagonal 7, knight move 11).  The nearest background
pixel of every pixel is tracked as well.  Only numpy is used.
"""

import numpy as np

# mask name -> (forward offsets, backward offsets, orthogonal weight)
# Each offset is (d_row, d_col, weight).  Forward offsets point at
# pixels already visited in raster order; backward offsets mirror them.
_MASKS = {
    "chamfer34": (
        [(-1, -1, 4), (-1, 0, 3), (-1, 1, 4), (0, -1, 3)],
        [(0, 1, 3), (1, -1, 4), (1, 0, 3), (1, 1, 4)],
        3,
    ),
    "chamfer5711": (
        [(-2, -1, 11), (-2, 1, 11), (-1, -2, 11), (-1, -1, 7),
         (-1, 0, 5), (-1, 1, 7), (-1, 2, 11), (0, -2, 11), (0, -1, 5)],
        [(0, 1, 5), (0, 2, 11), (1, -2, 11), (1, -1, 7), (1, 0, 5),
         (1, 1, 7), (1, 2, 11), (2, -1, 11), (2, 1, 11)],
        5,
    ),
}


def _validate_binary(binary):
    """Return ``binary`` as a bool array, raising ValueError on bad input."""
    binary = np.asarray(binary)
    if binary.ndim != 2:
        raise ValueError(
            f"expected a 2D binary image, got shape {binary.shape}"
        )
    if binary.size == 0:
        raise ValueError("image must not be empty")
    if binary.dtype == bool:
        return binary
    if not np.issubdtype(binary.dtype, np.number):
        raise ValueError(f"image dtype {binary.dtype} is not numeric")
    values = np.unique(binary)
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(
            "binary image may only contain 0 and 1, "
            f"got values {values[:5]}"
        )
    return binary.astype(bool)


def _validate_mask(mask):
    if mask not in _MASKS:
        raise ValueError(
            f"mask must be one of {sorted(_MASKS)}, got {mask!r}"
        )
    return _MASKS[mask]


def _shifted(dist, d_row, d_col, weight):
    """``dist`` shifted so entry (r, c) holds dist[r + d_row, c + d_col].

    Out-of-bounds entries are +inf (no neighbour there).
    """
    h, w = dist.shape
    out = np.full(dist.shape, np.inf)
    r_src = slice(max(0, d_row), h - max(0, -d_row))
    c_src = slice(max(0, d_col), w - max(0, -d_col))
    r_dst = slice(max(0, -d_row), h - max(0, d_row))
    c_dst = slice(max(0, -d_col), w - max(0, d_col))
    out[r_dst, c_dst] = dist[r_src, c_src] + weight
    return out


def _chamfer_passes(binary, mask):
    """Run the two chamfer passes; return (dist, nearest_row, nearest_col).

    ``dist`` is in raw chamfer units (divide by the orthogonal weight
    for approximate pixel units).  ``nearest_*`` hold, for every pixel,
    the coordinates of its nearest background pixel.
    """
    forward, backward, _ = _validate_mask(mask)
    h, w = binary.shape
    dist = np.where(binary, np.inf, 0.0)
    nearest_r = np.zeros((h, w), dtype=np.int64)
    nearest_c = np.zeros((h, w), dtype=np.int64)
    rows, cols = np.indices((h, w))
    nearest_r[:, :] = rows
    nearest_c[:, :] = cols

    # Vectorised offsets propagate only a few steps per sweep (the
    # classic raster scan propagates arbitrarily far in one pass
    # because each pixel sees already-updated neighbours), so repeat
    # forward + backward sweeps until nothing changes.  Every update
    # is a valid path to the background, so the fixed point is the
    # true minimum-path (chamfer) distance.
    for _ in range(h + w):
        changed = False
        for offsets in (forward, backward):
            for d_row, d_col, weight in offsets:
                cand = _shifted(dist, d_row, d_col, weight)
                better = cand < dist
                if np.any(better):
                    changed = True
                    dist = np.where(better, cand, dist)
                    # The nearest background of the neighbour that won.
                    src_r = _shifted(nearest_r.astype(np.float64),
                                     d_row, d_col, 0)
                    src_c = _shifted(nearest_c.astype(np.float64),
                                     d_row, d_col, 0)
                    nearest_r = np.where(
                        better, src_r, nearest_r).astype(np.int64)
                    nearest_c = np.where(
                        better, src_c, nearest_c).astype(np.int64)
        if not changed:
            break
    return dist, nearest_r, nearest_c


def distance_transform(binary, mask="chamfer34", scale=True):
    """Chamfer distance transform of a binary image.

    Parameters
    ----------
    binary : (H, W) array
        Binary image; nonzero (foreground) pixels get their distance
        to the nearest zero (background) pixel.  Background pixels
        have distance 0.
    mask : {"chamfer34", "chamfer5711"}
        Chamfer mask; 5-7-11 is the more accurate approximation.
    scale : bool
        If True (default), divide by the mask's orthogonal weight so
        distances are in approximate pixel units; if False, return
        raw chamfer units.

    Returns
    -------
    numpy.ndarray
        Float64 array of the same shape with the distance of every
        pixel to the nearest background pixel.
    """
    binary = _validate_binary(binary)
    if not np.any(~binary):
        raise ValueError(
            "image has no background pixels (all foreground); "
            "distances are undefined"
        )
    _, _, orthogonal = _validate_mask(mask)
    dist, _, _ = _chamfer_passes(binary, mask)
    if scale:
        dist = dist / float(orthogonal)
    return dist


def nearest_background(binary, mask="chamfer34"):
    """Nearest background pixel of every pixel.

    Parameters
    ----------
    binary : (H, W) array
        Binary image (same convention as :func:`distance_transform`).
    mask : {"chamfer34", "chamfer5711"}

    Returns
    -------
    (nearest_row, nearest_col) : tuple of int64 arrays
        For pixel (r, c), ``(nearest_row[r, c], nearest_col[r, c])``
        is the location of its nearest background pixel under the
        chamfer metric.  Background pixels map to themselves.
    """
    binary = _validate_binary(binary)
    if not np.any(~binary):
        raise ValueError(
            "image has no background pixels (all foreground); "
            "nearest background is undefined"
        )
    _validate_mask(mask)
    _, nearest_r, nearest_c = _chamfer_passes(binary, mask)
    return nearest_r, nearest_c
