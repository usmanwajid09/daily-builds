"""Simple watershed segmentation from scratch (numpy only).

Segments a binary mask into regions, one per "basin".  The distance
transform of the mask is computed with a two-pass chamfer algorithm,
its regional maxima become markers, and a priority flood (highest
distance first) grows each marker's basin until basins meet, where a
watershed line (label -1) is drawn.  Touching objects that are
separate in the distance transform — e.g. two overlapping discs — are
split into two segments.
"""

import heapq

import numpy as np

WATERSHED = -1

__all__ = ["distance_transform", "find_markers", "watershed",
           "segment_count", "watershed_lines", "WATERSHED"]


def _validate_mask(mask):
    """Return the mask as a bool array, raising on bad input."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be a 2D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("mask must not be empty")
    return arr.astype(bool)


def distance_transform(mask):
    """Approximate Euclidean distance transform of a binary mask.

    Returns a float array where each foreground pixel holds its
    (chamfer-approximated) distance to the nearest background pixel
    and background pixels hold 0.
    """
    mask = _validate_mask(mask)
    height, width = mask.shape
    dist = np.where(mask, height + width, 0).astype(np.float64)
    sqrt2 = float(np.sqrt(2.0))
    # Forward pass: top, left, top-left, top-right neighbours.
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            best = dist[y, x]
            if y > 0:
                best = min(best, dist[y - 1, x] + 1.0)
            if x > 0:
                best = min(best, dist[y, x - 1] + 1.0)
            if y > 0 and x > 0:
                best = min(best, dist[y - 1, x - 1] + sqrt2)
            if y > 0 and x < width - 1:
                best = min(best, dist[y - 1, x + 1] + sqrt2)
            dist[y, x] = best
    # Backward pass: bottom, right, bottom-right, bottom-left.
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            if not mask[y, x]:
                continue
            best = dist[y, x]
            if y < height - 1:
                best = min(best, dist[y + 1, x] + 1.0)
            if x < width - 1:
                best = min(best, dist[y, x + 1] + 1.0)
            if y < height - 1 and x < width - 1:
                best = min(best, dist[y + 1, x + 1] + sqrt2)
            if y < height - 1 and x > 0:
                best = min(best, dist[y + 1, x - 1] + sqrt2)
            dist[y, x] = best
    return dist


def _label_components(binary, shape):
    """Label 8-connected components of ``binary`` (1..N), 0 elsewhere."""
    height, width = shape
    labels = np.zeros(shape, dtype=np.int64)
    label = 0
    for y in range(height):
        for x in range(width):
            if binary[y, x] and labels[y, x] == 0:
                label += 1
                stack = [(y, x)]
                labels[y, x] = label
                while stack:
                    cy, cx = stack.pop()
                    for ny in range(max(0, cy - 1), min(height, cy + 2)):
                        for nx in range(max(0, cx - 1), min(width, cx + 2)):
                            if binary[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = label
                                stack.append((ny, nx))
    return labels


def find_markers(mask, min_distance=1.0):
    """Find markers as regional maxima of the distance transform.

    Returns an int array: 0 for background, a positive id per marker
    region.  ``min_distance`` drops maxima shallower than that many
    pixels (useful to ignore noise bumps).
    """
    mask = _validate_mask(mask)
    if min_distance < 0:
        raise ValueError("min_distance must be >= 0")
    if not mask.any():
        raise ValueError("mask is empty: no foreground pixels to segment")
    dist = distance_transform(mask)
    height, width = dist.shape
    padded = np.pad(dist, 1, mode="constant", constant_values=-1.0)
    is_max = np.ones_like(dist, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            is_max &= dist >= padded[dy:dy + height, dx:dx + width]
    candidates = mask & is_max & (dist >= min_distance)
    return _label_components(candidates, dist.shape)


def _check_markers_cover(mask, markers):
    """Raise if any connected foreground component lacks a marker."""
    components = _label_components(mask, mask.shape)
    for component_id in range(1, int(components.max()) + 1):
        if not np.any(markers[components == component_id] > 0):
            raise ValueError(
                "every connected component of the mask must contain "
                "at least one marker")


def watershed(mask, markers=None, min_distance=5.0):
    """Segment a binary mask by priority-flood watershed.

    Parameters
    ----------
    mask : 2D bool array
        Foreground to segment.
    markers : 2D int array, optional
        Seed labels (0 = unseeded).  If omitted, markers come from
        :func:`find_markers` with ``min_distance``.
    min_distance : float
        Passed to :func:`find_markers` when ``markers`` is None.

    Returns
    -------
    int array, same shape as ``mask``: positive ids are segments,
    0 is background, -1 (``WATERSHED``) is a watershed line.
    """
    mask = _validate_mask(mask)
    if not mask.any():
        raise ValueError("mask is empty: no foreground pixels to segment")
    if markers is None:
        markers = find_markers(mask, min_distance=min_distance)
    markers = np.asarray(markers)
    if markers.shape != mask.shape:
        raise ValueError(
            f"markers shape {markers.shape} does not match mask shape "
            f"{mask.shape}")
    if markers.min() < 0:
        raise ValueError("markers must be >= 0")
    if np.any((markers > 0) & ~mask):
        raise ValueError("markers must lie inside the mask foreground")
    if markers.max() == 0:
        raise ValueError("no markers found: cannot segment the mask")
    _check_markers_cover(mask, markers)

    dist = distance_transform(mask)
    height, width = mask.shape
    labels = markers.astype(np.int64).copy()
    heap = []
    for y, x in zip(*np.nonzero(markers > 0)):
        heapq.heappush(heap, (-dist[y, x], int(y), int(x)))
    while heap:
        _, y, x = heapq.heappop(heap)
        current = int(labels[y, x])
        if current == WATERSHED:
            continue
        # If neighbours already belong to two different basins, this
        # pixel is on the boundary: mark it as a watershed line.
        seen = set()
        for ny in range(max(0, y - 1), min(height, y + 2)):
            for nx in range(max(0, x - 1), min(width, x + 2)):
                value = int(labels[ny, nx])
                if value > 0:
                    seen.add(value)
        if len(seen) >= 2:
            labels[y, x] = WATERSHED
            continue
        for ny in range(max(0, y - 1), min(height, y + 2)):
            for nx in range(max(0, x - 1), min(width, x + 2)):
                if mask[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = current
                    heapq.heappush(heap, (-dist[ny, nx], ny, nx))
    return labels


def segment_count(labels):
    """Return the number of segments (positive labels) in a label map."""
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError(f"labels must be a 2D array, got shape {arr.shape}")
    return int(len(np.unique(arr[arr > 0])))


def watershed_lines(labels):
    """Return a bool mask of watershed-line pixels (label -1)."""
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError(f"labels must be a 2D array, got shape {arr.shape}")
    return arr == WATERSHED
