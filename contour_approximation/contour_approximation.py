"""Contour approximation from scratch (numpy only).

A binary image's outer boundary is traced with Moore-Neighbour
tracing, then simplified with the Douglas-Peucker algorithm into a
polygon with far fewer vertices. The result reports the simplified
vertices, the polygon area (shoelace formula) and the perimeter.

Moore-Neighbour tracing walks the boundary keeping the foreground on
one side: from the current boundary pixel it scans the eight
neighbours clockwise, starting just after the direction it came
from, and steps to the first foreground neighbour. Tracing stops
when it returns to the start pixel heading for the second pixel
(Jacob's stopping criterion).

Douglas-Peucker simplifies an open polyline recursively: keep the
endpoints, find the interior point farthest from the line between
them, and keep it (splitting there) only if its distance exceeds
``epsilon``. A closed contour is split at the point farthest from
its start pixel and simplified as two open chains.
"""

import numpy as np

__all__ = [
    "approximate_contour",
    "douglas_peucker",
    "polygon_area",
    "polygon_perimeter",
    "trace_contour",
]

# Neighbour offsets in clockwise order, starting east.
# (row, col): E, SE, S, SW, W, NW, N, NE.
_DIRECTIONS = (
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
)


def _validate_binary(binary):
    binary = np.asarray(binary)
    if binary.ndim != 2:
        raise ValueError(f"binary image must be 2-D, got {binary.ndim}-D")
    if binary.size == 0:
        raise ValueError("binary image must not be empty")
    if binary.dtype != bool:
        raise ValueError(f"binary image must be boolean, got dtype {binary.dtype}")
    return binary


def _first_foreground(binary):
    """Return (row, col) of the topmost, then leftmost, foreground pixel."""
    rows, cols = np.nonzero(binary)
    if len(rows) == 0:
        return None
    order = np.lexsort((cols, rows))
    return int(rows[order[0]]), int(cols[order[0]])


def trace_contour(binary):
    """Trace the outer boundary of the foreground region.

    Args:
        binary: 2-D boolean array. Only the region containing the
            topmost-leftmost foreground pixel is traced.

    Returns:
        ``(N, 2)`` int array of ``(x, y)`` points in trace order,
        without repeating the start pixel at the end. Returns an empty
        ``(0, 2)`` array when there is no foreground pixel.

    Raises:
        ValueError: the image is not a non-empty 2-D boolean array.
    """
    binary = _validate_binary(binary)
    start = _first_foreground(binary)
    if start is None:
        return np.empty((0, 2), dtype=int)

    height, width = binary.shape
    # We "enter" the start pixel from the west, so the backtrack
    # direction (start -> previous pixel) is W, index 4.
    backtrack = 4
    current = start
    points = [(start[1], start[0])]  # (x, y)
    while True:
        found = None
        for step in range(1, 9):
            direction = (backtrack + step) % 8
            dr, dc = _DIRECTIONS[direction]
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < height and 0 <= nc < width and binary[nr, nc]:
                found = (nr, nc, direction)
                break
        if found is None:
            # Isolated single pixel: nothing to walk to.
            break
        nr, nc, direction = found
        # Backtrack direction from the new pixel to the current one.
        new_backtrack = (direction + 4) % 8
        if (nr, nc) == start:
            # Back at the start pixel: the loop is closed (Jacob's
            # stopping criterion). Do not repeat the start pixel.
            break
        points.append((nc, nr))
        current = (nr, nc)
        backtrack = new_backtrack
        if len(points) > binary.size + 2:  # safety net, never expected
            raise RuntimeError("contour trace did not terminate")
    return np.array(points, dtype=int)


def _perpendicular_distances(points, start, end):
    """Distance of each point from the line through ``start``-``end``."""
    points = np.asarray(points, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    length = np.hypot(segment[0], segment[1])
    if length == 0:
        return np.hypot(points[:, 0] - start[0], points[:, 1] - start[1])
    # |cross(segment, point - start)| / |segment|
    cross = np.abs(segment[0] * (points[:, 1] - start[1])
                   - segment[1] * (points[:, 0] - start[0]))
    return cross / length


def _simplify_open(points, epsilon):
    """Douglas-Peucker on an open polyline (endpoints always kept)."""
    points = np.asarray(points, dtype=float)
    if len(points) <= 2:
        return points
    distances = _perpendicular_distances(points, points[0], points[-1])
    interior = distances[1:-1]
    if len(interior) == 0:
        return points
    k = int(np.argmax(interior)) + 1
    if interior[k - 1] <= epsilon:
        return np.array([points[0], points[-1]])
    left = _simplify_open(points[:k + 1], epsilon)
    right = _simplify_open(points[k:], epsilon)
    return np.vstack([left[:-1], right])


def douglas_peucker(points, epsilon):
    """Simplify a polyline with the Douglas-Peucker algorithm.

    Args:
        points: ``(N, 2)`` array of ``(x, y)`` points. If the first and
            last points coincide the polyline is treated as closed.
        epsilon: maximum allowed deviation; points closer than this
            to the approximating segment are dropped. Must be >= 0.

    Returns:
        Simplified ``(M, 2)`` float array, ``M <= N``.

    Raises:
        ValueError: fewer than 2 points, a bad shape, or a negative
            epsilon.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    if len(points) < 2:
        raise ValueError(f"need at least 2 points, got {len(points)}")
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")

    closed = np.array_equal(points[0], points[-1])
    if closed:
        points = points[:-1]
    if len(points) <= 2 or epsilon == 0:
        result = points.copy()
    elif closed:
        # Split the loop at the point farthest from the start pixel,
        # simplify the two open chains, then re-join them.
        distances = np.hypot(points[:, 0] - points[0, 0],
                             points[:, 1] - points[0, 1])
        k = int(np.argmax(distances))
        first = _simplify_open(points[:k + 1], epsilon)
        second = _simplify_open(np.vstack([points[k:], points[:1]]), epsilon)
        result = np.vstack([first[:-1], second[:-1]])
    else:
        result = _simplify_open(points, epsilon)

    if closed:
        result = np.vstack([result, result[:1]])
    return result


def polygon_area(points):
    """Shoelace area of a polygon given as an ``(N, 2)`` point array."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(points):
    """Closed-polygon perimeter of an ``(N, 2)`` point array."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    steps = np.diff(np.vstack([points, points[:1]]), axis=0)
    return float(np.hypot(steps[:, 0], steps[:, 1]).sum())


def approximate_contour(binary, epsilon=1.0):
    """Trace and simplify the outer contour of a binary image.

    Args:
        binary: 2-D boolean array.
        epsilon: Douglas-Peucker tolerance in pixels.

    Returns:
        Dict with ``vertices`` (``(M, 2)`` float array, closed by
        repeating the first vertex at the end), ``area`` (shoelace
        area of the simplified polygon), ``perimeter``, and
        ``raw_points`` (the ``(N, 2)`` traced contour before
        simplification). An empty image gives zero vertices, zero
        area and zero perimeter.
    """
    binary = _validate_binary(binary)
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    raw = trace_contour(binary)
    if len(raw) == 0:
        return {"vertices": np.empty((0, 2)), "area": 0.0,
                "perimeter": 0.0, "raw_points": raw}
    closed = np.vstack([raw, raw[:1]])
    vertices = douglas_peucker(closed, epsilon)
    return {
        "vertices": vertices,
        "area": polygon_area(vertices),
        "perimeter": polygon_perimeter(vertices),
        "raw_points": raw,
    }
