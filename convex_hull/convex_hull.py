"""Convex hull from scratch (numpy only).

Two classic algorithms are implemented and cross-checked against
each other:

- **Graham scan** (``graham_scan``): pick the lowest point as pivot,
  sort the rest by polar angle around it, then walk the sorted list
  keeping only left turns. Runs in O(n log n).
- **Jarvis march / gift wrapping** (``jarvis_march``): from the
  leftmost point, repeatedly pick the most counter-clockwise next
  point until the hull closes. Runs in O(n h) for h hull points.

Both return the hull vertices counter-clockwise, starting at the
lowest point, without repeating the start vertex. Collinear points
on a hull edge are dropped so only the extreme endpoints are kept.

Also provided: ``polygon_area`` (shoelace formula) and
``point_in_hull`` (a point is inside a counter-clockwise convex
polygon exactly when it is to the left of every edge).
"""

import numpy as np

__all__ = [
    "graham_scan",
    "jarvis_march",
    "point_in_hull",
    "polygon_area",
]


def _validate_points(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    if len(points) == 0:
        raise ValueError("points must not be empty")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite (no NaN or inf)")
    return points


def _cross(o, a, b):
    """Z-component of (a - o) x (b - o); > 0 means b is left of o->a."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _dist2(a, b):
    return float((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def graham_scan(points):
    """Convex hull of ``points`` via the Graham scan.

    Args:
        points: ``(N, 2)`` array of ``(x, y)`` points.

    Returns:
        ``(H, 2)`` float array of hull vertices counter-clockwise.
        A single unique point returns that point; two unique points
        (or any collinear set) return the two extreme endpoints.

    Raises:
        ValueError: bad shape, empty input, or non-finite coordinates.
    """
    points = _validate_points(points)
    unique = np.unique(points, axis=0)
    if len(unique) <= 2:
        return unique

    # Pivot: lowest y, then lowest x.
    pivot_idx = int(np.lexsort((unique[:, 0], unique[:, 1]))[0])
    pivot = unique[pivot_idx]
    rest = np.delete(unique, pivot_idx, axis=0)

    # Sort by polar angle around the pivot (atan2), breaking ties by
    # distance so nearer collinear points come first and are dropped.
    angles = np.arctan2(rest[:, 1] - pivot[1], rest[:, 0] - pivot[0])
    dists = (rest[:, 0] - pivot[0]) ** 2 + (rest[:, 1] - pivot[1]) ** 2
    order = np.lexsort((dists, angles))
    ordered = rest[order]

    hull = [pivot, ordered[0]]
    for point in ordered[1:]:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)
    return np.array(hull, dtype=float)


def jarvis_march(points):
    """Convex hull of ``points`` via the Jarvis march (gift wrapping).

    Same contract as :func:`graham_scan`; slower on large inputs but
    independent, so it serves as a cross-check in the tests.
    """
    points = _validate_points(points)
    unique = np.unique(points, axis=0)
    if len(unique) <= 2:
        return unique

    # Start at the leftmost point (lowest x, then lowest y).
    start_idx = int(np.lexsort((unique[:, 1], unique[:, 0]))[0])
    start = unique[start_idx]
    hull = [start]
    current = start
    while True:
        # Pick the point most counter-clockwise from `current`;
        # on a tie (collinear) take the farthest one.
        candidate = unique[0] if not np.array_equal(unique[0], current) else unique[1]
        for point in unique:
            if np.array_equal(point, current):
                continue
            cross = _cross(current, candidate, point)
            if cross > 0 or (cross == 0
                             and _dist2(current, point) > _dist2(current, candidate)):
                candidate = point
        if np.array_equal(candidate, start):
            break
        hull.append(candidate)
        current = candidate
        if len(hull) > len(unique):  # safety net, never expected
            raise RuntimeError("Jarvis march did not terminate")
    return np.array(hull, dtype=float)


def polygon_area(points):
    """Shoelace area of a polygon given as an ``(N, 2)`` point array."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_in_hull(point, hull):
    """Test whether ``point`` lies inside (or on) a convex hull.

    Args:
        point: ``(x, y)`` pair.
        hull: ``(H, 2)`` counter-clockwise hull vertices as returned
            by :func:`graham_scan` or :func:`jarvis_march`.

    Returns:
        True when the point is inside the hull or on its boundary.
        A 1-point hull matches only that point; a 2-point hull
        matches points on the segment between them.
    """
    point = np.asarray(point, dtype=float)
    if point.shape != (2,):
        raise ValueError(f"point must have shape (2,), got {point.shape}")
    hull = _validate_points(hull)
    if len(hull) == 1:
        return bool(np.array_equal(point, hull[0]))
    if len(hull) == 2:
        a, b = hull
        if _cross(a, b, point) != 0:
            return False
        return bool(min(a[0], b[0]) <= point[0] <= max(a[0], b[0])
                    and min(a[1], b[1]) <= point[1] <= max(a[1], b[1]))
    # Inside a CCW convex polygon iff the point is left of (or on)
    # every directed edge.
    for i in range(len(hull)):
        if _cross(hull[i], hull[(i + 1) % len(hull)], point) < 0:
            return False
    return True
