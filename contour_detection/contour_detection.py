"""Contour tracing from scratch (numpy only).

Finds the outer contours of the foreground regions in a binary image:
connected components are labelled with a flood fill, then each
component's outer boundary is traced with Moore-Neighbour tracing
(Jacob's stopping criterion).  Each contour reports its points, the
polygon area (shoelace formula, through pixel centres), the perimeter
(sum of step lengths) and its axis-aligned bounding box.

Only outer contours are returned; holes inside a region are not traced
as separate contours.
"""

import numpy as np
from collections import namedtuple

__all__ = [
    "Contour",
    "bounding_box",
    "contour_area",
    "contour_perimeter",
    "find_contours",
]

Contour = namedtuple("Contour", ["points", "area", "perimeter", "bbox"])

# 8-neighbour directions, clockwise from north: N, NE, E, SE, S, SW, W, NW.
_DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
_NEIGHBOURS_8 = _DIRECTIONS


def _validate_binary(binary):
    """Return the binary image as a bool array, raising on bad input."""
    arr = np.asarray(binary)
    if arr.ndim != 2:
        raise ValueError(f"binary must be a 2D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("binary must not be empty")
    return arr != 0


def _label_components(binary):
    """Label 8-connected foreground components; return (labels, count).

    8-connectivity matches the boundary tracer, so diagonally touching
    pixels belong to one contour.
    """
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)
    count = 0
    for r in range(height):
        for c in range(width):
            if not binary[r, c] or labels[r, c] != 0:
                continue
            count += 1
            stack = [(r, c)]
            labels[r, c] = count
            while stack:
                y, x = stack.pop()
                for dy, dx in _NEIGHBOURS_8:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height and 0 <= nx < width
                            and binary[ny, nx] and labels[ny, nx] == 0):
                        labels[ny, nx] = count
                        stack.append((ny, nx))
    return labels, count


def _trace_boundary(labels, label, start):
    """Moore-Neighbour trace of one component's outer boundary.

    Returns a list of (row, col) pixels in order, closed by repeating
    the start pixel at the end (a single isolated pixel returns [start]).
    """
    height, width = labels.shape
    contour = [start]
    current = start
    backtrack = 6  # we "arrived" from the west neighbour
    max_steps = labels.size + 10
    for _ in range(max_steps):
        nxt = None
        nxt_dir = None
        for k in range(8):
            direction = (backtrack + 1 + k) % 8
            dy, dx = _DIRECTIONS[direction]
            ny, nx = current[0] + dy, current[1] + dx
            if (0 <= ny < height and 0 <= nx < width
                    and labels[ny, nx] == label):
                nxt = (ny, nx)
                nxt_dir = direction
                break
        if nxt is None:
            break
        # Jacob's stopping criterion: back at the start and about to
        # repeat the first step -> the loop is closed.
        if len(contour) > 1 and current == start and nxt == contour[1]:
            break
        contour.append(nxt)
        backtrack = (nxt_dir + 4) % 8
        current = nxt
    return contour


def _validate_points(points):
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {pts.shape}")
    return pts


def contour_area(points):
    """Polygon area of a contour via the shoelace formula.

    Points are (x, y) pixel centres, so this is the area of the polygon
    through the boundary pixels, not the count of foreground pixels.
    Returns 0.0 for fewer than 3 points.
    """
    pts = _validate_points(points)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0].astype(float)
    y = pts[:, 1].astype(float)
    return abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def contour_perimeter(points):
    """Perimeter of a contour: sum of distances between consecutive points.

    The contour is treated as closed, so the segment from the last
    point back to the first is included (it is zero-length for the
    already-closed contours returned by :func:`find_contours`).
    Returns 0.0 for fewer than 2 points.
    """
    pts = _validate_points(points)
    if len(pts) < 2:
        return 0.0
    closed = np.vstack([pts.astype(float), pts[:1].astype(float)])
    steps = np.diff(closed, axis=0)
    return float(np.sum(np.hypot(steps[:, 0], steps[:, 1])))


def bounding_box(points):
    """Axis-aligned bounding box of a contour as (x_min, y_min, x_max, y_max)."""
    pts = _validate_points(points)
    if len(pts) == 0:
        raise ValueError("points must not be empty")
    return (int(pts[:, 0].min()), int(pts[:, 1].min()),
            int(pts[:, 0].max()), int(pts[:, 1].max()))


def find_contours(binary):
    """Find the outer contours of the foreground regions in a binary image.

    Parameters
    ----------
    binary : 2D array
        Non-zero pixels are foreground.

    Returns
    -------
    list of Contour, largest area first.  Each contour's ``points`` is
    an (N, 2) int array of (x, y) pixel coordinates in trace order,
    closed by repeating the start pixel at the end.
    """
    binary = _validate_binary(binary)
    labels, count = _label_components(binary)
    contours = []
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        start = (int(ys[0]), int(xs[0]))
        traced = _trace_boundary(labels, label, start)
        points = np.array([(x, y) for (y, x) in traced], dtype=int)
        contours.append(Contour(
            points=points,
            area=contour_area(points),
            perimeter=contour_perimeter(points),
            bbox=bounding_box(points),
        ))
    contours.sort(key=lambda c: c.area, reverse=True)
    return contours
