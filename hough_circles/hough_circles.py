"""Hough circle detection from scratch (numpy only).

Every edge pixel votes for all centres that could lie on a circle of a
given radius through it.  Votes accumulate in a 3D accumulator indexed by
``(row, col, radius)``; peaks in the accumulator are circles.  Peaks are
picked out with non-maximum suppression so one real circle is reported
once even if neighbouring radii or centres also score highly.

A circle is reported as :class:`Circle` with ``x`` (column), ``y``
(row), ``radius`` and ``votes`` (the accumulator peak value, roughly the
number of edge pixels on the circle's circumference).
"""

import numpy as np
from collections import namedtuple

__all__ = [
    "Circle",
    "circle_edge_map",
    "detect_circles",
    "hough_circle_transform",
    "sobel_edges",
]

Circle = namedtuple("Circle", ["x", "y", "radius", "votes"])


def _validate_edge_map(edge_map):
    """Return the edge map as a bool array, raising on bad input."""
    arr = np.asarray(edge_map)
    if arr.ndim != 2:
        raise ValueError(f"edge_map must be a 2D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("edge_map must not be empty")
    return arr.astype(bool)


def _validate_radii(radii):
    """Return radii as a list of positive ints, raising on bad input."""
    try:
        radii = list(radii)
    except TypeError:
        raise ValueError(f"radii must be a sequence of positive integers, got {radii!r}")
    if len(radii) == 0:
        raise ValueError("radii must not be empty")
    out = []
    for radius in radii:
        if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)):
            raise ValueError(f"radii must be positive integers, got {radius!r}")
        if radius <= 0:
            raise ValueError(f"radii must be positive, got {radius}")
        out.append(int(radius))
    return out


def _convolve2d(image, kernel):
    """2D convolution with zero padding (numpy only, kernel must be odd)."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)))
    out = np.zeros_like(image, dtype=float)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + image.shape[0], j:j + image.shape[1]]
    return out


def sobel_edges(gray, threshold=50.0):
    """Return a boolean edge map from a grayscale image (Sobel magnitude).

    Parameters
    ----------
    gray : 2D array
        Grayscale image.
    threshold : float
        Gradient magnitude above which a pixel counts as an edge.

    Returns
    -------
    2D bool array, True where the gradient magnitude exceeds ``threshold``.
    """
    arr = np.asarray(gray, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"gray must be a 2D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("gray must not be empty")
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    gx = _convolve2d(arr, kx)
    gy = _convolve2d(arr, ky)
    return np.hypot(gx, gy) > threshold


def circle_edge_map(shape, circles):
    """Build a synthetic boolean edge map with circles drawn on it.

    Parameters
    ----------
    shape : (height, width)
        Size of the output image.
    circles : iterable of (x, y, radius)
        Circle centres (``x`` = column, ``y`` = row) and radii in pixels.

    Returns
    -------
    2D bool array with True on each circle's circumference.
    """
    if len(shape) != 2:
        raise ValueError(f"shape must be (height, width), got {shape!r}")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"shape must have positive dimensions, got {shape!r}")
    edges = np.zeros((height, width), dtype=bool)
    angles = np.deg2rad(np.arange(0, 360, 1.0))
    for circle in circles:
        x, y, radius = circle
        if radius <= 0:
            raise ValueError(f"circle radius must be positive, got {radius}")
        xs = np.round(x + radius * np.cos(angles)).astype(int)
        ys = np.round(y + radius * np.sin(angles)).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        edges[ys[valid], xs[valid]] = True
    return edges


def hough_circle_transform(edge_map, radii, angle_step=1.0):
    """Compute the Hough accumulator for circles.

    Parameters
    ----------
    edge_map : 2D array
        Boolean (or non-zero = edge) edge map.
    radii : sequence of positive ints
        Radii to search for, in pixels.
    angle_step : float
        Angular step in degrees used when voting for centres.

    Returns
    -------
    (accumulator, radii)
        ``accumulator`` is an int64 array of shape
        ``(height, width, len(radii))``; ``radii`` is the validated list.
    """
    edges = _validate_edge_map(edge_map)
    radii = _validate_radii(radii)
    if angle_step <= 0 or angle_step > 90:
        raise ValueError(f"angle_step must be in (0, 90], got {angle_step}")
    height, width = edges.shape
    acc = np.zeros((height, width, len(radii)), dtype=np.int64)
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return acc, radii
    angles = np.deg2rad(np.arange(0, 360, angle_step))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    for r_idx, radius in enumerate(radii):
        dx = np.round(radius * cos_a).astype(np.int64)
        dy = np.round(radius * sin_a).astype(np.int64)
        # Several angles round to the same integer offset; vote once per
        # distinct offset so a peak counts edge pixels, not angles.
        offsets = np.unique(np.stack([dx, dy], axis=1), axis=0)
        dx, dy = offsets[:, 0], offsets[:, 1]
        cx = (xs[:, None] - dx[None, :]).ravel()
        cy = (ys[:, None] - dy[None, :]).ravel()
        valid = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
        flat = (cy[valid] * width + cx[valid]) * len(radii) + r_idx
        np.add.at(acc.ravel(), flat, 1)
    return acc, radii


def detect_circles(edge_map, radii, threshold=None, min_distance=5, angle_step=1.0):
    """Detect circles in an edge map.

    Parameters
    ----------
    edge_map : 2D array
        Boolean (or non-zero = edge) edge map.
    radii : sequence of positive ints
        Radii to search for, in pixels.
    threshold : int or None
        Minimum accumulator votes for a peak to count as a circle.
        Defaults to ``max(8, int(0.5 * 2 * pi * min(radii)))``.
    min_distance : int
        Centres closer than this (with radii within 2 of each other)
        are treated as the same circle; only the strongest is kept.
    angle_step : float
        Angular step in degrees used when voting for centres.

    Returns
    -------
    list of Circle, strongest first.
    """
    edges = _validate_edge_map(edge_map)
    radii = _validate_radii(radii)
    if min_distance < 1:
        raise ValueError(f"min_distance must be >= 1, got {min_distance}")
    if threshold is not None and threshold < 1:
        raise ValueError(f"threshold must be at least 1, got {threshold}")
    acc, radii = hough_circle_transform(edges, radii, angle_step=angle_step)
    if threshold is None:
        threshold = max(8, int(0.5 * 2 * np.pi * min(radii)))
    ys, xs, r_idx = np.nonzero(acc >= threshold)
    if len(xs) == 0:
        return []
    votes = acc[ys, xs, r_idx]
    order = np.argsort(-votes, kind="stable")
    circles = []
    for idx in order:
        x, y, r = int(xs[idx]), int(ys[idx]), radii[int(r_idx[idx])]
        duplicate = False
        for found in circles:
            if abs(found.radius - r) <= 2 and (found.x - x) ** 2 + (found.y - y) ** 2 < min_distance ** 2:
                duplicate = True
                break
        if not duplicate:
            circles.append(Circle(x=x, y=y, radius=r, votes=int(votes[idx])))
    return circles
