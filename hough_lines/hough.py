"""Hough transform line detection from scratch (numpy only).

Implements the classic Hough transform for straight lines: every edge
pixel votes for all (rho, theta) lines that could pass through it, then
the accumulator peaks are picked out as detected lines.  A line is
parameterised as ``x * cos(theta) + y * sin(theta) = rho`` with ``theta``
in degrees in [-90, 90) and ``rho`` in pixels.
"""

import numpy as np

__all__ = ["hough_transform", "detect_lines", "line_endpoints"]


def _validate_edge_map(edge_map):
    """Return the edge map as an array, raising on bad input."""
    arr = np.asarray(edge_map)
    if arr.ndim != 2:
        raise ValueError(f"edge_map must be a 2D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("edge_map must not be empty")
    return arr


def hough_transform(edge_map, theta_step=1.0, rho_step=1.0):
    """Compute the Hough accumulator for a binary edge map.

    Parameters
    ----------
    edge_map : 2D array
        Non-zero entries are treated as edge pixels.
    theta_step : float
        Angle resolution in degrees (must be positive).
    rho_step : float
        Distance resolution in pixels (must be positive).

    Returns
    -------
    (accumulator, thetas, rhos)
        ``accumulator`` is an ``(len(rhos), len(thetas))`` int array of
        votes, ``thetas`` are angles in radians, ``rhos`` are distances
        in pixels.
    """
    arr = _validate_edge_map(edge_map)
    if theta_step <= 0:
        raise ValueError("theta_step must be positive")
    if rho_step <= 0:
        raise ValueError("rho_step must be positive")
    edges = arr != 0
    height, width = arr.shape
    thetas = np.deg2rad(np.arange(-90.0, 90.0, theta_step))
    max_rho = float(np.hypot(height, width))
    rhos = np.arange(-max_rho, max_rho + rho_step, rho_step)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int64)
    ys, xs = np.nonzero(edges)
    if xs.size == 0:
        return accumulator, thetas, rhos
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    for x, y in zip(xs.tolist(), ys.tolist()):
        rho_values = x * cos_t + y * sin_t
        rho_idx = np.round((rho_values - rhos[0]) / rho_step).astype(np.int64)
        valid = (rho_idx >= 0) & (rho_idx < len(rhos))
        np.add.at(accumulator, (rho_idx[valid], np.nonzero(valid)[0]), 1)
    return accumulator, thetas, rhos


def detect_lines(edge_map, threshold=10, min_rho_dist=10.0,
                 min_theta_dist=10.0, theta_step=1.0, rho_step=1.0):
    """Detect lines in an edge map via Hough peak picking.

    Parameters
    ----------
    edge_map : 2D array
        Non-zero entries are treated as edge pixels.
    threshold : int
        Minimum accumulator votes for a peak (must be >= 1).
    min_rho_dist, min_theta_dist : float
        Minimum separation (pixels / degrees) between two reported
        lines; closer peaks are suppressed (non-maximum suppression).
    theta_step, rho_step : float
        Accumulator resolutions, passed to :func:`hough_transform`.

    Returns
    -------
    list of (rho, theta_degrees, votes)
        Sorted by votes, highest first.  Empty list if nothing passes
        the threshold.
    """
    if threshold < 1:
        raise ValueError("threshold must be at least 1 vote")
    if min_rho_dist < 0 or min_theta_dist < 0:
        raise ValueError("min_rho_dist and min_theta_dist must be >= 0")
    accumulator, thetas, rhos = hough_transform(
        edge_map, theta_step=theta_step, rho_step=rho_step)
    n_theta = len(thetas)
    candidates = np.argwhere(accumulator >= threshold)
    if len(candidates) == 0:
        return []
    votes = accumulator[candidates[:, 0], candidates[:, 1]]
    # Stable sort so ties resolve deterministically (lowest rho first).
    order = np.argsort(-votes, kind="stable")
    accepted = []
    for idx in order:
        rho_val = float(rhos[candidates[idx][0]])
        theta_val = float(np.rad2deg(thetas[candidates[idx][1]]))
        vote = int(accumulator[candidates[idx][0], candidates[idx][1]])
        clash = False
        for acc_rho, acc_theta, _ in accepted:
            raw_theta = abs(theta_val - acc_theta)
            theta_dist = min(raw_theta, 180.0 - raw_theta)
            if theta_dist > min_theta_dist:
                continue
            # Same line, or the same line with (rho, theta) flipped to
            # (-rho, theta + 180 deg): both collapse to one detection.
            if (abs(rho_val - acc_rho) <= min_rho_dist or
                    abs(rho_val + acc_rho) <= min_rho_dist):
                clash = True
                break
        if not clash:
            accepted.append((rho_val, theta_val, vote))
    return accepted


def line_endpoints(rho, theta_degrees, shape):
    """Return the two endpoints of a line clipped to an image.

    Parameters
    ----------
    rho : float
        Perpendicular distance from the origin (top-left corner).
    theta_degrees : float
        Angle of the line normal, in degrees.
    shape : (height, width)
        Image shape the line is clipped to.

    Returns
    -------
    ((x1, y1), (x2, y2))
        Endpoints in (column, row) coordinates, on the image border.
    """
    if len(shape) != 2:
        raise ValueError(f"shape must be (height, width), got {shape}")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("shape dimensions must be positive")
    theta = np.deg2rad(float(theta_degrees))
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    points = []
    if abs(sin_t) > 1e-12:
        for x in (0.0, float(width - 1)):
            y = (rho - x * cos_t) / sin_t
            if -1e-9 <= y <= height - 1 + 1e-9:
                points.append((x, min(max(y, 0.0), float(height - 1))))
    if abs(cos_t) > 1e-12:
        for y in (0.0, float(height - 1)):
            x = (rho - y * sin_t) / cos_t
            if -1e-9 <= x <= width - 1 + 1e-9:
                points.append((min(max(x, 0.0), float(width - 1)), y))
    unique = []
    for point in points:
        if not any(abs(point[0] - q[0]) < 1e-9 and
                   abs(point[1] - q[1]) < 1e-9 for q in unique):
            unique.append(point)
    if len(unique) < 2:
        raise ValueError("line does not cross the image")
    best = (unique[0], unique[1])
    best_dist = -1.0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            dist = (unique[i][0] - unique[j][0]) ** 2 + \
                   (unique[i][1] - unique[j][1]) ** 2
            if dist > best_dist:
                best_dist, best = dist, (unique[i], unique[j])
    return best
