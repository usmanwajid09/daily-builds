"""Lucas-Kanade optical flow from scratch (numpy only).

Implements the classic Lucas-Kanade method: for a small window around a
point, assume the motion is constant and brightness is conserved, so
``Ix * u + Iy * v + It = 0`` for every pixel in the window.  The flow
``(u, v)`` is the least-squares solution of that over-determined system.
Windows whose gradient matrix is (near-)singular -- flat regions, or a
single straight edge (the aperture problem) -- are reported as zero flow.
"""

import numpy as np

__all__ = ["dense_flow", "lucas_kanade", "spatial_gradients"]


def _validate_image(image, name="image"):
    """Return the image as a float64 2D array, raising on bad input."""
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _validate_window_size(window_size):
    if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)):
        raise ValueError(f"window_size must be an odd integer, got {window_size!r}")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be an odd integer >= 3, got {window_size}")
    return int(window_size)


def spatial_gradients(image):
    """Return ``(Ix, Iy)`` spatial gradients of a grayscale image.

    Central differences via :func:`numpy.gradient`.  ``Ix`` is the
    derivative along columns (x), ``Iy`` along rows (y).
    """
    arr = _validate_image(image)
    iy, ix = np.gradient(arr)
    return ix, iy


def _min_eigenvalue(sxx, sxy, syy):
    """Smallest eigenvalue of the symmetric 2x2 matrix [[sxx, sxy], [sxy, syy]]."""
    trace = sxx + syy
    disc = np.sqrt(max((sxx - syy) ** 2 + 4 * sxy ** 2, 0.0))
    return 0.5 * (trace - disc)


def _flow_at_point(ix, iy, it, x, y, half, min_eigenvalue):
    """Lucas-Kanade flow at one point, or (0, 0) if unreliable."""
    height, width = ix.shape
    y0, y1 = max(y - half, 0), min(y + half + 1, height)
    x0, x1 = max(x - half, 0), min(x + half + 1, width)
    if y1 - y0 < 3 or x1 - x0 < 3:
        return 0.0, 0.0
    wx = ix[y0:y1, x0:x1].ravel()
    wy = iy[y0:y1, x0:x1].ravel()
    wt = it[y0:y1, x0:x1].ravel()
    sxx = float(np.dot(wx, wx))
    sxy = float(np.dot(wx, wy))
    syy = float(np.dot(wy, wy))
    if _min_eigenvalue(sxx, sxy, syy) < min_eigenvalue:
        return 0.0, 0.0
    sxt = float(np.dot(wx, -wt))
    syt = float(np.dot(wy, -wt))
    det = sxx * syy - sxy * sxy
    if abs(det) < 1e-12:
        return 0.0, 0.0
    u = (syy * sxt - sxy * syt) / det
    v = (sxx * syt - sxy * sxt) / det
    return float(u), float(v)


def lucas_kanade(prev, curr, points, window_size=15, min_eigenvalue=1e-3):
    """Estimate optical flow at given points between two frames.

    Parameters
    ----------
    prev, curr : 2D arrays
        Consecutive grayscale frames, same shape.
    points : (N, 2) array of (x, y)
        Points (``x`` = column, ``y`` = row) to track.
    window_size : odd int >= 3
        Side of the square window used for the least-squares fit.
    min_eigenvalue : float
        Windows whose gradient matrix has a smaller minimum eigenvalue
        than this are unreliable and get flow (0, 0).

    Returns
    -------
    (N, 2) float array of ``(u, v)`` displacements: ``u`` along x
    (columns), ``v`` along y (rows).  Unreliable points get (0, 0).
    """
    prev = _validate_image(prev, "prev")
    curr = _validate_image(curr, "curr")
    if prev.shape != curr.shape:
        raise ValueError(
            f"prev and curr must have the same shape, got {prev.shape} and {curr.shape}"
        )
    window_size = _validate_window_size(window_size)
    if min_eigenvalue < 0:
        raise ValueError(f"min_eigenvalue must be >= 0, got {min_eigenvalue}")
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {pts.shape}")
    if len(pts) == 0:
        raise ValueError("points must not be empty")
    height, width = prev.shape
    if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= width):
        raise ValueError(f"point x coordinates must be in [0, {width})")
    if np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= height):
        raise ValueError(f"point y coordinates must be in [0, {height})")
    ix, iy = spatial_gradients(prev)
    it = curr - prev
    half = window_size // 2
    flows = np.zeros((len(pts), 2), dtype=float)
    for i, (x, y) in enumerate(pts):
        flows[i] = _flow_at_point(ix, iy, it, int(x), int(y), half, min_eigenvalue)
    return flows


def dense_flow(prev, curr, window_size=15, step=8, min_eigenvalue=1e-3):
    """Lucas-Kanade flow on a regular grid of points.

    Parameters
    ----------
    prev, curr : 2D arrays
        Consecutive grayscale frames, same shape.
    window_size : odd int >= 3
        Side of the square window used for the least-squares fit.
    step : int
        Spacing in pixels between grid points.
    min_eigenvalue : float
        Reliability threshold, as in :func:`lucas_kanade`.

    Returns
    -------
    (U, V, xs, ys)
        ``U`` and ``V`` are float arrays of shape ``(len(ys), len(xs))``
        with the flow at each grid point; ``xs`` and ``ys`` are the
        1D arrays of grid coordinates.
    """
    prev = _validate_image(prev, "prev")
    curr = _validate_image(curr, "curr")
    if prev.shape != curr.shape:
        raise ValueError(
            f"prev and curr must have the same shape, got {prev.shape} and {curr.shape}"
        )
    window_size = _validate_window_size(window_size)
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    height, width = prev.shape
    half = window_size // 2
    xs = np.arange(half, width - half, step)
    ys = np.arange(half, height - half, step)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(
            f"image {prev.shape} is too small for window_size={window_size} and step={step}"
        )
    grid = np.array([(x, y) for y in ys for x in xs], dtype=float)
    flows = lucas_kanade(prev, curr, grid, window_size=window_size,
                         min_eigenvalue=min_eigenvalue)
    u = flows[:, 0].reshape(len(ys), len(xs))
    v = flows[:, 1].reshape(len(ys), len(xs))
    return u, v, xs, ys
