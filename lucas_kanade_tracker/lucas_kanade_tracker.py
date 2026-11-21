"""Lucas-Kanade sparse optical-flow point tracking from scratch.

For each point, the classic Lucas-Kanade least-squares solution is
computed over a small window: with image gradients ``Ix, Iy`` and
temporal difference ``It``, the displacement ``d`` minimises
``sum (Ix*dx + Iy*dy + It)^2`` over the window, i.e. it solves the
2x2 system ``(A^T A) d = A^T b`` with ``A = [Ix, Iy]`` and
``b = -It``. The estimate is refined by a few iterations of warping
(bilinear resampling of the second frame). Points whose window has
too little texture (smallest eigenvalue of ``A^T A`` below a
threshold) or that leave the frame are reported as lost instead of
returning a meaningless displacement.

Points are ``(y, x)`` row/column coordinates, matching numpy image
indexing.
"""

import numpy as np

__all__ = [
    "LucasKanadeTracker",
    "track_points",
    "textured_disc_frame",
]


def _check_frame(frame, name="frame"):
    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D grayscale image, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _check_points(points):
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"points must have shape (n_points, 2) as (y, x), got {arr.shape}")
    if arr.size == 0:
        raise ValueError("points must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("points must contain only finite values")
    return arr


def _bilinear_sample(image, ys, xs):
    """Sample ``image`` at float coordinates ``ys, xs`` (arrays, same shape)."""
    height, width = image.shape
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = ys - y0
    wx = xs - x0
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    top = image[y0c, x0c] * (1 - wx) + image[y0c, x1c] * wx
    bottom = image[y1c, x0c] * (1 - wx) + image[y1c, x1c] * wx
    return top * (1 - wy) + bottom * wy


def textured_disc_frame(shape=(64, 64), center=(32, 32), radius=10,
                        seed=0, background=0.0, intensity=1.0):
    """Synthetic frame: a disc filled with fixed random texture.

    The texture gives every point inside the disc a usable gradient,
    which is what Lucas-Kanade needs. Moving the centre between two
    frames produces a known ground-truth translation.
    """
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError(f"shape must be (height, width) of positive ints, got {shape!r}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius!r}")
    rng = np.random.default_rng(seed)
    # Texture lives in disc-local coordinates, so it translates with
    # the disc instead of staying fixed to the frame. It is smoothed
    # (two passes of a [1, 2, 1] kernel) because Lucas-Kanade relies
    # on the linear brightness-constancy approximation, which fails
    # on pixel-wise white noise: a shift of a few pixels decorrelates
    # it completely.
    patch = rng.random(size=(2 * radius + 1, 2 * radius + 1)) * intensity
    kernel = np.array([1.0, 2.0, 1.0]) / 4.0
    for _ in range(3):
        padded = np.pad(patch, 1, mode="edge")
        patch = (padded[:-2, 1:-1] * kernel[0] + padded[1:-1, 1:-1] * kernel[1]
                 + padded[2:, 1:-1] * kernel[2])
        padded = np.pad(patch, 1, mode="edge")
        patch = (padded[1:-1, :-2] * kernel[0] + padded[1:-1, 1:-1] * kernel[1]
                 + padded[1:-1, 2:] * kernel[2])
    frame = np.full(shape, float(background))
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dy = yy - center[0]
    dx = xx - center[1]
    mask = (dy ** 2 + dx ** 2 <= radius ** 2) & (np.abs(dy) <= radius) & (np.abs(dx) <= radius)
    frame[mask] = patch[dy[mask] + radius, dx[mask] + radius]
    return frame


def track_points(prev, curr, points, window_size=15, max_iterations=10,
                 min_eigenvalue=1e-3):
    """Track ``points`` from ``prev`` to ``curr`` with Lucas-Kanade.

    Parameters
    ----------
    prev, curr : 2D arrays, same shape
        Consecutive grayscale frames.
    points : (n_points, 2) array of ``(y, x)`` coordinates in ``prev``.
    window_size : int, odd, >= 3
        Side of the square window around each point.
    max_iterations : int
        Warping refinement iterations per point.
    min_eigenvalue : float
        Minimum smallest-eigenvalue of ``A^T A`` for a point to count
        as trackable (texture test).

    Returns
    -------
    new_points : (n_points, 2) array
        Estimated ``(y, x)`` in ``curr`` (input point if lost).
    status : (n_points,) int array
        1 if tracked, 0 if lost (no texture or left the frame).
    errors : (n_points,) float array
        Mean absolute intensity residual in the window after tracking
        (nan for lost points).
    """
    prev = _check_frame(prev, "prev")
    curr = _check_frame(curr, "curr")
    if prev.shape != curr.shape:
        raise ValueError(
            f"prev and curr must have the same shape, got {prev.shape} and {curr.shape}")
    points = _check_points(points)
    if not isinstance(window_size, (int, np.integer)) or int(window_size) < 3 \
            or int(window_size) % 2 == 0:
        raise ValueError(f"window_size must be an odd integer >= 3, got {window_size!r}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations!r}")
    window_size = int(window_size)
    height, width = prev.shape
    half = window_size // 2
    gy, gx = np.gradient(prev)
    offsets = np.arange(-half, half + 1)
    oy, ox = np.meshgrid(offsets, offsets, indexing="ij")
    oy = oy.ravel()
    ox = ox.ravel()

    new_points = points.copy()
    status = np.zeros(len(points), dtype=int)
    errors = np.full(len(points), np.nan)
    for i, (py, px) in enumerate(points):
        cy, cx = int(round(py)), int(round(px))
        if cy - half < 0 or cy + half >= height or cx - half < 0 or cx + half >= width:
            continue  # window does not fit in the frame
        # Sample the template and gradients at the point's true
        # (fractional) position, not the rounded window: sampling at
        # integer positions would pull every point onto the pixel grid.
        sample_y0 = py + oy
        sample_x0 = px + ox
        template = _bilinear_sample(prev, sample_y0, sample_x0)
        Ix = _bilinear_sample(gx, sample_y0, sample_x0)
        Iy = _bilinear_sample(gy, sample_y0, sample_x0)
        ata = np.array([[np.sum(Ix * Ix), np.sum(Ix * Iy)],
                        [np.sum(Ix * Iy), np.sum(Iy * Iy)]])
        if np.linalg.eigvalsh(ata)[0] < min_eigenvalue:
            continue  # flat window: displacement is unobservable
        displacement = np.zeros(2)  # (dy, dx)
        residual = np.full(len(oy), np.nan)
        for _ in range(int(max_iterations)):
            sample_y = py + displacement[0] + oy
            sample_x = px + displacement[1] + ox
            if sample_y.min() < 0 or sample_y.max() > height - 1 \
                    or sample_x.min() < 0 or sample_x.max() > width - 1:
                break
            warped = _bilinear_sample(curr, sample_y, sample_x)
            it = warped - template
            # Gradients come from the first frame (fixed Hessian).
            # solve() returns (delta_x, delta_y): ata is ordered
            # (x, y), so map the components back to (dy, dx).
            atb = np.array([-np.sum(Ix * it), -np.sum(Iy * it)])
            try:
                delta_xy = np.linalg.solve(ata, atb)
            except np.linalg.LinAlgError:
                break
            displacement = displacement + np.array([delta_xy[1], delta_xy[0]])
            residual = it
            if np.hypot(delta_xy[0], delta_xy[1]) < 0.01:
                break
        else:
            residual = it
        new_points[i] = points[i] + displacement
        status[i] = 1
        errors[i] = float(np.mean(np.abs(residual)))
    return new_points, status, errors


class LucasKanadeTracker:
    """Stateful wrapper: track the same points across many frames."""

    def __init__(self, window_size=15, max_iterations=10, min_eigenvalue=1e-3):
        self.window_size = window_size
        self.max_iterations = max_iterations
        self.min_eigenvalue = min_eigenvalue
        self.points = None

    def init(self, frame, points):
        """Set the points to track (``(y, x)``, in ``frame``)."""
        _check_frame(frame)
        self.points = _check_points(points)
        return self.points.copy()

    def update(self, prev, curr):
        """Advance the tracked points from ``prev`` to ``curr``.

        Returns ``(new_points, status, errors)``; lost points keep
        their last position and are reported with status 0.
        """
        if self.points is None:
            raise RuntimeError("tracker is not initialised; call init() first")
        new_points, status, errors = track_points(
            prev, curr, self.points, window_size=self.window_size,
            max_iterations=self.max_iterations,
            min_eigenvalue=self.min_eigenvalue)
        self.points = new_points
        return new_points, status, errors
