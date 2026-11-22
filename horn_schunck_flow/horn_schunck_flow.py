"""Horn-Schunck dense optical flow, from scratch (numpy only).

Implements the classic Horn & Schunck (1981) algorithm: it estimates a
dense flow field ``(u, v)`` between two grayscale frames by minimising

    E = ∬ (Ix·u + Iy·v + It)² + α²·(‖∇u‖² + ‖∇v‖²) dx dy

where the first term is the brightness-constancy constraint and the
second term enforces smoothness of the flow field.  The minimiser is
found with the iterative update

    u = ū − Ix·(Ix·ū + Iy·v̄ + It) / (α² + Ix² + Iy²)
    v = v̄ − Iy·(Ix·ū + Iy·v̄ + It) / (α² + Ix² + Iy²)

where ū, v̄ are 4-neighbour averages of the current flow estimates.
Derivatives use the standard Horn-Schunck 2×2 kernels, averaged over
the two frames.
"""

import numpy as np

__all__ = [
    "HornSchunckFlow",
    "flow_magnitude",
    "flow_angle",
    "mean_flow",
    "warp_image",
]


def _as_float_image(image, name):
    """Validate a grayscale image and return it as a float64 array."""
    try:
        arr = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D grayscale array, got {arr.ndim}D")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _neighbour_average(field):
    """4-neighbour average of a 2D array (edge values replicated)."""
    padded = np.pad(field, 1, mode="edge")
    return 0.25 * (
        padded[:-2, 1:-1]  # up
        + padded[2:, 1:-1]  # down
        + padded[1:-1, :-2]  # left
        + padded[1:-1, 2:]  # right
    )


def _derivatives(img1, img2):
    """Horn-Schunck spatial/temporal derivatives (2x2 kernels, both frames).

    Returns Ix, Iy, It with the same shape as the inputs; the last row
    and column (where the 2x2 kernel has no partner pixel) are zero.
    """
    h, w = img1.shape
    ix = np.zeros((h, w), dtype=np.float64)
    iy = np.zeros((h, w), dtype=np.float64)
    it = np.zeros((h, w), dtype=np.float64)

    # 2x2 kernel, averaged over the two frames:
    #   Ix = 1/4 * [(I[y,x+1]-I[y,x]) + (I[y+1,x+1]-I[y+1,x])] per frame
    #   Iy = 1/4 * [(I[y+1,x]-I[y,x]) + (I[y+1,x+1]-I[y,x+1])] per frame
    #   It = 1/4 * sum of (I2 - I1) over the four kernel pixels
    a1, b1 = img1[:-1, :-1], img1[:-1, 1:]
    c1, d1 = img1[1:, :-1], img1[1:, 1:]
    a2, b2 = img2[:-1, :-1], img2[:-1, 1:]
    c2, d2 = img2[1:, :-1], img2[1:, 1:]

    ix[:-1, :-1] = 0.25 * ((b1 - a1) + (d1 - c1) + (b2 - a2) + (d2 - c2))
    iy[:-1, :-1] = 0.25 * ((c1 - a1) + (d1 - b1) + (c2 - a2) + (d2 - b2))
    it[:-1, :-1] = 0.25 * ((a2 - a1) + (b2 - b1) + (c2 - c1) + (d2 - d1))
    return ix, iy, it


def flow_magnitude(u, v):
    """Per-pixel flow speed: sqrt(u² + v²). Always non-negative."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"u and v must have the same shape, got {u.shape} and {v.shape}")
    return np.hypot(u, v)


def flow_angle(u, v):
    """Per-pixel flow direction in degrees, in (-180, 180].

    0° is motion to the right (+x), 90° is motion down (+y, image
    coordinates), -90° is motion up. Uses atan2(v, u).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"u and v must have the same shape, got {u.shape} and {v.shape}")
    return np.degrees(np.arctan2(v, u))


def mean_flow(u, v):
    """Mean flow vector (mean_u, mean_v) over the whole field."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"u and v must have the same shape, got {u.shape} and {v.shape}")
    if u.size == 0:
        raise ValueError("flow arrays must be non-empty")
    return float(np.mean(u)), float(np.mean(v))


def warp_image(image, u, v):
    """Backward-warp ``image`` by flow ``(u, v)`` (bilinear, numpy only).

    out[y, x] = image[y + v[y, x], x + u[y, x]]: a pixel that moved
    by (u, v) from frame 1 to frame 2 sits at (y+v, x+u) in frame 2,
    so warping frame 2 by the flow from frame 1 to frame 2
    approximately reconstructs frame 1. Pixels sampling outside the
    image are filled with 0.
    """
    img = _as_float_image(image, "image")
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != img.shape or v.shape != img.shape:
        raise ValueError(
            f"u and v must match the image shape {img.shape}, "
            f"got {u.shape} and {v.shape}"
        )
    h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    src_y = ys + v
    src_x = xs + u

    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    fx = src_x - x0
    fy = src_y - y0

    def _sample(dy, dx):
        yy = np.clip(y0 + dy, 0, h - 1)
        xx = np.clip(x0 + dx, 0, w - 1)
        return img[yy, xx]

    warped = (
        (1 - fx) * (1 - fy) * _sample(0, 0)
        + fx * (1 - fy) * _sample(0, 1)
        + (1 - fx) * fy * _sample(1, 0)
        + fx * fy * _sample(1, 1)
    )
    inside = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)
    return np.where(inside, warped, 0.0)


class HornSchunckFlow:
    """Dense optical flow via the Horn-Schunck iterative method.

    Parameters
    ----------
    alpha : float, default 1.0
        Smoothness weight. Larger values give a smoother (more
        regularised) flow field; smaller values follow the image
        gradients more closely.
    n_iterations : int, default 100
        Maximum number of Gauss-Seidel-style iterations.
    tolerance : float, default 1e-4
        Early-stop threshold: iteration stops when the largest change
        in either flow component between two iterations falls below
        this value. Set to 0 to always run all ``n_iterations``.
    """

    def __init__(self, alpha=1.0, n_iterations=100, tolerance=1e-4):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if int(n_iterations) != n_iterations or n_iterations < 1:
            raise ValueError(f"n_iterations must be an integer >= 1, got {n_iterations}")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")
        self.alpha = float(alpha)
        self.n_iterations = int(n_iterations)
        self.tolerance = float(tolerance)
        self.iterations_run_ = 0

    def compute_flow(self, img1, img2):
        """Estimate dense flow from ``img1`` to ``img2``.

        Returns ``(u, v)`` float64 arrays with the same shape as the
        inputs: ``u`` is horizontal displacement (positive = right),
        ``v`` is vertical displacement (positive = down).
        """
        img1 = _as_float_image(img1, "img1")
        img2 = _as_float_image(img2, "img2")
        if img1.shape != img2.shape:
            raise ValueError(
                f"img1 and img2 must have the same shape, "
                f"got {img1.shape} and {img2.shape}"
            )

        ix, iy, it = _derivatives(img1, img2)
        denom = self.alpha ** 2 + ix ** 2 + iy ** 2

        u = np.zeros_like(img1)
        v = np.zeros_like(img1)

        for i in range(self.n_iterations):
            u_bar = _neighbour_average(u)
            v_bar = _neighbour_average(v)
            common = (ix * u_bar + iy * v_bar + it) / denom
            u_new = u_bar - ix * common
            v_new = v_bar - iy * common
            change = max(
                float(np.max(np.abs(u_new - u))),
                float(np.max(np.abs(v_new - v))),
            )
            u, v = u_new, v_new
            self.iterations_run_ = i + 1
            if change < self.tolerance:
                break

        return u, v
