"""Active contour (snake) segmentation, implemented from scratch.

A snake is a closed curve of ``N`` points that evolves to minimise an
energy functional with three terms (Kass, Witkin & Terzopoulos, 1988):

* **elasticity** — the squared first derivative, ``|p[i+1] - p[i]|^2``,
  which keeps neighbouring points evenly spaced;
* **rigidity** — the squared second derivative,
  ``|p[i-1] - 2*p[i] + p[i+1]|^2``, which keeps the curve smooth;
* **external energy** — the negative gradient magnitude of a
  Gaussian-smoothed image, so the curve is pulled toward edges.

Minimisation uses the greedy algorithm of Williams & Shah (1992): each
point, in turn, moves to the lowest-energy position inside a small
(``2 * search_radius + 1``) x (``2 * search_radius + 1``) neighbourhood.
Each energy term is normalised to ``[0, 1]`` over the candidate
neighbourhood before the weighted sum is taken, so the weights
``alpha``, ``beta`` and ``gamma`` stay comparable even though the raw
terms live on very different scales.  The greedy update is robust and
needs no linear-system solve, which is why it is used here instead of
the pentadiagonal matrix solve from the original Kass et al. paper.

Coordinates are ``(x, y)`` everywhere: ``x`` is the column index,
``y`` is the row index.  Contours are ``(N, 2)`` float arrays.

Only numpy is used.
"""

import numpy as np

__all__ = [
    "ActiveContour",
    "contour_length",
    "enclosed_area",
    "gaussian_kernel_1d",
    "gaussian_smooth",
    "edge_map",
]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def gaussian_kernel_1d(sigma):
    """Return a normalised 1D Gaussian kernel for ``sigma``.

    The kernel spans ``ceil(3 * sigma)`` on each side of the centre,
    which captures > 99% of the Gaussian's mass.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be positive and finite, got {sigma!r}")
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _convolve_1d(image, kernel, axis):
    """Convolve ``image`` with 1D ``kernel`` along ``axis`` (reflect pad)."""
    radius = len(kernel) // 2
    padded = np.pad(image, [(radius, radius) if a == axis else (0, 0)
                             for a in range(image.ndim)], mode="reflect")
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), axis, padded)


def gaussian_smooth(image, sigma=1.0):
    """Gaussian-smooth a 2D image with a separable kernel (numpy only)."""
    image = _validate_image(image)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    kernel = gaussian_kernel_1d(sigma)
    smoothed = _convolve_1d(image, kernel, axis=0)
    smoothed = _convolve_1d(smoothed, kernel, axis=1)
    return smoothed


def edge_map(image, sigma=1.0):
    """Gradient magnitude of a Gaussian-smoothed image, scaled to [0, 1].

    The smoothing is the point: on a raw image a one-pixel step gives a
    one-pixel-wide gradient spike, and the snake's neighbourhood search
    can step straight over it.  Smoothing with ``sigma`` spreads the
    edge over a few pixels so the snake feels it from a few pixels away.
    """
    smoothed = gaussian_smooth(image, sigma=sigma)
    gy, gx = np.gradient(smoothed)
    magnitude = np.hypot(gx, gy)
    peak = magnitude.max()
    if peak > 0:
        magnitude = magnitude / peak
    return magnitude


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_image(image):
    """Return ``image`` as a float64 2D array, or raise ValueError."""
    if isinstance(image, (list, tuple)):
        image = np.asarray(image)
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"image must be a 2D numpy array, got {type(image).__name__}")
    if image.ndim != 2:
        raise ValueError(
            f"image must be 2D grayscale, got shape {image.shape}")
    if image.size == 0:
        raise ValueError("image must be non-empty")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(
            f"image must hold numbers, got dtype {image.dtype}")
    image = image.astype(np.float64)
    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain only finite values (no NaN/inf)")
    return image


def _validate_contour(contour):
    """Return ``contour`` as a float64 (N, 2) array, or raise ValueError."""
    if isinstance(contour, (list, tuple)):
        contour = np.asarray(contour)
    if not isinstance(contour, np.ndarray):
        raise ValueError(
            f"contour must be a numpy array, got {type(contour).__name__}")
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError(
            f"contour must have shape (N, 2), got {contour.shape}")
    if contour.shape[0] < 3:
        raise ValueError(
            f"contour needs at least 3 points, got {contour.shape[0]}")
    if not np.issubdtype(contour.dtype, np.number):
        raise ValueError(
            f"contour must hold numbers, got dtype {contour.dtype}")
    contour = contour.astype(np.float64)
    if not np.all(np.isfinite(contour)):
        raise ValueError(
            "contour must contain only finite values (no NaN/inf)")
    return contour


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def contour_length(contour):
    """Perimeter of a closed contour: sum of segment lengths."""
    contour = _validate_contour(contour)
    shifted = np.roll(contour, -1, axis=0)
    return float(np.sum(np.hypot(*(shifted - contour).T)))


def enclosed_area(contour):
    """Area enclosed by a closed contour (shoelace formula)."""
    contour = _validate_contour(contour)
    x, y = contour[:, 0], contour[:, 1]
    return float(0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _mean_radius(contour, center):
    """Mean distance of the contour points from ``center`` (x, y)."""
    contour = _validate_contour(contour)
    return float(np.mean(np.hypot(contour[:, 0] - center[0],
                                   contour[:, 1] - center[1])))


# ---------------------------------------------------------------------------
# The snake
# ---------------------------------------------------------------------------

class ActiveContour:
    """Greedy active contour (snake) for segmenting a bright object.

    Parameters
    ----------
    alpha : float, default 0.5
        Elasticity weight — keeps points evenly spaced along the curve.
    beta : float, default 0.2
        Rigidity weight — keeps the curve smooth (penalises curvature).
    gamma : float, default 1.0
        External-energy weight — how strongly edges attract the curve.
        Because every term is normalised to [0, 1] over the search
        neighbourhood before weighting, gamma lives on the same scale
        as alpha and beta, so 1.0 (not 0.1) is the natural default;
        with gamma = 0.1 the internal terms win and the snake stalls.
    n_iterations : int, default 100
        Maximum number of greedy sweeps over the contour.
    n_points : int, default 50
        Number of points in circles built by :meth:`initialise_circle`.
    sigma : float, default 1.5
        Gaussian smoothing applied before the gradient is taken.  1.5
        (not 1.0) is the default because the edge signal has to reach
        the snake: with sigma = 1.0 the gradient magnitude 10 px from
        a step edge is ~1e-22, below the level the greedy search can
        feel, and a snake initialised outside the object never moves.
        sigma = 1.5 spreads the edge over ~4 px each side, so the
        normalised external term still points downhill toward it.
    search_radius : int, default 2
        Half-width of the neighbourhood each point searches per sweep
        (a 5x5 window when 2), so a point moves at most 2 px per sweep.
    """

    def __init__(self, alpha=0.5, beta=0.2, gamma=1.0, n_iterations=100,
                 n_points=50, sigma=1.5, search_radius=2):
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha!r}")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta!r}")
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma!r}")
        if not isinstance(n_iterations, (int, np.integer)) or n_iterations < 1:
            raise ValueError(
                f"n_iterations must be an integer >= 1, got {n_iterations!r}")
        if not isinstance(n_points, (int, np.integer)) or n_points < 3:
            raise ValueError(
                f"n_points must be an integer >= 3, got {n_points!r}")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma!r}")
        if not isinstance(search_radius, (int, np.integer)) or search_radius < 1:
            raise ValueError(
                f"search_radius must be an integer >= 1, got {search_radius!r}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.n_iterations = int(n_iterations)
        self.n_points = int(n_points)
        self.sigma = float(sigma)
        self.search_radius = int(search_radius)

    # -- initialisation ----------------------------------------------------

    def initialise_circle(self, center, radius, n_points=None,
                          image_shape=None):
        """Return ``n_points`` evenly spaced points on a circle.

        ``center`` is ``(x, y)`` and ``radius`` is in pixels.  When
        ``image_shape`` is ``(height, width)``, the centre must lie
        inside the image; otherwise only ``radius > 0`` is checked.
        """
        if n_points is None:
            n_points = self.n_points
        if not isinstance(n_points, (int, np.integer)) or n_points < 3:
            raise ValueError(
                f"n_points must be an integer >= 3, got {n_points!r}")
        center = np.asarray(center, dtype=np.float64)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError(
                f"center must be a finite (x, y) pair, got {center!r}")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius!r}")
        if image_shape is not None:
            height, width = image_shape
            if not (0 <= center[0] < width and 0 <= center[1] < height):
                raise ValueError(
                    f"center {tuple(center)} is outside image of shape "
                    f"{image_shape} (width {width}, height {height})")
        angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        contour = np.column_stack(
            (center[0] + radius * np.cos(angles),
             center[1] + radius * np.sin(angles)))
        return contour

    # -- energies ----------------------------------------------------------

    def energy(self, image, contour):
        """Total snake energy of ``contour`` on ``image`` (lower is better).

        ``alpha * E_elastic + beta * E_rigid - gamma * sum(edge)`` where
        the elastic term is the sum of squared first differences, the
        rigid term the sum of squared second differences, and the edge
        term the summed (normalised) gradient magnitude under the points.
        """
        image = _validate_image(image)
        contour = _validate_contour(contour)
        edges = edge_map(image, sigma=self.sigma)
        first = np.roll(contour, -1, axis=0) - contour
        elastic = float(np.sum(first ** 2))
        second = (np.roll(contour, 1, axis=0)
                  - 2.0 * contour
                  + np.roll(contour, -1, axis=0))
        rigid = float(np.sum(second ** 2))
        # Bilinear sample of the edge map at each (sub-pixel) point.
        height, width = image.shape
        x = np.clip(contour[:, 0], 0, width - 1)
        y = np.clip(contour[:, 1], 0, height - 1)
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        fx = x - x0
        fy = y - y0
        sampled = (edges[y0, x0] * (1 - fx) * (1 - fy)
                   + edges[y0, x1] * fx * (1 - fy)
                   + edges[y1, x0] * (1 - fx) * fy
                   + edges[y1, x1] * fx * fy)
        external = -float(np.sum(sampled))
        return self.alpha * elastic + self.beta * rigid + self.gamma * external

    # -- fitting -----------------------------------------------------------

    def fit(self, image, contour):
        """Evolve ``contour`` on ``image`` and return the new contour.

        The input contour is not modified.  Points are kept inside the
        image by clipping candidate positions to the image bounds.
        Sweeps stop early when no point moves by more than 0.5 px.
        """
        image = _validate_image(image)
        contour = _validate_contour(contour).copy()
        height, width = image.shape
        edges = edge_map(image, sigma=self.sigma)
        n = len(contour)
        offsets = np.arange(-self.search_radius, self.search_radius + 1)
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        neighbourhood = np.column_stack(
            (dx.ravel(), dy.ravel())).astype(np.float64)

        for _ in range(self.n_iterations):
            max_move = 0.0
            # Mean spacing of the current contour: the target spacing
            # for the continuity (elasticity) term.
            spacings = np.hypot(
                *(np.roll(contour, -1, axis=0) - contour).T)
            avg_spacing = float(np.mean(spacings))
            for i in range(n):
                prev = contour[i - 1]
                nxt = contour[(i + 1) % n]
                point = contour[i]
                candidates = point + neighbourhood
                candidates[:, 0] = np.clip(candidates[:, 0], 0, width - 1)
                candidates[:, 1] = np.clip(candidates[:, 1], 0, height - 1)

                # Continuity: distance to the previous point vs the
                # average spacing (Williams & Shah).
                d_prev = np.hypot(*(candidates - prev).T)
                e_cont = (d_prev - avg_spacing) ** 2
                # Curvature: second difference with both neighbours.
                second = prev - 2.0 * candidates + nxt
                e_curv = np.sum(second ** 2, axis=1)
                # External: strong edges are low energy.
                cx = np.rint(candidates[:, 0]).astype(int)
                cy = np.rint(candidates[:, 1]).astype(int)
                e_ext = -edges[cy, cx]

                # Normalise each term to [0, 1] over the candidates so
                # the weights stay comparable; a flat term contributes
                # nothing instead of amplifying noise.
                total = np.zeros(len(candidates))
                for weight, term in ((self.alpha, e_cont),
                                     (self.beta, e_curv),
                                     (self.gamma, e_ext)):
                    if weight == 0:
                        continue
                    span = term.max() - term.min()
                    if span > 1e-12:
                        total += weight * (term - term.min()) / span
                best = int(np.argmin(total))
                move = float(np.hypot(*(candidates[best] - point)))
                max_move = max(max_move, move)
                contour[i] = candidates[best]
            if max_move < 0.5:
                break
        return contour
