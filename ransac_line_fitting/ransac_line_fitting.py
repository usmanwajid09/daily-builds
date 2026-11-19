"""RANSAC line fitting on noisy 2-D points, from scratch (numpy only).

Ordinary least squares is dragged off course by even a few gross
outliers, because every point pulls on the fit. RANSAC instead
samples minimal subsets — two points define a line — scores each
candidate by how many points lie within ``threshold`` of it, keeps
the candidate with the most inliers, and finally refits by least
squares on the inliers alone. With enough trials the best line is
found even when a large fraction of the points are outliers.

The line is stored as ``y = slope * x + intercept``. Vertical lines
cannot be written that way, so a sampled pair with (near-)identical
x coordinates is skipped; for data that really is vertical, fit
``x`` as a function of ``y`` instead (see :func:`least_squares_line`).
"""

import numpy as np

__all__ = ["RansacResult", "least_squares_line", "ransac_line"]


class RansacResult:
    """Result of :func:`ransac_line`."""

    def __init__(self, slope, intercept, inliers, n_trials):
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.inliers = np.asarray(inliers, dtype=bool)
        self.n_trials = int(n_trials)

    @property
    def n_inliers(self):
        return int(self.inliers.sum())

    def predict(self, x):
        """Evaluate the fitted line at ``x`` (scalar or array)."""
        return self.slope * np.asarray(x, dtype=float) + self.intercept

    def __repr__(self):
        return (f"RansacResult(slope={self.slope:.4f}, "
                f"intercept={self.intercept:.4f}, "
                f"inliers={self.n_inliers}, trials={self.n_trials})")


def _validate_points(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    if len(points) < 2:
        raise ValueError(f"need at least 2 points, got {len(points)}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite (no NaN or inf)")
    return points


def least_squares_line(points):
    """Ordinary least-squares fit ``y = slope * x + intercept``.

    Args:
        points: ``(N, 2)`` array of ``(x, y)`` points, N >= 2.

    Returns:
        ``(slope, intercept)`` as floats.

    Raises:
        ValueError: bad shape, fewer than 2 points, non-finite values,
            or (near-)identical x coordinates — a vertical line has no
            finite slope.
    """
    points = _validate_points(points)
    x, y = points[:, 0], points[:, 1]
    x_mean, y_mean = x.mean(), y.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    if denom == 0:
        raise ValueError("all x coordinates are identical: vertical line, no finite slope")
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    return slope, float(y_mean - slope * x_mean)


def _distances_to_line(points, slope, intercept):
    """Perpendicular distance of each point to ``y = slope*x + intercept``."""
    x, y = points[:, 0], points[:, 1]
    return np.abs(slope * x - y + intercept) / np.hypot(slope, 1.0)


def ransac_line(points, n_trials=200, threshold=1.0, min_inliers=2,
                seed=None):
    """Fit a line to ``points`` with RANSAC, ignoring outliers.

    Args:
        points: ``(N, 2)`` array of ``(x, y)`` points, N >= 2.
        n_trials: number of random two-point samples to try.
        threshold: a point counts as an inlier when its perpendicular
            distance to the candidate line is at most this. Must be > 0.
        min_inliers: minimum inliers a candidate needs before it can
            win; must be >= 2.
        seed: seed for the sampler (``None`` = non-deterministic).

    Returns:
        A :class:`RansacResult` with the refitted slope/intercept, a
        boolean inlier mask, and the trial count. The refit is least
        squares over the winning inlier set.

    Raises:
        ValueError: bad points, ``n_trials`` < 1, ``threshold`` <= 0,
            ``min_inliers`` < 2 or larger than the point count, or no
            candidate reached ``min_inliers`` inliers (e.g. the data
            is a vertical line, or the threshold is far too small).
    """
    points = _validate_points(points)
    n = len(points)
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if min_inliers < 2:
        raise ValueError(f"min_inliers must be at least 2, got {min_inliers}")
    if min_inliers > n:
        raise ValueError(f"min_inliers ({min_inliers}) exceeds point count ({n})")

    rng = np.random.default_rng(seed)
    best_inliers = None
    best_count = 0
    for _ in range(n_trials):
        i, j = rng.choice(n, size=2, replace=False)
        x1, y1 = points[i]
        x2, y2 = points[j]
        if x1 == x2:
            continue  # vertical pair: no finite slope, skip the trial
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        inliers = _distances_to_line(points, slope, intercept) <= threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            if best_count == n:
                break
    if best_inliers is None or best_count < min_inliers:
        raise ValueError(
            f"no line reached {min_inliers} inliers in {n_trials} trials "
            f"(best: {best_count}); try a larger threshold or more trials"
        )

    # Refit by least squares on the winning inliers only.
    slope, intercept = least_squares_line(points[best_inliers])
    # Recompute the inlier mask against the refitted line, so the
    # reported mask matches the returned line.
    inliers = _distances_to_line(points, slope, intercept) <= threshold
    return RansacResult(slope, intercept, inliers, n_trials)
