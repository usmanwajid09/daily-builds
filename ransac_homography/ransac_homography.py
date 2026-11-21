"""RANSAC homography estimation from point correspondences.

A homography is a 3x3 matrix mapping points in one image plane to
another (planar scenes, pure camera rotation, or a change of viewpoint
of a plane). Given noisy correspondences with outliers, RANSAC samples
minimal 4-point sets, fits a homography with the direct linear
transform (DLT, Hartley-normalised), scores it by reprojection error,
and refits on all inliers. Only numpy is used.
"""

import numpy as np


def _validate_points(points, name, min_count=None):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape (N, 2), got shape {points.shape}"
        )
    if min_count is not None and points.shape[0] < min_count:
        raise ValueError(
            f"{name} needs at least {min_count} points, "
            f"got {points.shape[0]}"
        )
    if points.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points


def _normalise_points(points):
    """Hartley normalisation: centroid at origin, mean distance sqrt(2).

    Returns (normalised_points, T) with ``T`` the 3x3 transform so
    that ``normalised = (T @ [x, y, 1])[:2]``.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    mean_dist = np.mean(np.sqrt((centred ** 2).sum(axis=1)))
    if mean_dist == 0.0:
        raise ValueError("points are all identical; cannot normalise")
    scale = np.sqrt(2.0) / mean_dist
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    homogeneous = np.column_stack([points, np.ones(len(points))])
    normalised = (transform @ homogeneous.T).T[:, :2]
    return normalised, transform


def compute_homography(src, dst):
    """Fit a homography mapping ``src`` points onto ``dst`` points.

    ``src`` and ``dst`` are (N, 2) arrays with N >= 4. Uses the direct
    linear transform on Hartley-normalised points. Returns a 3x3
    matrix normalised so ``H[2, 2] == 1``.

    Raises ValueError on too few / degenerate (e.g. collinear) points.
    """
    src = _validate_points(src, "src", min_count=4)
    dst = _validate_points(dst, "dst", min_count=4)
    if src.shape[0] != dst.shape[0]:
        raise ValueError(
            f"src and dst must have the same number of points, "
            f"got {src.shape[0]} and {dst.shape[0]}"
        )
    src_n, t_src = _normalise_points(src)
    dst_n, t_dst = _normalise_points(dst)
    n = len(src_n)
    a = np.zeros((2 * n, 9))
    x, y = src_n[:, 0], src_n[:, 1]
    xp, yp = dst_n[:, 0], dst_n[:, 1]
    a[0::2, 0] = -x
    a[0::2, 1] = -y
    a[0::2, 2] = -1.0
    a[0::2, 6] = x * xp
    a[0::2, 7] = y * xp
    a[0::2, 8] = xp
    a[1::2, 3] = -x
    a[1::2, 4] = -y
    a[1::2, 5] = -1.0
    a[1::2, 6] = x * yp
    a[1::2, 7] = y * yp
    a[1::2, 8] = yp
    _, singular, vh = np.linalg.svd(a)
    # An exact fit has one ~zero singular value (the null space). The
    # configuration is degenerate only if the null space is bigger
    # than one dimension, i.e. the second-smallest value is ~zero too.
    if singular[-2] < 1e-8 * singular[0]:
        raise ValueError(
            "degenerate point configuration (e.g. collinear points); "
            "cannot fit a homography"
        )
    h = vh[-1].reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h @ t_src
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("degenerate homography (H[2, 2] is ~0)")
    return h / h[2, 2]


def apply_homography(homography, points):
    """Map ``points`` (N, 2) through a 3x3 homography. Returns (N, 2)."""
    h = np.asarray(homography, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError(
            f"homography must have shape (3, 3), got shape {h.shape}"
        )
    if not np.all(np.isfinite(h)):
        raise ValueError("homography must contain only finite values")
    points = _validate_points(points, "points")
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = (h @ homogeneous.T).T
    w = mapped[:, 2]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("homography maps a point to infinity (w ~ 0)")
    return mapped[:, :2] / w[:, None]


def reprojection_errors(homography, src, dst):
    """Euclidean reprojection error per correspondence. Returns (N,)."""
    src = _validate_points(src, "src")
    dst = _validate_points(dst, "dst")
    if src.shape[0] != dst.shape[0]:
        raise ValueError(
            f"src and dst must have the same number of points, "
            f"got {src.shape[0]} and {dst.shape[0]}"
        )
    return np.linalg.norm(apply_homography(homography, src) - dst, axis=1)


def ransac_homography(src, dst, threshold=3.0, max_iterations=1000,
                      min_inliers=None, seed=None):
    """Estimate a homography robustly with RANSAC.

    Parameters
    ----------
    src, dst : (N, 2) arrays of correspondences, N >= 4.
    threshold : float, max reprojection error (pixels) for an inlier.
    max_iterations : int, number of random 4-point samples to try.
    min_inliers : int or None; stop early once this many inliers are
        found (defaults to N, i.e. never stop early).
    seed : int or None, for reproducibility.

    Returns ``(H, inliers, iterations)`` where ``H`` is the 3x3
    homography refit on all inliers, ``inliers`` is a boolean mask of
    length N, and ``iterations`` is how many samples were tried. A
    model must explain at least 5 correspondences (4 when N == 4),
    since a 4-point sample trivially fits its own points.

    Raises ValueError on bad input and RuntimeError if no valid model
    is found.
    """
    src = _validate_points(src, "src", min_count=4)
    dst = _validate_points(dst, "dst", min_count=4)
    if src.shape[0] != dst.shape[0]:
        raise ValueError(
            f"src and dst must have the same number of points, "
            f"got {src.shape[0]} and {dst.shape[0]}"
        )
    if not isinstance(threshold, (int, float, np.integer, np.floating)) \
            or float(threshold) <= 0:
        raise ValueError(f"threshold must be a positive number, got {threshold}")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError(
            f"max_iterations must be a positive integer, got {max_iterations}"
        )
    n = len(src)
    if min_inliers is None:
        min_inliers = n
    if not isinstance(min_inliers, (int, np.integer)) or not 4 <= min_inliers <= n:
        raise ValueError(
            f"min_inliers must be an integer in [4, {n}], got {min_inliers}"
        )
    rng = np.random.default_rng(seed)

    # A 4-point sample always fits its own points exactly, so a model
    # only counts if it explains at least one further point (or all
    # points when there are only 4).
    min_useful = 5 if n > 4 else 4
    best_h = None
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        sample = rng.choice(n, size=4, replace=False)
        try:
            candidate = compute_homography(src[sample], dst[sample])
        except ValueError:
            continue
        errors = reprojection_errors(candidate, src, dst)
        inliers = errors <= float(threshold)
        count = int(inliers.sum())
        if count > best_count and count >= min_useful:
            best_count = count
            best_inliers = inliers
            best_h = candidate
            if count >= min_inliers:
                break
    if best_h is None or best_count < min_useful:
        raise RuntimeError(
            "RANSAC found no homography with at least "
            f"{min_useful} inliers; "
            "try a larger threshold or more iterations"
        )
    # Refit on all inliers for the final model.
    try:
        best_h = compute_homography(src[best_inliers], dst[best_inliers])
    except ValueError:
        pass  # keep the best sampled model
    errors = reprojection_errors(best_h, src, dst)
    best_inliers = errors <= float(threshold)
    return best_h, best_inliers, iterations
