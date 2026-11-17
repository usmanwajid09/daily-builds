"""Perspective (homography) warp from scratch (numpy only).

A homography is a 3x3 matrix mapping points between two planes.  It is
estimated from four point correspondences with the Direct Linear
Transform (DLT) plus Hartley normalisation, solved by SVD.  Warping uses
inverse mapping -- every output pixel is mapped back into the source
image and sampled with bilinear interpolation -- so the output has no
holes.  Grayscale (2D) and colour (H, W, C) images are supported.
"""

import numpy as np

__all__ = ["apply_homography", "compute_homography", "warp_perspective"]


def _validate_points(points, name):
    """Return points as a float64 (4, 2) array, raising on bad input."""
    pts = np.asarray(points, dtype=float)
    if pts.shape != (4, 2):
        raise ValueError(f"{name} must have shape (4, 2), got {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError(f"{name} must contain only finite values")
    return pts


def _triangle_area(a, b, c):
    """Signed area of the triangle through three 2D points."""
    return 0.5 * abs(
        (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    )


def _check_non_degenerate(pts, name):
    """Raise if any three of the four points are (near-)collinear."""
    for i in range(4):
        others = [pts[j] for j in range(4) if j != i]
        if _triangle_area(*others) < 1e-9:
            raise ValueError(
                f"{name} are degenerate: three of the points are collinear"
            )


def _normalise_points(pts):
    """Hartley normalisation: centroid at origin, mean distance sqrt(2)."""
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    mean_dist = np.mean(np.hypot(centred[:, 0], centred[:, 1]))
    if mean_dist < 1e-12:
        raise ValueError("points are degenerate: all points coincide")
    scale = np.sqrt(2.0) / mean_dist
    transform = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1],
    ])
    normalised = np.column_stack([pts, np.ones(4)]) @ transform.T
    return normalised[:, :2], transform


def compute_homography(src_points, dst_points):
    """Compute the 3x3 homography mapping src_points onto dst_points.

    Parameters
    ----------
    src_points, dst_points : (4, 2) arrays
        Corresponding points; each row is (x, y).

    Returns
    -------
    3x3 float array with ``H[2, 2] == 1``.

    Raises
    ------
    ValueError
        If either set of points is not (4, 2), is non-finite, or is
        degenerate (three points collinear).
    """
    src = _validate_points(src_points, "src_points")
    dst = _validate_points(dst_points, "dst_points")
    _check_non_degenerate(src, "src_points")
    _check_non_degenerate(dst, "dst_points")
    src_n, t_src = _normalise_points(src)
    dst_n, t_dst = _normalise_points(dst)
    rows = []
    for (x, y), (xp, yp) in zip(src_n, dst_n):
        rows.append([-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp])
        rows.append([0, 0, 0, -x, -y, -1, yp * x, yp * y, yp])
    a = np.array(rows, dtype=float)
    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    homography = np.linalg.inv(t_dst) @ h @ t_src
    if abs(homography[2, 2]) < 1e-12:
        raise ValueError("points are degenerate: homography is singular")
    return homography / homography[2, 2]


def _validate_homography(homography):
    h = np.asarray(homography, dtype=float)
    if h.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got shape {h.shape}")
    if not np.all(np.isfinite(h)):
        raise ValueError("homography must contain only finite values")
    if abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("homography is singular (determinant is ~0)")
    return h


def apply_homography(homography, points):
    """Map (N, 2) points through a homography; returns an (N, 2) array."""
    h = _validate_homography(homography)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must contain only finite values")
    if len(pts) == 0:
        raise ValueError("points must not be empty")
    homogeneous = np.column_stack([pts, np.ones(len(pts))]) @ h.T
    w = homogeneous[:, 2]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("homography maps a point to infinity (w is ~0)")
    return homogeneous[:, :2] / w[:, None]


def _bilinear_sample(image, xs, ys, fill_value):
    """Sample a 2D image at float coordinates with bilinear interpolation."""
    height, width = image.shape
    # A small tolerance: homography round-off can map a border pixel a
    # hair outside the source image; that is still the border pixel.
    tol = 1e-9
    valid = (xs >= -tol) & (xs <= width - 1 + tol) & (ys >= -tol) & (ys <= height - 1 + tol)
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, width - 1)
    y0 = np.clip(np.floor(ys).astype(int), 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    fx = np.clip(xs - x0, 0, 1)
    fy = np.clip(ys - y0, 0, 1)
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    sampled = top * (1 - fy) + bottom * fy
    return np.where(valid, sampled, fill_value)


def warp_perspective(image, homography, output_shape, fill_value=0.0):
    """Warp an image through a homography (inverse mapping + bilinear).

    Parameters
    ----------
    image : 2D (grayscale) or 3D (H, W, C) array.
    homography : 3x3 array mapping source coordinates to output
        coordinates.
    output_shape : (height, width)
        Size of the output image.
    fill_value : float
        Value for output pixels that map outside the source image.

    Returns
    -------
    Warped image as a float64 array with the same number of channels
    as the input.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {img.shape}")
    if img.size == 0:
        raise ValueError("image must not be empty")
    h = _validate_homography(homography)
    if len(output_shape) != 2:
        raise ValueError(f"output_shape must be (height, width), got {output_shape!r}")
    out_h, out_w = int(output_shape[0]), int(output_shape[1])
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"output_shape must have positive dimensions, got {output_shape!r}")
    h_inv = np.linalg.inv(h)
    yy, xx = np.mgrid[0:out_h, 0:out_w]
    src = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)]) 
    mapped = h_inv @ src
    w = mapped[2]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("homography maps output pixels to infinity")
    xs = (mapped[0] / w).reshape(out_h, out_w)
    ys = (mapped[1] / w).reshape(out_h, out_w)
    if img.ndim == 2:
        return _bilinear_sample(img, xs, ys, fill_value)
    channels = [
        _bilinear_sample(img[:, :, c], xs, ys, fill_value)
        for c in range(img.shape[2])
    ]
    return np.stack(channels, axis=2)
