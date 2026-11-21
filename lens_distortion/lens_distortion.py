"""Radial lens distortion: apply and correct, from scratch (numpy).

The Brown model with two radial coefficients (k1, k2):

    r_distorted = r * (1 + k1 * r^2 + k2 * r^4)

where r is the distance from the distortion centre, normalised by the
image's larger side so k1/k2 are scale-free. k1 > 0 is barrel
distortion (magnification grows with radius), k1 < 0 is pincushion.
Distorting / correcting an image uses inverse mapping with bilinear
sampling; correcting is the iterative inverse of distorting a point.
Only numpy is used.
"""

import numpy as np


def _validate_image(image, name="image"):
    image = np.asarray(image)
    if image.ndim == 2:
        pass
    elif image.ndim == 3 and image.shape[2] == 3:
        pass
    else:
        raise ValueError(
            f"{name} must be a 2D grayscale or 3-channel RGB image, "
            f"got shape {image.shape}"
        )
    if image.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"{name} dtype {image.dtype} is not numeric")
    return image


def _validate_coeffs(k1, k2):
    for name, value in (("k1", k1), ("k2", k2)):
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a number, got {value!r}")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value}")
    return float(k1), float(k2)


def _validate_points(points, name="points"):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape (N, 2), got shape {points.shape}"
        )
    if points.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points


def _centre_and_scale(shape, centre):
    """Distortion centre (cx, cy) and normalising scale for an image."""
    height, width = shape[0], shape[1]
    if centre is None:
        centre = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    else:
        centre = np.asarray(centre, dtype=np.float64)
        if centre.shape != (2,):
            raise ValueError(
                f"centre must have shape (2,), got {centre.shape}"
            )
        if not np.all(np.isfinite(centre)):
            raise ValueError("centre must contain only finite values")
    return centre, float(max(width, height))


def _radial_factor(r2, k1, k2):
    return 1.0 + k1 * r2 + k2 * r2 ** 2


def distort_points(points, k1, k2=0.0, centre=None, image_shape=None):
    """Apply radial distortion to (N, 2) pixel coordinates.

    ``centre`` defaults to the centre of ``image_shape`` (required
    when ``centre`` is not given); the normalising scale is the larger
    side of ``image_shape``.
    """
    k1, k2 = _validate_coeffs(k1, k2)
    points = _validate_points(points)
    if centre is None and image_shape is None:
        raise ValueError(
            "centre is required when image_shape is not given"
        )
    if image_shape is not None:
        centre, scale = _centre_and_scale(image_shape, centre)
    else:
        centre = np.asarray(centre, dtype=np.float64)
        if centre.shape != (2,):
            raise ValueError(
                f"centre must have shape (2,), got {centre.shape}"
            )
        scale = 1.0
    offset = points - centre
    r2 = (offset ** 2).sum(axis=1) / scale ** 2
    factor = _radial_factor(r2, k1, k2)
    return centre + offset * factor[:, None]


def undistort_points(points, k1, k2=0.0, centre=None, image_shape=None,
                     iterations=20):
    """Invert radial distortion on (N, 2) pixel coordinates.

    Iteratively solves for the undistorted point whose distorted
    position equals the input (fixed-point iteration; converges fast
    for realistic |k1|, |k2|).
    """
    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError(
            f"iterations must be a positive integer, got {iterations}"
        )
    k1, k2 = _validate_coeffs(k1, k2)
    points = _validate_points(points)
    if centre is None and image_shape is None:
        raise ValueError(
            "centre is required when image_shape is not given"
        )
    if image_shape is not None:
        centre, scale = _centre_and_scale(image_shape, centre)
    else:
        centre = np.asarray(centre, dtype=np.float64)
        if centre.shape != (2,):
            raise ValueError(
                f"centre must have shape (2,), got {centre.shape}"
            )
        scale = 1.0
    # Fixed point: start at the distorted point, repeatedly remove the
    # distortion the current estimate would produce.
    estimate = points.copy()
    target = points - centre
    for _ in range(int(iterations)):
        offset = estimate - centre
        r2 = (offset ** 2).sum(axis=1) / scale ** 2
        factor = _radial_factor(r2, k1, k2)
        estimate = centre + target / factor[:, None]
    return estimate


def _sample_bilinear(image, xs, ys):
    """Sample ``image`` (H, W) or (H, W, 3) at float coords; 0 outside."""
    height, width = image.shape[0], image.shape[1]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = xs - x0
    fy = ys - y0

    def _gather(x, y):
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if image.ndim == 3:
            valid = valid[:, None]
        xc = np.clip(x, 0, width - 1)
        yc = np.clip(y, 0, height - 1)
        return np.where(valid, image[yc, xc], 0.0)

    top = _gather(x0, y0) * (1 - fx)[:, None] + _gather(x1, y0) * fx[:, None] \
        if image.ndim == 3 else \
        _gather(x0, y0) * (1 - fx) + _gather(x1, y0) * fx
    bottom = _gather(x0, y1) * (1 - fx)[:, None] + _gather(x1, y1) * fx[:, None] \
        if image.ndim == 3 else \
        _gather(x0, y1) * (1 - fx) + _gather(x1, y1) * fx
    if image.ndim == 3:
        return top * (1 - fy)[:, None] + bottom * fy[:, None]
    return top * (1 - fy) + bottom * fy


def _remap(image, sample_points):
    """Sample ``image`` at ``sample_points`` (H*W, 2) -> same shape."""
    height, width = image.shape[0], image.shape[1]
    arr = image.astype(np.float64)
    sampled = _sample_bilinear(arr, sample_points[:, 0], sample_points[:, 1])
    return sampled.reshape(image.shape)


def distort_image(image, k1, k2=0.0, centre=None):
    """Return ``image`` with radial distortion applied.

    Output pixel (x, y) takes the value of the undistorted source
    pixel, so straight lines bow outward for k1 > 0 (barrel).
    """
    image = _validate_image(image)
    k1, k2 = _validate_coeffs(k1, k2)
    height, width = image.shape[0], image.shape[1]
    centre, _ = _centre_and_scale(image.shape, centre)
    ys, xs = np.mgrid[0:height, 0:width]
    grid = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    sources = undistort_points(grid, k1, k2, centre=centre,
                               image_shape=image.shape)
    result = _remap(image, sources)
    return np.clip(np.round(result), 0, 255).astype(image.dtype) \
        if image.dtype == np.uint8 else result


def correct_image(image, k1, k2=0.0, centre=None):
    """Undo radial distortion: return the corrected image.

    Output pixel (x, y) takes the value of the distorted source
    pixel, i.e. the inverse mapping of :func:`distort_image`.
    """
    image = _validate_image(image)
    k1, k2 = _validate_coeffs(k1, k2)
    height, width = image.shape[0], image.shape[1]
    centre, _ = _centre_and_scale(image.shape, centre)
    ys, xs = np.mgrid[0:height, 0:width]
    grid = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    sources = distort_points(grid, k1, k2, centre=centre,
                             image_shape=image.shape)
    result = _remap(image, sources)
    return np.clip(np.round(result), 0, 255).astype(image.dtype) \
        if image.dtype == np.uint8 else result
