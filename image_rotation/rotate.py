"""Image rotation, implemented from scratch (numpy only)."""

import numpy as np

_METHODS = ("nearest", "bilinear")


def _validate_image(image):
    """Return ``image`` as a float64 array (2D or (H, W, 3)) or raise."""
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"image must be numeric, got dtype {arr.dtype}")
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] != 3):
        raise ValueError(
            f"image must be 2D grayscale or (H, W, 3) RGB, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("image must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    return arr


def _validate_angle(angle):
    """Return ``angle`` as a float in [0, 360) or raise."""
    if isinstance(angle, bool) or not isinstance(angle, (int, float, np.floating)):
        raise ValueError(f"angle must be a number of degrees, got {angle!r}")
    angle = float(angle)
    if not np.isfinite(angle):
        raise ValueError(f"angle must be finite, got {angle}")
    # Normalising first makes 360 degrees exactly equal to 0 degrees.
    return angle % 360.0


def _validate_method(method):
    """Return the normalised method name or raise."""
    if not isinstance(method, str) or method.lower() not in _METHODS:
        raise ValueError(
            f"method must be one of {list(_METHODS)}, got {method!r}"
        )
    return method.lower()


def _validate_fill(fill):
    """Return ``fill`` as a float or raise."""
    if isinstance(fill, bool) or not isinstance(fill, (int, float, np.floating)):
        raise ValueError(f"fill must be a number, got {fill!r}")
    fill = float(fill)
    if not np.isfinite(fill):
        raise ValueError(f"fill must be finite, got {fill}")
    return fill


def _source_coords(height, width, angle_deg):
    """Source coordinates for every output pixel (inverse mapping).

    Returns two float64 arrays of shape (H, W): the x and y coordinates
    in the source image that each output pixel samples from. The image
    is rotated counter-clockwise by ``angle_deg`` around its centre.
    """
    theta = np.deg2rad(angle_deg)
    # Snap to 12 decimals: cos(90 degrees) is 6e-17 in floating point,
    # not 0, and that hair would push border pixels a fraction outside
    # the image (and into the fill value) at exact multiples of 90.
    cos_t, sin_t = round(float(np.cos(theta)), 12), round(float(np.sin(theta)), 12)
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    x_rel, y_rel = xs - cx, ys - cy
    # Inverse of the counter-clockwise rotation (in screen coordinates,
    # where y grows downwards): a destination pixel samples the source
    # pixel obtained by rotating its offset clockwise by the angle.
    x_src = cx + cos_t * x_rel - sin_t * y_rel
    y_src = cy + sin_t * x_rel + cos_t * y_rel
    return x_src, y_src


def _sample_nearest(channel, x_src, y_src, fill):
    """Sample one 2D channel at ``(x_src, y_src)`` by nearest neighbour."""
    height, width = channel.shape
    inside = (
        (x_src >= 0) & (x_src <= width - 1)
        & (y_src >= 0) & (y_src <= height - 1)
    )
    xi = np.clip(np.rint(x_src), 0, width - 1).astype(int)
    yi = np.clip(np.rint(y_src), 0, height - 1).astype(int)
    out = channel[yi, xi]
    return np.where(inside, out, fill)


def _sample_bilinear(channel, x_src, y_src, fill):
    """Sample one 2D channel at ``(x_src, y_src)`` by bilinear interpolation.

    Neighbour indices are clamped to the image, so a source point inside
    the image interpolates only from real pixels: the fill value can
    never leak into the interior through an out-of-range neighbour.
    """
    height, width = channel.shape
    inside = (
        (x_src >= 0) & (x_src <= width - 1)
        & (y_src >= 0) & (y_src <= height - 1)
    )
    # Clamp first: points inside the image then interpolate purely
    # from image pixels; points outside are replaced by fill below.
    x = np.clip(x_src, 0, width - 1)
    y = np.clip(y_src, 0, height - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx, fy = x - x0, y - y0
    top = channel[y0, x0] * (1 - fx) + channel[y0, x1] * fx
    bottom = channel[y1, x0] * (1 - fx) + channel[y1, x1] * fx
    out = top * (1 - fy) + bottom * fy
    return np.where(inside, out, fill)


def rotate_image(image, angle, method="bilinear", fill=0.0):
    """Return ``image`` rotated counter-clockwise by ``angle`` degrees.

    The rotation is done from scratch with numpy only, by inverse
    mapping: each output pixel samples the source image at the location
    obtained by rotating its offset from the centre back by ``angle``.

    Args:
        image: 2D grayscale array, or (H, W, 3) RGB array.
        angle: rotation angle in degrees; counter-clockwise is positive.
            Any finite number works (720 is the same as 0).
        method: ``"bilinear"`` (default) for smooth interpolation, or
            ``"nearest"`` for nearest-neighbour sampling.
        fill: value used for output pixels whose source location falls
            outside the image (default 0.0).

    Returns:
        A float64 array with the SAME shape as the input. The output
        canvas is NOT enlarged, so for large angles the corners of the
        rotated image are cropped and the vacated corners are filled
        with ``fill``. That is the documented behaviour, not a bug: if
        you need the whole rotated image, pad the input first.

    Raises:
        ValueError: on a bad image (see :func:`_validate_image`), a
            non-finite or non-numeric ``angle`` / ``fill``, or an
            unknown ``method``.
    """
    arr = _validate_image(image)
    angle = _validate_angle(angle)
    method = _validate_method(method)
    fill = _validate_fill(fill)

    height, width = arr.shape[0], arr.shape[1]
    x_src, y_src = _source_coords(height, width, angle)
    sample = _sample_bilinear if method == "bilinear" else _sample_nearest
    if arr.ndim == 2:
        return sample(arr, x_src, y_src, fill)
    return np.stack(
        [sample(arr[:, :, c], x_src, y_src, fill) for c in range(3)], axis=2
    )
