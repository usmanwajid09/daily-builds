"""Image moments: raw, central, normalised and Hu moments.

Implemented from scratch with numpy only — no OpenCV. Moments describe
a binary or grayscale image's shape: area, centroid, orientation and the
seven Hu invariants, which are (approximately) invariant to translation,
scale and rotation.
"""

import numpy as np

__all__ = [
    "raw_moments",
    "central_moments",
    "normalised_moments",
    "hu_moments",
    "centroid",
    "orientation",
    "area",
    "moments_summary",
]

_MAX_ORDER = 3


def _validate_image(image):
    """Return the image as a 2D float64 array, or raise ValueError."""
    if image is None:
        raise ValueError("image must not be None")
    try:
        arr = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("image must be a numeric array") from None
    if arr.ndim != 2:
        raise ValueError(f"image must be 2D, got {arr.ndim}D")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("image must contain only finite values")
    if np.any(arr < 0):
        raise ValueError("image values must be non-negative (use a binary or grayscale image)")
    return arr


def _validate_order(order):
    if not isinstance(order, (int, np.integer)) or isinstance(order, bool):
        raise ValueError(f"order must be an integer, got {order!r}")
    if order < 0 or order > _MAX_ORDER:
        raise ValueError(f"order must be between 0 and {_MAX_ORDER}, got {order}")
    return int(order)


def raw_moments(image, order=3):
    """Return raw moments m_pq for p + q <= order as a dict.

    m_pq = sum_x sum_y x^p * y^q * I(x, y), with x = column, y = row.
    Keys are ``"m00"``, ``"m10"``, ... e.g. ``raw_moments(img)["m10"]``.
    """
    arr = _validate_image(image)
    order = _validate_order(order)
    height, width = arr.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    moments = {}
    for p in range(order + 1):
        for q in range(order + 1 - p):
            moments[f"m{p}{q}"] = float(np.sum((xs ** p) * (ys ** q) * arr))
    return moments


def centroid(image):
    """Return the centroid (x, y) of the image mass.

    Raises ValueError for an all-zero image (no mass to locate).
    """
    arr = _validate_image(image)
    m = raw_moments(arr, order=1)
    if m["m00"] == 0.0:
        raise ValueError("cannot compute centroid of an all-zero image: total mass is 0")
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def area(image):
    """Return the total mass m00 (pixel count for a binary image)."""
    return raw_moments(_validate_image(image), order=0)["m00"]


def central_moments(image, order=3):
    """Return central moments mu_pq about the centroid as a dict.

    mu_pq = sum_x sum_y (x - x_bar)^p * (y - y_bar)^q * I(x, y).
    """
    arr = _validate_image(image)
    order = _validate_order(order)
    cx, cy = centroid(arr)
    height, width = arr.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    dx, dy = xs - cx, ys - cy
    moments = {}
    for p in range(order + 1):
        for q in range(order + 1 - p):
            moments[f"mu{p}{q}"] = float(np.sum((dx ** p) * (dy ** q) * arr))
    return moments


def normalised_moments(image, order=3):
    """Return normalised central moments eta_pq = mu_pq / mu00^gamma.

    gamma = (p + q) / 2 + 1. eta00 is 1 by construction. These are
    invariant to translation and scale.
    """
    arr = _validate_image(image)
    order = _validate_order(order)
    mu = central_moments(arr, order=order)
    if mu["mu00"] == 0.0:
        raise ValueError("cannot normalise moments of an all-zero image")
    eta = {}
    for p in range(order + 1):
        for q in range(order + 1 - p):
            gamma = (p + q) / 2.0 + 1.0
            eta[f"eta{p}{q}"] = float(mu[f"mu{p}{q}"] / (mu["mu00"] ** gamma))
    return eta


def hu_moments(image):
    """Return the seven Hu moment invariants as a length-7 float array.

    Computed from the normalised central moments eta_pq (order <= 3).
    Invariant to translation, scale and rotation (up to discretisation).
    """
    eta = normalised_moments(_validate_image(image), order=3)
    e20, e02 = eta["eta20"], eta["eta02"]
    e11 = eta["eta11"]
    e30, e12 = eta["eta30"], eta["eta12"]
    e21, e03 = eta["eta21"], eta["eta03"]
    h1 = e20 + e02
    h2 = (e20 - e02) ** 2 + 4 * e11 ** 2
    h3 = (e30 - 3 * e12) ** 2 + (3 * e21 - e03) ** 2
    h4 = (e30 + e12) ** 2 + (e21 + e03) ** 2
    h5 = ((e30 - 3 * e12) * (e30 + e12) * ((e30 + e12) ** 2 - 3 * (e21 + e03) ** 2)
          + (3 * e21 - e03) * (e21 + e03) * (3 * (e30 + e12) ** 2 - (e21 + e03) ** 2))
    h6 = ((e20 - e02) * ((e30 + e12) ** 2 - (e21 + e03) ** 2)
          + 4 * e11 * (e30 + e12) * (e21 + e03))
    h7 = ((3 * e21 - e03) * (e30 + e12) * ((e30 + e12) ** 2 - 3 * (e21 + e03) ** 2)
          - (e30 - 3 * e12) * (e21 + e03) * (3 * (e30 + e12) ** 2 - (e21 + e03) ** 2))
    return np.array([h1, h2, h3, h4, h5, h6, h7], dtype=np.float64)


def orientation(image):
    """Return the orientation angle (radians) of the shape's major axis.

    From the second-order central moments: theta = 0.5 * atan2(2*mu11,
    mu20 - mu02). Result lies in [-pi/2, pi/2]. A horizontal bar gives
    ~0, a vertical bar gives ~pi/2 (or -pi/2).
    """
    mu = central_moments(_validate_image(image), order=2)
    return float(0.5 * np.arctan2(2.0 * mu["mu11"], mu["mu20"] - mu["mu02"]))


def moments_summary(image):
    """Return a dict with area, centroid, orientation and Hu moments."""
    arr = _validate_image(image)
    cx, cy = centroid(arr)
    return {
        "area": float(area(arr)),
        "centroid": (float(cx), float(cy)),
        "orientation_rad": float(orientation(arr)),
        "hu_moments": hu_moments(arr),
        "raw_moments": raw_moments(arr, order=3),
    }
