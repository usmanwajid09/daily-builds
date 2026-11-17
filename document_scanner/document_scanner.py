"""Document scanner from scratch (numpy only).

Detect the four corners of a bright quadrilateral on a dark background
and perspective-warp it to a flat rectangle.

Public API:
    find_document_corners(image, threshold=None) -> (4, 2) float64 corners
    warp_document(image, corners, width, height) -> warped image
    scan_document(image, width=400, height=500, threshold=None)
        -> (warped image, corners)
    compute_homography(src, dst) -> (3, 3) homography matrix
"""

import numpy as np

__all__ = [
    "find_document_corners",
    "warp_document",
    "scan_document",
    "compute_homography",
    "order_corners",
]


def _to_grayscale(image):
    """Validate an image and return it as a 2D float64 grayscale array."""
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array, got "
                         f"{type(image).__name__}")
    if image.ndim == 2:
        gray = image.astype(np.float64)
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        rgb = image[:, :, :3].astype(np.float64)
        gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
                + 0.114 * rgb[:, :, 2])
    else:
        raise ValueError(
            "image must be 2D grayscale (H, W) or 3D RGB/RGBA (H, W, 3|4), "
            f"got shape {image.shape}")
    if gray.size == 0:
        raise ValueError("image must not be empty")
    if not np.all(np.isfinite(gray)):
        raise ValueError("image must contain only finite values")
    return gray


def _otsu_threshold(gray):
    """Otsu's threshold for a grayscale image (values assumed 0-255 scale)."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(hist, np.arange(256, dtype=np.float64))
    sum_b = 0.0
    w_b = 0
    max_var = -1.0
    threshold = 127.0
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = float(i)
    # The between-class variance is flat across any empty gap between the
    # two classes, so the first maximiser can sit right next to the dark
    # peak (e.g. 28 for peaks at 20 and 220) and admit noisy background
    # pixels. Refine to the midpoint of the two class means instead.
    below = gray <= threshold
    above = ~below
    if below.any() and above.any():
        threshold = 0.5 * (gray[below].mean() + gray[above].mean())
    return threshold


def _convex_hull(points):
    """Monotone-chain convex hull of (N, 2) points; returns hull (M, 2)."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull


def order_corners(points):
    """Order 4 (x, y) points as TL, TR, BR, BL.

    Uses the sum/difference trick: TL has the smallest x+y, BR the
    largest x+y, TR the largest x-y, BL the smallest x-y.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError(f"order_corners expects 4 points of shape (4, 2), "
                         f"got shape {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("corners must contain only finite values")
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    ordered = np.empty((4, 2), dtype=np.float64)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[1] = pts[np.argmax(d)]   # top-right
    ordered[3] = pts[np.argmin(d)]   # bottom-left
    if len(np.unique(ordered, axis=0)) != 4:
        raise ValueError("corners must be 4 distinct points")
    return ordered


def find_document_corners(image, threshold=None):
    """Find the 4 corners of the bright document in a dark image.

    Parameters
    ----------
    image : (H, W) grayscale or (H, W, 3|4) RGB array.
    threshold : float or None. Brightness above this counts as document.
        If None, Otsu's method picks it automatically.

    Returns
    -------
    (4, 2) float64 array of (x, y) corners ordered TL, TR, BR, BL.

    Raises
    ------
    ValueError
        If the image shape is invalid, or no bright quadrilateral is found
        (e.g. an all-dark image), or the bright region is degenerate.
    """
    gray = _to_grayscale(image)
    if threshold is None:
        thr = _otsu_threshold(gray)
    else:
        thr = float(threshold)
        if not np.isfinite(thr):
            raise ValueError("threshold must be a finite number")
        if thr < 0 or thr > 255:
            raise ValueError(f"threshold must be in [0, 255], got {thr}")

    mask = gray > thr
    # Drop isolated bright pixels (single-pixel sensor noise): a bright
    # pixel with no bright 8-neighbour cannot belong to the document.
    padded = np.pad(mask, 1)
    neighbours = np.zeros_like(mask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbours |= padded[1 + dy:1 + dy + mask.shape[0],
                                 1 + dx:1 + dx + mask.shape[1]]
    mask = mask & neighbours
    n_bright = int(mask.sum())
    if n_bright < 20:
        raise ValueError(
            "no bright quadrilateral found: only "
            f"{n_bright} pixels above threshold {thr:.1f} "
            "(image may be all dark)")
    # A document must be a minority bright region on a dark background;
    # if almost everything is bright there is no quadrilateral to find.
    if n_bright > 0.95 * gray.size:
        raise ValueError(
            "no bright quadrilateral found: "
            f"{n_bright} of {gray.size} pixels are bright "
            "(image may be all bright)")

    ys, xs = np.nonzero(mask)
    points = np.column_stack([xs, ys]).astype(np.float64)
    hull = _convex_hull(points)
    if len(hull) < 4:
        raise ValueError(
            "no bright quadrilateral found: bright region has only "
            f"{len(hull)} convex-hull points")

    # The 4 extreme points of the hull are the document corners.
    corners = order_corners(np.array([
        hull[np.argmin(hull.sum(axis=1))],
        hull[np.argmax(hull[:, 0] - hull[:, 1])],
        hull[np.argmax(hull.sum(axis=1))],
        hull[np.argmin(hull[:, 0] - hull[:, 1])],
    ]))

    # Degeneracy check: the quadrilateral must have real area.
    x, y = corners[:, 0], corners[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area < 20.0:
        raise ValueError(
            f"no bright quadrilateral found: detected region area "
            f"is only {area:.1f} px^2")
    return corners


def compute_homography(src, dst):
    """Return the 3x3 homography mapping src points to dst points.

    src, dst: (4, 2) arrays of corresponding (x, y) points.
    Solves the 8x8 linear system from the four point correspondences.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError(
            "compute_homography expects src and dst of shape (4, 2), "
            f"got {src.shape} and {dst.shape}")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        raise ValueError("homography points must contain only finite values")

    a = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        a[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        a[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i] = u
        b[2 * i + 1] = v
    try:
        h = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "cannot compute homography: source points are degenerate "
            "(e.g. collinear)") from exc
    h_mat = np.append(h, 1.0).reshape(3, 3)
    return h_mat


def _validate_corners(corners):
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError(f"corners must have shape (4, 2), got {corners.shape}")
    if not np.all(np.isfinite(corners)):
        raise ValueError("corners must contain only finite values")
    if len(np.unique(corners, axis=0)) != 4:
        raise ValueError("corners must be 4 distinct points")
    return corners


def _bilinear_sample(gray, xs, ys):
    """Bilinear interpolation of a 2D image at float coords (xs, ys)."""
    h, w = gray.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = xs - x0
    fy = ys - y0
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    top = gray[y0c, x0c] * (1 - fx) + gray[y0c, x1c] * fx
    bottom = gray[y1c, x0c] * (1 - fx) + gray[y1c, x1c] * fx
    return top * (1 - fy) + bottom * fy


def warp_document(image, corners, width, height):
    """Perspective-warp the quadrilateral at `corners` to a flat rectangle.

    Parameters
    ----------
    image : (H, W) grayscale or (H, W, 3) RGB array.
    corners : (4, 2) (x, y) points ordered TL, TR, BR, BL.
    width, height : positive ints, size of the output rectangle.

    Returns the warped image with the same dtype/channel layout as the
    input (grayscale stays 2D, colour stays 3D).
    """
    if not isinstance(width, (int, np.integer)) or isinstance(width, bool):
        raise ValueError(f"width must be a positive int, got {width!r}")
    if not isinstance(height, (int, np.integer)) or isinstance(height, bool):
        raise ValueError(f"height must be a positive int, got {height!r}")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")
    corners = _validate_corners(corners)
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array, got "
                         f"{type(image).__name__}")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError(
            "image must be 2D grayscale (H, W) or 3D RGB/RGBA (H, W, 3|4), "
            f"got shape {image.shape}")
    if image.size == 0:
        raise ValueError("image must not be empty")

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                    [0, height - 1]], dtype=np.float64)
    h_mat = compute_homography(dst, corners)  # dst -> source
    gx, gy = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones_like(gx, dtype=np.float64)
    dst_h = np.stack([gx.astype(np.float64), gy.astype(np.float64), ones])
    src_h = np.einsum("ij,jkl->ikl", h_mat, dst_h)
    src_x = src_h[0] / src_h[2]
    src_y = src_h[1] / src_h[2]

    if image.ndim == 2:
        return _bilinear_sample(image.astype(np.float64), src_x, src_y)
    channels = [_bilinear_sample(image[:, :, c].astype(np.float64), src_x, src_y)
                for c in range(image.shape[2])]
    return np.stack(channels, axis=-1)


def scan_document(image, width=400, height=500, threshold=None):
    """Detect a document's corners and warp it flat in one call.

    Returns (warped image, corners) with corners ordered TL, TR, BR, BL.
    Raises ValueError if no bright quadrilateral is found.
    """
    corners = find_document_corners(image, threshold=threshold)
    warped = warp_document(image, corners, width, height)
    return warped, corners
