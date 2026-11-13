"""Harris corner detection, implemented from scratch (numpy only).

The pipeline is the classic Harris & Stephens one:

1. Convert the input to grayscale and take Sobel gradients ``Ix, Iy``.
2. Build the structure tensor at each pixel from the products
   ``Ix^2, Iy^2, Ix*Iy``, smoothed with a small box window.
3. Corner response ``R = det(M) - k * trace(M)^2``. Corners have large
   gradients in *both* directions, so both eigenvalues of ``M`` are
   large and ``R`` is large and positive; edges have one large
   eigenvalue and give ``R <= 0``; flat areas give ``R ~ 0``.
4. Keep local maxima of ``R`` above a relative threshold (non-maximum
   suppression), then return the strongest ``max_corners`` of them.

No OpenCV, no scipy — the Sobel convolution is nine shifted adds, and
the window sums are done with an integral image.
"""

import numpy as np

_SOBEL_X = np.array([[-1.0, 0.0, 1.0],
                     [-2.0, 0.0, 2.0],
                     [-1.0, 0.0, 1.0]])

_SOBEL_Y = np.array([[-1.0, -2.0, -1.0],
                     [0.0, 0.0, 0.0],
                     [1.0, 2.0, 1.0]])

_LUMINANCE = np.array([0.299, 0.587, 0.114])


def _to_grayscale(image):
    """Return a 2D float64 grayscale view of ``image`` or raise."""
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"image must be numeric, got dtype {arr.dtype}")
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr @ _LUMINANCE
    elif arr.ndim != 2:
        raise ValueError(
            f"image must be 2D grayscale or (H, W, 3) RGB, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("image must not be empty")
    gray = arr.astype(np.float64)
    if not np.isfinite(gray).all():
        raise ValueError("image must contain only finite values")
    return gray


def _convolve3x3(image, kernel):
    """Convolve a 2D array with a 3x3 kernel (replicate padding)."""
    padded = np.pad(image, 1, mode="edge")
    out = np.zeros_like(image, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * padded[i:i + image.shape[0], j:j + image.shape[1]]
    return out


def _box_sum(image, radius):
    """Sum of ``image`` over a (2*radius+1) square window per pixel.

    Uses an integral image, so the cost is independent of the window
    size. Padded with zeros, so windows at the border only sum the
    pixels that exist.
    """
    if radius == 0:
        return image.copy()
    padded = np.pad(image, radius, mode="constant", constant_values=0.0)
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1))
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    h, w = image.shape
    size = 2 * radius + 1
    top, left = 0, 0
    bottom, right = top + h + size - 1, left + w + size - 1
    # Window for output pixel (r, c) covers integral rows r..r+size-1
    # and columns c..c+size-1, shifted by the padding offset.
    total = (integral[size:size + h, size:size + w]
             - integral[0:h, size:size + w]
             - integral[size:size + h, 0:w]
             + integral[0:h, 0:w])
    return total


def _validate_k(k):
    if isinstance(k, bool) or not isinstance(k, (int, float, np.floating)):
        raise ValueError(f"k must be a number in (0, 1), got {k!r}")
    if not np.isfinite(k) or not 0.0 < float(k) < 1.0:
        raise ValueError(f"k must be in (0, 1), got {k}")


def harris_response(image, k=0.04, window_radius=1):
    """Return the Harris corner response ``R`` for ``image``.

    Args:
        image: 2D grayscale array, or (H, W, 3) RGB array.
        k: Harris sensitivity in (0, 1); 0.04-0.06 is the usual range.
        window_radius: radius of the box window (in pixels) used to
            smooth the structure-tensor products; must be >= 0.

    Returns:
        A float64 array with the same (H, W) shape as the input.

    Raises:
        ValueError: on a bad image, a ``k`` outside (0, 1), or a
            negative ``window_radius``.
    """
    _validate_k(k)
    if isinstance(window_radius, bool) or not isinstance(window_radius, (int, np.integer)):
        raise ValueError(f"window_radius must be a non-negative integer, got {window_radius!r}")
    if int(window_radius) < 0:
        raise ValueError(f"window_radius must be >= 0, got {window_radius}")
    gray = _to_grayscale(image)
    ix = _convolve3x3(gray, _SOBEL_X)
    iy = _convolve3x3(gray, _SOBEL_Y)
    ixx = _box_sum(ix * ix, int(window_radius))
    iyy = _box_sum(iy * iy, int(window_radius))
    ixy = _box_sum(ix * iy, int(window_radius))
    det = ixx * iyy - ixy * ixy
    trace = ixx + iyy
    return det - float(k) * trace * trace


def _local_maxima(response, radius):
    """Boolean mask of pixels that are the max in their neighbourhood."""
    padded = np.pad(response, radius, mode="constant",
                    constant_values=-np.inf)
    h, w = response.shape
    centre = padded[radius:radius + h, radius:radius + w]
    is_max = np.ones_like(response, dtype=bool)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            neighbour = padded[radius + dr:radius + dr + h,
                               radius + dc:radius + dc + w]
            is_max &= centre >= neighbour
    return is_max


def detect_corners(image, max_corners=10, k=0.04, window_radius=1,
                   threshold_rel=0.01, nms_radius=2):
    """Return the strongest corners in ``image`` as (row, col) pairs.

    Args:
        image: 2D grayscale array, or (H, W, 3) RGB array.
        max_corners: maximum number of corners to return (>= 1).
        k: Harris sensitivity in (0, 1); see :func:`harris_response`.
        window_radius: structure-tensor window radius; see
            :func:`harris_response`.
        threshold_rel: relative threshold in [0, 1]; a corner must have
            response at least ``threshold_rel`` times the strongest
            response in the image.
        nms_radius: non-maximum suppression radius; corners closer
            than this are merged, keeping the stronger one.

    Returns:
        A list of ``(row, col)`` integer tuples, strongest first, with
        at most ``max_corners`` entries (possibly fewer, or empty for a
        flat image).

    Raises:
        ValueError: on a bad image, ``max_corners`` < 1, bad ``k`` /
            ``window_radius``, a ``threshold_rel`` outside [0, 1], or a
            negative ``nms_radius``.
    """
    _validate_k(k)
    if isinstance(max_corners, bool) or not isinstance(max_corners, (int, np.integer)):
        raise ValueError(f"max_corners must be a positive integer, got {max_corners!r}")
    if int(max_corners) < 1:
        raise ValueError(f"max_corners must be >= 1, got {max_corners}")
    if isinstance(nms_radius, bool) or not isinstance(nms_radius, (int, np.integer)):
        raise ValueError(f"nms_radius must be a non-negative integer, got {nms_radius!r}")
    if int(nms_radius) < 0:
        raise ValueError(f"nms_radius must be >= 0, got {nms_radius}")
    if isinstance(threshold_rel, bool) or not isinstance(threshold_rel, (int, float, np.floating)):
        raise ValueError(f"threshold_rel must be a number in [0, 1], got {threshold_rel!r}")
    if not 0.0 <= float(threshold_rel) <= 1.0:
        raise ValueError(f"threshold_rel must be in [0, 1], got {threshold_rel}")
    response = harris_response(image, k=k, window_radius=window_radius)
    peak = response.max()
    if peak <= 0.0:
        return []
    keep = _local_maxima(response, max(int(nms_radius), 1))
    keep &= (response >= float(threshold_rel) * peak) & (response > 0)
    # Border pixels are excluded: their windows are incomplete, so the
    # response there is unreliable.
    keep[:1, :] = False
    keep[-1:, :] = False
    keep[:, :1] = False
    keep[:, -1:] = False
    rows, cols = np.nonzero(keep)
    order = np.argsort(-response[rows, cols])[:int(max_corners)]
    return [(int(rows[i]), int(cols[i])) for i in order]
