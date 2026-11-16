"""FAST corner detection (Features from Accelerated Segment Test).

A 16-pixel Bresenham circle of radius 3 is drawn around each pixel. A pixel
is a corner if at least ``n_contiguous`` pixels on that circle are all
brighter than ``centre + threshold`` or all darker than ``centre - threshold``.
The classic FAST-9 variant uses ``n_contiguous=9``; the stricter FAST-12
variant uses 12. Note that a perfect 90-degree step corner produces at most
an 11-pixel arc on the 16-pixel circle, so FAST-12 only fires on sharper
(acute) corners, while FAST-9 fires on ordinary square corners.

Only NumPy is used.
"""

import numpy as np

# 16 pixels of a Bresenham circle of radius 3, clockwise from (0, +3).
# Each entry is a (row_offset, col_offset) pair.
CIRCLE_OFFSETS = (
    (0, 3), (1, 3), (2, 2), (3, 1),
    (3, 0), (3, -1), (2, -2), (1, -3),
    (0, -3), (-1, -3), (-2, -2), (-3, -1),
    (-3, 0), (-3, 1), (-2, 2), (-1, 3),
)

# Indices (0-based) of circle pixels 1, 5, 9 and 13, used by the fast
# pre-test: at least three of these four must be brighter (or darker) than
# the centre before the full 16-pixel test is run.
_PRETEST_INDICES = (0, 4, 8, 12)

_RADIUS = 3

__all__ = [
    "CIRCLE_OFFSETS",
    "corner_score",
    "detect_corners",
    "is_corner",
]


def _validate(image, threshold, n_contiguous, nms_radius):
    """Validate the common arguments, raising ValueError with a clear message."""
    try:
        arr = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("image must be a numeric 2D array") from None
    if arr.ndim != 2:
        raise ValueError(
            f"image must be a 2D grayscale array, got {arr.ndim}D array"
        )
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("image must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("image must contain only finite values")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if not 1 <= n_contiguous <= 16:
        raise ValueError(
            f"n_contiguous must be between 1 and 16, got {n_contiguous}"
        )
    if nms_radius < 1:
        raise ValueError(f"nms_radius must be >= 1, got {nms_radius}")
    return arr


def _longest_run(flags):
    """Longest run of True in ``flags``, treating the list as circular.

    Returns ``(length, start_index)`` of the longest contiguous run.
    """
    n = len(flags)
    doubled = list(flags) + list(flags)
    best_len, best_start = 0, 0
    cur_len, cur_start = 0, 0
    for i, flag in enumerate(doubled):
        if flag:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len and cur_len <= n:
                best_len, best_start = cur_len, cur_start % n
        else:
            cur_len = 0
    return best_len, best_start


def _test_pixel(circle_values, centre, threshold, n_contiguous):
    """Run the FAST segment test on one pixel's 16 circle values.

    Returns ``(is_corner, score)``. The score is the largest sum of absolute
    differences from the centre over a qualifying contiguous run (0 if the
    pixel is not a corner).
    """
    diffs = [v - centre for v in circle_values]
    bright = [d > threshold for d in diffs]
    dark = [d < -threshold for d in diffs]

    # Fast pre-test: pixels 1, 5, 9, 13. A contiguous run of n circle pixels
    # always contains at least n // 4 of these four cardinal pixels, so if
    # fewer than n // 4 of them are brighter than centre + threshold (and
    # fewer than n // 4 are darker than centre - threshold), no qualifying
    # run can exist and the full 16-pixel test is skipped. For the default
    # n_contiguous=12 this is the classic "at least 3 of the 4" pre-test;
    # for n_contiguous=9 it is "at least 2 of the 4", which is what lets a
    # 90-degree corner (whose four cardinal pixels split 2 bright / 2 dark)
    # pass through to the full test.
    pre_min = n_contiguous // 4
    if pre_min > 0:
        pre_bright = sum(bright[i] for i in _PRETEST_INDICES)
        pre_dark = sum(dark[i] for i in _PRETEST_INDICES)
        if pre_bright < pre_min and pre_dark < pre_min:
            return False, 0.0

    is_corner = False
    best_score = 0.0
    for flags in (bright, dark):
        length, start = _longest_run(flags)
        if length >= n_contiguous:
            is_corner = True
            score = sum(abs(diffs[(start + k) % 16]) for k in range(length))
            best_score = max(best_score, score)
    return is_corner, best_score


def _circle_values(image, row, col):
    """The 16 circle pixel values around ``(row, col)``."""
    return [image[row + dr, col + dc] for dr, dc in CIRCLE_OFFSETS]


def is_corner(image, row, col, threshold=20, n_contiguous=12):
    """True if the pixel at ``(row, col)`` passes the FAST segment test.

    ``row`` and ``col`` must be at least 3 pixels from every border, since
    the radius-3 circle has to fit inside the image; otherwise ValueError.
    """
    arr = _validate(image, threshold, n_contiguous, nms_radius=1)
    height, width = arr.shape
    if not (_RADIUS <= row < height - _RADIUS
            and _RADIUS <= col < width - _RADIUS):
        raise ValueError(
            f"pixel ({row}, {col}) is too close to the border of a "
            f"{height}x{width} image: the radius-3 circle must fit inside"
        )
    centre = float(arr[row, col])
    corner, _ = _test_pixel(_circle_values(arr, row, col), centre,
                            threshold, n_contiguous)
    return corner


def corner_score(image, row, col, threshold=20, n_contiguous=12):
    """Score of the pixel at ``(row, col)``: 0 if it is not a corner.

    The score is the largest sum of absolute differences from the centre
    over a qualifying contiguous run of brighter or darker circle pixels,
    so stronger, higher-contrast corners score higher.
    """
    arr = _validate(image, threshold, n_contiguous, nms_radius=1)
    height, width = arr.shape
    if not (_RADIUS <= row < height - _RADIUS
            and _RADIUS <= col < width - _RADIUS):
        raise ValueError(
            f"pixel ({row}, {col}) is too close to the border of a "
            f"{height}x{width} image: the radius-3 circle must fit inside"
        )
    centre = float(arr[row, col])
    _, score = _test_pixel(_circle_values(arr, row, col), centre,
                           threshold, n_contiguous)
    return float(score)


def _non_maximum_suppression(rows, cols, scores, nms_radius):
    """Greedy NMS: keep corners in score order, dropping any corner within
    ``nms_radius`` (Chebyshev distance) of a stronger kept corner."""
    order = sorted(range(len(scores)),
                   key=lambda i: (-scores[i], rows[i], cols[i]))
    kept = []
    for i in order:
        r, c = rows[i], cols[i]
        if all(max(abs(r - kr), abs(c - kc)) >= nms_radius
               for kr, kc in kept):
            kept.append((r, c))
    kept_set = set(kept)
    keep_idx = [i for i in range(len(scores))
                if (rows[i], cols[i]) in kept_set]
    # Preserve score order (strongest first) in the output.
    keep_idx.sort(key=lambda i: (-scores[i], rows[i], cols[i]))
    return keep_idx


def detect_corners(image, threshold=20, n_contiguous=12, nms=True,
                   nms_radius=3):
    """Detect FAST corners in a 2D grayscale image.

    Parameters
    ----------
    image : 2D array-like
        Grayscale image (any numeric dtype).
    threshold : float, default 20
        A circle pixel counts as brighter/darker only if it differs from
        the centre by more than this. Must be > 0.
    n_contiguous : int, default 12
        Minimum length of a contiguous run of brighter (or darker) circle
        pixels for a corner. Must be between 1 and 16. Use 9 (FAST-9) for
        ordinary 90-degree corners and 12 (FAST-12) for sharper corners.
    nms : bool, default True
        Apply non-maximum suppression to collapse each cluster of corner
        pixels into a single corner.
    nms_radius : int, default 3
        With NMS on, no two returned corners are closer than this
        (Chebyshev distance). Must be >= 1.

    Returns
    -------
    corners : ndarray of shape (k, 2), dtype int
        ``(row, col)`` of each detected corner, strongest score first.
    scores : ndarray of shape (k,), dtype float
        Corner score of each returned corner.
    """
    arr = _validate(image, threshold, n_contiguous, nms_radius)
    height, width = arr.shape

    rows, cols, scores = [], [], []
    for r in range(_RADIUS, height - _RADIUS):
        for c in range(_RADIUS, width - _RADIUS):
            centre = float(arr[r, c])
            corner, score = _test_pixel(_circle_values(arr, r, c), centre,
                                        threshold, n_contiguous)
            if corner:
                rows.append(r)
                cols.append(c)
                scores.append(score)

    if not rows:
        return (np.empty((0, 2), dtype=int),
                np.empty((0,), dtype=float))

    if nms:
        keep = _non_maximum_suppression(rows, cols, scores, nms_radius)
        rows = [rows[i] for i in keep]
        cols = [cols[i] for i in keep]
        scores = [scores[i] for i in keep]
    else:
        order = sorted(range(len(scores)),
                       key=lambda i: (-scores[i], rows[i], cols[i]))
        rows = [rows[i] for i in order]
        cols = [cols[i] for i in order]
        scores = [scores[i] for i in order]

    corners = np.array(list(zip(rows, cols)), dtype=int)
    return corners, np.array(scores, dtype=float)
