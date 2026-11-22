"""Panorama image stitching from scratch (numpy only).

Two images that show overlapping views of the same scene are aligned
by searching integer translations for the one with the highest
normalized cross-correlation (NCC) over the overlap, then placed on a
canvas covering their union and blended with a linear feather ramp
across the overlap so the seam is smooth.

Only integer translations are estimated (no rotation, scaling or
perspective). Grayscale (H, W) and colour (H, W, C) images are both
supported. The returned panorama is always a ``float64`` array.

Public API
----------
``normalized_cross_correlation(a, b)``
``overlap_region(shape1, shape2, shift)``
``find_best_shift(img1, img2, max_shift=None)``
``stitch_images(img1, img2, shift=None)``
"""

import numpy as np

__all__ = [
    "normalized_cross_correlation",
    "overlap_region",
    "find_best_shift",
    "stitch_images",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _is_int(value):
    """True for real ints (bool excluded, floats excluded)."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_image(image, name="image"):
    """Return ``image`` as a float64 ndarray, or raise ValueError."""
    try:
        arr = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric array") from None
    if arr.dtype == object or arr.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric array")
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"{name} must be a 2D grayscale or 3D colour array, "
            f"got {arr.ndim}D"
        )
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")
    if arr.ndim == 3 and arr.shape[2] == 0:
        raise ValueError(f"{name} must have at least one channel")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_pair(img1, img2):
    """Validate two images and return them as float64 arrays."""
    a = _validate_image(img1, "img1")
    b = _validate_image(img2, "img2")
    if (a.ndim == 2) != (b.ndim == 2):
        raise ValueError(
            "img1 and img2 must have the same number of dimensions "
            f"(got {a.ndim}D and {b.ndim}D)"
        )
    if a.ndim == 3 and a.shape[2] != b.shape[2]:
        raise ValueError(
            "img1 and img2 must have the same number of channels "
            f"(got {a.shape[2]} and {b.shape[2]})"
        )
    return a, b


def _validate_shift(shift):
    """Validate a (dy, dx) shift; return it as a pair of ints."""
    if isinstance(shift, (str, bytes)) or not hasattr(shift, "__len__"):
        raise ValueError("shift must be a pair of ints (dy, dx)")
    if len(shift) != 2:
        raise ValueError("shift must be a pair of ints (dy, dx)")
    dy, dx = shift
    if not _is_int(dy) or not _is_int(dx):
        raise ValueError("shift must be a pair of ints (dy, dx)")
    return int(dy), int(dx)


def _validate_max_shift(max_shift):
    if not _is_int(max_shift):
        raise ValueError("max_shift must be a non-negative int")
    if max_shift < 0:
        raise ValueError("max_shift must be a non-negative int")
    return int(max_shift)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def normalized_cross_correlation(a, b):
    """NCC between two same-shaped patches, in [-1, 1].

    1.0 means the patches are identical up to a positive affine change
    in brightness. Returns 0.0 when either patch is constant (the
    correlation is undefined there).
    """
    a = _validate_image(a, "a")
    b = _validate_image(b, "b")
    if a.shape != b.shape:
        raise ValueError(
            f"a and b must have the same shape (got {a.shape} and {b.shape})"
        )
    a_flat = a.ravel()
    b_flat = b.ravel()
    a_centred = a_flat - a_flat.mean()
    b_centred = b_flat - b_flat.mean()
    denom = np.sqrt((a_centred ** 2).sum() * (b_centred ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float((a_centred * b_centred).sum() / denom)


def overlap_region(shape1, shape2, shift):
    """Overlap of img2 (placed at ``shift``) with img1.

    ``shift`` is (dy, dx): img2's top-left corner sits at row ``dy``,
    column ``dx`` in img1's coordinates. Returns
    ``(y0, y1, x0, x1, y0b, y1b, x0b, x1b)`` -- the row/column ranges
    of the overlap in img1's coordinates first, then in img2's -- or
    ``None`` when the two images do not overlap at all.
    """
    dy, dx = _validate_shift(shift)
    h1, w1 = int(shape1[0]), int(shape1[1])
    h2, w2 = int(shape2[0]), int(shape2[1])
    y0 = max(0, dy)
    y1 = min(h1, h2 + dy)
    x0 = max(0, dx)
    x1 = min(w1, w2 + dx)
    if y1 <= y0 or x1 <= x0:
        return None
    return (y0, y1, x0, x1, y0 - dy, y1 - dy, x0 - dx, x1 - dx)


def find_best_shift(img1, img2, max_shift=None):
    """Find the integer (dy, dx) translation aligning img2 onto img1.

    Every candidate shift with ``|dy|, |dx| <= max_shift`` is scored
    by the normalized cross-correlation over the overlap region, and
    the best-scoring shift is returned together with its score as
    ``((dy, dx), score)``. ``max_shift`` defaults to half the smaller
    image side. Ties are broken in favour of the smallest shift, so
    identical images align at (0, 0).
    """
    a, b = _validate_pair(img1, img2)
    if max_shift is None:
        max_shift = min(a.shape[0], a.shape[1], b.shape[0], b.shape[1]) // 2
    max_shift = _validate_max_shift(max_shift)

    best_shift = None
    best_score = -np.inf
    # Search smallest shifts first so ties keep the smallest shift.
    candidates = [
        (dy, dx)
        for dy in range(-max_shift, max_shift + 1)
        for dx in range(-max_shift, max_shift + 1)
    ]
    candidates.sort(key=lambda s: (abs(s[0]) + abs(s[1]), s[0], s[1]))
    for dy, dx in candidates:
        region = overlap_region(a.shape, b.shape, (dy, dx))
        if region is None:
            continue
        y0, y1, x0, x1, y0b, y1b, x0b, x1b = region
        if (y1 - y0) * (x1 - x0) < 4:
            continue
        patch_a = a[y0:y1, x0:x1]
        patch_b = b[y0b:y1b, x0b:x1b]
        # Skip degenerate (constant) patches: NCC is undefined there.
        if patch_a.std() == 0.0 or patch_b.std() == 0.0:
            continue
        score = normalized_cross_correlation(patch_a, patch_b)
        if score > best_score + 1e-12:
            best_score = score
            best_shift = (dy, dx)
    if best_shift is None:
        raise ValueError(
            "no candidate shift gives a valid overlap; "
            "images may be constant or max_shift may be too small"
        )
    return best_shift, float(best_score)


def stitch_images(img1, img2, shift=None):
    """Stitch img1 and img2 into one panorama (float64 array).

    img1 sits at the origin; img2 is placed at ``shift`` = (dy, dx)
    relative to img1 (found automatically with
    :func:`find_best_shift` when ``shift`` is None). The canvas covers
    the union of both images. Where only one image covers the canvas
    it is copied exactly; in the overlap the two are blended with a
    linear feather ramp (img2's weight ramps 0 -> 1 across the overlap
    width, or across its height when the overlap is one pixel wide).
    Images placed apart (no overlap) still produce a valid canvas,
    with zeros in the gap.
    """
    a, b = _validate_pair(img1, img2)
    if shift is None:
        shift, _ = find_best_shift(a, b)
    else:
        shift = _validate_shift(shift)
    dy, dx = shift

    h1, w1 = a.shape[0], a.shape[1]
    h2, w2 = b.shape[0], b.shape[1]
    top = min(0, dy)
    left = min(0, dx)
    bottom = max(h1, h2 + dy)
    right = max(w1, w2 + dx)
    canvas_h = bottom - top
    canvas_w = right - left

    canvas = np.zeros((canvas_h, canvas_w) + a.shape[2:], dtype=np.float64)
    # Place img1, then img2; the overlap is blended afterwards.
    r1, c1 = -top, -left
    canvas[r1:r1 + h1, c1:c1 + w1] = a
    r2, c2 = dy - top, dx - left
    canvas[r2:r2 + h2, c2:c2 + w2] = b

    region = overlap_region(a.shape, b.shape, (dy, dx))
    if region is not None:
        y0, y1, x0, x1, y0b, y1b, x0b, x1b = region
        ov_h = y1 - y0
        ov_w = x1 - x0
        patch_a = a[y0:y1, x0:x1]
        patch_b = b[y0b:y1b, x0b:x1b]
        if ov_w >= 2:
            # Feather left -> right: img2 weight 0 at the left edge,
            # 1 at the right edge of the overlap.
            ramp = np.linspace(0.0, 1.0, ov_w, dtype=np.float64)
            weight = np.tile(ramp, (ov_h, 1))
        else:
            # One-pixel-wide overlap: feather top -> bottom instead.
            ramp = np.linspace(0.0, 1.0, ov_h, dtype=np.float64)
            weight = np.tile(ramp[:, np.newaxis], (1, ov_w))
        if a.ndim == 3:
            weight = weight[..., np.newaxis]
        blended = (1.0 - weight) * patch_a + weight * patch_b
        canvas[y0 - top:y1 - top, x0 - left:x1 - left] = blended
    return canvas
