"""Skin segmentation in YCbCr colour space from scratch (numpy only).

Skin pixels cluster tightly in the chrominance (Cb, Cr) plane, so a
simple rule-based test on YCbCr values — Cb in [77, 127] and Cr in
[133, 173], the classic Chai & Ngan bounds — segments skin without any
training data. The module also reports skin fraction and the bounding
box of the largest connected skin region (4-connectivity flood fill).
"""

import numpy as np
from collections import deque

CB_RANGE = (77, 127)
CR_RANGE = (133, 173)


def _validate_rgb(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must have shape (H, W, 3), got {image.shape}"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(image.astype(np.float64)).all():
        raise ValueError("image must contain only finite values")
    if image.min() < 0 or image.max() > 255:
        raise ValueError("image values must lie in [0, 255]")
    return image.astype(np.float64)


def rgb_to_ycbcr(image):
    """Convert an RGB image (values 0-255) to YCbCr (BT.601, studio swing)."""
    rgb = _validate_rgb(image)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def detect_skin(image, cb_range=CB_RANGE, cr_range=CR_RANGE):
    """Return a boolean mask of skin pixels in an RGB ``image``.

    A pixel counts as skin when its Cb lies in ``cb_range`` and its Cr
    lies in ``cr_range`` (defaults are the classic Chai & Ngan bounds).
    """
    if len(cb_range) != 2 or cb_range[0] > cb_range[1]:
        raise ValueError(f"invalid cb_range {cb_range!r}")
    if len(cr_range) != 2 or cr_range[0] > cr_range[1]:
        raise ValueError(f"invalid cr_range {cr_range!r}")
    ycbcr = rgb_to_ycbcr(image)
    cb, cr = ycbcr[..., 1], ycbcr[..., 2]
    return (cb >= cb_range[0]) & (cb <= cb_range[1]) & \
        (cr >= cr_range[0]) & (cr <= cr_range[1])


def _largest_component(mask):
    """Return ``(bbox, pixel_count)`` of the largest 4-connected region.

    ``bbox`` is ``(row_min, col_min, row_max, col_max)`` inclusive, or
    ``None`` when the mask is empty.
    """
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_bbox, best_count = None, 0
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or visited[r, c]:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            r_min = r_max = r
            c_min = c_max = c
            count = 0
            while queue:
                cr_, cc_ = queue.popleft()
                count += 1
                r_min, r_max = min(r_min, cr_), max(r_max, cr_)
                c_min, c_max = min(c_min, cc_), max(c_max, cc_)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr_ + dr, cc_ + dc
                    if (0 <= nr < height and 0 <= nc < width
                            and mask[nr, nc] and not visited[nr, nc]):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            if count > best_count:
                best_count = count
                best_bbox = (r_min, c_min, r_max, c_max)
    return best_bbox, best_count


def skin_stats(image, cb_range=CB_RANGE, cr_range=CR_RANGE):
    """Summarise skin in an RGB ``image``.

    Returns a dict with:
    - ``skin_fraction``: fraction of pixels classified as skin.
    - ``skin_pixels``: number of skin pixels.
    - ``largest_bbox``: ``(row_min, col_min, row_max, col_max)`` of the
      largest skin region, or ``None`` when no skin is found.
    - ``largest_pixels``: pixel count of the largest skin region.
    """
    mask = detect_skin(image, cb_range, cr_range)
    bbox, count = _largest_component(mask)
    total = mask.size
    return {
        "skin_fraction": float(mask.sum() / total),
        "skin_pixels": int(mask.sum()),
        "largest_bbox": bbox,
        "largest_pixels": int(count),
    }
