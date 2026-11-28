"""Simple face detection via skin-colour segmentation (numpy only).

Detects faces in an RGB image without any trained model: pixels whose
YCbCr chroma falls in the skin range are segmented, connected skin
regions are labelled, and regions that are face-sized, roughly
elliptical (a face fills most of its bounding box but not all of it)
and not too stretched are reported as faces with bounding boxes.
"""

import numpy as np
from collections import deque


def rgb_to_ycbcr(image):
    """Convert an RGB image (0-255) to YCbCr (ITU-R BT.601, 0-255)."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"image must be an RGB array of shape (H, W, 3), got {arr.shape}")
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def skin_mask(image, cb_range=(77, 127), cr_range=(133, 173)):
    """Boolean mask of skin-coloured pixels (YCbCr chroma ranges)."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"image must be an RGB array of shape (H, W, 3), got {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    ycbcr = rgb_to_ycbcr(arr)
    cb, cr = ycbcr[..., 1], ycbcr[..., 2]
    return ((cb >= cb_range[0]) & (cb <= cb_range[1]) &
            (cr >= cr_range[0]) & (cr <= cr_range[1]))


def _label_components(mask):
    """4-connected component labelling; returns a label image."""
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current = 0
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            current += 1
            queue = deque([(r, c)])
            labels[r, c] = current
            while queue:
                y, x = queue.popleft()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= ny < height and 0 <= nx < width
                            and mask[ny, nx] and labels[ny, nx] == 0):
                        labels[ny, nx] = current
                        queue.append((ny, nx))
    return labels, current


def detect_faces(image, min_area=200, max_fill=0.95, min_fill=0.45,
                 min_aspect=0.5, max_aspect=2.0):
    """Detect faces; returns a list of (top, left, bottom, right) boxes.

    ``bottom`` and ``right`` are exclusive. Regions are kept when
    their area is at least ``min_area``, their aspect ratio (width /
    height) is within ``[min_aspect, max_aspect]``, and their fill
    ratio (area / bounding-box area) is within ``[min_fill,
    max_fill]`` — a face ellipse fills ~0.79 of its box, while a solid
    rectangle fills 1.0 and scattered skin fills far less.
    """
    mask = skin_mask(image)
    labels, n_labels = _label_components(mask)
    faces = []
    for label in range(1, n_labels + 1):
        rows, cols = np.where(labels == label)
        area = len(rows)
        if area < min_area:
            continue
        top, bottom = int(rows.min()), int(rows.max()) + 1
        left, right = int(cols.min()), int(cols.max()) + 1
        box_h, box_w = bottom - top, right - left
        if box_h == 0 or box_w == 0:
            continue
        aspect = box_w / box_h
        if not (min_aspect <= aspect <= max_aspect):
            continue
        fill = area / float(box_h * box_w)
        if not (min_fill <= fill <= max_fill):
            continue
        faces.append((top, left, bottom, right))
    faces.sort()
    return faces
