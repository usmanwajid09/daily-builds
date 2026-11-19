"""Zhang-Suen skeleton thinning, from scratch (numpy only).

Thinning peels a binary shape down to a 1-pixel-wide skeleton that
keeps its topology — a thick stroke becomes its centreline, a blob
becomes a thin spine. The Zhang-Suen algorithm deletes border pixels
in two alternating sub-iterations, removing a pixel only when its
neighbourhood says the shape stays connected and no endpoint or
hole is destroyed.
"""

import numpy as np


def _validate_binary(image):
    """Return ``image`` as a 2D bool array or raise ValueError.

    Accepts bool arrays, 0/1 arrays, and 0/255 arrays.
    """
    arr = np.asarray(image)
    if arr.dtype.kind not in "bfiu":
        raise ValueError(f"image must be numeric or bool, got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"image must be a 2D binary image, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if arr.dtype.kind == "b":
        return arr.copy()
    if arr.dtype.kind == "f":
        values = set(np.unique(arr).tolist())
        if not values <= {0.0, 1.0, 255.0}:
            raise ValueError(
                f"image must be binary (0/1 or 0/255), got values {sorted(values)}")
    else:
        values = set(np.unique(arr).tolist())
        if not values <= {0, 1, 255}:
            raise ValueError(
                f"image must be binary (0/1 or 0/255), got values {sorted(values)}")
    return arr.astype(bool)


def _deletable(img, first_subiteration):
    """Return the list of pixels one Zhang-Suen sub-iteration deletes.

    Neighbours P2..P9 run clockwise from north around P1::

        P9 P2 P3
        P8 P1 P4
        P7 P6 P5

    A foreground pixel is deleted when 2 <= B(P1) <= 6 (B = foreground
    neighbours), A(P1) == 1 (A = 0->1 transitions around it, so the
    shape stays connected), and the sub-iteration's product condition
    holds (it protects a different diagonal in each pass).
    """
    height, width = img.shape
    doomed = []
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if not img[row, col]:
                continue
            p2 = int(img[row - 1, col])
            p3 = int(img[row - 1, col + 1])
            p4 = int(img[row, col + 1])
            p5 = int(img[row + 1, col + 1])
            p6 = int(img[row + 1, col])
            p7 = int(img[row + 1, col - 1])
            p8 = int(img[row, col - 1])
            p9 = int(img[row - 1, col - 1])
            neighbours = [p2, p3, p4, p5, p6, p7, p8, p9]
            count_b = sum(neighbours)
            if count_b < 2 or count_b > 6:
                continue
            ring = neighbours + [p2]
            count_a = sum(1 for a, b in zip(ring, ring[1:]) if a == 0 and b == 1)
            if count_a != 1:
                continue
            if first_subiteration:
                if p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
                    doomed.append((row, col))
            else:
                if p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
                    doomed.append((row, col))
    return doomed


def zhang_suen_thin(image):
    """Thin a binary image to a 1-pixel-wide skeleton.

    Args:
        image: 2D bool array, or 0/1 or 0/255 numeric array.

    Returns:
        A 2D bool array: the skeleton. It is a subset of the input
        foreground, and thinning it again changes nothing.

    Raises:
        ValueError: if the image is not a 2D binary array.
    """
    img = _validate_binary(image).copy()
    while True:
        changed = False
        for first in (True, False):
            doomed = _deletable(img, first)
            if doomed:
                changed = True
                for row, col in doomed:
                    img[row, col] = False
        if not changed:
            break
    return img


def skeletonize(image):
    """Alias for :func:`zhang_suen_thin`."""
    return zhang_suen_thin(image)


def skeleton_pixel_count(skeleton):
    """Count the foreground pixels in a skeleton (bool array)."""
    arr = np.asarray(skeleton)
    if arr.ndim != 2:
        raise ValueError(f"skeleton must be 2D, got shape {arr.shape}")
    return int(np.count_nonzero(arr))


def is_skeleton(image):
    """Return True when ``image`` is already 1 pixel wide (thinning is a no-op)."""
    binary = _validate_binary(image)
    return bool(np.array_equal(zhang_suen_thin(binary), binary))
