"""Content-aware image resizing via seam carving (numpy only).

A vertical seam is a connected path of one pixel per row, where consecutive
rows differ in column by at most one. The seam with the lowest total energy
(gradient magnitude) is found with dynamic programming and removed; repeating
the process shrinks the image width while preserving high-energy content
(edges, objects) and discarding low-energy (smooth) regions.

Supports grayscale (2D) and RGB (3D) images.
"""

import numpy as np

__all__ = [
    "to_grayscale",
    "energy_map",
    "find_vertical_seam",
    "remove_seam",
    "remove_vertical_seams",
    "shrink_width",
]


def _validate_image(image):
    """Validate that *image* is a non-empty 2D or 3D numpy array."""
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"image must be a 2D (grayscale) or 3D (RGB) array, got {arr.ndim}D"
        )
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("image must be non-empty (height and width > 0)")
    if arr.ndim == 3 and arr.shape[2] == 0:
        raise ValueError("image must have at least one channel")
    return arr


def to_grayscale(image):
    """Convert a 2D or 3D image to a float grayscale array.

    Grayscale input is returned as float64 unchanged (values preserved);
    RGB input uses the standard luminance weights.
    """
    arr = _validate_image(image).astype(np.float64)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 1:
        return arr[:, :, 0]
    if arr.shape[2] == 3:
        return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    # Generic multi-channel fallback: mean across channels.
    return arr.mean(axis=2)


def energy_map(image):
    """Gradient-magnitude energy map of an image.

    Energy at each pixel is |dx| + |dy| of the grayscale image, computed with
    Sobel-style finite differences (central differences in the interior,
    one-sided at the borders) via ``numpy.gradient``.
    """
    gray = to_grayscale(image)
    dy, dx = np.gradient(gray)
    return np.abs(dx) + np.abs(dy)


def _cumulative_energy(energy):
    """DP table: cumulative minimum energy to reach each pixel from the top.

    cumulative[i, j] = energy[i, j] + min(cumulative[i-1, j-1..j+1]).
    """
    h, w = energy.shape
    cumulative = energy.astype(np.float64).copy()
    for i in range(1, h):
        prev = cumulative[i - 1]
        left = np.roll(prev, 1)
        left[0] = np.inf
        right = np.roll(prev, -1)
        right[-1] = np.inf
        cumulative[i] += np.minimum(np.minimum(left, prev), right)
    return cumulative


def find_vertical_seam(energy):
    """Return the lowest-energy vertical seam as an array of column indices.

    One column index per row; consecutive rows differ by at most 1.
    """
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 2:
        raise ValueError(f"energy must be a 2D array, got {energy.ndim}D")
    if energy.size == 0 or energy.shape[0] == 0 or energy.shape[1] == 0:
        raise ValueError("energy map must be non-empty")
    cumulative = _cumulative_energy(energy)
    h, w = energy.shape
    seam = np.empty(h, dtype=int)
    seam[-1] = int(np.argmin(cumulative[-1]))
    for i in range(h - 2, -1, -1):
        j = seam[i + 1]
        lo = max(j - 1, 0)
        hi = min(j + 1, w - 1)
        seam[i] = lo + int(np.argmin(cumulative[i, lo:hi + 1]))
    return seam


def remove_seam(image, seam):
    """Remove one vertical seam from the image; returns a new array."""
    arr = _validate_image(image)
    seam = np.asarray(seam)
    h, w = arr.shape[0], arr.shape[1]
    if seam.shape != (h,):
        raise ValueError(
            f"seam must have one column per row ({h}), got shape {seam.shape}"
        )
    if w <= 1:
        raise ValueError("cannot remove a seam from an image of width 1")
    if np.any(seam < 0) or np.any(seam >= w):
        raise ValueError("seam column indices out of bounds")
    if arr.ndim == 2:
        out = np.empty((h, w - 1), dtype=arr.dtype)
        for i in range(h):
            out[i] = np.delete(arr[i], seam[i])
    else:
        out = np.empty((h, w - 1, arr.shape[2]), dtype=arr.dtype)
        for i in range(h):
            out[i] = np.delete(arr[i], seam[i], axis=0)
    return out


def remove_vertical_seams(image, n):
    """Remove *n* vertical seams, recomputing the energy after each removal.

    Returns a new array with width reduced by *n* and height unchanged.
    """
    arr = _validate_image(image)
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
        raise ValueError(f"n must be a non-negative integer, got {n!r}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n >= arr.shape[1]:
        raise ValueError(
            f"cannot remove {n} seams from image of width {arr.shape[1]} "
            "(must leave at least one column)"
        )
    result = np.array(arr, copy=True)
    for _ in range(n):
        seam = find_vertical_seam(energy_map(result))
        result = remove_seam(result, seam)
    return result


def shrink_width(image, target_width):
    """Shrink the image to *target_width* columns via seam carving."""
    arr = _validate_image(image)
    if not isinstance(target_width, (int, np.integer)) or isinstance(
        target_width, bool
    ):
        raise ValueError(
            f"target_width must be an integer, got {target_width!r}"
        )
    if target_width <= 0:
        raise ValueError(f"target_width must be > 0, got {target_width}")
    if target_width >= arr.shape[1]:
        raise ValueError(
            f"target_width ({target_width}) must be smaller than the current "
            f"width ({arr.shape[1]})"
        )
    return remove_vertical_seams(arr, arr.shape[1] - target_width)
