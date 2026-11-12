"""Sobel edge detection from scratch.

The pipeline is the classic one, built only on numpy:

1. Convert the input to a single-channel grayscale image.
2. Convolve with the horizontal and vertical Sobel kernels to get the
   image gradients ``gx`` and ``gy``.
3. Gradient magnitude ``sqrt(gx**2 + gy**2)`` is the edge strength.
4. (Optional) Thin the response with non-maximum suppression along the
   gradient direction, so edges come out one pixel wide.
5. Threshold the magnitude, relative to the strongest edge in the image,
   to produce a boolean edge map.

No OpenCV, no PIL, no scipy — the 3x3 convolution is nine shifted adds.
"""

import numpy as np

_SOBEL_X = np.array([[-1.0, 0.0, 1.0],
                     [-2.0, 0.0, 2.0],
                     [-1.0, 0.0, 1.0]])

_SOBEL_Y = np.array([[-1.0, -2.0, -1.0],
                     [0.0, 0.0, 0.0],
                     [1.0, 2.0, 1.0]])

_LUMINANCE = np.array([0.299, 0.587, 0.114])


def to_grayscale(image):
    """Return a 2D float64 grayscale view of ``image``.

    Accepts a 2D grayscale array, a (H, W, 3) RGB array (luminance
    weights), or a (H, W, 1) array. Raises ``ValueError`` for anything
    else, for empty images, or for non-finite values.
    """
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
    """Convolve a 2D array with a 3x3 kernel (replicate padding).

    Replicate padding (border values repeated) means a flat image stays
    flat at the border — zero padding would invent a fake edge there,
    because the kernel would see a step from 0 to the border value.
    """
    padded = np.pad(image, 1, mode="edge")
    out = np.zeros_like(image, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * padded[i:i + image.shape[0], j:j + image.shape[1]]
    return out


def sobel_gradients(image):
    """Return ``(gx, gy, magnitude)`` Sobel gradients of ``image``.

    ``gx`` responds to vertical edges, ``gy`` to horizontal edges, and
    ``magnitude`` is the per-pixel edge strength.
    """
    gray = to_grayscale(image)
    gx = _convolve3x3(gray, _SOBEL_X)
    gy = _convolve3x3(gray, _SOBEL_Y)
    magnitude = np.hypot(gx, gy)
    return gx, gy, magnitude


def _non_maximum_suppression(magnitude, gx, gy):
    """Thin ``magnitude`` to one-pixel-wide ridges along the gradient.

    Each pixel survives only if its magnitude is at least as large as the
    two neighbours lying along its gradient direction (quantised to 0, 45,
    90 or 135 degrees). Border pixels are never edges.
    """
    angle = np.degrees(np.arctan2(gy, gx)) % 180.0
    # Quantise the direction into 4 sectors: 0, 45, 90, 135 degrees.
    sector = ((angle + 22.5) // 45).astype(int) % 4
    padded = np.pad(magnitude, 1, mode="constant", constant_values=0.0)
    h, w = magnitude.shape
    centre = padded[1:h + 1, 1:w + 1]
    # Neighbour pairs for each sector: (row offset, col offset) of one side.
    offsets = [(0, 1), (1, 1), (1, 0), (1, -1)]  # 0, 45, 90, 135 degrees
    suppressed = np.zeros_like(magnitude)
    for s, (dr, dc) in enumerate(offsets):
        mask = sector == s
        if not mask.any():
            continue
        a = padded[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc]
        b = padded[1 - dr:h + 1 - dr, 1 - dc:w + 1 - dc]
        keep = mask & (centre >= a) & (centre >= b)
        suppressed[keep] = centre[keep]
    return suppressed


def detect_edges(image, threshold=0.1, nms=True):
    """Return a boolean edge map for ``image``.

    Args:
        image: 2D grayscale array, or (H, W, 3) RGB array.
        threshold: relative threshold in [0, 1]; a pixel is an edge when
            its gradient magnitude is at least ``threshold`` times the
            strongest magnitude in the image. 0 keeps every non-zero
            gradient, 1 keeps only the single strongest ridge.
        nms: if True (default), thin edges with non-maximum suppression
            before thresholding.

    Returns:
        A boolean ``numpy`` array with the same (H, W) shape as the input.

    Raises:
        ValueError: on a bad image (see :func:`to_grayscale`) or a
            threshold outside [0, 1].
    """
    if not isinstance(threshold, (int, float, np.floating)) or isinstance(threshold, bool):
        raise ValueError(f"threshold must be a number in [0, 1], got {threshold!r}")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    gray = to_grayscale(image)
    gx, gy, magnitude = sobel_gradients(gray)
    if nms:
        magnitude = _non_maximum_suppression(magnitude, gx, gy)
    peak = magnitude.max()
    if peak == 0.0:
        return np.zeros(gray.shape, dtype=bool)
    if float(threshold) == 0.0:
        return magnitude > 0.0
    return magnitude >= float(threshold) * peak


def edge_pixel_count(edge_map):
    """Return the number of edge pixels in a boolean edge map."""
    arr = np.asarray(edge_map)
    if arr.dtype != bool:
        raise ValueError("edge_map must be a boolean array")
    return int(arr.sum())
