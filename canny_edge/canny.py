"""Canny edge detection from scratch.

The classic Canny pipeline, built only on numpy:

1. Smooth the image with a Gaussian kernel (separable 1D passes) to
   suppress noise.
2. Convolve with the horizontal and vertical Sobel kernels to get the
   image gradients ``gx`` and ``gy`` and the gradient magnitude.
3. Thin the magnitude with non-maximum suppression along the gradient
   direction, so edges come out one pixel wide.
4. Double threshold: pixels above the high threshold are strong edges,
   pixels between the low and high thresholds are weak edges.
5. Hysteresis: keep a weak edge only if it connects (8-connected) to a
   strong edge, directly or through other weak edges.

All convolutions use replicate padding (border values repeated), so a
flat image stays flat at the border — zero padding would invent a fake
edge there, because the kernel would see a step from 0 to the border
value.

No OpenCV, no PIL, no scipy.
"""

import numpy as np

_SOBEL_X = np.array([[-1.0, 0.0, 1.0],
                     [-2.0, 0.0, 2.0],
                     [-1.0, 0.0, 1.0]])

_SOBEL_Y = np.array([[-1.0, -2.0, -1.0],
                     [0.0, 0.0, 0.0],
                     [1.0, 2.0, 1.0]])

_LUMINANCE = np.array([0.299, 0.587, 0.114])


def gaussian_kernel_1d(sigma, kernel_size=None):
    """Return a normalised 1D Gaussian kernel.

    Args:
        sigma: standard deviation of the Gaussian; must be > 0.
        kernel_size: odd positive integer, the kernel length. Defaults to
            ``2 * ceil(3 * sigma) + 1`` so the kernel covers +/-3 sigma.

    Returns:
        A 1D float64 array of length ``kernel_size`` that sums to 1.

    Raises:
        ValueError: if ``sigma`` is not a positive finite number or
            ``kernel_size`` is not an odd positive integer.
    """
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float, np.floating)):
        raise ValueError(f"sigma must be a positive number, got {sigma!r}")
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be a positive number, got {sigma}")
    if kernel_size is None:
        kernel_size = 2 * int(np.ceil(3 * sigma)) + 1
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, (int, np.integer)):
        raise ValueError(f"kernel_size must be an odd positive integer, got {kernel_size!r}")
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be an odd positive integer, got {kernel_size}")
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _to_grayscale(image):
    """Return ``image`` as a 2D float64 grayscale array or raise."""
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


def _validate_2d_float(name, arr):
    """Return ``arr`` as a 2D float64 array or raise ``ValueError``."""
    arr = np.asarray(arr)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _blur_2d(channel, kernel):
    """Blur one 2D channel with a separable kernel (replicate padding)."""
    radius = len(kernel) // 2
    # Convolve along rows, then along columns: separable means two 1D
    # passes instead of one 2D pass (2k ops per pixel instead of k^2).
    # Each pass pads only the axis it convolves along.
    padded = np.pad(channel, ((0, 0), (radius, radius)), mode="edge")
    tmp = np.zeros_like(channel, dtype=np.float64)
    for i, weight in enumerate(kernel):
        tmp += weight * padded[:, i:i + channel.shape[1]]
    padded = np.pad(tmp, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(channel, dtype=np.float64)
    for j, weight in enumerate(kernel):
        out += weight * padded[j:j + channel.shape[0], :]
    return out


def _convolve3x3(image, kernel):
    """Convolve a 2D array with a 3x3 kernel (replicate padding)."""
    padded = np.pad(image, 1, mode="edge")
    out = np.zeros_like(image, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * padded[i:i + image.shape[0], j:j + image.shape[1]]
    return out


def sobel_gradients(image):
    """Return ``(gx, gy, magnitude)`` Sobel gradients of ``image``.

    ``gx`` responds to vertical edges, ``gy`` to horizontal edges, and
    ``magnitude`` is the per-pixel edge strength (``hypot(gx, gy)``).
    The image is converted to grayscale first; see :func:`canny_edges`
    for the accepted shapes.
    """
    gray = _to_grayscale(image)
    gx = _convolve3x3(gray, _SOBEL_X)
    gy = _convolve3x3(gray, _SOBEL_Y)
    magnitude = np.hypot(gx, gy)
    return gx, gy, magnitude


def non_maximum_suppression(magnitude, gx, gy):
    """Thin ``magnitude`` to one-pixel-wide ridges along the gradient.

    Each pixel survives only if its magnitude is at least as large as the
    two neighbours lying along its gradient direction (quantised to 0, 45,
    90 or 135 degrees). All other pixels are set to 0. Neighbours outside
    the image are taken as the border pixel itself (replicate padding),
    so a flat magnitude stays flat at the border. The comparison is
    strict on one side (``>=`` on one neighbour, ``>`` on the other) so
    a plateau of equal magnitudes thins to a single pixel instead of
    surviving two pixels wide — a symmetric ramp would otherwise keep
    both of its equal peak pixels.

    Args:
        magnitude: 2D array of gradient magnitudes.
        gx, gy: 2D arrays of horizontal / vertical gradients, same shape
            as ``magnitude``.

    Returns:
        A float64 array of the same shape, with non-maxima zeroed out.
    """
    magnitude = _validate_2d_float("magnitude", magnitude)
    gx = _validate_2d_float("gx", gx)
    gy = _validate_2d_float("gy", gy)
    if gx.shape != magnitude.shape or gy.shape != magnitude.shape:
        raise ValueError(
            f"magnitude, gx and gy must have the same shape, got "
            f"{magnitude.shape}, {gx.shape}, {gy.shape}"
        )
    angle = np.degrees(np.arctan2(gy, gx)) % 180.0
    # Quantise the direction into 4 sectors: 0, 45, 90, 135 degrees.
    sector = ((angle + 22.5) // 45).astype(int) % 4
    padded = np.pad(magnitude, 1, mode="edge")
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
        # Strict on one side so equal-magnitude plateaus thin to one pixel.
        keep = mask & (centre >= a) & (centre > b)
        suppressed[keep] = centre[keep]
    return suppressed


def _validate_threshold(name, value):
    """Return ``value`` as a float in [0, 1] or raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def hysteresis(magnitude, low, high):
    """Apply double threshold + hysteresis to a magnitude array.

    Pixels at or above ``high`` are strong edges. Pixels at or above
    ``low`` are weak edges, kept only if they connect (8-connected) to a
    strong edge, directly or through other weak edges. Isolated weak
    pixels are dropped. A threshold of 0 means "any non-zero magnitude"
    rather than "every pixel", so a flat (all-zero) magnitude still
    yields no edges.

    Args:
        magnitude: 2D array of (non-maximum-suppressed) gradient
            magnitudes.
        low: low threshold, an absolute magnitude value in [0, 1] when
            the magnitude is normalised, or any non-negative number in
            general; must satisfy ``0 <= low <= high``.
        high: high threshold, same units as ``low``.

    Returns:
        A boolean array of the same shape: True where an edge survives.
    """
    magnitude = _validate_2d_float("magnitude", magnitude)
    if isinstance(low, bool) or not isinstance(low, (int, float, np.floating)):
        raise ValueError(f"low must be a number, got {low!r}")
    if isinstance(high, bool) or not isinstance(high, (int, float, np.floating)):
        raise ValueError(f"high must be a number, got {high!r}")
    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"low and high must be finite, got low={low}, high={high}")
    if low < 0 or high < 0:
        raise ValueError(
            f"low and high must be non-negative, got low={low}, high={high}"
        )
    if low > high:
        raise ValueError(f"low must not exceed high, got low={low}, high={high}")
    if high == 0.0:
        strong = magnitude > 0.0
    else:
        strong = magnitude >= high
    if low == 0.0:
        weak = magnitude > 0.0
    else:
        weak = magnitude >= low
    if not strong.any():
        return np.zeros(magnitude.shape, dtype=bool)
    # Grow the strong set into neighbouring weak pixels until it stops.
    # Neighbours outside the image do not exist (zero padding here is
    # about connectivity, not convolution, so it invents no edges).
    edges = strong.copy()
    while True:
        padded = np.pad(edges, 1, mode="constant", constant_values=False)
        h, w = edges.shape
        neighbour = np.zeros_like(edges)
        for dr in range(3):
            for dc in range(3):
                if dr == 1 and dc == 1:
                    continue
                neighbour |= padded[dr:dr + h, dc:dc + w]
        new = weak & neighbour & ~edges
        if not new.any():
            break
        edges |= new
    return edges


def canny_edges(image, sigma=1.0, low_ratio=0.1, high_ratio=0.3):
    """Return a boolean Canny edge map for ``image``.

    Args:
        image: 2D grayscale array, or (H, W, 3) RGB array (converted to
            grayscale with luminance weights).
        sigma: standard deviation of the Gaussian smoothing; must be > 0.
            Larger values suppress more noise but blur fine edges away.
        low_ratio: low threshold as a fraction of the strongest
            (post-suppression) gradient magnitude in the image.
        high_ratio: high threshold as a fraction of the strongest
            gradient magnitude. Must satisfy ``0 <= low_ratio <=
            high_ratio <= 1``.

    Returns:
        A boolean ``numpy`` array with the same (H, W) shape as the
        input. A flat image yields no edges at all, including at the
        borders.

    Raises:
        ValueError: on a bad image (non-numeric dtype, wrong shape,
            empty, or non-finite values), a bad ``sigma``, or thresholds
            outside ``0 <= low_ratio <= high_ratio <= 1``.
    """
    gray = _to_grayscale(image)
    kernel = gaussian_kernel_1d(sigma)  # validates sigma
    low_ratio = _validate_threshold("low_ratio", low_ratio)
    high_ratio = _validate_threshold("high_ratio", high_ratio)
    if low_ratio > high_ratio:
        raise ValueError(
            f"low_ratio must not exceed high_ratio, got low_ratio={low_ratio}, "
            f"high_ratio={high_ratio}"
        )
    blurred = _blur_2d(gray, kernel)
    gx, gy, magnitude = sobel_gradients(blurred)
    thinned = non_maximum_suppression(magnitude, gx, gy)
    peak = thinned.max()
    # Floating-point residue: Sobel on a (blurred) flat image leaves a
    # magnitude of ~1 ulp of the image scale, not exactly 0. Relative
    # thresholds would then call the whole flat image an edge, so treat
    # a peak that is negligible next to the image values as no edge at
    # all. A real edge weaker than this is far below one grey level.
    scale = max(1.0, float(np.abs(gray).max()))
    if peak <= 1e-9 * scale:
        return np.zeros(gray.shape, dtype=bool)
    return hysteresis(thinned, low_ratio * peak, high_ratio * peak)


def edge_pixel_count(edge_map):
    """Return the number of edge pixels in a boolean edge map."""
    arr = np.asarray(edge_map)
    if arr.dtype != bool:
        raise ValueError("edge_map must be a boolean array")
    return int(arr.sum())
