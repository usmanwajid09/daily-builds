"""Laplacian-of-Gaussian (LoG) blob detection, from scratch (numpy only).

A blob is a bright, roughly circular region on a dark background (or the
reverse, with the sign flipped). The LoG filter responds most strongly
when its scale matches the blob's size, so scanning a range of scales
and keeping the local maxima of the scale-normalised response — across
space *and* scale — finds blobs and estimates their radii in one pass.

The classic reference is Lindeberg's scale-space blob detection; the
radius convention ``radius = sqrt(2) * sigma`` is the standard one:
a LoG filter of scale sigma responds maximally to a disc of that radius.
"""

import numpy as np


def _validate_sigma(sigma, name="sigma"):
    """Return ``sigma`` as a float or raise ValueError."""
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float, np.floating)):
        raise ValueError(f"{name} must be a positive number, got {sigma!r}")
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"{name} must be a positive number, got {sigma}")
    return sigma


def _validate_image(image):
    """Return ``image`` as a 2D float64 array or raise ValueError."""
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"image must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"image must be 2D grayscale, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    return arr


def log_kernel(sigma, kernel_size=None):
    """Return the 2D Laplacian-of-Gaussian kernel at scale ``sigma``.

    The kernel is

        LoG(x, y) = -1 / (pi * sigma**4)
                    * (1 - (x**2 + y**2) / (2 * sigma**2))
                    * exp(-(x**2 + y**2) / (2 * sigma**2))

    Args:
        sigma: standard deviation of the Gaussian; must be > 0.
        kernel_size: odd positive integer, the kernel side length.
            Defaults to ``2 * ceil(3 * sigma) + 1`` (covers +/-3 sigma).

    Returns:
        A 2D float64 array of shape ``(kernel_size, kernel_size)``. The
        kernel is shifted to sum to exactly zero (the continuous LoG
        integrates to zero), so a constant image gives a zero response.

    Raises:
        ValueError: if ``sigma`` is not a positive finite number or
            ``kernel_size`` is not an odd positive integer.
    """
    sigma = _validate_sigma(sigma)
    if kernel_size is None:
        kernel_size = 2 * int(np.ceil(3 * sigma)) + 1
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, (int, np.integer)):
        raise ValueError(
            f"kernel_size must be an odd positive integer, got {kernel_size!r}"
        )
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be an odd positive integer, got {kernel_size}"
        )
    radius = kernel_size // 2
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float64)
    r2 = x ** 2 + y ** 2
    kernel = (
        -1.0 / (np.pi * sigma ** 4)
        * (1.0 - r2 / (2.0 * sigma ** 2))
        * np.exp(-r2 / (2.0 * sigma ** 2))
    )
    # The continuous LoG integrates to zero; force the discrete kernel
    # to sum to exactly zero so flat regions give a zero response.
    kernel = kernel - kernel.mean()
    return kernel


def _convolve_2d(image, kernel):
    """Convolve a 2D image with a 2D kernel using replicate padding.

    Border pixels are replicated outward (``np.pad(..., mode="edge")``),
    so a constant image stays constant right up to the border instead of
    inventing fake structure the way zero padding would.
    """
    radius = kernel.shape[0] // 2
    padded = np.pad(image, radius, mode="edge")
    out = np.zeros_like(image, dtype=np.float64)
    for i, row in enumerate(kernel):
        for j, weight in enumerate(row):
            if weight != 0.0:
                out += weight * padded[i:i + image.shape[0], j:j + image.shape[1]]
    return out


def log_response(image, sigma):
    """Return the sigma-normalised LoG response of ``image`` at ``sigma``.

    The response is ``-sigma**2 * (LoG * image)``: the minus sign makes
    bright blobs on a dark background give a *positive* peak at the
    blob centre, and the ``sigma**2`` factor normalises the response
    across scales so peaks at different scales are comparable.

    Args:
        image: 2D grayscale array.
        sigma: LoG scale; must be > 0.

    Returns:
        A float64 array with the same shape as ``image``.

    Raises:
        ValueError: on a bad image (see :func:`_validate_image`) or a
            bad ``sigma`` (see :func:`_validate_sigma`).
    """
    arr = _validate_image(image)
    sigma = _validate_sigma(sigma)
    kernel = log_kernel(sigma)
    return -(sigma ** 2) * _convolve_2d(arr, kernel)


def _local_maxima_3d(stack):
    """Boolean mask of voxels that are maxima in their 3x3x3 neighbourhood.

    ``stack`` has shape ``(num_sigma, height, width)``. Neighbourhoods
    are compared with replicate padding, so a peak on the image border
    is still a peak.
    """
    num_sigma, height, width = stack.shape
    padded = np.pad(stack, 1, mode="edge")
    maxima = np.full_like(stack, -np.inf)
    for ds in range(3):
        for dr in range(3):
            for dc in range(3):
                maxima = np.maximum(
                    maxima,
                    padded[ds:ds + num_sigma, dr:dr + height, dc:dc + width],
                )
    return stack >= maxima


def _circle_overlap_area(r1, r2, distance):
    """Area of the intersection of two circles, given their radii and
    centre distance."""
    if distance >= r1 + r2:
        return 0.0
    if distance <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    d2 = distance ** 2
    area = (
        r1 ** 2 * np.arccos((d2 + r1 ** 2 - r2 ** 2) / (2 * distance * r1))
        + r2 ** 2 * np.arccos((d2 + r2 ** 2 - r1 ** 2) / (2 * distance * r2))
        - 0.5
        * np.sqrt(
            (-distance + r1 + r2)
            * (distance + r1 - r2)
            * (distance - r1 + r2)
            * (distance + r1 + r2)
        )
    )
    return float(area)


def _prune_overlapping_blobs(blobs, overlap):
    """Drop blobs that overlap an already-kept, stronger blob.

    ``blobs`` must be sorted by descending response. A blob is dropped
    when the intersection of its circle with a kept blob's circle covers
    more than ``overlap`` of the smaller circle's area.
    """
    kept = []
    for blob in blobs:
        row, col, radius, _response = blob
        overlaps = False
        for kept_row, kept_col, kept_radius, _kept_response in kept:
            distance = np.hypot(row - kept_row, col - kept_col)
            area = _circle_overlap_area(radius, kept_radius, distance)
            smaller = np.pi * min(radius, kept_radius) ** 2
            if area / smaller > overlap:
                overlaps = True
                break
        if not overlaps:
            kept.append(blob)
    return kept


def detect_blobs(image, min_sigma=1.0, max_sigma=10.0, num_sigma=5, threshold=0.1,
                 overlap=0.5):
    """Detect bright blobs in a grayscale image with LoG scale-space.

    Builds the sigma-normalised LoG response at ``num_sigma`` scales
    between ``min_sigma`` and ``max_sigma``, then keeps every pixel
    that is a local maximum across space and scale and whose response
    is at least ``threshold``.

    Args:
        image: 2D grayscale array (bright blobs on a dark background).
        min_sigma: smallest LoG scale; must be > 0.
        max_sigma: largest LoG scale; must be >= ``min_sigma``.
        num_sigma: number of scales sampled between ``min_sigma`` and
            ``max_sigma``; must be an integer >= 1.
        threshold: minimum response for a blob to be reported; must be
            >= 0. It is an absolute value on the normalised response,
            so scale it to the image amplitude (for 0-255 images the
            responses are ~100x larger than for 0-1 images).
        overlap: fraction in (0, 1] of the smaller circle's area; a
            blob is dropped when its circle overlaps a stronger blob's
            circle by more than this fraction. ``1.0`` disables the
            pruning. Defaults to 0.5.

    Returns:
        A list of ``(row, col, radius, response)`` tuples, sorted by
        descending response. ``row`` and ``col`` are ints (the blob
        centre), ``radius`` is ``sqrt(2) * sigma`` at the best-matching
        scale, and ``response`` is the normalised LoG response there.

    Raises:
        ValueError: on a bad image or bad ``min_sigma`` / ``max_sigma``
            / ``num_sigma`` / ``threshold`` / ``overlap``.
    """
    arr = _validate_image(image)
    min_sigma = _validate_sigma(min_sigma, "min_sigma")
    max_sigma = _validate_sigma(max_sigma, "max_sigma")
    if max_sigma < min_sigma:
        raise ValueError(
            f"max_sigma ({max_sigma}) must be >= min_sigma ({min_sigma})"
        )
    if isinstance(num_sigma, bool) or not isinstance(num_sigma, (int, np.integer)):
        raise ValueError(f"num_sigma must be an integer >= 1, got {num_sigma!r}")
    num_sigma = int(num_sigma)
    if num_sigma < 1:
        raise ValueError(f"num_sigma must be an integer >= 1, got {num_sigma}")
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float, np.floating)
    ):
        raise ValueError(f"threshold must be a number >= 0, got {threshold!r}")
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be a number >= 0, got {threshold}")
    if isinstance(overlap, bool) or not isinstance(
        overlap, (int, float, np.floating)
    ):
        raise ValueError(f"overlap must be a number in (0, 1], got {overlap!r}")
    overlap = float(overlap)
    if not np.isfinite(overlap) or overlap <= 0 or overlap > 1:
        raise ValueError(f"overlap must be a number in (0, 1], got {overlap}")

    sigmas = np.linspace(min_sigma, max_sigma, num_sigma)
    stack = np.stack([log_response(arr, sigma) for sigma in sigmas])
    is_max = _local_maxima_3d(stack)

    blobs = []
    for s, sigma in enumerate(sigmas):
        rows, cols = np.nonzero(is_max[s] & (stack[s] >= threshold))
        for row, col in zip(rows, cols):
            blobs.append(
                (
                    int(row),
                    int(col),
                    float(np.sqrt(2.0) * sigma),
                    float(stack[s, row, col]),
                )
            )
    blobs.sort(key=lambda blob: blob[3], reverse=True)
    return _prune_overlapping_blobs(blobs, overlap)
