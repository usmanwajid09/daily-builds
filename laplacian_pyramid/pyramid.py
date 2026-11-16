"""Gaussian and Laplacian image pyramids, from scratch (numpy only).

A Gaussian pyramid is a stack of progressively blurred, downsampled
copies of an image; a Laplacian pyramid stores the detail lost between
successive Gaussian levels, so the original can be rebuilt exactly by
expanding and adding the levels back together. Laplacian pyramids are
the classic tool for seamless multi-scale image blending.
"""

import numpy as np

__all__ = [
    "gaussian_kernel_1d",
    "gaussian_kernel_2d",
    "reduce",
    "expand",
    "build_gaussian_pyramid",
    "build_laplacian_pyramid",
    "reconstruct",
    "blend_images",
]

# 5-tap binomial kernel [1, 4, 6, 4, 1] / 16 (Burt & Adelson, 1983).
_KERNEL_1D = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def gaussian_kernel_1d():
    """Return the 5-tap binomial smoothing kernel ``[1, 4, 6, 4, 1] / 16``."""
    return _KERNEL_1D.copy()


def gaussian_kernel_2d():
    """Return the 5x5 Gaussian kernel (outer product of the 1D kernel)."""
    return np.outer(_KERNEL_1D, _KERNEL_1D)


def _validate_image(image):
    """Check ``image`` is a non-empty 2D (grayscale) or 3D (RGB) array."""
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"image must be a 2D (grayscale) or 3D (RGB) array, got {arr.ndim}D"
        )
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("image must be non-empty (height and width > 0)")
    if arr.ndim == 3 and arr.shape[2] == 0:
        raise ValueError("image must be non-empty (channels > 0)")
    return arr


def _validate_levels(levels, height, width):
    """Check ``levels`` is a positive int the image size can support."""
    if isinstance(levels, bool) or not isinstance(levels, (int, np.integer)):
        raise ValueError(f"levels must be an integer >= 1, got {levels!r}")
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    # Each reduce halves the size, so after levels - 1 reductions the
    # smallest side must still be at least 1 pixel.
    max_levels = int(np.floor(np.log2(min(height, width)))) + 1
    if levels > max_levels:
        raise ValueError(
            f"levels={levels} is too many for a {height}x{width} image "
            f"(maximum is {max_levels})"
        )


def _blur_2d(channel, scale=1.0):
    """Blur one 2D channel with the separable 5-tap binomial kernel.

    ``scale`` multiplies the kernel (2.0 per pass in ``expand`` so the
    two passes together give the x4 energy correction).
    """
    kernel = _KERNEL_1D * scale
    padded = np.pad(channel, 2, mode="reflect")
    # Convolve along rows, then columns (separable 5x5 kernel).
    rows = (
        kernel[0] * padded[:, :-4]
        + kernel[1] * padded[:, 1:-3]
        + kernel[2] * padded[:, 2:-2]
        + kernel[3] * padded[:, 3:-1]
        + kernel[4] * padded[:, 4:]
    )
    # `rows` still carries the 2 reflect-padded rows top and bottom, so
    # the column pass consumes them directly (no second padding).
    blurred = (
        kernel[0] * rows[:-4, :]
        + kernel[1] * rows[1:-3, :]
        + kernel[2] * rows[2:-2, :]
        + kernel[3] * rows[3:-1, :]
        + kernel[4] * rows[4:, :]
    )
    return blurred


def _blur(image, scale=1.0):
    """Blur a 2D or 3D (H, W, C) float image with the 5x5 Gaussian kernel."""
    if image.ndim == 2:
        return _blur_2d(image, scale=scale)
    return np.stack(
        [_blur_2d(image[:, :, c], scale=scale) for c in range(image.shape[2])],
        axis=2,
    )


def reduce(image):
    """Blur ``image`` and downsample it by 2 (one pyramid step down).

    Returns a float64 array roughly half the height and width.
    """
    arr = _validate_image(image).astype(np.float64)
    return _blur(arr)[::2, ::2]


def expand(image, target_shape=None):
    """Upsample ``image`` by 2 (one pyramid step up).

    Zero-insertion upsampling halves the brightness, so the blur kernel
    is scaled by 4 (x2 per separable pass) to preserve energy. If
    ``target_shape`` is given, the result is cropped/padded to match it
    (needed when the level below had an odd size).
    """
    arr = _validate_image(image).astype(np.float64)
    height, width = arr.shape[0], arr.shape[1]
    if arr.ndim == 2:
        up = np.zeros((2 * height, 2 * width), dtype=np.float64)
    else:
        up = np.zeros((2 * height, 2 * width, arr.shape[2]), dtype=np.float64)
    up[::2, ::2] = arr
    # Blur with the kernel scaled by 2 per separable pass (x4 total),
    # compensating for the zero-insertion halving the brightness.
    out = _blur(up, scale=2.0)
    if target_shape is not None:
        t_h, t_w = target_shape[0], target_shape[1]
        out = out[:t_h, :t_w]
        if out.shape[0] < t_h or out.shape[1] < t_w:
            pad_h = max(t_h - out.shape[0], 0)
            pad_w = max(t_w - out.shape[1], 0)
            pad = ((0, pad_h), (0, pad_w)) + ((0, 0),) * (out.ndim - 2)
            out = np.pad(out, pad, mode="edge")
    return out


def build_gaussian_pyramid(image, levels):
    """Build a Gaussian pyramid with ``levels`` levels.

    Level 0 is the image itself (as float64); each later level is
    ``reduce`` applied to the previous one, so shapes halve each step.
    """
    arr = _validate_image(image).astype(np.float64)
    _validate_levels(levels, arr.shape[0], arr.shape[1])
    pyramid = [arr]
    for _ in range(levels - 1):
        pyramid.append(reduce(pyramid[-1]))
    return pyramid


def build_laplacian_pyramid(image, levels):
    """Build a Laplacian pyramid with ``levels`` levels.

    Each level is a Gaussian level minus the expansion of the next
    (smaller) Gaussian level; the last level is the smallest Gaussian
    level itself, so the pyramid holds all the information needed to
    rebuild the image exactly.
    """
    gaussian = build_gaussian_pyramid(image, levels)
    pyramid = []
    for i in range(levels - 1):
        expanded = expand(gaussian[i + 1], target_shape=gaussian[i].shape[:2])
        pyramid.append(gaussian[i] - expanded)
    pyramid.append(gaussian[-1])
    return pyramid


def reconstruct(laplacian_pyramid):
    """Rebuild the original image from a Laplacian pyramid.

    Starts from the smallest level and repeatedly expands and adds the
    next detail level. Returns a float64 array.
    """
    if not laplacian_pyramid:
        raise ValueError("laplacian_pyramid must contain at least one level")
    levels = [_validate_image(level).astype(np.float64) for level in laplacian_pyramid]
    current = levels[-1]
    for i in range(len(levels) - 2, -1, -1):
        expanded = expand(current, target_shape=levels[i].shape[:2])
        current = expanded + levels[i]
    return current


def blend_images(image_a, image_b, mask, levels):
    """Blend ``image_a`` and ``image_b`` with a soft ``mask``.

    ``mask`` is a 2D float array the same height/width as the images,
    with 1 meaning "take image A" and 0 meaning "take image B". Each
    image is split into a Laplacian pyramid and the mask into a Gaussian
    pyramid; the pyramids are combined level by level and the result is
    reconstructed, so the seam blends smoothly at every scale. The
    result is clipped to the value range spanned by the two inputs.
    """
    a = _validate_image(image_a).astype(np.float64)
    b = _validate_image(image_b).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"image_a and image_b must have the same shape, "
            f"got {a.shape} and {b.shape}"
        )
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be a 2D array, got {m.ndim}D")
    if m.shape != a.shape[:2]:
        raise ValueError(
            f"mask shape {m.shape} must match image size {a.shape[:2]}"
        )
    if m.size == 0:
        raise ValueError("mask must be non-empty")
    _validate_levels(levels, a.shape[0], a.shape[1])
    mask = np.clip(m.astype(np.float64), 0.0, 1.0)

    lap_a = build_laplacian_pyramid(a, levels)
    lap_b = build_laplacian_pyramid(b, levels)
    gauss_mask = build_gaussian_pyramid(mask, levels)

    blended_levels = []
    for la, lb, gm in zip(lap_a, lap_b, gauss_mask):
        if la.ndim == 3:
            gm = gm[:, :, np.newaxis]
        blended_levels.append(la * gm + lb * (1.0 - gm))
    blended = reconstruct(blended_levels)
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    return np.clip(blended, lo, hi)
