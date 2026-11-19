"""Blur skin-coloured regions of an image for privacy (numpy only).

Detects skin with the same YCbCr rule as ``skin_detector`` — Cb in
[77, 127] and Cr in [133, 173], the classic Chai & Ngan bounds — then
replaces the masked pixels with a box-blurred version of the image, so
faces and other skin regions become unrecognisable while everything
else stays pixel-identical. The blur is a padded mean filter computed
with an integral image, so it is fast even for large kernels.
"""

import numpy as np

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


def _validate_ranges(cb_range, cr_range):
    if len(cb_range) != 2 or cb_range[0] > cb_range[1]:
        raise ValueError(f"invalid cb_range {cb_range!r}")
    if len(cr_range) != 2 or cr_range[0] > cr_range[1]:
        raise ValueError(f"invalid cr_range {cr_range!r}")


def _validate_kernel_size(kernel_size):
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise ValueError(
            f"kernel_size must be an odd integer >= 1, got {kernel_size!r}"
        )
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be an odd integer >= 1, got {kernel_size!r}"
        )


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
    _validate_ranges(cb_range, cr_range)
    ycbcr = rgb_to_ycbcr(image)
    cb, cr = ycbcr[..., 1], ycbcr[..., 2]
    return (cb >= cb_range[0]) & (cb <= cb_range[1]) & \
        (cr >= cr_range[0]) & (cr <= cr_range[1])


def _box_blur(image, kernel_size):
    """Mean filter with a ``kernel_size`` x ``kernel_size`` window.

    Edges are handled by replicating the border pixels, and the sums
    come from an integral image so the cost does not grow with the
    kernel size.
    """
    pad = kernel_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    integral = np.zeros(
        (padded.shape[0] + 1, padded.shape[1] + 1, image.shape[2])
    )
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    height, width = image.shape[:2]
    k = kernel_size
    sums = (
        integral[k:k + height, k:k + width]
        - integral[:height, k:k + width]
        - integral[k:k + height, :width]
        + integral[:height, :width]
    )
    return sums / (k * k)


def blur_region(image, mask, kernel_size=5):
    """Return ``image`` with the pixels where ``mask`` is true blurred.

    Masked pixels are replaced by a ``kernel_size`` x ``kernel_size``
    box blur of the image; every unmasked pixel is returned exactly
    unchanged. ``kernel_size`` must be an odd integer >= 1.
    """
    _validate_kernel_size(kernel_size)
    rgb = _validate_rgb(image)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError(
            f"mask must have shape {rgb.shape[:2]}, got {mask.shape}"
        )
    if kernel_size == 1 or not mask.any():
        return np.asarray(image).copy()
    blurred = _box_blur(rgb, kernel_size)
    out = rgb.copy()
    out[mask] = blurred[mask]
    out = np.clip(out, 0, 255)
    if np.asarray(image).dtype == np.uint8:
        return np.rint(out).astype(np.uint8)
    return out


def privacy_blur(image, kernel_size=5, cb_range=CB_RANGE, cr_range=CR_RANGE):
    """Blur the skin regions of an RGB ``image`` for privacy.

    Returns ``(blurred_image, skin_mask)``: the image with every
    skin pixel replaced by its box-blurred value, and the boolean
    skin mask used.
    """
    mask = detect_skin(image, cb_range, cr_range)
    return blur_region(image, mask, kernel_size), mask


def blur_skin(image, kernel_size=5, cb_range=CB_RANGE, cr_range=CR_RANGE):
    """Return ``image`` with its skin regions blurred (mask discarded)."""
    blurred, _mask = privacy_blur(image, kernel_size, cb_range, cr_range)
    return blurred


def blur_stats(image, kernel_size=5, cb_range=CB_RANGE, cr_range=CR_RANGE):
    """Summarise the skin blur applied to an RGB ``image``.

    Returns a dict with:
    - ``skin_pixels``: number of pixels classified as skin.
    - ``skin_fraction``: fraction of pixels classified as skin.
    - ``blurred_pixels``: number of skin pixels whose value actually
      changed after blurring (uniform interiors stay the same, so
      this can be smaller than ``skin_pixels``).
    """
    blurred, mask = privacy_blur(image, kernel_size, cb_range, cr_range)
    rgb = np.asarray(image, dtype=np.float64)
    changed = mask & np.any(blurred.astype(np.float64) != rgb, axis=-1)
    return {
        "skin_pixels": int(mask.sum()),
        "skin_fraction": float(mask.sum() / mask.size),
        "blurred_pixels": int(changed.sum()),
    }
