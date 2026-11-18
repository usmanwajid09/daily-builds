"""Non-local means image denoising, from scratch with numpy only.

Every output pixel is a weighted average of pixels in a search window
around it, where the weight of a neighbour depends on how similar the
small patch around it is to the patch around the target pixel (not on
how close it is in space). That lets the filter average over many
similar-looking pixels anywhere in the window — including repeated
texture — while barely averaging across an edge. Works on grayscale
and RGB images.
"""
from __future__ import annotations

import numpy as np


def _validate(image, search_radius, patch_radius, h):
    try:
        arr = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("image must be a numeric 2D or 3D array") from None
    if arr.ndim not in (2, 3):
        raise ValueError(f"image must be 2D (grayscale) or 3D (RGB), got {arr.ndim}D")
    if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"colour image must have 1, 3 or 4 channels, got {arr.shape[2]}")
    if arr.size == 0:
        raise ValueError("image must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("image must contain only finite values")
    if not isinstance(search_radius, int) or isinstance(search_radius, bool) \
            or search_radius < 1:
        raise ValueError(f"search_radius must be an integer >= 1, got {search_radius}")
    if not isinstance(patch_radius, int) or isinstance(patch_radius, bool) \
            or patch_radius < 0:
        raise ValueError(f"patch_radius must be an integer >= 0, got {patch_radius}")
    if h <= 0:
        raise ValueError(f"h must be > 0, got {h}")
    if arr.shape[0] <= 2 * patch_radius or arr.shape[1] <= 2 * patch_radius:
        raise ValueError("image is too small for the chosen patch_radius")
    return arr


def _patch(image, r, c, patch_radius):
    """Padded patch centred at (r, c); image must already be padded."""
    p = patch_radius
    return image[r:r + 2 * p + 1, c:c + 2 * p + 1]


def non_local_means(image, search_radius=5, patch_radius=1, h=10.0):
    """Denoise an image with non-local means.

    Parameters
    ----------
    image : 2D grayscale or 3D (H, W, C) array.
    search_radius : int
        Neighbours are searched in a (2*search_radius+1)^2 window.
    patch_radius : int
        Half-size of the comparison patch (patch is 2*patch_radius+1
        on a side; 0 compares single pixels).
    h : float
        Filtering strength: larger h averages more aggressively.
        A good starting point is around the noise standard deviation.

    Returns a float64 array of the same shape as the input.
    """
    arr = _validate(image, search_radius, patch_radius, h)
    gray = arr.ndim == 2
    img = arr[:, :, None] if gray else arr
    height, width, channels = img.shape
    pad = search_radius + patch_radius
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(img)
    patch_area = (2 * patch_radius + 1) ** 2 * channels
    for r in range(height):
        for c in range(width):
            target = _patch(padded, r + search_radius, c + search_radius,
                            patch_radius)
            total_weight = 0.0
            acc = np.zeros(channels)
            for dr in range(-search_radius, search_radius + 1):
                for dc in range(-search_radius, search_radius + 1):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    neighbour = _patch(padded, nr + search_radius,
                                       nc + search_radius, patch_radius)
                    dist2 = float(np.sum((target - neighbour) ** 2)) / patch_area
                    w = np.exp(-dist2 / (h * h))
                    acc += w * img[nr, nc]
                    total_weight += w
            out[r, c] = acc / total_weight
    return out[:, :, 0] if gray else out


def psnr(reference, denoised, peak=255.0) -> float:
    """Peak signal-to-noise ratio in dB between two same-shaped images."""
    ref = np.asarray(reference, dtype=np.float64)
    den = np.asarray(denoised, dtype=np.float64)
    if ref.shape != den.shape:
        raise ValueError(
            f"shapes differ: reference {ref.shape} vs denoised {den.shape}")
    if ref.size == 0:
        raise ValueError("images must be non-empty")
    mse = float(np.mean((ref - den) ** 2))
    if mse == 0:
        return float("inf")
    if peak <= 0:
        raise ValueError(f"peak must be > 0, got {peak}")
    return float(10.0 * np.log10(peak ** 2 / mse))
