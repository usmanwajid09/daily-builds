"""Joint (cross) bilateral filter from scratch (numpy only).

Like a bilateral filter, but the range weights come from a *guidance*
image instead of the image being filtered. This lets a clean guidance
image steer the smoothing of a noisy image: pixels are only averaged
with neighbours that look similar in the guidance, so edges present
in the guidance are preserved even if the input image is too noisy
to show them. The classic use is upsampling / denoising a depth map
guided by a colour photo.
"""

import numpy as np


def _as_2d(image, name):
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D grayscale array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def gaussian_kernel_2d(diameter, sigma):
    """Return a normalised 2D Gaussian kernel (odd ``diameter``)."""
    if diameter % 2 == 0 or diameter < 1:
        raise ValueError(f"diameter must be a positive odd integer, got {diameter}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    radius = diameter // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel_1d = np.exp(-0.5 * (axis / sigma) ** 2)
    kernel = np.outer(kernel_1d, kernel_1d)
    return kernel / kernel.sum()


def joint_bilateral_filter(image, guidance, diameter=5, sigma_color=25.0,
                           sigma_space=2.0):
    """Smooth ``image`` guided by ``guidance`` (both 2D, same shape).

    Each output pixel is a weighted average of its neighbourhood, where
    the weight is a spatial Gaussian times a range Gaussian evaluated
    on the *guidance* values. Returns a float64 array of the same shape.
    """
    img = _as_2d(image, "image")
    guide = _as_2d(guidance, "guidance")
    if img.shape != guide.shape:
        raise ValueError(
            f"image and guidance must have the same shape, got {img.shape} "
            f"and {guide.shape}")
    if diameter % 2 == 0 or diameter < 1:
        raise ValueError(f"diameter must be a positive odd integer, got {diameter}")
    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("sigma_color and sigma_space must be positive")
    radius = diameter // 2
    spatial = gaussian_kernel_2d(diameter, sigma_space)
    padded_img = np.pad(img, radius, mode="edge")
    padded_guide = np.pad(guide, radius, mode="edge")
    out = np.zeros_like(img)
    height, width = img.shape
    for y in range(height):
        for x in range(width):
            patch = padded_img[y:y + diameter, x:x + diameter]
            guide_patch = padded_guide[y:y + diameter, x:x + diameter]
            diff = guide_patch - guide[y, x]
            range_w = np.exp(-0.5 * (diff / sigma_color) ** 2)
            weights = spatial * range_w
            out[y, x] = (weights * patch).sum() / weights.sum()
    return out
