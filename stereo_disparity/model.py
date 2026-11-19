"""Stereo disparity by block matching, from scratch with numpy only.

Given a rectified stereo pair (left and right images of the same scene
taken from cameras displaced horizontally), each pixel's disparity is
the horizontal shift that best aligns a small window around it in the
left image with a window in the right image, measured by the sum of
absolute differences (SAD). Larger disparity means the object is
closer to the cameras, so the disparity map orders scene depth: near
objects have larger disparities than far ones.
"""
from __future__ import annotations

import numpy as np


def _validate_image(image) -> np.ndarray:
    try:
        arr = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("image must be a numeric 2D array") from None
    if arr.ndim != 2:
        raise ValueError(f"image must be a 2D grayscale array, got {arr.ndim}D array")
    if arr.size == 0:
        raise ValueError("image must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("image must contain only finite values")
    return arr


def make_synthetic_pair(height=60, width=80, seed=0):
    """Build a synthetic stereo pair with objects at known disparities.

    Left image: a near square (disparity 8), a far square (disparity 3),
    and a textured background. The right image is the left image shifted
    left by each object's disparity (objects move, background stays).
    A little noise is added to both.

    Returns (left, right, ground_truth) where ground_truth is the
    disparity map (0 for background, 8 for the near square, 3 for the
    far square).
    """
    rng = np.random.default_rng(seed)
    left = rng.normal(128.0, 12.0, size=(height, width))
    truth = np.zeros((height, width), dtype=np.int32)
    # Far square: rows 10-29, cols 10-29, disparity 3. Textured (not
    # flat) so block matching can pin down its disparity uniquely.
    left[10:30, 10:30] = 200.0 + rng.normal(0, 5.0, size=(20, 20))
    truth[10:30, 10:30] = 3
    # Near square: rows 35-54, cols 45-64, disparity 8.
    left[35:55, 45:65] = 60.0 + rng.normal(0, 5.0, size=(20, 20))
    truth[35:55, 45:65] = 8
    right = np.zeros_like(left)
    # Background (disparity 0) copies straight across.
    right[:] = left
    # Shift each object left by its disparity in the right image.
    right[10:30, 10 - 3:30 - 3] = left[10:30, 10:30]
    right[35:55, 45 - 8:65 - 8] = left[35:55, 45:65]
    # Where the shifted objects vacated, restore background noise.
    right[10:30, 27:30] = rng.normal(128.0, 12.0, size=(20, 3))
    right[35:55, 57:65] = rng.normal(128.0, 12.0, size=(20, 8))
    left = left + rng.normal(0, 2.0, size=left.shape)
    right = right + rng.normal(0, 2.0, size=right.shape)
    return left, right, truth


def block_match(left, right, max_disparity=16, window_radius=2):
    """Compute a disparity map by SAD block matching.

    For each pixel in ``left``, windows centred on it are compared
    against windows in ``right`` shifted left by d = 0..max_disparity;
    the d with the smallest sum of absolute differences wins. Pixels
    too close to the border for the window are set to -1 (unknown).

    Returns an int32 array of the same shape: disparity per pixel,
    or -1 where it could not be computed.
    """
    left = _validate_image(left)
    right = _validate_image(right)
    if left.shape != right.shape:
        raise ValueError(
            f"images must have the same shape, got {left.shape} vs {right.shape}")
    if not isinstance(max_disparity, int) or isinstance(max_disparity, bool) \
            or max_disparity < 0:
        raise ValueError(f"max_disparity must be an integer >= 0, got {max_disparity}")
    if not isinstance(window_radius, int) or isinstance(window_radius, bool) \
            or window_radius < 0:
        raise ValueError(f"window_radius must be an integer >= 0, got {window_radius}")
    height, width = left.shape
    if width <= max_disparity + 2 * window_radius:
        raise ValueError("image is too narrow for max_disparity and window_radius")
    disparity = np.full((height, width), -1, dtype=np.int32)
    w = window_radius
    for r in range(w, height - w):
        for c in range(w + max_disparity, width - w):
            target = left[r - w:r + w + 1, c - w:c + w + 1]
            best_d, best_sad = 0, np.inf
            for d in range(max_disparity + 1):
                candidate = right[r - w:r + w + 1, c - d - w:c - d + w + 1]
                sad = float(np.sum(np.abs(target - candidate)))
                if sad < best_sad:
                    best_sad, best_d = sad, d
            disparity[r, c] = best_d
    return disparity


def disparity_accuracy(disparity, ground_truth):
    """Fraction of valid pixels whose disparity equals ground truth."""
    disparity = np.asarray(disparity)
    ground_truth = np.asarray(ground_truth)
    if disparity.shape != ground_truth.shape:
        raise ValueError("disparity and ground truth must have the same shape")
    valid = disparity >= 0
    if not valid.any():
        raise ValueError("no valid disparities to score")
    return float(np.mean(disparity[valid] == ground_truth[valid]))


def depth_order_ok(disparity, near_region, far_region):
    """Check that the near region has larger median disparity than far.

    ``near_region`` and ``far_region`` are (row_slice, col_slice)
    tuples. Returns True when median(near) > median(far) and both
    regions contain valid disparities.
    """
    disparity = np.asarray(disparity)
    near = disparity[near_region]
    far = disparity[far_region]
    near_valid = near[near >= 0]
    far_valid = far[far >= 0]
    if near_valid.size == 0 or far_valid.size == 0:
        raise ValueError("regions contain no valid disparities")
    return float(np.median(near_valid)) > float(np.median(far_valid))
