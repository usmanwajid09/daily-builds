"""ORB-style features from scratch with numpy only.

A FAST corner detector (16-pixel Bresenham circle of radius 3, with a
contiguous-arc test) finds keypoints, and a BRIEF-like binary descriptor
is built at each keypoint by comparing intensities at a fixed set of
randomly chosen point pairs inside a patch around the keypoint.
Descriptors are matched with brute-force Hamming distance, with an
optional Lowe ratio test. The descriptor pairs are fixed by a seed so
results are reproducible.
"""
from __future__ import annotations

import numpy as np

# 16 pixels of a Bresenham circle of radius 3, clockwise from (0, +3).
CIRCLE_OFFSETS = (
    (0, 3), (1, 3), (2, 2), (3, 1),
    (3, 0), (3, -1), (2, -2), (1, -3),
    (0, -3), (-1, -3), (-2, -2), (-3, -1),
    (-3, 0), (-3, 1), (-2, 2), (-1, 3),
)

_RADIUS = 3


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


def _has_contiguous_arc(flags: np.ndarray, n_contiguous: int) -> bool:
    """True if any run of ``n_contiguous`` True values wraps the circle."""
    doubled = np.concatenate([flags, flags])
    run = 0
    for value in doubled:
        run = run + 1 if value else 0
        if run >= n_contiguous:
            return True
    return False


def is_corner(image, row, col, threshold=20.0, n_contiguous=12) -> bool:
    """True if pixel (row, col) is a FAST corner."""
    arr = _validate_image(image)
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if not 1 <= n_contiguous <= 16:
        raise ValueError(f"n_contiguous must be between 1 and 16, got {n_contiguous}")
    if not (0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]):
        raise ValueError(f"(row, col)=({row}, {col}) is outside the image")
    centre = arr[row, col]
    circle = np.array(
        [arr[row + dr, col + dc] for dr, dc in CIRCLE_OFFSETS]
        if _RADIUS <= row < arr.shape[0] - _RADIUS
        and _RADIUS <= col < arr.shape[1] - _RADIUS
        else [np.nan] * 16
    )
    if np.isnan(circle).any():
        return False
    bright = circle > centre + threshold
    dark = circle < centre - threshold
    return _has_contiguous_arc(bright, n_contiguous) or \
        _has_contiguous_arc(dark, n_contiguous)


def corner_score(image, row, col, threshold=20.0, n_contiguous=12) -> float:
    """Strength of a corner: max absolute circle-minus-centre difference."""
    arr = _validate_image(image)
    if not (_RADIUS <= row < arr.shape[0] - _RADIUS
            and _RADIUS <= col < arr.shape[1] - _RADIUS):
        return 0.0
    centre = arr[row, col]
    circle = np.array([arr[row + dr, col + dc] for dr, dc in CIRCLE_OFFSETS])
    diffs = np.abs(circle - centre)
    return float(np.max(diffs)) if is_corner(image, row, col, threshold,
                                              n_contiguous) else 0.0


def detect_keypoints(image, threshold=20.0, n_contiguous=12, max_keypoints=500):
    """Detect FAST corners, strongest first, with simple non-max suppression.

    Returns a list of (row, col) tuples, strongest corners first.
    """
    arr = _validate_image(image)
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if not 1 <= n_contiguous <= 16:
        raise ValueError(f"n_contiguous must be between 1 and 16, got {n_contiguous}")
    if max_keypoints < 1:
        raise ValueError(f"max_keypoints must be at least 1, got {max_keypoints}")
    found = []
    for r in range(_RADIUS, arr.shape[0] - _RADIUS):
        for c in range(_RADIUS, arr.shape[1] - _RADIUS):
            if is_corner(arr, r, c, threshold, n_contiguous):
                found.append((r, c, corner_score(arr, r, c, threshold,
                                                  n_contiguous)))
    found.sort(key=lambda t: -t[2])
    kept = []
    for r, c, _s in found:
        if all(abs(r - kr) > 3 or abs(c - kc) > 3 for kr, kc in kept):
            kept.append((r, c))
        if len(kept) >= max_keypoints:
            break
    return kept


def make_descriptor_pairs(n_bits=256, patch_radius=15, seed=0):
    """Fixed random point pairs for BRIEF descriptors, in a patch.

    Returns an (n_bits, 2, 2) int array: for each bit, two (dr, dc)
    offsets inside the patch, drawn from a Gaussian with std
    patch_radius / 5 (the usual BRIEF sampling).
    """
    if n_bits < 1:
        raise ValueError(f"n_bits must be at least 1, got {n_bits}")
    if patch_radius < 1:
        raise ValueError(f"patch_radius must be at least 1, got {patch_radius}")
    rng = np.random.default_rng(seed)
    pairs = rng.normal(0, patch_radius / 5.0, size=(n_bits, 2, 2))
    pairs = np.clip(np.round(pairs), -patch_radius, patch_radius).astype(int)
    return pairs


def describe(image, keypoints, n_bits=256, patch_radius=15, seed=0):
    """Compute a BRIEF-like binary descriptor for each keypoint.

    Returns a bool array of shape (n_keypoints, n_bits): bit i is True
    when the first sampled point is brighter than the second. Keypoints
    too close to the border for the patch are dropped; the returned
    keypoints list matches the descriptor rows.
    """
    arr = _validate_image(image)
    pairs = make_descriptor_pairs(n_bits, patch_radius, seed)
    kept_kp, rows = [], []
    for r, c in keypoints:
        if not (patch_radius <= r < arr.shape[0] - patch_radius
                and patch_radius <= c < arr.shape[1] - patch_radius):
            continue
        bits = []
        for (dr1, dc1), (dr2, dc2) in pairs:
            bits.append(arr[r + dr1, c + dc1] < arr[r + dr2, c + dc2])
        kept_kp.append((r, c))
        rows.append(bits)
    descriptors = np.array(rows, dtype=bool) if rows else np.zeros((0, n_bits), dtype=bool)
    return kept_kp, descriptors


def hamming_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between two sets of bool descriptors."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("descriptors must be 2D bool arrays")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"descriptor lengths differ: {a.shape[1]} vs {b.shape[1]}")
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError("cannot match empty descriptor sets")
    return (a[:, None, :] != b[None, :, :]).sum(axis=2)


def match_descriptors(desc1, desc2, ratio=0.75):
    """Brute-force matching with Lowe's ratio test.

    Returns a list of (i, j, distance) matches sorted by distance:
    each descriptor in ``desc1`` is matched to its nearest neighbour
    in ``desc2`` only if that neighbour is clearly closer than the
    second-nearest (nearest < ratio * second-nearest). ``ratio=None``
    disables the ratio test.
    """
    d = hamming_distance(desc1, desc2)
    matches = []
    for i in range(d.shape[0]):
        order = np.argsort(d[i])
        best, second = int(d[i, order[0]]), int(d[i, order[1]]) \
            if d.shape[1] > 1 else None
        if ratio is not None:
            if second is None or best >= ratio * second:
                continue
        matches.append((i, int(order[0]), best))
    matches.sort(key=lambda m: m[2])
    return matches


def extract_orb(image, threshold=20.0, n_contiguous=12, n_bits=256,
                patch_radius=15, max_keypoints=500, seed=0):
    """Detect keypoints and compute descriptors in one call.

    Returns (keypoints, descriptors) with matching rows.
    """
    keypoints = detect_keypoints(image, threshold, n_contiguous, max_keypoints)
    return describe(image, keypoints, n_bits, patch_radius, seed)
