"""Colour quantisation by k-means clustering from scratch (numpy only).

Reduces an RGB image to ``n_colors`` representative colours (a
palette). Each pixel is replaced by the nearest palette colour, which
compresses the image's colour information and is a handy first step
for palette-based art, GIF-style storage, or segmentation by colour.
"""

import numpy as np


def _validate_image(image):
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"image must be an RGB array of shape (H, W, 3), got {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    return arr


def _kmeans_plus_plus(points, n_clusters, rng):
    """Pick well-spread starting centres (k-means++)."""
    n = points.shape[0]
    centres = [points[rng.integers(n)]]
    closest_sq = ((points - centres[0]) ** 2).sum(axis=1)
    for _ in range(1, n_clusters):
        total = closest_sq.sum()
        if total <= 0:
            centres.append(points[rng.integers(n)])
            continue
        probs = closest_sq / total
        centres.append(points[rng.choice(n, p=probs)])
        dist_sq = ((points - centres[-1]) ** 2).sum(axis=1)
        closest_sq = np.minimum(closest_sq, dist_sq)
    return np.stack(centres)


def kmeans_quantize(image, n_colors=8, max_iter=100, tol=1e-4, seed=0,
                    n_init=5):
    """Quantise an RGB image to ``n_colors`` colours.

    Runs ``n_init`` k-means++ restarts and keeps the run with the
    lowest within-cluster error, so a bad random start cannot merge
    two real colours. Returns ``(quantized, palette)`` where
    ``quantized`` is a float64 array with the input shape whose pixels
    are palette colours, and ``palette`` is a float64 array of shape
    ``(n_colors, 3)``.
    """
    arr = _validate_image(image)
    if not isinstance(n_colors, (int, np.integer)) or n_colors < 1:
        raise ValueError(f"n_colors must be a positive integer, got {n_colors}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    height, width, _ = arr.shape
    pixels = arr.reshape(-1, 3)
    n_pixels = pixels.shape[0]
    if n_colors > n_pixels:
        raise ValueError(
            f"n_colors ({n_colors}) exceeds the number of pixels ({n_pixels})")
    rng = np.random.default_rng(seed)
    best_centres, best_labels, best_inertia = None, None, np.inf
    for _ in range(max(1, int(n_init))):
        centres = _kmeans_plus_plus(pixels, n_colors, rng)
        labels = np.zeros(n_pixels, dtype=int)
        for _ in range(max_iter):
            dists = ((pixels[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            new_labels = dists.argmin(axis=1)
            new_centres = centres.copy()
            for k in range(n_colors):
                members = pixels[new_labels == k]
                if len(members) > 0:
                    new_centres[k] = members.mean(axis=0)
            shift = np.abs(new_centres - centres).max()
            centres, labels = new_centres, new_labels
            if shift < tol:
                break
        inertia = float((((pixels - centres[labels]) ** 2).sum(axis=1)).sum())
        if inertia < best_inertia:
            best_centres, best_labels, best_inertia = centres, labels, inertia
    centres, labels = best_centres, best_labels
    quantized = centres[labels].reshape(height, width, 3)
    return quantized, centres


def quantization_mse(image, quantized):
    """Mean squared error between an image and its quantised version."""
    a = _validate_image(image)
    b = _validate_image(quantized)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(((a - b) ** 2).mean())
