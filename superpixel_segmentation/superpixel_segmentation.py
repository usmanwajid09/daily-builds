"""SLIC-style superpixel segmentation from scratch (NumPy only).

A superpixel is a small group of neighbouring pixels that share a similar
colour. SLIC (Achanta et al., 2012) finds them with a localised k-means:
cluster centres are seeded on a regular grid and each pixel is only compared
against the centres whose 2S x 2S neighbourhood covers it, using a distance
that mixes colour difference and spatial distance, weighted by a
``compactness`` parameter. A final connectivity pass guarantees every label
is a single connected region.

Public API
----------
slic(image, n_segments=100, compactness=10.0, max_iter=10, seed=0)
    -> (labels, centers)
enforce_connectivity(labels, min_size=None) -> labels
"""

import numpy as np
from collections import deque


def _validate_image(image):
    """Return the image as a float array in [0, 1] with shape (H, W, C)."""
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array, got "
                         f"{type(image).__name__}")
    if image.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2D (grayscale) or 3D (H, W, C), got {image.ndim}D "
            f"with shape {image.shape}")
    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise ValueError(
            f"3D image must have 1 or 3 channels, got {image.shape[2]} "
            f"channels (shape {image.shape})")
    if image.size == 0:
        raise ValueError("image must contain at least one pixel")

    if image.dtype == np.uint8:
        img = image.astype(np.float64) / 255.0
    elif np.issubdtype(image.dtype, np.floating):
        img = image.astype(np.float64)
        if img.min() < 0.0 or img.max() > 1.0:
            raise ValueError(
                f"float images must lie in [0, 1], got range "
                f"[{img.min():.4g}, {img.max():.4g}]")
    else:
        raise ValueError(
            f"image dtype must be uint8 or a float type, got {image.dtype}")

    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img


def _validate_params(n_segments, compactness, max_iter, n_pixels):
    if isinstance(n_segments, bool) or not isinstance(n_segments, (int, np.integer)):
        raise ValueError(f"n_segments must be an integer, got {n_segments!r}")
    if n_segments < 2:
        raise ValueError(f"n_segments must be >= 2, got {n_segments}")
    if n_segments > n_pixels:
        raise ValueError(
            f"n_segments ({n_segments}) must be <= number of pixels "
            f"({n_pixels})")
    if not isinstance(compactness, (int, float, np.floating)) or isinstance(compactness, bool):
        raise ValueError(f"compactness must be a number, got {compactness!r}")
    if compactness <= 0:
        raise ValueError(f"compactness must be > 0, got {compactness}")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
        raise ValueError(f"max_iter must be an integer, got {max_iter!r}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")


def _gradient_magnitude(img):
    """Simple gradient magnitude (summed over channels) for centre seeding."""
    gx = np.zeros(img.shape[:2])
    gy = np.zeros(img.shape[:2])
    gx[:, 1:-1] = np.abs(img[:, 2:, :] - img[:, :-2, :]).sum(axis=2)
    gy[1:-1, :] = np.abs(img[2:, :, :] - img[:-2, :, :]).sum(axis=2)
    return gx + gy


def _initialise_centers(img, n_segments, rng):
    """Seed centres on a grid, nudged to the lowest-gradient nearby pixel.

    The grid shape (rows x cols) is chosen so rows * cols is as close as
    possible to ``n_segments``. Returns centers as (K, 2 + C):
    [y, x, c0, c1, ...].
    """
    height, width, n_channels = img.shape
    best = None
    for rows in range(1, n_segments + 1):
        cols = max(1, int(round(n_segments / rows)))
        product = rows * cols
        # Prefer a product close to n_segments and a cell aspect ratio
        # close to the image aspect ratio.
        aspect_penalty = abs((height / rows) / max(width / cols, 1e-9) - 1.0)
        score = (abs(product - n_segments), aspect_penalty)
        if best is None or score < best[0]:
            best = (score, rows, cols)
    _, rows, cols = best
    step_y = height / rows
    step_x = width / cols
    grad = _gradient_magnitude(img)

    centers = []
    for r in range(rows):
        for c in range(cols):
            y = int((r + 0.5) * step_y)
            x = int((c + 0.5) * step_x)
            y = min(max(y, 0), height - 1)
            x = min(max(x, 0), width - 1)
            # Nudge to the lowest-gradient pixel in the 3x3 neighbourhood.
            best_y, best_x, best_g = y, x, grad[y, x]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and grad[ny, nx] < best_g:
                        best_y, best_x, best_g = ny, nx, grad[ny, nx]
            centers.append([best_y, best_x, *img[best_y, best_x]])
    return np.array(centers, dtype=np.float64)


def enforce_connectivity(labels, min_size=None):
    """Relabel so every label is exactly one connected region.

    Small disconnected fragments (smaller than ``min_size``, defaulting to
    a quarter of the mean segment size) are merged into the most common
    neighbouring label; larger fragments keep their label but are split off
    as new labels. 4-connectivity is used.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape {labels.shape}")
    height, width = labels.shape
    if min_size is None:
        min_size = max(1, labels.size // max(1, len(np.unique(labels))) // 4)

    out = np.full((height, width), -1, dtype=np.int64)
    visited = np.zeros((height, width), dtype=bool)
    next_label = 0

    # Collect every connected component first (4-connectivity).
    components = []
    for sy in range(height):
        for sx in range(width):
            if visited[sy, sx]:
                continue
            original = labels[sy, sx]
            component = []
            queue = deque([(sy, sx)])
            visited[sy, sx] = True
            while queue:
                y, x = queue.popleft()
                component.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= ny < height and 0 <= nx < width
                            and not visited[ny, nx] and labels[ny, nx] == original):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            components.append(component)

    def _neighbour_label(component):
        """Most common already-assigned label adjacent to a component."""
        neighbour_counts = {}
        for y, x in component:
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and out[ny, nx] != -1:
                    neighbour_counts[out[ny, nx]] = neighbour_counts.get(out[ny, nx], 0) + 1
        if neighbour_counts:
            return max(neighbour_counts, key=neighbour_counts.get)
        return None

    # Large components first, so small fragments always have an assigned
    # neighbour to merge into (a fragment scanned before its neighbours
    # would otherwise be orphaned).
    for component in sorted(components, key=len, reverse=True):
        if len(component) >= min_size:
            new_label = next_label
            next_label += 1
        else:
            new_label = _neighbour_label(component)
            if new_label is None:
                new_label = next_label
                next_label += 1
        for y, x in component:
            out[y, x] = new_label

    # Renumber to a dense 0..K-1 range.
    _, out = np.unique(out, return_inverse=True)
    return out.reshape(height, width)


def slic(image, n_segments=100, compactness=10.0, max_iter=10, seed=0):
    """Segment an image into superpixels (SLIC, from scratch).

    Parameters
    ----------
    image : (H, W) or (H, W, 3) array
        Grayscale or RGB image, float in [0, 1] or uint8 in [0, 255].
    n_segments : int
        Target number of superpixels (>= 2, <= number of pixels).
    compactness : float
        Weight of spatial distance vs colour distance (> 0). Larger values
        give more compact, grid-like superpixels; smaller values follow
        colour boundaries more closely.
    max_iter : int
        Maximum k-means iterations (>= 1).
    seed : int
        Kept for API determinism; the algorithm itself is deterministic
        (grid seeding), so the seed has no effect on the result.

    Returns
    -------
    labels : (H, W) int array
        Superpixel id of every pixel, 0..K-1, each label one connected region.
    centers : (K, 2 + C) float array
        Final cluster centres as [y, x, channel values...].
    """
    img = _validate_image(image)
    height, width, n_channels = img.shape
    _validate_params(n_segments, compactness, max_iter, height * width)

    rng = np.random.default_rng(seed)  # reserved; seeding is deterministic
    centers = _initialise_centers(img, n_segments, rng)
    n_centers = len(centers)
    interval = np.sqrt(height * width / n_centers)  # grid spacing S
    m_over_s = compactness / max(interval, 1e-6)
    # Colour channels live in [0, 1], so a raw colour distance (max ~1.7)
    # would be swamped by the spatial term. Scale colours to a 0-100 range
    # (as in Lab space, where classic SLIC uses compactness ~10) so the
    # two terms are comparable.
    color_scale = 100.0

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    labels = np.zeros((height, width), dtype=np.int64)

    for _ in range(max_iter):
        # Distance to the nearest centre, searched only in a 2S window.
        best_dist = np.full((height, width), np.inf)
        for k in range(n_centers):
            cy, cx = centers[k, 0], centers[k, 1]
            y0 = max(0, int(cy - 2 * interval))
            y1 = min(height, int(cy + 2 * interval) + 1)
            x0 = max(0, int(cx - 2 * interval))
            x1 = min(width, int(cx + 2 * interval) + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            dc = np.zeros((y1 - y0, x1 - x0))
            for ch in range(n_channels):
                dc += ((img[y0:y1, x0:x1, ch] - centers[k, 2 + ch])
                       * color_scale) ** 2
            ds = ((yy[y0:y1, x0:x1] - cy) ** 2
                  + (xx[y0:y1, x0:x1] - cx) ** 2)
            dist = dc + (m_over_s ** 2) * ds
            window = best_dist[y0:y1, x0:x1]
            closer = dist < window
            window[closer] = dist[closer]
            labels[y0:y1, x0:x1][closer] = k

        # Update centres as the mean of their assigned pixels.
        new_centers = centers.copy()
        max_shift = 0.0
        for k in range(n_centers):
            mask = labels == k
            if not np.any(mask):
                continue  # keep the old centre for an empty cluster
            new_centers[k, 0] = yy[mask].mean()
            new_centers[k, 1] = xx[mask].mean()
            for ch in range(n_channels):
                new_centers[k, 2 + ch] = img[:, :, ch][mask].mean()
            max_shift = max(max_shift,
                            np.abs(new_centers[k] - centers[k]).max())
        centers = new_centers
        if max_shift < 1e-3:
            break

    labels = enforce_connectivity(labels)
    # Recompute centres from the final (connectivity-enforced) labels.
    unique = np.unique(labels)
    final_centers = np.zeros((len(unique), 2 + n_channels))
    for i, lab in enumerate(unique):
        mask = labels == lab
        final_centers[i, 0] = yy[mask].mean()
        final_centers[i, 1] = xx[mask].mean()
        for ch in range(n_channels):
            final_centers[i, 2 + ch] = img[:, :, ch][mask].mean()
    return labels, final_centers


__all__ = ["slic", "enforce_connectivity"]
