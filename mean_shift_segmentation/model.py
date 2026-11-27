"""Mean-shift image segmentation (numpy only).

Pixels are mapped into a joint colour + spatial feature space
(colour channels scaled to [0, 1], spatial coordinates scaled by
``spatial_weight``) and shifted towards the local mode of a Gaussian
kernel density estimate. Pixels whose modes end up close together are
merged into one segment.

Public API: mean_shift_segment, segment_image, psnr (not needed here),
segment_count.
"""

import numpy as np

__all__ = ["mean_shift_segment", "segment_image", "segment_count"]


def _to_rgb_float(image):
    """Validate an image and return it as float64 RGB in [0, 255]."""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"image must be a numpy array, got {type(image).__name__}"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 3:
        pass
    else:
        raise ValueError(
            "image must be 2D grayscale (H, W) or 3D RGB (H, W, 3); "
            f"got shape {image.shape}"
        )
    if np.issubdtype(image.dtype, np.floating):
        if image.min() < 0 or image.max() > 255:
            raise ValueError(
                "float image values must lie in [0, 255]; "
                f"got range [{image.min()}, {image.max()}]"
            )
        return image.astype(np.float64)
    if np.issubdtype(image.dtype, np.integer):
        if image.min() < 0 or image.max() > 255:
            raise ValueError(
                "integer image values must lie in [0, 255]; "
                f"got range [{image.min()}, {image.max()}]"
            )
        return image.astype(np.float64)
    raise TypeError(
        f"unsupported image dtype {image.dtype}; use uint8 or float in [0, 255]"
    )


def _features(image, spatial_weight):
    """Joint colour + spatial feature vectors, one row per pixel."""
    height, width, _ = image.shape
    yy, xx = np.mgrid[0:height, 0:width]
    colour = image / 255.0
    spatial = np.stack(
        [yy / max(height, 1), xx / max(width, 1)], axis=-1
    ) * spatial_weight
    return np.concatenate(
        [colour.reshape(-1, 3), spatial.reshape(-1, 2)], axis=1
    )


def _mean_shift_modes(points, bandwidth, max_iter, tol):
    """Run Gaussian-kernel mean shift from every point; return the modes."""
    modes = points.copy()
    for _ in range(max_iter):
        max_shift = 0.0
        for i in range(len(modes)):
            diff = points - modes[i]
            dist2 = np.einsum("ij,ij->i", diff, diff)
            weights = np.exp(-dist2 / (2.0 * bandwidth ** 2))
            total = weights.sum()
            if total <= 0:
                continue
            new_mode = (weights[:, None] * points).sum(axis=0) / total
            shift = np.linalg.norm(new_mode - modes[i])
            max_shift = max(max_shift, shift)
            modes[i] = new_mode
        if max_shift < tol:
            break
    return modes


def _merge_modes(modes, merge_bandwidth):
    """Cluster modes closer than ``merge_bandwidth``; return centres."""
    centres = []
    for mode in modes:
        for centre in centres:
            if np.linalg.norm(mode - centre) <= merge_bandwidth:
                break
        else:
            centres.append(mode.copy())
    return np.array(centres)


def mean_shift_segment(image, bandwidth=0.25, spatial_weight=0.5,
                       max_iter=20, tol=1e-3):
    """Segment ``image`` with mean shift; return integer labels (H, W).

    ``bandwidth`` is the Gaussian kernel width in feature space (colour
    in [0, 1], spatial in [0, ``spatial_weight``]). Larger bandwidths
    merge more pixels into fewer, larger segments. Labels are 0..K-1.
    """
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    if spatial_weight < 0:
        raise ValueError(
            f"spatial_weight must be non-negative, got {spatial_weight}"
        )
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be a positive int, got {max_iter}")
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")

    img = _to_rgb_float(image)
    height, width, _ = img.shape
    points = _features(img, spatial_weight)
    modes = _mean_shift_modes(points, bandwidth, max_iter, tol)
    centres = _merge_modes(modes, bandwidth)
    # Assign each pixel to its nearest mode centre.
    diff = points[:, None, :] - centres[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    labels = np.argmin(dist2, axis=1).reshape(height, width)
    return labels.astype(np.int32)


def segment_count(labels):
    """Number of distinct segments in a label map from mean_shift_segment."""
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("labels must not be empty")
    return int(len(np.unique(labels)))


def segment_image(image, bandwidth=0.25, spatial_weight=0.5,
                  max_iter=20, tol=1e-3):
    """Return a uint8 RGB image where each pixel is replaced by the mean
    colour of its segment (a flat, posterized view of the segmentation)."""
    img = _to_rgb_float(image)
    labels = mean_shift_segment(
        image, bandwidth=bandwidth, spatial_weight=spatial_weight,
        max_iter=max_iter, tol=tol,
    )
    out = np.zeros_like(img)
    for label in np.unique(labels):
        mask = labels == label
        out[mask] = img[mask].mean(axis=0)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
