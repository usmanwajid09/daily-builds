"""Cartoonizer: turn an image into a cartoon / posterized version.

Pipeline (all numpy + stdlib):
1. edge-preserving smoothing (small bilateral-style weighted average),
2. per-channel colour quantization (uniform bins),
3. darkening pixels where the gradient magnitude is high (edge mask).

Public API: cartoonize, edge_preserving_smooth, quantize_colors, edge_mask.
"""

import numpy as np


def _to_rgb_uint8(image):
    """Validate an image and return it as a uint8 RGB array (H, W, 3).

    Accepts 2D grayscale or 3D RGB uint8 arrays; float arrays in [0, 255]
    are converted. Grayscale input is broadcast to 3 identical channels,
    so every function in this module always returns RGB.
    """
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
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if image.min() < 0 or image.max() > 255:
            raise ValueError(
                "float image values must lie in [0, 255]; "
                f"got range [{image.min()}, {image.max()}]"
            )
        return np.clip(np.round(image), 0, 255).astype(np.uint8)
    if np.issubdtype(image.dtype, np.integer):
        if image.min() < 0 or image.max() > 255:
            raise ValueError(
                "integer image values must lie in [0, 255]; "
                f"got range [{image.min()}, {image.max()}]"
            )
        return image.astype(np.uint8)
    raise TypeError(
        f"unsupported image dtype {image.dtype}; use uint8 or float in [0, 255]"
    )


def edge_preserving_smooth(image, iterations=1, sigma_s=1.0, sigma_r=25.0):
    """Edge-preserving smoothing via a small bilateral-style filter.

    Each pixel is replaced by a weighted average of its 5x5 neighbourhood,
    where weights combine a spatial Gaussian (sigma_s) and a range Gaussian
    (sigma_r) on intensity difference. Flat areas get averaged (noise drops);
    across a sharp step the range weight is ~0, so the edge stays intact.

    Returns a uint8 RGB array of the same shape as the input.
    """
    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError(f"iterations must be a positive int, got {iterations}")
    if sigma_s <= 0:
        raise ValueError(f"sigma_s must be positive, got {sigma_s}")
    if sigma_r <= 0:
        raise ValueError(f"sigma_r must be positive, got {sigma_r}")

    img = _to_rgb_uint8(image).astype(np.float64)
    radius = 2
    # Spatial Gaussian weights for the 5x5 window (fixed, reused each pass).
    ys, xs = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    spatial = np.exp(-(xs ** 2 + ys ** 2) / (2.0 * sigma_s ** 2))

    out = img
    for _ in range(iterations):
        padded = np.pad(out, ((radius, radius), (radius, radius), (0, 0)),
                        mode="edge")
        acc = np.zeros_like(out)
        wsum = np.zeros(out.shape[:2], dtype=np.float64)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                patch = padded[radius + dy: radius + dy + out.shape[0],
                               radius + dx: radius + dx + out.shape[1]]
                # Range weight from the mean absolute channel difference.
                diff = np.abs(patch - out).mean(axis=2)
                w = spatial[dy + radius, dx + radius] * np.exp(
                    -(diff ** 2) / (2.0 * sigma_r ** 2)
                )
                acc += patch * w[:, :, None]
                wsum += w
        out = acc / np.maximum(wsum[:, :, None], 1e-12)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def quantize_colors(image, n_colors=8):
    """Reduce the palette by uniform per-channel quantization.

    Each channel is mapped to ``n_colors`` evenly spaced bins; the output
    value is the centre of its bin. With n_colors bins per channel the image
    can contain at most n_colors**3 distinct colours.
    """
    if isinstance(n_colors, bool) or not isinstance(n_colors, (int, np.integer)):
        raise TypeError(f"n_colors must be an int, got {type(n_colors).__name__}")
    if n_colors < 2:
        raise ValueError(f"n_colors must be >= 2, got {n_colors}")

    img = _to_rgb_uint8(image).astype(np.float64)
    step = 256.0 / n_colors
    binned = np.floor(img / step)
    binned = np.clip(binned, 0, n_colors - 1)
    centres = (binned + 0.5) * step - 0.5
    return np.clip(np.round(centres), 0, 255).astype(np.uint8)


def _grayscale(image):
    """Luminance (float) of a uint8 RGB array."""
    return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2])


def _sobel_magnitude(gray):
    """Gradient magnitude via 3x3 Sobel kernels (numpy only)."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    padded = np.pad(gray, 1, mode="edge")
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for dy in range(3):
        for dx in range(3):
            patch = padded[dy:dy + gray.shape[0], dx:dx + gray.shape[1]]
            gx += kx[dy, dx] * patch
            gy += ky[dy, dx] * patch
    return np.hypot(gx, gy)


def edge_mask(image, threshold=30.0):
    """Boolean edge mask: True where the Sobel gradient magnitude exceeds
    ``threshold``. A flat image yields an all-False mask."""
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    img = _to_rgb_uint8(image)
    magnitude = _sobel_magnitude(_grayscale(img))
    return magnitude > threshold


def cartoonize(image, n_colors=8, edge_threshold=30.0, smooth_iterations=1):
    """Return a uint8 RGB cartoon version of ``image``.

    Smooths (edge-preserving), quantizes colours, then sets edge pixels to
    black. Grayscale input is returned as RGB (channels duplicated).
    """
    if isinstance(n_colors, bool) or not isinstance(n_colors, (int, np.integer)):
        raise TypeError(f"n_colors must be an int, got {type(n_colors).__name__}")
    if n_colors < 2:
        raise ValueError(f"n_colors must be >= 2, got {n_colors}")
    if edge_threshold < 0:
        raise ValueError(
            f"edge_threshold must be non-negative, got {edge_threshold}"
        )
    if not isinstance(smooth_iterations, (int, np.integer)) or smooth_iterations < 1:
        raise ValueError(
            f"smooth_iterations must be a positive int, got {smooth_iterations}"
        )

    smoothed = edge_preserving_smooth(image, iterations=smooth_iterations)
    quantized = quantize_colors(smoothed, n_colors=n_colors)
    edges = edge_mask(smoothed, threshold=edge_threshold)
    result = quantized.copy()
    result[edges] = 0
    return result
