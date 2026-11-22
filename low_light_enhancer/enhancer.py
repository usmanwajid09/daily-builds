"""Low-light image enhancement from scratch (numpy only).

Pipeline::

    dark image -> median denoise -> adaptive gamma -> CLAHE -> bright image

- **Adaptive gamma** — the gamma exponent is chosen from the image's own
  mean brightness so the result lands near a target mean (default 0.5).
- **CLAHE** — contrast-limited adaptive histogram equalisation on the
  luminance channel, tile by tile, with histogram clipping so noise is
  not amplified. Colour is preserved by scaling each channel with the
  luminance ratio.
- **Denoise** — a 3x3 median filter, applied before brightening so sensor
  noise is not amplified along with the signal.

``metrics()`` reports brightness (mean luminance), contrast (standard
deviation) and dynamic range before/after, so the improvement is
measurable rather than a matter of taste.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "metrics",
    "gamma_correct",
    "adaptive_gamma",
    "clahe_enhance",
    "denoise",
    "enhance",
    "make_dark_image",
]


def _to_float(image):
    """Normalise a grayscale or RGB image to float in [0, 1]."""
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D or 3-D, got shape {image.shape}.")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(f"RGB images must have 3 channels, got shape {image.shape}.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image must not be empty, got shape {image.shape}.")
    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.0
    else:
        image = image.astype(np.float64)
        if image.min() < 0.0 or image.max() > 1.0:
            raise ValueError(
                "float images must have values in [0, 1]; "
                f"got range [{image.min():.3f}, {image.max():.3f}]."
            )
    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain only finite values.")
    return image


def _luminance(image):
    if image.ndim == 2:
        return image
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def metrics(image):
    """Brightness / contrast / dynamic-range summary of an image."""
    img = _to_float(image)
    lum = _luminance(img)
    return {
        "brightness": float(lum.mean()),
        "contrast": float(lum.std()),
        "dynamic_range": float(lum.max() - lum.min()),
        "min": float(lum.min()),
        "max": float(lum.max()),
    }


def gamma_correct(image, gamma):
    """Apply ``out = image ** gamma`` (gamma < 1 brightens, > 1 darkens)."""
    if not isinstance(gamma, (int, float)) or gamma <= 0:
        raise ValueError(f"gamma must be a positive number, got {gamma!r}.")
    img = _to_float(image)
    return np.clip(img ** gamma, 0, 1)


def adaptive_gamma(image, target_mean=0.5):
    """Gamma-correct so the mean luminance moves towards ``target_mean``.

    Returns (corrected_image, gamma_used). A pure-black image cannot be
    brightened by gamma (0 ** gamma == 0) and is returned unchanged with
    gamma 1.0.
    """
    if not 0 < target_mean < 1:
        raise ValueError(f"target_mean must be in (0, 1), got {target_mean!r}.")
    img = _to_float(image)
    mean = float(_luminance(img).mean())
    if mean <= 1e-6:
        return img.copy(), 1.0
    gamma = float(np.log(target_mean) / np.log(mean))
    gamma = float(np.clip(gamma, 0.2, 3.0))
    return gamma_correct(img, gamma), gamma


def _equalise_tile(tile, clip_limit):
    """Contrast-limited histogram equalisation of one tile (values in [0, 1])."""
    flat = np.clip((tile * 255).astype(int), 0, 255).reshape(-1)
    hist = np.bincount(flat, minlength=256).astype(np.float64)
    if clip_limit is not None:
        limit = clip_limit * (flat.size / 256.0)
        excess = np.maximum(hist - limit, 0).sum()
        hist = np.minimum(hist, limit)
        hist += excess / 256.0
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0][0] if np.any(cdf > 0) else 0.0
    denom = flat.size - cdf_min
    if denom <= 0:
        return tile.copy()  # flat tile: nothing to equalise
    lut = (cdf - cdf_min) / denom
    return lut[np.clip((tile * 255).astype(int), 0, 255)]


def clahe_enhance(image, clip_limit=2.0, tile_size=8):
    """CLAHE on the luminance channel; colour channels are scaled to match.

    Each tile is equalised independently (nearest-tile mapping), which
    boosts local contrast without amplifying noise the way a global
    equalisation would.
    """
    if clip_limit is not None and (not isinstance(clip_limit, (int, float)) or clip_limit <= 0):
        raise ValueError(f"clip_limit must be positive, got {clip_limit!r}.")
    if not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError(f"tile_size must be a positive int, got {tile_size!r}.")
    img = _to_float(image)
    lum = _luminance(img)
    h, w = lum.shape
    pad_h = (-h) % tile_size
    pad_w = (-w) % tile_size
    padded = np.pad(lum, ((0, pad_h), (0, pad_w)), mode="edge")
    out = np.zeros_like(padded)
    for y in range(0, padded.shape[0], tile_size):
        for x in range(0, padded.shape[1], tile_size):
            out[y:y + tile_size, x:x + tile_size] = _equalise_tile(
                padded[y:y + tile_size, x:x + tile_size], clip_limit)
    enhanced_lum = out[:h, :w]
    if img.ndim == 2:
        return np.clip(enhanced_lum, 0, 1)
    # Preserve hue: scale each channel by the luminance ratio.
    ratio = np.where(lum > 1e-6, enhanced_lum / np.maximum(lum, 1e-6), 1.0)
    return np.clip(img * ratio[..., None], 0, 1)


def denoise(image, size=3):
    """3x3 median filter (per channel for RGB)."""
    if not isinstance(size, int) or size <= 0 or size % 2 == 0:
        raise ValueError(f"size must be a positive odd int, got {size!r}.")
    img = _to_float(image)
    pad = size // 2

    def _median_2d(channel):
        padded = np.pad(channel, pad, mode="edge")
        windows = sliding_window_view(padded, (size, size))
        return np.median(windows, axis=(2, 3))

    if img.ndim == 2:
        return _median_2d(img)
    return np.stack([_median_2d(img[..., c]) for c in range(3)], axis=-1)


def enhance(image, target_mean=0.5, clip_limit=2.0, tile_size=8,
            denoise_first=True):
    """Full pipeline: denoise -> adaptive gamma -> CLAHE.

    Returns (enhanced_image, metrics_before, metrics_after, gamma_used).
    """
    img = _to_float(image)
    before = metrics(img)
    if denoise_first:
        img = denoise(img)
    img, gamma = adaptive_gamma(img, target_mean=target_mean)
    img = clahe_enhance(img, clip_limit=clip_limit, tile_size=tile_size)
    after = metrics(img)
    return np.clip(img, 0, 1), before, after, gamma


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def make_dark_image(size=64, seed=0):
    """A synthetic dark scene: dim shapes on a near-black background."""
    rng = np.random.default_rng(seed)
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    img = np.full((size, size, 3), 0.04, dtype=np.float64)
    yy, xx = np.mgrid[0:size, 0:size]
    circle = (yy - size // 3) ** 2 + (xx - size // 3) ** 2 <= (size // 8) ** 2
    img[circle] = [0.25, 0.10, 0.10]   # dim red circle
    square = ((np.abs(yy - 2 * size // 3) <= size // 10)
              & (np.abs(xx - 2 * size // 3) <= size // 10))
    img[square] = [0.10, 0.28, 0.12]   # dim green square
    img += rng.normal(0, 0.01, size=img.shape)
    return np.clip(img, 0, 1)
