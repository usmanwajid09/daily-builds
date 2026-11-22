"""Single-image dehazing from scratch via the dark channel prior.

Implements the classic He, Sun & Tang (2009) pipeline with numpy only:

1. **Dark channel** — per-pixel minimum over RGB, then a minimum filter
   over a local patch. Haze-free outdoor images have a dark channel near
   zero almost everywhere; haze lifts it.
2. **Atmospheric light** — the brightest pixels of the hazy image among
   the top 0.1% of dark-channel values.
3. **Transmission map** — ``t = 1 - omega * dark_channel(I / A)``,
   lightly smoothed with a box filter.
4. **Recovery** — ``J = (I - A) / max(t, t0) + A``, clipped to [0, 1].

Reference: K. He, J. Sun, X. Tang, "Single Image Haze Removal Using the
Dark Channel Prior", CVPR 2009.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "dark_channel",
    "estimate_atmospheric_light",
    "estimate_transmission",
    "recover",
    "dehaze",
    "make_hazy_image",
    "make_clean_scene",
]


def _to_float_rgb(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must be RGB (H, W, 3), got shape {image.shape}."
        )
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


def _check_patch_size(patch_size):
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError(f"patch_size must be a positive int, got {patch_size!r}.")
    if patch_size % 2 == 0:
        raise ValueError(f"patch_size must be odd, got {patch_size}.")


def _min_filter(channel, patch_size):
    """Minimum filter with edge replication at the borders."""
    pad = patch_size // 2
    padded = np.pad(channel, pad, mode="edge")
    windows = sliding_window_view(padded, (patch_size, patch_size))
    return windows.min(axis=(2, 3))


def _mean_filter(channel, patch_size):
    """Box (mean) filter with edge replication at the borders."""
    pad = patch_size // 2
    padded = np.pad(channel, pad, mode="edge")
    windows = sliding_window_view(padded, (patch_size, patch_size))
    return windows.mean(axis=(2, 3))


def dark_channel(image, patch_size=15):
    """Dark channel of an RGB image: min over channels, then min filter."""
    _check_patch_size(patch_size)
    rgb = _to_float_rgb(image)
    channel_min = rgb.min(axis=2)
    return _min_filter(channel_min, patch_size)


def estimate_atmospheric_light(image, dark=None, top_fraction=0.001):
    """Estimate the atmospheric light A (an RGB triple in [0, 1]).

    Takes the pixels with the largest dark-channel values (the haziest
    0.1%) and returns the per-channel maximum of the hazy image there.
    """
    rgb = _to_float_rgb(image)
    if dark is None:
        dark = dark_channel(rgb)
    dark = np.asarray(dark, dtype=np.float64)
    if dark.shape != rgb.shape[:2]:
        raise ValueError(
            f"dark channel shape {dark.shape} does not match image {rgb.shape[:2]}."
        )
    if not 0 < top_fraction <= 1:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction!r}.")
    n_top = max(1, int(np.ceil(top_fraction * dark.size)))
    flat_dark = dark.reshape(-1)
    top_idx = np.argpartition(flat_dark, -n_top)[-n_top:]
    flat_rgb = rgb.reshape(-1, 3)
    brightest = flat_rgb[top_idx]
    return brightest.max(axis=0)


def estimate_transmission(image, atmospheric_light, omega=0.95, patch_size=15):
    """Transmission map t in (0, 1]; 1 means no haze, 0 means fully hazed."""
    _check_patch_size(patch_size)
    if not 0 < omega <= 1:
        raise ValueError(f"omega must be in (0, 1], got {omega!r}.")
    rgb = _to_float_rgb(image)
    atmospheric_light = np.asarray(atmospheric_light, dtype=np.float64)
    if atmospheric_light.shape != (3,):
        raise ValueError(
            f"atmospheric_light must be an RGB triple, got shape {atmospheric_light.shape}."
        )
    if np.any(atmospheric_light <= 0):
        raise ValueError("atmospheric_light must be positive in every channel.")
    normalised = rgb / atmospheric_light
    dark = dark_channel(np.clip(normalised, 0, 1), patch_size=patch_size)
    t = 1.0 - omega * dark
    # Light smoothing removes block artefacts from the patch minimum.
    t = _mean_filter(t, patch_size)
    return np.clip(t, 0.05, 1.0)


def recover(hazy_image, atmospheric_light, transmission, t0=0.1):
    """Recover the haze-free image from the haze model."""
    rgb = _to_float_rgb(hazy_image)
    transmission = np.asarray(transmission, dtype=np.float64)
    if transmission.shape != rgb.shape[:2]:
        raise ValueError(
            f"transmission shape {transmission.shape} does not match image {rgb.shape[:2]}."
        )
    if not 0 < t0 < 1:
        raise ValueError(f"t0 must be in (0, 1), got {t0!r}.")
    atmospheric_light = np.asarray(atmospheric_light, dtype=np.float64)
    if atmospheric_light.shape != (3,):
        raise ValueError("atmospheric_light must be an RGB triple.")
    t = np.maximum(transmission, t0)[..., None]
    recovered = (rgb - atmospheric_light) / t + atmospheric_light
    return np.clip(recovered, 0, 1)


def dehaze(hazy_image, omega=0.95, patch_size=15, t0=0.1):
    """Full pipeline. Returns (recovered_image, info dict)."""
    rgb = _to_float_rgb(hazy_image)
    dark = dark_channel(rgb, patch_size=patch_size)
    atmospheric_light = estimate_atmospheric_light(rgb, dark=dark)
    transmission = estimate_transmission(rgb, atmospheric_light,
                                         omega=omega, patch_size=patch_size)
    recovered = recover(rgb, atmospheric_light, transmission, t0=t0)
    info = {
        "atmospheric_light": atmospheric_light,
        "mean_transmission": float(transmission.mean()),
        "min_transmission": float(transmission.min()),
        "mean_dark_channel": float(dark.mean()),
    }
    return recovered, info


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def make_clean_scene(size=64, seed=0):
    """A colourful synthetic scene with a dark channel near zero in places."""
    rng = np.random.default_rng(seed)
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size, 3))
    img[..., 0] = (xx / size) * 0.8          # red gradient
    img[..., 1] = (yy / size) * 0.8          # green gradient
    img[..., 2] = 0.4 + 0.4 * np.sin(xx / 6.0)  # blue waves
    # A dark circle and a bright square so the scene has real structure.
    circle = (yy - size // 3) ** 2 + (xx - size // 3) ** 2 <= (size // 8) ** 2
    img[circle] = [0.05, 0.05, 0.05]
    square = ((np.abs(yy - 2 * size // 3) <= size // 10)
              & (np.abs(xx - 2 * size // 3) <= size // 10))
    img[square] = [0.9, 0.85, 0.1]
    img += rng.normal(0, 0.01, size=img.shape)
    return np.clip(img, 0, 1)


def make_hazy_image(clean_image, transmission=0.55, atmospheric_light=(0.9, 0.9, 0.9)):
    """Apply the haze model I = J * t + A * (1 - t) with constant t."""
    rgb = _to_float_rgb(clean_image)
    if not 0 < transmission <= 1:
        raise ValueError(f"transmission must be in (0, 1], got {transmission!r}.")
    atmospheric_light = np.asarray(atmospheric_light, dtype=np.float64)
    if atmospheric_light.shape != (3,):
        raise ValueError("atmospheric_light must be an RGB triple.")
    A = atmospheric_light.reshape(1, 1, 3)
    return np.clip(rgb * transmission + A * (1 - transmission), 0, 1)
