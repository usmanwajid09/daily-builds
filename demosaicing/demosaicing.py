"""Bayer demosaicing, implemented from scratch (numpy only)."""

import numpy as np

# Channel indices in each 2x2 Bayer tile: 0 = red, 1 = green, 2 = blue.
_PATTERNS = {
    "RGGB": np.array([[0, 1], [1, 2]]),
    "BGGR": np.array([[2, 1], [1, 0]]),
    "GRBG": np.array([[1, 0], [2, 1]]),
    "GBRG": np.array([[1, 2], [0, 1]]),
}

# Bilinear weights for the red/blue channels (known samples sit on a
# 2-pixel grid; a missing sample is the mean of its same-channel
# neighbours: left/right, up/down, or the four diagonals).
_RB_KERNEL = np.array(
    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float64
)
# Green samples sit on a checkerboard; a missing green sample is the
# mean of its four orthogonal green neighbours.
_G_KERNEL = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def _validate_pattern(pattern):
    if pattern not in _PATTERNS:
        raise ValueError(
            f"unknown Bayer pattern {pattern!r}; expected one of "
            f"{sorted(_PATTERNS)}"
        )
    return _PATTERNS[pattern]


def _convolve3x3(image, kernel):
    """Convolve a 2D array with a 3x3 kernel, zero padding outside."""
    padded = np.pad(image, 1, mode="constant", constant_values=0.0)
    out = np.zeros_like(image, dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            weight = kernel[di, dj]
            if weight != 0.0:
                out += weight * padded[di:di + image.shape[0], dj:dj + image.shape[1]]
    return out


def _channel_mask(pattern_tile, channel, shape):
    """Boolean mask of the pixels where ``channel`` was sampled."""
    height, width = shape
    tile = np.tile(pattern_tile, (height // 2 + 1, width // 2 + 1))
    return tile[:height, :width] == channel


def _interpolate_channel(bayer, mask, kernel):
    """Fill the unmeasured pixels of one channel by normalised
    convolution: convolve the known samples (zeros elsewhere) with the
    kernel and divide by the kernel applied to the mask, so each
    output is the weighted mean of its known neighbours."""
    known = np.where(mask, bayer, 0.0)
    numerator = _convolve3x3(known, kernel)
    denominator = _convolve3x3(mask.astype(np.float64), kernel)
    filled = np.divide(
        numerator, denominator, out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return np.where(mask, bayer, filled)


def mosaic(rgb, pattern="RGGB"):
    """Sample an RGB image down to a Bayer mosaic (the inverse of
    :func:`demosaic`).

    Parameters
    ----------
    rgb : np.ndarray
        HxWx3 image with even height and width.
    pattern : str
        Bayer pattern: "RGGB", "BGGR", "GRBG" or "GBRG".

    Returns
    -------
    np.ndarray
        2D Bayer mosaic, same dtype as the input.
    """
    if not isinstance(rgb, np.ndarray):
        raise ValueError("rgb must be a numpy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"rgb must be a HxWx3 image, got shape {rgb.shape}"
        )
    if rgb.size == 0:
        raise ValueError("rgb must not be empty")
    if rgb.shape[0] % 2 != 0 or rgb.shape[1] % 2 != 0:
        raise ValueError(
            "rgb height and width must both be even for a Bayer mosaic, "
            f"got {rgb.shape[0]}x{rgb.shape[1]}"
        )
    if not np.issubdtype(rgb.dtype, np.number):
        raise ValueError("rgb must have a numeric dtype")
    if not np.all(np.isfinite(rgb.astype(np.float64))):
        raise ValueError("rgb must contain only finite values")
    tile = _validate_pattern(pattern)
    height, width = rgb.shape[:2]
    tiled = np.tile(tile, (height // 2 + 1, width // 2 + 1))[:height, :width]
    bayer = np.empty((height, width), dtype=rgb.dtype)
    for channel in range(3):
        bayer[tiled == channel] = rgb[:, :, channel][tiled == channel]
    return bayer


def demosaic(bayer, pattern="RGGB"):
    """Reconstruct an RGB image from a Bayer mosaic by bilinear
    interpolation of the missing colour samples.

    Parameters
    ----------
    bayer : np.ndarray
        2D Bayer mosaic with even height and width.
    pattern : str
        Bayer pattern: "RGGB", "BGGR", "GRBG" or "GBRG".

    Returns
    -------
    np.ndarray
        HxWx3 float64 image. Pixels that were actually measured keep
        their exact value; the rest are bilinear estimates.
    """
    if not isinstance(bayer, np.ndarray):
        raise ValueError("bayer must be a numpy array")
    if bayer.ndim != 2:
        raise ValueError(f"bayer must be 2D, got {bayer.ndim}D")
    if bayer.size == 0:
        raise ValueError("bayer must not be empty")
    if bayer.shape[0] % 2 != 0 or bayer.shape[1] % 2 != 0:
        raise ValueError(
            "bayer height and width must both be even, "
            f"got {bayer.shape[0]}x{bayer.shape[1]}"
        )
    if not np.issubdtype(bayer.dtype, np.number):
        raise ValueError("bayer must have a numeric dtype")
    arr = bayer.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("bayer must contain only finite values")
    tile = _validate_pattern(pattern)

    out = np.empty((arr.shape[0], arr.shape[1], 3), dtype=np.float64)
    for channel, kernel in ((0, _RB_KERNEL), (1, _G_KERNEL), (2, _RB_KERNEL)):
        mask = _channel_mask(tile, channel, arr.shape)
        out[:, :, channel] = _interpolate_channel(arr, mask, kernel)
    return out


def psnr(reference, reconstructed, max_value=255.0):
    """Peak signal-to-noise ratio in dB between two same-shape images.

    Returns ``inf`` when the images are identical.
    """
    if not isinstance(reference, np.ndarray) or not isinstance(reconstructed, np.ndarray):
        raise ValueError("both images must be numpy arrays")
    if reference.shape != reconstructed.shape:
        raise ValueError(
            f"shape mismatch: {reference.shape} vs {reconstructed.shape}"
        )
    if max_value <= 0:
        raise ValueError("max_value must be positive")
    mse = float(np.mean((reference.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(max_value ** 2 / mse)
