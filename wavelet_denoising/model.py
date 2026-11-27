"""Haar wavelet denoising (numpy only).

A 2D Haar discrete wavelet transform (multi-level) decomposes the
image into an approximation subband plus horizontal / vertical /
diagonal detail subbands. Noise lives mostly in the detail
coefficients, so thresholding them (soft or hard, with the universal
threshold estimated from the finest detail subband) and inverting the
transform removes noise while keeping edges.

Public API: haar_dwt2, haar_idwt2, wavelet_denoise, psnr.
"""

import numpy as np

__all__ = ["haar_dwt2", "haar_idwt2", "wavelet_denoise", "psnr"]


def _validate_gray(image):
    """Validate a grayscale image; return float64 in [0, 255]."""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"image must be a numpy array, got {type(image).__name__}"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.ndim != 2:
        raise ValueError(
            f"image must be 2D grayscale (H, W); got shape {image.shape}"
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


def _haar_step_1d(values):
    """One Haar step along the last axis (length must be even).

    First half: scaled averages, second half: scaled differences.
    """
    if values.shape[-1] % 2 != 0:
        raise ValueError(
            f"Haar step needs an even length, got {values.shape[-1]}"
        )
    even = values[..., 0::2]
    odd = values[..., 1::2]
    scale = np.sqrt(2.0)
    return np.concatenate([(even + odd) / scale, (even - odd) / scale],
                          axis=-1)


def _haar_inverse_1d(values):
    """Inverse of _haar_step_1d along the last axis."""
    half = values.shape[-1] // 2
    avg = values[..., :half]
    diff = values[..., half:]
    scale = np.sqrt(2.0)
    out = np.empty_like(values)
    out[..., 0::2] = (avg + diff) / scale
    out[..., 1::2] = (avg - diff) / scale
    return out


def _pad_to_multiple(image, levels):
    """Pad with edge replication so both dims are multiples of 2**levels."""
    factor = 2 ** levels
    height, width = image.shape
    pad_h = (-height) % factor
    pad_w = (-width) % factor
    if pad_h == 0 and pad_w == 0:
        return image, (height, width)
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, (height, width)


def haar_dwt2(image, levels=2):
    """Multi-level 2D Haar DWT; returns coefficients (same shape as the
    padded image) and the original (height, width) for exact inversion."""
    if not isinstance(levels, (int, np.integer)) or levels < 1:
        raise ValueError(f"levels must be a positive int, got {levels}")
    img = _validate_gray(image)
    padded, orig_shape = _pad_to_multiple(img, levels)
    coeffs = padded.copy()
    height, width = padded.shape
    for level in range(levels):
        h = height // (2 ** level)
        w = width // (2 ** level)
        sub = coeffs[:h, :w]
        sub = _haar_step_1d(sub)            # along columns
        sub = _haar_step_1d(sub.T).T        # along rows
        coeffs[:h, :w] = sub
    return coeffs, orig_shape


def haar_idwt2(coeffs, orig_shape, levels=2):
    """Inverse of haar_dwt2; returns a float64 image of ``orig_shape``."""
    if not isinstance(levels, (int, np.integer)) or levels < 1:
        raise ValueError(f"levels must be a positive int, got {levels}")
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.ndim != 2:
        raise ValueError(
            f"coeffs must be 2D; got shape {coeffs.shape}"
        )
    height, width = coeffs.shape
    out = coeffs.copy()
    for level in reversed(range(levels)):
        h = height // (2 ** level)
        w = width // (2 ** level)
        sub = out[:h, :w]
        sub = _haar_inverse_1d(sub.T).T     # undo rows
        sub = _haar_inverse_1d(sub)         # undo columns
        out[:h, :w] = sub
    oh, ow = orig_shape
    return out[:oh, :ow]


def _estimate_sigma(coeffs, levels):
    """Noise sigma from the MAD of the finest-level HH detail subband."""
    height, width = coeffs.shape
    # Finest detail is the level-1 HH subband: bottom-right quadrant.
    hh = coeffs[height // 2:, width // 2:]
    mad = np.median(np.abs(hh - np.median(hh)))
    return mad / 0.6745 if mad > 0 else 0.0


def _soft_threshold(values, threshold):
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _hard_threshold(values, threshold):
    return np.where(np.abs(values) >= threshold, values, 0.0)


def wavelet_denoise(image, levels=2, mode="soft", threshold=None):
    """Denoise a grayscale image with Haar wavelet thresholding.

    ``mode`` is "soft" or "hard". With ``threshold=None`` the universal
    threshold sigma * sqrt(2 * ln N) is used, with sigma estimated from
    the finest detail subband. Returns a uint8 image.
    """
    if mode not in ("soft", "hard"):
        raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")
    if threshold is not None and threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    img = _validate_gray(image)
    coeffs, orig_shape = haar_dwt2(img, levels=levels)
    if threshold is None:
        sigma = _estimate_sigma(coeffs, levels)
        threshold = sigma * np.sqrt(2.0 * np.log(img.size)) if sigma > 0 else 0.0
    thresh = _soft_threshold if mode == "soft" else _hard_threshold
    height, width = coeffs.shape
    # Threshold every detail subband, keep the level-`levels` approximation.
    approx_h = height // (2 ** levels)
    approx_w = width // (2 ** levels)
    cleaned = coeffs.copy()
    mask = np.ones_like(cleaned, dtype=bool)
    mask[:approx_h, :approx_w] = False
    cleaned[mask] = thresh(cleaned[mask], threshold)
    restored = haar_idwt2(cleaned, orig_shape, levels=levels)
    return np.clip(np.round(restored), 0, 255).astype(np.uint8)


def psnr(reference, other, max_value=255.0):
    """Peak signal-to-noise ratio in dB (inf for identical images)."""
    ref = np.asarray(reference, dtype=np.float64)
    oth = np.asarray(other, dtype=np.float64)
    if ref.shape != oth.shape:
        raise ValueError(
            f"shape mismatch: {ref.shape} vs {oth.shape}"
        )
    mse = float(np.mean((ref - oth) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(max_value ** 2 / mse)
