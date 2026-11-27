"""HDR merge: combine multiple exposures into one radiance map (numpy only).

Follows Debevec & Malik (1997) with an identity camera response
(linear sensor assumed): for each pixel, the log-radiance is the
hat-weighted average of (ln pixel_value - ln exposure_time) over the
exposures. The hat weight is zero for saturated (0 / 255) pixels, so
blown-out and crushed pixels never contribute. A Reinhard tone map
then compresses the radiance map back to a displayable 8-bit image.

Public API: hat_weight, merge_hdr, reinhard_tone_map, psnr.
"""

import numpy as np

__all__ = ["hat_weight", "merge_hdr", "reinhard_tone_map", "psnr"]


def _validate_exposures(images, exposure_times):
    """Validate the exposure stack; return float64 images + float times."""
    if not isinstance(images, (list, tuple)) or len(images) < 2:
        raise ValueError(
            f"need at least 2 exposure images, got {len(images) if isinstance(images, (list, tuple)) else type(images).__name__}"
        )
    if len(exposure_times) != len(images):
        raise ValueError(
            f"{len(images)} images but {len(exposure_times)} exposure times"
        )
    times = np.asarray(exposure_times, dtype=np.float64)
    if np.any(times <= 0) or not np.all(np.isfinite(times)):
        raise ValueError(
            f"exposure times must be positive and finite, got {exposure_times}"
        )
    validated = []
    shape = None
    for i, img in enumerate(images):
        if not isinstance(img, np.ndarray):
            raise TypeError(
                f"exposure {i} must be a numpy array, got {type(img).__name__}"
            )
        if img.size == 0:
            raise ValueError(f"exposure {i} must not be empty")
        if img.ndim not in (2, 3):
            raise ValueError(
                f"exposure {i} must be 2D grayscale or 3D RGB; "
                f"got shape {img.shape}"
            )
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if shape is None:
            shape = img.shape
        elif img.shape != shape:
            raise ValueError(
                f"exposure {i} has shape {img.shape}, expected {shape}"
            )
        if np.issubdtype(img.dtype, np.floating):
            if img.min() < 0 or img.max() > 255:
                raise ValueError(
                    f"exposure {i} float values must lie in [0, 255]"
                )
        elif np.issubdtype(img.dtype, np.integer):
            if img.min() < 0 or img.max() > 255:
                raise ValueError(
                    f"exposure {i} integer values must lie in [0, 255]"
                )
        else:
            raise TypeError(
                f"exposure {i} has unsupported dtype {img.dtype}"
            )
        validated.append(img.astype(np.float64))
    return validated, times


def hat_weight(values):
    """Debevec hat weight: 0 at 0 and 255, peaking at 127.5.

    Accepts a scalar or an array of pixel values in [0, 255].
    """
    values = np.asarray(values, dtype=np.float64)
    return np.minimum(values, 255.0 - values)


def merge_hdr(images, exposure_times):
    """Merge an exposure stack into a float64 HDR radiance map (H, W, 3).

    Log-radiance per pixel = sum(w * (ln z - ln dt)) / sum(w), with the
    hat weight w. Pixels where every exposure is saturated get the
    unweighted mean of (ln z - ln dt) instead, so no pixel is left
    undefined.
    """
    validated, times = _validate_exposures(images, exposure_times)
    log_times = np.log(times)
    weighted_sum = np.zeros(validated[0].shape, dtype=np.float64)
    weight_sum = np.zeros(validated[0].shape, dtype=np.float64)
    fallback_sum = np.zeros(validated[0].shape, dtype=np.float64)
    for img, log_dt in zip(validated, log_times):
        weights = hat_weight(img)
        log_values = np.log(np.maximum(img, 1e-6)) - log_dt
        weighted_sum += weights * log_values
        weight_sum += weights
        fallback_sum += log_values
    n = len(validated)
    log_radiance = np.where(
        weight_sum > 0,
        weighted_sum / np.maximum(weight_sum, 1e-12),
        fallback_sum / n,
    )
    return np.exp(log_radiance)


def reinhard_tone_map(radiance, white_point=None, gamma=1.0):
    """Reinhard global tone map: Ld = L * (1 + L / Lw^2) / (1 + L),
    then scale so the white point maps to 255 and apply gamma.

    ``radiance`` is a float array (H, W) or (H, W, 3). With
    ``white_point=None`` the 99.9th percentile of the radiance is used;
    the curve maps that white point to exactly 1.0 (255).
    Returns a uint8 image.
    """
    radiance = np.asarray(radiance, dtype=np.float64)
    if radiance.size == 0:
        raise ValueError("radiance must not be empty")
    if np.any(radiance < 0) or not np.all(np.isfinite(radiance)):
        raise ValueError("radiance must be finite and non-negative")
    if white_point is not None and white_point <= 0:
        raise ValueError(f"white_point must be positive, got {white_point}")
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if white_point is None:
        white_point = float(np.percentile(radiance, 99.9))
        if white_point <= 0:
            white_point = float(radiance.max()) or 1.0
    luminance = radiance if radiance.ndim == 2 else radiance.mean(axis=2)
    # Per-channel scaling by the luminance tone curve keeps colours.
    # The Reinhard curve already maps the white point to exactly 1.0.
    tone = luminance * (1.0 + luminance / white_point ** 2) / (1.0 + luminance)
    scale = np.where(luminance > 0, tone / np.maximum(luminance, 1e-12), 0.0)
    if radiance.ndim == 3:
        mapped = radiance * scale[:, :, None]
    else:
        mapped = tone
    mapped = np.clip(mapped, 0.0, 1.0) ** (1.0 / gamma)
    return np.clip(np.round(mapped * 255.0), 0, 255).astype(np.uint8)


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
