"""Phase correlation: FFT-based translation estimation between images.

Implemented from scratch with numpy only — no OpenCV. The normalised
cross-power spectrum of two images is inverse-transformed; its peak
locates the translation between them, robust to uniform brightness
changes. A parabolic fit around the peak refines the shift to
sub-pixel accuracy.
"""

import numpy as np

__all__ = [
    "phase_correlate",
    "estimate_translation",
    "shift_image_fourier",
]


def _validate_image(image, name="image"):
    if image is None:
        raise ValueError(f"{name} must not be None")
    try:
        arr = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric array") from None
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D grayscale, got {arr.ndim}D")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _hann_window(shape):
    """Separable Hann window; reduces edge artefacts in the FFT."""
    wy = np.hanning(shape[0]) if shape[0] > 1 else np.ones(1)
    wx = np.hanning(shape[1]) if shape[1] > 1 else np.ones(1)
    return np.outer(wy, wx)


def _parabolic_offset(left, centre, right):
    """Sub-pixel offset of a peak from three samples (parabolic fit)."""
    denom = left - 2.0 * centre + right
    if denom == 0.0:
        return 0.0
    offset = 0.5 * (left - right) / denom
    return float(np.clip(offset, -1.0, 1.0))


def shift_image_fourier(image, dy, dx):
    """Shift a grayscale image by (dy, dx) pixels via the Fourier shift theorem.

    Supports fractional shifts. Content wraps around the borders
    (periodic shift), as phase correlation itself assumes.
    """
    arr = _validate_image(image)
    if not np.isfinite(dy) or not np.isfinite(dx):
        raise ValueError("shift must be finite")
    height, width = arr.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    phase = np.exp(-2j * np.pi * (fy * dy + fx * dx))
    shifted = np.fft.ifft2(np.fft.fft2(arr) * phase)
    return np.real(shifted)


def phase_correlate(image1, image2, apply_window=True, subpixel=True):
    """Return ``(dy, dx, peak)`` — the shift taking image1 onto image2.

    Positive ``dy``/``dx`` mean image2 is shifted down/right relative to
    image1. ``peak`` is the correlation-peak height in [0, 1]; ~1 for
    identical images up to a pure translation. Raises ValueError for
    mismatched shapes, empty or constant images (no phase information).
    """
    a = _validate_image(image1, "image1")
    b = _validate_image(image2, "image2")
    if a.shape != b.shape:
        raise ValueError(f"images must have the same shape, got {a.shape} and {b.shape}")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        raise ValueError("phase correlation needs non-constant images (zero variance)")
    height, width = a.shape
    if apply_window:
        window = _hann_window(a.shape)
        a = (a - a.mean()) * window
        b = (b - b.mean()) * window
    else:
        a = a - a.mean()
        b = b - b.mean()
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = np.conj(fa) * fb
    magnitude = np.abs(cross)
    magnitude[magnitude == 0] = 1.0
    corr = np.real(np.fft.ifft2(cross / magnitude))
    peak = float(corr.max())
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = py if py <= height // 2 else py - height
    dx = px if px <= width // 2 else px - width
    if subpixel:
        # Parabolic refinement along each axis (indices wrap around).
        ym, yp = (py - 1) % height, (py + 1) % height
        xm, xp = (px - 1) % width, (px + 1) % width
        dy += _parabolic_offset(corr[ym, px], corr[py, px], corr[yp, px])
        dx += _parabolic_offset(corr[py, xm], corr[py, px], corr[py, xp])
    return (float(dy), float(dx), peak)


def estimate_translation(image1, image2, apply_window=True, subpixel=True):
    """Return a dict with the estimated shift and correlation peak."""
    dy, dx, peak = phase_correlate(image1, image2,
                                   apply_window=apply_window, subpixel=subpixel)
    return {
        "dy": dy,
        "dx": dx,
        "shift": (dy, dx),
        "peak": peak,
        "magnitude": float(np.hypot(dy, dx)),
    }
