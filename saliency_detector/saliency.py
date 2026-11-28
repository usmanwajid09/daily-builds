"""Spectral-residual saliency detection from scratch (numpy only).

Implements the method of Hou & Zhang (2007): the log amplitude
spectrum of an image is mostly smooth, so subtracting a smoothed
version of it leaves a "spectral residual" that marks the unusual
parts of the scene. Recombining the residual with the original phase
and transforming back gives a saliency map whose peaks are the
regions a human eye would look at first.
"""

import numpy as np


def _box_blur(image, size):
    """Mean filter with a square window (edge-padded)."""
    if size % 2 == 0 or size < 1:
        raise ValueError(f"size must be a positive odd integer, got {size}")
    radius = size // 2
    padded = np.pad(image, radius, mode="edge")
    out = np.zeros_like(image, dtype=np.float64)
    for dy in range(size):
        for dx in range(size):
            out += padded[dy:dy + image.shape[0], dx:dx + image.shape[1]]
    return out / float(size * size)


def _gaussian_blur(image, sigma):
    """Small separable Gaussian blur (radius = ceil(3 * sigma))."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    radius = max(1, int(np.ceil(3 * sigma)))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (axis / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = np.pad(image, ((radius, radius), (0, 0)), mode="edge")
    temp = np.zeros_like(image, dtype=np.float64)
    for i, w in enumerate(kernel):
        temp += w * padded[i:i + image.shape[0], :]
    padded = np.pad(temp, ((0, 0), (radius, radius)), mode="edge")
    out = np.zeros_like(image, dtype=np.float64)
    for j, w in enumerate(kernel):
        out += w * padded[:, j:j + image.shape[1]]
    return out


def spectral_residual_saliency(image, blur_size=3, smooth_sigma=2.0):
    """Return a saliency map in [0, 1] for a 2D grayscale image.

    Steps: FFT → log amplitude spectrum → subtract a box-blurred
    version (the residual) → inverse FFT with the original phase →
    squared magnitude → Gaussian smoothing → normalise to [0, 1].
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"image must be a 2D grayscale array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    spectrum = np.fft.fft2(arr)
    amplitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    log_amp = np.log(amplitude + 1e-8)
    residual = log_amp - _box_blur(log_amp, blur_size)
    rebuilt = np.exp(residual + 1j * phase)
    saliency = np.abs(np.fft.ifft2(rebuilt)) ** 2
    saliency = _gaussian_blur(saliency, smooth_sigma)
    peak = saliency.max()
    if peak <= 0:
        return np.zeros_like(saliency)
    return saliency / peak


def saliency_peak(saliency_map):
    """Return (row, col) of the most salient pixel in a saliency map."""
    arr = np.asarray(saliency_map, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("saliency_map must be a non-empty 2D array")
    return tuple(int(v) for v in np.unravel_index(np.argmax(arr), arr.shape))
