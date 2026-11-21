"""Wiener deconvolution image deblurring from scratch (numpy FFT).

A blurred, noisy image ``g = h * f + n`` (``h`` the point-spread
function, ``n`` noise) is restored in the frequency domain with the
Wiener filter

    F = conj(H) / (|H|^2 + NSR) * G

where ``H`` and ``G`` are the Fourier transforms of the PSF and the
blurred image and ``NSR`` is the noise-to-signal power ratio. Where
the blur attenuates a frequency strongly (``|H|`` small) the filter
backs off instead of amplifying the noise there, which is what
plain inverse filtering would do. ``NSR`` can be estimated from a
noise variance and the image variance, or passed directly.
"""

import numpy as np

__all__ = [
    "apply_blur",
    "estimate_nsr",
    "make_gaussian_psf",
    "make_motion_psf",
    "psnr",
    "wiener_deconvolve",
]


def _check_image(image, name="image"):
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D grayscale image, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _check_psf(psf):
    arr = np.asarray(psf, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"psf must be a 2D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("psf must not be empty")
    if arr.shape[0] % 2 == 0 or arr.shape[1] % 2 == 0:
        raise ValueError(
            f"psf must have odd height and width (centred kernel), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("psf must contain only finite values")
    if arr.sum() == 0:
        raise ValueError("psf must not sum to zero")
    return arr / arr.sum()


def make_gaussian_psf(size=15, sigma=2.0):
    """Centred, normalised 2D Gaussian PSF (``size`` odd)."""
    if not isinstance(size, (int, np.integer)) or int(size) < 1 or int(size) % 2 == 0:
        raise ValueError(f"size must be a positive odd integer, got {size!r}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    size = int(size)
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=float)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return psf / psf.sum()


def make_motion_psf(length=15, angle_degrees=0.0):
    """Centred, normalised straight-line motion-blur PSF.

    ``length`` is the blur length in pixels (odd), ``angle_degrees``
    the direction of motion measured from the x-axis.
    """
    if not isinstance(length, (int, np.integer)) or int(length) < 1 \
            or int(length) % 2 == 0:
        raise ValueError(f"length must be a positive odd integer, got {length!r}")
    length = int(length)
    angle = np.deg2rad(float(angle_degrees))
    psf = np.zeros((length, length))
    half = length // 2
    for t in range(-half, half + 1):
        y = int(round(half + t * np.sin(angle)))
        x = int(round(half + t * np.cos(angle)))
        psf[y, x] += 1.0  # accumulate: diagonal steps can share a pixel
    return psf / psf.sum()


def _psf_fft(psf, shape):
    """FFT of ``psf`` (centred kernel) padded to ``shape``."""
    padded = np.zeros(shape)
    h, w = psf.shape
    padded[:h, :w] = psf
    # Roll so the kernel centre lands on the origin (circular conv).
    padded = np.roll(padded, shift=(-(h // 2), -(w // 2)), axis=(0, 1))
    return np.fft.fft2(padded)


def apply_blur(image, psf):
    """Blur ``image`` with ``psf`` (circular convolution via FFT)."""
    image = _check_image(image)
    psf = _check_psf(psf)
    if psf.shape[0] > image.shape[0] or psf.shape[1] > image.shape[1]:
        raise ValueError(
            f"psf {psf.shape} must not be larger than image {image.shape}")
    H = _psf_fft(psf, image.shape)
    blurred = np.fft.ifft2(np.fft.fft2(image) * H)
    return np.real(blurred)


def estimate_nsr(noise_var, signal_var):
    """Noise-to-signal power ratio from noise and signal variances."""
    noise_var, signal_var = float(noise_var), float(signal_var)
    if noise_var < 0:
        raise ValueError(f"noise_var must be non-negative, got {noise_var!r}")
    if signal_var <= 0:
        raise ValueError(f"signal_var must be positive, got {signal_var!r}")
    return noise_var / signal_var


def wiener_deconvolve(blurred, psf, nsr=0.01):
    """Restore ``blurred`` given the blur ``psf`` and ``nsr``.

    ``nsr`` is the noise-to-signal power ratio: larger values
    regularise more (less sharpening, less noise amplification).
    Use ``nsr=0`` only for noise-free images — it reduces to plain
    inverse filtering.
    Returns a float image of the same shape (not clipped).
    """
    blurred = _check_image(blurred, name="blurred")
    psf = _check_psf(psf)
    if psf.shape[0] > blurred.shape[0] or psf.shape[1] > blurred.shape[1]:
        raise ValueError(
            f"psf {psf.shape} must not be larger than image {blurred.shape}")
    if nsr < 0:
        raise ValueError(f"nsr must be non-negative, got {nsr!r}")
    H = _psf_fft(psf, blurred.shape)
    G = np.fft.fft2(blurred)
    F = np.conj(H) / (np.abs(H) ** 2 + float(nsr)) * G
    return np.real(np.fft.ifft2(F))


def psnr(original, restored, data_range=None):
    """Peak signal-to-noise ratio in dB between two images."""
    original = _check_image(original, name="original")
    restored = np.asarray(restored, dtype=float)
    if restored.shape != original.shape:
        raise ValueError(
            f"restored must have the same shape as original, got {restored.shape} "
            f"vs {original.shape}")
    if data_range is None:
        data_range = float(original.max() - original.min()) or 1.0
    mse = float(np.mean((original - restored) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))
