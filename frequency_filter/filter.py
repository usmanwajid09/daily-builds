"""FFT-based frequency-domain image filtering (numpy only).

Filter an image in the frequency domain: take the 2D FFT, multiply by a
mask that keeps or removes frequency bands, then inverse-FFT back to
pixels. Low-pass keeps smooth structure and drops fine detail / noise;
high-pass keeps edges and drops the smooth background.
"""

import numpy as np

_KINDS = ("ideal", "gaussian")


def _validate(image, cutoff, kind):
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError(
            f"image must be a 2D (grayscale) or 3D (RGB) array, got {image.ndim}D"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.ndim == 3 and image.shape[2] == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(f"cutoff must be > 0, got {cutoff}")
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    return image


def make_mask(shape, cutoff, kind="gaussian"):
    """Return a frequency-domain mask of shape ``shape`` (2D).

    The mask is centred (matches ``np.fft.fftshift`` output): 1 at DC,
    falling off with distance from the centre. ``cutoff`` is the
    radius (in pixels / frequency bins) at which the mask drops to
    0.5 for Gaussian, or the hard radius for the ideal mask.

    * ``'ideal'`` — binary: 1 inside the cutoff radius, 0 outside.
    * ``'gaussian'`` — ``exp(-(r^2) / (2 * cutoff^2))``, smooth, no ringing.
    """
    if len(shape) != 2:
        raise ValueError(f"shape must be a 2D (rows, cols) tuple, got {shape}")
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"shape must be non-empty, got {shape}")
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(f"cutoff must be > 0, got {cutoff}")
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")

    rows, cols = shape
    cy, cx = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    if kind == "ideal":
        return (dist <= cutoff).astype(float)
    return np.exp(-(dist ** 2) / (2.0 * cutoff ** 2))


def _filter_channel(channel, cutoff, kind, invert):
    spectrum = np.fft.fftshift(np.fft.fft2(channel))
    mask = make_mask(channel.shape, cutoff, kind)
    if invert:
        mask = 1.0 - mask
    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * mask))
    return np.real(filtered)


def _apply(image, cutoff, kind, invert):
    image = _validate(image, cutoff, kind)
    image = image.astype(float)
    if image.ndim == 2:
        return _filter_channel(image, cutoff, kind, invert)
    out = np.empty_like(image)
    for c in range(image.shape[2]):
        out[..., c] = _filter_channel(image[..., c], cutoff, kind, invert)
    return out


def low_pass(image, cutoff, kind="gaussian"):
    """Keep frequencies within ``cutoff`` of DC; drop the rest.

    Returns a float array with the same shape as ``image``.
    """
    return _apply(image, cutoff, kind, invert=False)


def high_pass(image, cutoff, kind="gaussian"):
    """Drop frequencies within ``cutoff`` of DC; keep the rest.

    The result has (near) zero mean for smooth images — the DC / smooth
    background component is removed.
    """
    return _apply(image, cutoff, kind, invert=True)


def band_pass(image, low_cutoff, high_cutoff, kind="gaussian"):
    """Keep frequencies between ``low_cutoff`` and ``high_cutoff``."""
    if not np.isfinite(low_cutoff) or low_cutoff <= 0:
        raise ValueError(f"low_cutoff must be > 0, got {low_cutoff}")
    if not np.isfinite(high_cutoff) or high_cutoff <= 0:
        raise ValueError(f"high_cutoff must be > 0, got {high_cutoff}")
    if high_cutoff <= low_cutoff:
        raise ValueError(
            f"high_cutoff ({high_cutoff}) must be > low_cutoff ({low_cutoff})"
        )
    image = _validate(image, low_cutoff, kind)
    image = image.astype(float)
    if image.ndim == 2:
        # band-pass = low-pass at high_cutoff minus low-pass at low_cutoff
        return _filter_channel(image, high_cutoff, kind, invert=False) - _filter_channel(
            image, low_cutoff, kind, invert=False
        )
    out = np.empty_like(image)
    for c in range(image.shape[2]):
        out[..., c] = (
            _filter_channel(image[..., c], high_cutoff, kind, invert=False)
            - _filter_channel(image[..., c], low_cutoff, kind, invert=False)
        )
    return out


__all__ = ["make_mask", "low_pass", "high_pass", "band_pass"]
