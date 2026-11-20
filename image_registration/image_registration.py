"""Translation image registration via phase correlation.

Phase correlation estimates the integer (optionally subpixel) shift
between two images from the phase of their cross-power spectrum: the
inverse FFT of the normalised cross-power spectrum peaks exactly at
the translation between the images.  Parabolic interpolation refines
the peak to subpixel accuracy.  Only numpy is used.
"""

import numpy as np


def _validate_grayscale(image, name="image"):
    """Return ``image`` as float64, raising ValueError on bad input."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D grayscale image, got shape {image.shape}"
        )
    if image.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"{name} dtype {image.dtype} is not numeric")
    return image.astype(np.float64)


def _cross_power_spectrum(reference, moving):
    """Normalised cross-power spectrum of two same-shape images."""
    f_ref = np.fft.fft2(reference)
    f_mov = np.fft.fft2(moving)
    product = f_ref * np.conj(f_mov)
    magnitude = np.abs(product)
    return product / np.maximum(magnitude, 1e-12)


def _parabolic_offset(corr, peak_row, peak_col):
    """Subpixel refinement of a correlation peak (parabolic fit).

    Fits a parabola through the peak and its two neighbours along
    each axis and returns the fractional offset in (-1, 1) per axis.
    """
    h, w = corr.shape

    def axis_offset(centre, minus, plus):
        denom = minus - 2.0 * centre + plus
        if denom == 0.0:
            return 0.0
        offset = 0.5 * (minus - plus) / denom
        return float(np.clip(offset, -1.0, 1.0))

    row_offset = axis_offset(
        corr[peak_row, peak_col],
        corr[(peak_row - 1) % h, peak_col],
        corr[(peak_row + 1) % h, peak_col],
    )
    col_offset = axis_offset(
        corr[peak_row, peak_col],
        corr[peak_row, (peak_col - 1) % w],
        corr[peak_row, (peak_col + 1) % w],
    )
    return row_offset, col_offset


def phase_correlation(reference, moving, subpixel=False):
    """Estimate the translation of ``moving`` relative to ``reference``.

    Parameters
    ----------
    reference, moving : (H, W) arrays
        Grayscale images of the same shape.
    subpixel : bool
        If True, refine the peak with parabolic interpolation.

    Returns
    -------
    (dy, dx) : tuple of float
        Shift such that ``moving`` is approximately ``reference``
        rolled by ``(dy, dx)`` (positive dy = down, positive dx =
        right).  Integer-valued unless ``subpixel`` is True.
    peak : float
        Height of the correlation peak (0 to 1; near 1 for a clean
        pure translation).
    """
    reference = _validate_grayscale(reference, "reference")
    moving = _validate_grayscale(moving, "moving")
    if reference.shape != moving.shape:
        raise ValueError(
            f"reference and moving must have the same shape, got "
            f"{reference.shape} and {moving.shape}"
        )
    if np.all(reference == reference.flat[0]) \
            or np.all(moving == moving.flat[0]):
        raise ValueError(
            "phase correlation needs images with some variation; "
            "a constant image has no phase information"
        )

    spectrum = _cross_power_spectrum(reference, moving)
    corr = np.fft.fftshift(np.fft.ifft2(spectrum).real)
    h, w = corr.shape
    peak_row, peak_col = np.unravel_index(np.argmax(corr), corr.shape)
    peak = float(corr[peak_row, peak_col])
    # The correlation peak sits at -shift (mod N): the inverse FFT of
    # exp(+2*pi*i*k*s/N) is a delta at index -s.
    dy = float(h // 2 - peak_row)
    dx = float(w // 2 - peak_col)
    if subpixel:
        row_off, col_off = _parabolic_offset(corr, peak_row, peak_col)
        dy -= row_off
        dx -= col_off
    return (dy, dx), peak


def estimate_shift(reference, moving, subpixel=False):
    """Estimate the translation of ``moving`` relative to ``reference``.

    Convenience wrapper around :func:`phase_correlation` returning
    only the ``(dy, dx)`` shift.
    """
    shift, _ = phase_correlation(reference, moving, subpixel=subpixel)
    return shift


def apply_shift(image, dy, dx):
    """Shift an image by ``(dy, dx)``, wrapping around the edges.

    Integer shifts use ``np.roll`` (exact); fractional shifts use
    the Fourier shift theorem.
    """
    image = _validate_grayscale(image)
    dy = float(dy)
    dx = float(dx)
    if dy == int(dy) and dx == int(dx):
        return np.roll(image, (int(dy), int(dx)), axis=(0, 1))
    h, w = image.shape
    freq_y = np.fft.fftfreq(h)[:, None]
    freq_x = np.fft.fftfreq(w)[None, :]
    phase = np.exp(-2j * np.pi * (freq_y * dy + freq_x * dx))
    # The Nyquist frequency is ambiguous for fractional shifts: the
    # bins at +-0.5 cycles/px are the same bin, so a complex phase
    # there would break the spectrum's Hermitian symmetry (and the
    # shifted image would gain an imaginary part).  The bandlimited
    # value is real: cos(pi * shift) for the Nyquist axis.
    if h % 2 == 0:
        phase[h // 2, :] = (np.cos(np.pi * dy)
                            * np.exp(-2j * np.pi * freq_x * dx))
    if w % 2 == 0:
        phase[:, w // 2] = (np.exp(-2j * np.pi * freq_y[:, 0] * dy)
                            * np.cos(np.pi * dx))
    if h % 2 == 0 and w % 2 == 0:
        phase[h // 2, w // 2] = np.cos(np.pi * dy) * np.cos(np.pi * dx)
    return np.fft.ifft2(np.fft.fft2(image) * phase).real


def register_translation(reference, moving, subpixel=False):
    """Register ``moving`` onto ``reference`` by pure translation.

    Parameters
    ----------
    reference, moving : (H, W) arrays
        Grayscale images of the same shape.
    subpixel : bool
        If True, refine the shift estimate to subpixel accuracy.

    Returns
    -------
    dict
        ``{"shift": (dy, dx), "registered": array, "peak": float}``,
        where ``registered`` is ``moving`` shifted back onto
        ``reference`` and ``peak`` is the correlation peak height.
    """
    shift, peak = phase_correlation(reference, moving, subpixel=subpixel)
    registered = apply_shift(moving, -shift[0], -shift[1])
    return {"shift": shift, "registered": registered, "peak": peak}
