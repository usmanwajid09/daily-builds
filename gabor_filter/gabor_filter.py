"""Gabor filter bank: orientation/scale-selective texture analysis.

A 2D Gabor filter is a sinusoidal plane wave multiplied by a Gaussian
envelope.  A bank of filters at several orientations and wavelengths
responds strongly to edges and textures aligned with each filter, so
the responses (and their local energy) are classic texture features.
Only numpy is used; convolution is done with sliding windows.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _validate_grayscale(image):
    """Return ``image`` as float64, raising ValueError on bad input."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            f"expected a 2D grayscale image, got shape {image.shape}"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"image dtype {image.dtype} is not numeric")
    return image.astype(np.float64)


def gabor_kernel(ksize, sigma, theta, wavelength, gamma=0.5, psi=0.0):
    """Build one 2D Gabor kernel.

    Parameters
    ----------
    ksize : int
        Kernel size in pixels; must be odd and at least 3.
    sigma : float
        Standard deviation of the Gaussian envelope; must be > 0.
    theta : float
        Orientation of the filter in radians (0 = vertical stripes).
    wavelength : float
        Wavelength of the sinusoidal carrier in pixels; must be > 0.
    gamma : float
        Spatial aspect ratio (ellipticity); must be > 0.
    psi : float
        Phase offset of the carrier in radians.

    Returns
    -------
    numpy.ndarray
        Zero-mean ``(ksize, ksize)`` float64 kernel.
    """
    if not isinstance(ksize, (int, np.integer)) or ksize < 3 or ksize % 2 == 0:
        raise ValueError(f"ksize must be an odd integer >= 3, got {ksize}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if wavelength <= 0:
        raise ValueError(f"wavelength must be > 0, got {wavelength}")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(
        -(x_theta ** 2 + gamma ** 2 * y_theta ** 2) / (2.0 * sigma ** 2)
    )
    carrier = np.cos(2.0 * np.pi * x_theta / wavelength + psi)
    kernel = envelope * carrier
    kernel = kernel - kernel.mean()  # remove the DC component
    kernel = kernel / np.sum(np.abs(kernel))  # L1 normalisation
    return kernel


def gabor_bank(n_orientations=8, wavelengths=(4.0, 8.0), ksize=21,
               sigma=4.0, gamma=0.5):
    """Build a bank of Gabor kernels.

    Parameters
    ----------
    n_orientations : int
        Number of orientations, evenly spaced over [0, pi); must be >= 1.
    wavelengths : sequence of float
        One wavelength per scale; each must be > 0.
    ksize, sigma, gamma
        Passed to :func:`gabor_kernel`.

    Returns
    -------
    list of numpy.ndarray
        ``len(wavelengths) * n_orientations`` kernels, ordered
        scale-major (all orientations of scale 0, then scale 1, ...).
    """
    if not isinstance(n_orientations, (int, np.integer)) or n_orientations < 1:
        raise ValueError(
            f"n_orientations must be an integer >= 1, got {n_orientations}"
        )
    wavelengths = tuple(float(w) for w in wavelengths)
    if len(wavelengths) == 0:
        raise ValueError("wavelengths must contain at least one value")
    if any(w <= 0 for w in wavelengths):
        raise ValueError(f"all wavelengths must be > 0, got {wavelengths}")

    kernels = []
    for wavelength in wavelengths:
        for i in range(n_orientations):
            theta = i * np.pi / n_orientations
            kernels.append(
                gabor_kernel(ksize, sigma, theta, wavelength, gamma)
            )
    return kernels


def _convolve2d(image, kernel):
    """Convolve a 2D image with a kernel (reflect padding, same size)."""
    kh, kw = kernel.shape
    padded = np.pad(image, ((kh // 2, kh // 2), (kw // 2, kw // 2)),
                    mode="reflect")
    windows = sliding_window_view(padded, (kh, kw))
    flipped = kernel[::-1, ::-1]  # correlation -> true convolution
    return np.tensordot(windows, flipped, axes=((2, 3), (0, 1)))


def apply_gabor_bank(image, kernels):
    """Filter ``image`` with every kernel in ``kernels``.

    Parameters
    ----------
    image : (H, W) array
        Grayscale image.
    kernels : sequence of 2D arrays
        Gabor kernels, e.g. from :func:`gabor_bank`.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(H, W, len(kernels))`` with one
        response map per kernel.
    """
    image = _validate_grayscale(image)
    kernels = list(kernels)
    if len(kernels) == 0:
        raise ValueError("kernels must contain at least one kernel")
    for kernel in kernels:
        kernel = np.asarray(kernel)
        if kernel.ndim != 2:
            raise ValueError(
                f"each kernel must be 2D, got shape {kernel.shape}"
            )
    responses = [_convolve2d(image, np.asarray(k, dtype=np.float64))
                 for k in kernels]
    return np.stack(responses, axis=-1)


def gabor_responses(image, n_orientations=8, wavelengths=(4.0, 8.0),
                    ksize=21, sigma=4.0, gamma=0.5):
    """Build a bank and filter ``image`` with it (convenience wrapper)."""
    kernels = gabor_bank(n_orientations, wavelengths, ksize, sigma, gamma)
    return apply_gabor_bank(image, kernels)


def _box_filter(image, window):
    """Uniform (box) filter via an integral image, same size out."""
    if window == 1:
        return image.copy()
    pad = window // 2
    padded = np.pad(image, pad, mode="reflect")
    integral = np.pad(
        np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0))
    )
    h, w = image.shape
    sums = (integral[window:window + h, window:window + w]
            - integral[:h, window:window + w]
            - integral[window:window + h, :w]
            + integral[:h, :w])
    return sums / float(window * window)


def texture_energy(responses, window=9):
    """Local texture energy: RMS of each response over a window.

    Parameters
    ----------
    responses : (H, W, K) array
        Gabor response maps, e.g. from :func:`apply_gabor_bank`.
    window : int
        Odd side length of the local averaging window; must be >= 1.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(H, W, K)`` with the local RMS energy
        of every response map.
    """
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 3:
        raise ValueError(
            f"expected responses of shape (H, W, K), got {responses.shape}"
        )
    if responses.size == 0:
        raise ValueError("responses must not be empty")
    if not isinstance(window, (int, np.integer)) or window < 1 \
            or window % 2 == 0:
        raise ValueError(f"window must be an odd integer >= 1, got {window}")
    energy = np.empty_like(responses)
    for k in range(responses.shape[2]):
        energy[:, :, k] = np.sqrt(
            _box_filter(responses[:, :, k] ** 2, window)
        )
    return energy
