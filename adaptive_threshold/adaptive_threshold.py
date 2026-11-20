"""Adaptive (local) thresholding.

A global threshold fails when illumination varies across an image.
Adaptive thresholding compares each pixel against the local mean (or
a Gaussian-weighted local mean) of its neighbourhood minus a constant
``C``, so dark and bright regions are binarised independently.  A
comparison helper against a plain global threshold is included.
Only numpy is used.
"""

import numpy as np


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


def _validate_params(block_size, c, method):
    if not isinstance(block_size, (int, np.integer)) \
            or block_size < 3 or block_size % 2 == 0:
        raise ValueError(
            f"block_size must be an odd integer >= 3, got {block_size}"
        )
    if c < 0:
        raise ValueError(f"C must be >= 0, got {c}")
    if method not in ("mean", "gaussian"):
        raise ValueError(
            f"method must be 'mean' or 'gaussian', got {method!r}"
        )


def _local_mean(image, block_size):
    """Mean of each ``block_size`` x ``block_size`` neighbourhood.

    Uses an integral image; windows are clipped at the image border
    and divided by their actual (smaller) pixel count.
    """
    pad = block_size // 2
    h, w = image.shape
    integral = np.pad(
        np.cumsum(np.cumsum(image, axis=0), axis=1), ((1, 0), (1, 0))
    )
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    r0 = np.clip(rows - pad, 0, h - 1)
    r1 = np.clip(rows + pad, 0, h - 1)
    c0 = np.clip(cols - pad, 0, w - 1)
    c1 = np.clip(cols + pad, 0, w - 1)
    sums = (integral[r1 + 1, c1 + 1] - integral[r0, c1 + 1]
            - integral[r1 + 1, c0] + integral[r0, c0])
    counts = (r1 - r0 + 1) * (c1 - c0 + 1)
    return sums / counts


def _gaussian_kernel_1d(sigma):
    """1D Gaussian kernel with radius ceil(3 * sigma)."""
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def _gaussian_blur(image, sigma):
    """Separable Gaussian blur (reflect padding), same size out."""
    kernel = _gaussian_kernel_1d(sigma)
    radius = len(kernel) // 2
    padded = np.pad(image, ((radius, radius), (0, 0)), mode="reflect")
    blurred_rows = np.stack(
        [np.convolve(padded[i, :], kernel, mode="valid")
         for i in range(padded.shape[0])]
    )
    padded = np.pad(blurred_rows, ((0, 0), (radius, radius)), mode="reflect")
    blurred = np.stack(
        [np.convolve(padded[:, j], kernel, mode="valid")
         for j in range(padded.shape[1])]
    , axis=1)
    return blurred


def adaptive_threshold(image, block_size=11, c=2.0, method="mean"):
    """Binarise ``image`` against a local threshold.

    Parameters
    ----------
    image : (H, W) array
        Grayscale image.
    block_size : int
        Odd side length of the neighbourhood; must be >= 3.
    c : float
        Constant subtracted from the local mean; must be >= 0.
    method : {"mean", "gaussian"}
        "mean" uses the plain neighbourhood mean, "gaussian" a
        Gaussian-weighted local mean with sigma = block_size / 6.

    Returns
    -------
    numpy.ndarray
        Boolean array, True where the pixel is above its local
        threshold (``image > local_mean - c``).
    """
    image = _validate_grayscale(image)
    _validate_params(block_size, c, method)
    if method == "mean":
        local = _local_mean(image, block_size)
    else:
        local = _gaussian_blur(image, sigma=block_size / 6.0)
    return image > (local - c)


def global_threshold(image, method="mean"):
    """Binarise ``image`` against one global threshold.

    Parameters
    ----------
    image : (H, W) array
        Grayscale image.
    method : {"mean", "median"}
        Which statistic of the whole image to threshold against.

    Returns
    -------
    numpy.ndarray
        Boolean array, True where the pixel is above the threshold.
    """
    image = _validate_grayscale(image)
    if method == "mean":
        threshold = float(image.mean())
    elif method == "median":
        threshold = float(np.median(image))
    else:
        raise ValueError(
            f"method must be 'mean' or 'median', got {method!r}"
        )
    return image > threshold


def compare_thresholds(image, block_size=11, c=2.0, method="mean"):
    """Run adaptive and global thresholding side by side.

    Returns a dict with both boolean masks plus foreground pixel
    counts and the number of pixels where the two masks disagree,
    so the benefit of the local threshold is easy to see.
    """
    image = _validate_grayscale(image)
    adaptive = adaptive_threshold(image, block_size, c, method)
    glob = global_threshold(image)
    return {
        "adaptive": adaptive,
        "global": glob,
        "adaptive_foreground": int(adaptive.sum()),
        "global_foreground": int(glob.sum()),
        "difference": int((adaptive != glob).sum()),
    }
