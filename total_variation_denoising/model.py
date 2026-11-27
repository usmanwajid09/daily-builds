"""Total-variation denoising via the Chambolle projection algorithm
(numpy only).

Minimises  |grad u| + (1 / (2 * weight)) * ||u - f||^2  -- i.e. the
image is smoothed where it is flat but sharp edges are preserved,
because the total-variation penalty grows linearly (not quadratically)
with the gradient. Chambolle (2004) solves it in the dual: iterate a
projected gradient ascent on the dual field p, then u = f - weight *
div p.

Public API: tv_denoise, psnr.
"""

import numpy as np

__all__ = ["tv_denoise", "psnr"]


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


def _gradient(field):
    """Forward differences: returns (gy, gx), zero on the last row/col."""
    gy = np.zeros_like(field)
    gx = np.zeros_like(field)
    gy[:-1, :] = field[1:, :] - field[:-1, :]
    gx[:, :-1] = field[:, 1:] - field[:, :-1]
    return gy, gx


def _divergence(py, px):
    """Backward differences (adjoint of _gradient)."""
    div = np.zeros_like(px)
    div[1:, :] += py[1:, :] - py[:-1, :]
    div[0, :] += py[0, :]
    div[:, 1:] += px[:, 1:] - px[:, :-1]
    div[:, 0] += px[:, 0]
    return div


def tv_denoise(image, weight=15.0, max_iter=200, tol=1e-3):
    """Denoise a grayscale image; return a float64 image in [0, 255].

    ``weight`` is the denoising weight (lambda in Chambolle's
    notation): larger values smooth more aggressively (the data term
    is (1 / (2 * weight)) * ||u - f||^2), smaller values stay closer
    to the noisy input. ``tau`` is fixed at 0.125, the stability
    limit for the 2D dual iteration.
    """
    if weight <= 0:
        raise ValueError(f"weight must be positive, got {weight}")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be a positive int, got {max_iter}")
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")

    noisy = _validate_gray(image)
    tau = 0.125
    px = np.zeros_like(noisy)
    py = np.zeros_like(noisy)
    denoised = noisy.copy()
    for _ in range(max_iter):
        div_p = _divergence(py, px)
        residual = div_p - noisy / weight
        gy, gx = _gradient(residual)
        norm = np.sqrt(gx ** 2 + gy ** 2)
        denom = 1.0 + tau * norm
        px = (px + tau * gx) / denom
        py = (py + tau * gy) / denom
        new = noisy - weight * _divergence(py, px)
        change = float(np.max(np.abs(new - denoised)))
        denoised = new
        if change < tol:
            break
    return np.clip(denoised, 0.0, 255.0)


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
