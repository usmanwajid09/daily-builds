"""Image inpainting by iterative diffusion, implemented from scratch.

Small holes (masked-out regions) are filled by repeatedly replacing
each unknown pixel with the average of its four known neighbours.
This is the discrete Laplace equation solved by Jacobi iteration:
the filled region smoothly interpolates the surrounding known values,
so a hole in a gradient is recovered as a smooth continuation of that
gradient instead of a flat patch.

Both grayscale (2-D) and RGB (3-D, ``H x W x 3``) images are
supported. The mask is a 2-D boolean array where ``True`` marks the
pixels to fill. Unknown pixels start at the mean of the known pixels
(per channel) and the iteration stops when the largest per-pixel
change drops below ``tolerance`` or ``max_iterations`` is reached.
"""

import numpy as np

__all__ = ["inpaint"]


def _validate(image, mask):
    """Check shapes/dtypes and return (image, mask) as clean arrays."""
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2-D (grayscale) or 3-D (H x W x C), got {image.ndim}-D"
        )
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"colour images must have 1, 3 or 4 channels, got {image.shape[2]}"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"image must be numeric, got dtype {image.dtype}")
    mask = np.asarray(mask)
    if mask.dtype != bool:
        raise ValueError(f"mask must be a boolean array, got dtype {mask.dtype}")
    if mask.ndim != 2 or mask.shape != image.shape[:2]:
        raise ValueError(
            f"mask must be 2-D with shape {image.shape[:2]}, got shape {mask.shape}"
        )
    if not np.any(~mask):
        raise ValueError("mask covers the whole image: no known pixels to diffuse from")
    return image, mask


def inpaint(image, mask, max_iterations=1000, tolerance=1e-3):
    """Fill the masked region of ``image`` by iterative diffusion.

    Args:
        image: 2-D grayscale or 3-D colour array (any numeric dtype).
        mask: 2-D boolean array, same height/width as ``image``;
            ``True`` marks pixels to reconstruct.
        max_iterations: hard cap on diffusion sweeps.
        tolerance: stop once the largest per-pixel change in one sweep
            is below this value.

    Returns:
        Array of the same shape and dtype as ``image`` with the masked
        pixels filled in. Known pixels are returned unchanged (up to
        dtype rounding). Use :func:`inpaint_with_info` when the
        iteration count and convergence flag are needed too.

    Raises:
        ValueError: bad shapes/dtypes, an empty image, or a mask that
            covers every pixel.
    """
    result, _iterations, _converged = inpaint_with_info(
        image, mask, max_iterations=max_iterations, tolerance=tolerance
    )
    return result


def inpaint_with_info(image, mask, max_iterations=1000, tolerance=1e-3):
    """Like :func:`inpaint`, but also returns iteration diagnostics.

    Returns:
        ``(filled, iterations, converged)`` where ``filled`` is the
        inpainted array, ``iterations`` the number of sweeps run, and
        ``converged`` whether the change dropped below ``tolerance``.
    """
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    image, mask = _validate(image, mask)

    work = image.astype(np.float64)
    known = ~mask
    if not np.any(mask):
        return image.copy(), 0, True

    # Work channel-by-channel so grayscale and colour share one path.
    channels = [work] if work.ndim == 2 else [work[:, :, c] for c in range(work.shape[2])]

    iterations = 0
    converged = False
    for iterations in range(1, max_iterations + 1):
        max_change = 0.0
        for channel in channels:
            # Initialise unknown pixels to the known mean on the first sweep.
            if iterations == 1:
                channel[mask] = channel[known].mean()
            # Neighbour sums/counts from the four direct neighbours.
            padded = np.pad(channel, 1, mode="edge")
            neighbours = (
                padded[:-2, 1:-1] + padded[2:, 1:-1]
                + padded[1:-1, :-2] + padded[1:-1, 2:]
            )
            # A neighbour on the image edge re-uses the edge pixel
            # (edge padding), so every unknown pixel always has four
            # values to average; border pixels just weigh the edge more.
            new_values = neighbours[mask] / 4.0
            change = np.abs(new_values - channel[mask]).max(initial=0.0)
            max_change = max(max_change, float(change))
            channel[mask] = new_values
        if max_change < tolerance:
            converged = True
            break

    if work.ndim == 2:
        filled = work
    else:
        filled = np.stack(channels, axis=2)

    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        filled = np.clip(np.rint(filled), info.min, info.max).astype(image.dtype)
    else:
        filled = filled.astype(image.dtype)
    return filled, iterations, converged
