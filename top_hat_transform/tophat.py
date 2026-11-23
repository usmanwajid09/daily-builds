"""Top-hat morphological transforms, implemented from scratch (numpy only).

White top-hat  = image - opening(image):  extracts bright details
smaller than the structuring element (e.g. small bright spots on a
dark background). Black top-hat = closing(image) - image: extracts
dark details smaller than the structuring element.
"""

import numpy as np


def _validate_image(image):
    """Return ``image`` as a float64 2D array or raise ValueError."""
    arr = np.asarray(image)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"image must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"image must be a 2D grayscale array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("image must not be empty")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("image must contain only finite values")
    return arr


def _validate_kernel_size(kernel_size):
    """Return ``kernel_size`` as an odd positive int or raise ValueError."""
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, (int, np.integer)):
        raise ValueError(f"kernel_size must be an odd positive integer, got {kernel_size!r}")
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be an odd positive integer, got {kernel_size}")
    return kernel_size


def structuring_element(kernel_size, shape="square"):
    """Return a boolean structuring element of the given size and shape.

    Args:
        kernel_size: odd positive integer (side length).
        shape: "square" (all True) or "disk" (True inside the inscribed
            circle, False in the corners).
    """
    kernel_size = _validate_kernel_size(kernel_size)
    if shape not in ("square", "disk"):
        raise ValueError(f"shape must be 'square' or 'disk', got {shape!r}")
    element = np.ones((kernel_size, kernel_size), dtype=bool)
    if shape == "disk":
        radius = kernel_size // 2
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        element = (xx ** 2 + yy ** 2) <= radius ** 2
    return element


def _morph(image, element, mode):
    """Erode (mode='erode', min) or dilate (mode='dilate', max) an image."""
    radius = element.shape[0] // 2
    padded = np.pad(image, radius, mode="edge")
    out = np.empty_like(image)
    offsets = np.argwhere(element) - radius
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighbourhood = [padded[i + di + radius, j + dj + radius] for di, dj in offsets]
            values = np.array(neighbourhood)
            out[i, j] = values.min() if mode == "erode" else values.max()
    return out


def erode(image, kernel_size=3, shape="square"):
    """Grayscale erosion with a square or disk structuring element."""
    image = _validate_image(image)
    element = structuring_element(kernel_size, shape)
    return _morph(image, element, "erode")


def dilate(image, kernel_size=3, shape="square"):
    """Grayscale dilation with a square or disk structuring element."""
    image = _validate_image(image)
    element = structuring_element(kernel_size, shape)
    return _morph(image, element, "dilate")


def opening(image, kernel_size=3, shape="square"):
    """Opening: erosion followed by dilation (removes small bright spots)."""
    return dilate(erode(image, kernel_size, shape), kernel_size, shape)


def closing(image, kernel_size=3, shape="square"):
    """Closing: dilation followed by erosion (removes small dark spots)."""
    return erode(dilate(image, kernel_size, shape), kernel_size, shape)


def white_tophat(image, kernel_size=3, shape="square"):
    """White top-hat: ``image - opening(image)`` (bright details only)."""
    image = _validate_image(image)
    return image - opening(image, kernel_size, shape)


def black_tophat(image, kernel_size=3, shape="square"):
    """Black top-hat: ``closing(image) - image`` (dark details only)."""
    image = _validate_image(image)
    return closing(image, kernel_size, shape) - image


def top_hat(image, kernel_size=3, shape="square"):
    """Return ``(white_tophat, black_tophat)`` for a grayscale image."""
    image = _validate_image(image)
    return (white_tophat(image, kernel_size, shape),
            black_tophat(image, kernel_size, shape))
