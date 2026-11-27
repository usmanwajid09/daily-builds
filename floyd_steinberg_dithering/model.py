"""Floyd-Steinberg error-diffusion dithering (numpy only).

Converts a grayscale image to black-and-white (or an RGB image to a
small palette) by quantizing each pixel in scan order and pushing the
quantization error forward to its not-yet-visited neighbours with the
classic Floyd-Steinberg weights:

    right 7/16, down-left 3/16, down 5/16, down-right 1/16.

Serpentine scanning (alternating left-to-right / right-to-left rows)
is supported and reduces directional streaking artefacts.

Public API: dither_grayscale, dither_color.
"""

import numpy as np

__all__ = ["dither_grayscale", "dither_color"]


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


def _validate_rgb(image):
    """Validate an RGB image; return float64 in [0, 255]."""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"image must be a numpy array, got {type(image).__name__}"
        )
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must be 3D RGB (H, W, 3); got shape {image.shape}"
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


def _quantize_levels(value, levels):
    """Nearest palette value for ``value`` with ``levels`` evenly spaced
    levels in [0, 255] (levels=2 gives {0, 255})."""
    if levels == 2:
        return 255.0 if value >= 127.5 else 0.0
    step = 255.0 / (levels - 1)
    return float(np.clip(round(value / step), 0, levels - 1) * step)


# (dy, dx, weight) for a left-to-right pass; mirrored for right-to-left.
_FORWARD_KERNEL = ((0, 1, 7 / 16), (1, -1, 3 / 16),
                   (1, 0, 5 / 16), (1, 1, 1 / 16))
_REVERSE_KERNEL = ((0, -1, 7 / 16), (1, 1, 3 / 16),
                   (1, 0, 5 / 16), (1, -1, 1 / 16))


def _diffuse(work, height, width, y, x, error, serpentine, forward):
    """Push ``error`` (scalar or 3-vector) to not-yet-visited neighbours."""
    kernel = _FORWARD_KERNEL if forward else _REVERSE_KERNEL
    for dy, dx, weight in kernel:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            work[ny, nx] += error * weight


def dither_grayscale(image, serpentine=True):
    """Floyd-Steinberg dither a grayscale image to binary uint8 (0/255).

    The mean brightness of the output closely matches the input (error
    diffusion preserves the local average), so gradients render as
    dot patterns instead of hard bands.
    """
    img = _validate_gray(image)
    height, width = img.shape
    work = img.copy()
    out = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        forward = True
        if serpentine and y % 2 == 1:
            forward = False
        xs = range(width) if forward else range(width - 1, -1, -1)
        for x in xs:
            old = work[y, x]
            new = 255.0 if old >= 127.5 else 0.0
            out[y, x] = int(new)
            _diffuse(work, height, width, y, x, old - new,
                     serpentine, forward)
    return out


def dither_color(image, levels=2, serpentine=True):
    """Floyd-Steinberg dither an RGB image to ``levels`` values per
    channel (levels=2 gives the 8 corners of the RGB cube). The
    quantization error is diffused per channel independently.

    Returns a uint8 RGB image whose channel values are palette values.
    """
    if not isinstance(levels, (int, np.integer)) or levels < 2:
        raise ValueError(f"levels must be an int >= 2, got {levels}")
    img = _validate_rgb(image)
    height, width, _ = img.shape
    work = img.copy()
    out = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        forward = True
        if serpentine and y % 2 == 1:
            forward = False
        xs = range(width) if forward else range(width - 1, -1, -1)
        for x in xs:
            old = work[y, x].copy()
            new = np.array([_quantize_levels(v, levels) for v in old])
            out[y, x] = new.astype(np.uint8)
            _diffuse(work, height, width, y, x, old - new,
                     serpentine, forward)
    return out
