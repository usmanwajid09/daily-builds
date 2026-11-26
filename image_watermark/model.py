"""Visible and invisible image watermarking, implemented with NumPy only.

Visible watermark
-----------------
A white checkerboard block pattern (``block_size`` x ``block_size`` squares,
alternating white / transparent) is tiled over a corner region of the image
and blended in at the given ``alpha``. The corner region is
``4 * block_size`` pixels on each side (clipped to the image size), so with
the default ``block_size=8`` it covers a 32x32 corner. Only the white
squares change the image; the transparent squares leave the original
pixels untouched, which is what makes the pattern read as a checkerboard
rather than a solid white patch.

Invisible watermark
-------------------
The key is hashed with SHA-256 (``hashlib.sha256``) and embedded, prefixed
by a 2-byte magic marker (``b"WM"``), into the least-significant bits of
the image pixels (row-major order, channel order for RGB). The 272-bit
pattern (16 magic bits + 256 digest bits) is repeated as many whole times
as fit, so a large image carries several copies; verification checks the
first copy. Changing an LSB moves a pixel value by at most 1, so the
watermark is invisible to the eye (PSNR is typically well above 40 dB).
An image smaller than 272 pixels cannot hold even one copy and raises
``ValueError``.
"""

import hashlib
import math

import numpy as np

__all__ = [
    "add_visible_watermark",
    "embed_invisible_watermark",
    "verify_invisible_watermark",
    "psnr",
]

_POSITIONS = ("bottom-right", "bottom-left", "top-right", "top-left")

# 2-byte magic prefix marking the start of an embedded pattern.
_MAGIC = b"WM"
# SHA-256 digest is 32 bytes = 256 bits; plus 16 magic bits.
_PATTERN_BITS = (len(_MAGIC) + hashlib.sha256(b"").digest_size) * 8  # 272


def _validate_image(image):
    """Return ``image`` after checking it is a non-empty uint8 array.

    Raises TypeError for a non-ndarray or non-uint8 input, and ValueError
    for a wrong number of dimensions or an empty image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"image must be a numpy array, got {type(image).__name__}"
        )
    if image.dtype != np.uint8:
        raise TypeError(
            f"image must have dtype uint8, got {image.dtype}"
        )
    if image.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2D (grayscale) or 3D (RGB), got {image.ndim}D"
        )
    if image.size == 0:
        raise ValueError("image must be non-empty")
    return image


def _validate_key(key):
    """Return ``key`` after checking it is a non-empty str."""
    if not isinstance(key, str):
        raise TypeError(f"key must be a str, got {type(key).__name__}")
    if not key:
        raise ValueError("key must be a non-empty string")
    return key


def _key_bits(key):
    """Bit array (0/1, MSB first) of magic prefix + SHA-256(key)."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    payload = _MAGIC + digest
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8))


def _corner_slices(shape, position, block_size):
    """Row/column slices of the watermarked corner region."""
    height, width = shape[0], shape[1]
    region_h = min(4 * block_size, height)
    region_w = min(4 * block_size, width)
    if position.startswith("top"):
        rows = slice(0, region_h)
    else:  # bottom
        rows = slice(height - region_h, height)
    if position.endswith("right"):
        cols = slice(width - region_w, width)
    else:  # left
        cols = slice(0, region_w)
    return rows, cols


def add_visible_watermark(image, alpha=0.3, position="bottom-right",
                          block_size=8):
    """Blend a white checkerboard watermark into one corner of an image.

    Parameters
    ----------
    image : numpy uint8 array, 2D (grayscale) or 3D (RGB).
    alpha : float in (0, 1]; blending strength of the white squares.
        ``alpha=1`` makes the white squares pure white (255).
    position : one of "bottom-right", "bottom-left", "top-right",
        "top-left".
    block_size : positive int; side of one checkerboard square in pixels.
        The watermarked corner region is ``4 * block_size`` pixels square
        (clipped to the image).

    Returns a new uint8 array of the same shape; the input is unchanged.
    Pixels outside the corner region are copied exactly.
    """
    _validate_image(image)
    if isinstance(alpha, bool) or not isinstance(
        alpha, (int, float, np.floating)
    ):
        raise TypeError(f"alpha must be a number, got {type(alpha).__name__}")
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if position not in _POSITIONS:
        raise ValueError(
            f"position must be one of {_POSITIONS}, got {position!r}"
        )
    if isinstance(block_size, bool) or not isinstance(
        block_size, (int, np.integer)
    ):
        raise TypeError(
            f"block_size must be an int, got {type(block_size).__name__}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    out = image.astype(np.float64).copy()
    rows, cols = _corner_slices(image.shape, position, block_size)
    region_h = rows.stop - rows.start
    region_w = cols.stop - cols.start

    # Checkerboard mask over the corner region: True on white squares.
    rr, cc = np.mgrid[0:region_h, 0:region_w]
    white = ((rr // block_size) + (cc // block_size)) % 2 == 0

    region = out[rows, cols]
    if image.ndim == 3:
        # Expand the 2D mask to (h, w, channels) so boolean indexing matches.
        white = np.broadcast_to(white[:, :, None], region.shape)
    # Blend only the white squares towards 255; other squares untouched.
    region[white] = (1.0 - alpha) * region[white] + alpha * 255.0
    out[rows, cols] = region
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def embed_invisible_watermark(image, key):
    """Embed an invisible LSB watermark derived from ``key``.

    The SHA-256 digest of the key (with a ``b"WM"`` magic prefix) is
    written into the least-significant bits of the image pixels and the
    272-bit pattern is repeated as many whole times as fit. Returns a new
    uint8 array of the same shape; each pixel changes by at most 1.

    Raises ValueError if the image has fewer than 272 pixels and cannot
    hold even one copy of the pattern.
    """
    _validate_image(image)
    _validate_key(key)
    if image.size < _PATTERN_BITS:
        raise ValueError(
            f"image too small for an invisible watermark: needs at least "
            f"{_PATTERN_BITS} pixels, got {image.size}"
        )

    bits = _key_bits(key)
    n_copies = image.size // _PATTERN_BITS
    pattern = np.tile(bits, n_copies)

    flat = image.reshape(-1).copy()
    # Clear the LSB, then set it to the pattern bit.
    flat[: pattern.size] = (flat[: pattern.size] & np.uint8(0xFE)) | pattern.astype(np.uint8)
    return flat.reshape(image.shape)


def verify_invisible_watermark(image, key):
    """Return True if the invisible watermark for ``key`` is present.

    Returns False for a wrong key or an unwatermarked image (and for an
    image too small to hold a pattern). Raises TypeError/ValueError for
    invalid inputs, as in :func:`embed_invisible_watermark`.
    """
    _validate_image(image)
    _validate_key(key)
    if image.size < _PATTERN_BITS:
        return False
    bits = (image.reshape(-1)[:_PATTERN_BITS] & np.uint8(1)).astype(np.uint8)
    return bool(np.array_equal(bits, _key_bits(key)))


def psnr(image_a, image_b):
    """Peak signal-to-noise ratio in dB between two uint8 images.

    Returns ``float("inf")`` for identical images. Raises ValueError if
    the shapes differ, and TypeError/ValueError for invalid inputs.
    """
    _validate_image(image_a)
    _validate_image(image_b)
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"shape mismatch: {image_a.shape} vs {image_b.shape}"
        )
    mse = float(np.mean((image_a.astype(np.float64) - image_b.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / mse)
