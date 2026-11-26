"""LSB image steganography: hide a UTF-8 message in the least significant
bits of a uint8 image (grayscale 2D or RGB 3D).

Payload format (written LSB-first across the flattened image):

    magic  : b"LSB1"            (4 bytes)
    length : message byte count (4 bytes, big-endian)
    data   : message bytes     (UTF-8)

Embedding only ever flips the lowest bit of a value, so no pixel value
ever changes by more than 1.
"""

import numpy as np

MAGIC = b"LSB1"
_HEADER_BYTES = len(MAGIC) + 4  # magic + length field = 8 bytes
_HEADER_BITS = _HEADER_BYTES * 8


def _validate_image(image):
    """Return the image after checking it is a non-empty uint8 2D/3D array."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise TypeError(f"image must have dtype uint8, got {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2D (grayscale) or 3D (RGB), got {image.ndim}D"
        )
    if image.size == 0:
        raise ValueError("image must be non-empty")
    return image


def capacity_bytes(image):
    """Return how many message bytes fit in ``image``.

    floor((total values - header bits) / 8), where the header is 8 bytes
    (4-byte magic + 4-byte length). Never negative: an image too small for
    even the header has capacity 0.
    """
    _validate_image(image)
    return max(0, (image.size - _HEADER_BITS) // 8)


def embed_message(image, message):
    """Embed ``message`` (str) into the LSBs of ``image``.

    Returns a new uint8 array of the same shape; the input is not modified.
    Raises ValueError if the message is too long for the image capacity,
    TypeError if ``message`` is not a str.
    """
    _validate_image(image)
    if not isinstance(message, str):
        raise TypeError(f"message must be a str, got {type(message).__name__}")

    payload = MAGIC + len(message.encode("utf-8")).to_bytes(4, "big") \
        + message.encode("utf-8")
    n_bits = len(payload) * 8
    if n_bits > image.size:
        raise ValueError(
            f"message of {len(payload) - _HEADER_BYTES} bytes needs "
            f"{n_bits} bits but the image only holds {image.size} bits "
            f"(capacity {capacity_bytes(image)} message bytes)"
        )

    # LSB-first bits of every payload byte, as a uint8 array of 0s and 1s.
    bits = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8), bitorder="little"
    )
    flat = image.reshape(-1).copy()
    flat[:n_bits] = (flat[:n_bits] & 0xFE) | bits
    return flat.reshape(image.shape)


def extract_message(image):
    """Extract the hidden message from ``image``.

    Raises ValueError if the magic bytes are wrong/absent, if the declared
    length exceeds the image capacity, or if the payload is truncated.
    """
    _validate_image(image)
    flat = image.reshape(-1)
    if flat.size < _HEADER_BITS:
        raise ValueError(
            "image too small to hold an LSB1 header "
            f"({flat.size} values < {_HEADER_BITS} bits)"
        )

    bits = (flat & 1).astype(np.uint8)
    header = np.packbits(bits[:_HEADER_BITS], bitorder="little").tobytes()
    magic, length_bytes = header[:4], header[4:]
    if magic != MAGIC:
        raise ValueError(
            f"magic bytes mismatch: expected {MAGIC!r}, got {magic!r} "
            "(no hidden message in this image)"
        )
    length = int.from_bytes(length_bytes, "big")
    if length > capacity_bytes(image):
        raise ValueError(
            f"declared message length {length} bytes exceeds capacity "
            f"{capacity_bytes(image)} bytes"
        )
    n_bits = _HEADER_BITS + length * 8
    if n_bits > flat.size:
        raise ValueError(
            f"payload truncated: declared {length} bytes needs {n_bits} bits "
            f"but the image only holds {flat.size} bits"
        )

    payload = np.packbits(bits[_HEADER_BITS:n_bits], bitorder="little").tobytes()
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"payload is not valid UTF-8: {exc}") from exc


def lsb_difference(image_a, image_b):
    """Return the count of values whose least significant bit differs.

    Both images must be uint8 arrays of the same shape; raises
    TypeError/ValueError otherwise.
    """
    _validate_image(image_a)
    _validate_image(image_b)
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"images must have the same shape, got {image_a.shape} "
            f"vs {image_b.shape}"
        )
    return int(np.count_nonzero((image_a & 1) != (image_b & 1)))
