"""Perceptual-hash image similarity from scratch (numpy only).

Average hash (aHash) and difference hash (dHash) turn an image into a
short bit string that captures its coarse structure. Two images that
look alike get hashes that differ in only a few bits, so the Hamming
distance between hashes is a cheap similarity score — the basis of
near-duplicate detection in photo libraries and reverse image search.

Public API
----------
average_hash(image, hash_size=8) -> int
difference_hash(image, hash_size=8) -> int
hamming_distance(hash1, hash2) -> int
find_most_similar(query, images, method="average") -> (index, distance)
"""

import numpy as np


def _validate_hash_size(hash_size):
    if isinstance(hash_size, bool) or not isinstance(hash_size, (int, np.integer)):
        raise ValueError(
            f"hash_size must be an integer >= 2, got {hash_size!r}"
        )
    if hash_size < 2:
        raise ValueError(
            f"hash_size must be at least 2, got {hash_size}"
        )


def _to_grayscale(image):
    """Validate an image and return it as a 2D float64 grayscale array."""
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"image must be a 2D grayscale or 3D RGB array, got {arr.ndim}D "
            f"array with shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("image must not be empty")
    if arr.dtype.kind not in "fiu":
        raise ValueError(
            f"image must have a numeric dtype, got {arr.dtype}"
        )
    gray = arr.astype(np.float64)
    if not np.all(np.isfinite(gray)):
        raise ValueError("image must not contain NaN or infinite values")
    if gray.ndim == 3:
        if gray.shape[2] == 1:
            gray = gray[:, :, 0]
        elif gray.shape[2] == 3:
            gray = (
                0.299 * gray[:, :, 0]
                + 0.587 * gray[:, :, 1]
                + 0.114 * gray[:, :, 2]
            )
        else:
            raise ValueError(
                "3D image must have 1 or 3 channels, "
                f"got {gray.shape[2]} channels"
            )
    return gray


def _resize_block_average(gray, out_h, out_w):
    """Resize a 2D array to (out_h, out_w) by simple block averaging.

    Each output pixel is the mean of the input block it covers, so the
    resize works for any input size, not only exact multiples.
    """
    in_h, in_w = gray.shape
    row_edges = np.linspace(0, in_h, out_h + 1).astype(int)
    col_edges = np.linspace(0, in_w, out_w + 1).astype(int)
    out = np.empty((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        r0, r1 = row_edges[i], max(row_edges[i + 1], row_edges[i] + 1)
        for j in range(out_w):
            c0, c1 = col_edges[j], max(col_edges[j + 1], col_edges[j] + 1)
            out[i, j] = gray[r0:r1, c0:c1].mean()
    return out


def _bits_to_int(bits):
    """Pack a flat bool array (row-major) into an int, MSB first."""
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bool(bit))
    return value


def average_hash(image, hash_size=8):
    """Average hash (aHash) of an image, returned as an int.

    The image is shrunk to ``hash_size`` x ``hash_size`` grayscale and
    each pixel becomes one bit: 1 when brighter than the mean pixel,
    0 otherwise. With the default ``hash_size=8`` this is a 64-bit hash.
    """
    _validate_hash_size(hash_size)
    gray = _to_grayscale(image)
    small = _resize_block_average(gray, hash_size, hash_size)
    bits = small > small.mean()
    return _bits_to_int(bits)


def difference_hash(image, hash_size=8):
    """Difference hash (dHash) of an image, returned as an int.

    The image is shrunk to ``hash_size`` x ``hash_size + 1`` grayscale
    and each bit records whether a pixel is brighter than its right-hand
    neighbour, capturing horizontal gradients. With the default
    ``hash_size=8`` this is a 64-bit hash.
    """
    _validate_hash_size(hash_size)
    gray = _to_grayscale(image)
    small = _resize_block_average(gray, hash_size, hash_size + 1)
    bits = small[:, :-1] > small[:, 1:]
    return _bits_to_int(bits)


def _validate_hash(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be a non-negative integer hash, got {value!r}"
        )
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}")


def hamming_distance(hash1, hash2):
    """Number of bit positions where two integer hashes differ."""
    _validate_hash(hash1, "hash1")
    _validate_hash(hash2, "hash2")
    return bin(int(hash1) ^ int(hash2)).count("1")


def _hash_with_method(image, method):
    if method == "average":
        return average_hash(image)
    if method == "difference":
        return difference_hash(image)
    raise ValueError(
        f"method must be 'average' or 'difference', got {method!r}"
    )


def find_most_similar(query, images, method="average"):
    """Find the image most similar to ``query`` by perceptual hash.

    Parameters
    ----------
    query : array-like
        The image to match (2D grayscale or 3D RGB).
    images : sequence of array-like
        Candidate images; must not be empty.
    method : {"average", "difference"}
        Which hash to compare with.

    Returns
    -------
    (index, distance) : tuple[int, int]
        Index of the closest image in ``images`` and its Hamming
        distance to the query hash (0 = identical hash).
    """
    if images is None or len(images) == 0:
        raise ValueError("images must be a non-empty sequence of images")
    query_hash = _hash_with_method(query, method)
    best_index, best_distance = 0, None
    for i, candidate in enumerate(images):
        distance = hamming_distance(
            query_hash, _hash_with_method(candidate, method)
        )
        if best_distance is None or distance < best_distance:
            best_index, best_distance = i, distance
    return best_index, best_distance
