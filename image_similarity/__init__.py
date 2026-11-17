"""Image similarity by perceptual hashing.

Exposes :func:`average_hash`, :func:`difference_hash`,
:func:`hamming_distance` and :func:`find_most_similar` from
:mod:`image_similarity.image_similarity`.
"""

from image_similarity.image_similarity import (
    average_hash,
    difference_hash,
    find_most_similar,
    hamming_distance,
)

__all__ = [
    "average_hash",
    "difference_hash",
    "hamming_distance",
    "find_most_similar",
]
